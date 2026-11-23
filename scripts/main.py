"""Run the reactive chess controller with Raspberry Pi sensors and motor board."""
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
from uuid import uuid4

from chessbot.input.camera import Camera
from chessbot.execution.drive import Drive, STOP
from chessbot.input.imu import Imu
from chessbot.execution.motor import Motors
from chessbot.runtime import Runtime, SensorFrame

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def write(log, **record):
    log.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")


def config_path(value):
    return HERE / Path(value).expanduser()


def run(mode, settings, cfg, camera_profile, drive, model):
    from chessbot.perception.vision import Vision

    test = mode == "test"
    mode_settings = settings["modes"][mode]
    seconds = mode_settings["seconds"]
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    precision = settings["yolo_precision"]
    threads = settings["yolo_threads"]
    if type(threads) is not int or threads < 1:
        raise ValueError("yolo_threads must be a positive integer")
    vision = Vision(cfg, camera_profile, model, precision=precision, threads=threads)
    preview = recorder = stepper = None
    if test:
        from chessbot.test_mode import TestStepper
        stepper = TestStepper()
    controller = Runtime(cfg, vision, drive, settings["max_camera_age_s"],
                         settings["max_imu_age_s"], mode_settings["one_cycle"], stepper)
    cam = Camera(settings["camera_device"], settings["camera_fps_requested"])
    imu = Imu(settings["imu_device"], settings["imu_baud"], settings["R_body_imu"])
    output = config_path(settings["log_dir"]) / f"{datetime.now():%Y%m%d_%H%M%S}_{mode}_{uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    output.joinpath("config_snapshot.json").write_text(
        json.dumps(dict(mode=mode, drive=not test, pi=settings, policy=cfg, camera=camera_profile,
                        calibration=load(config_path(settings["wheel_profile"])), model_dir=str(model),
                        yolo_precision=precision, yolo_threads=threads),
                   indent=2, ensure_ascii=False) + "\n")

    if test:
        print(f"TEST | Motors off | Ctrl+C to stop | Log: {output}", flush=True)
    reason = "interrupted"
    sequence = -1
    last_tick = -math.inf
    start = None
    skip_pending = False
    startup_deadline = time.monotonic() + settings["startup_timeout_s"]
    with (output / "ticks.jsonl").open("x", buffering=1) as log, ExitStack() as stack:
        cam.start()
        stack.callback(cam.close)
        imu.start()
        stack.callback(imu.close)
        motors = None
        try:
            if test:
                from chessbot.preview import Preview
                preview = Preview(settings["preview"])
                stack.callback(preview.close)
            elif mode == "record-run":
                from chessbot.recorder import Recorder
                recorder = stack.enter_context(Recorder(output, settings["recording"]))
            while not stop.is_set():
                if preview:
                    skip, exit_reason = preview.poll()
                    skip_pending = skip_pending or skip
                    if exit_reason:
                        reason = exit_reason
                        break
                image = cam.latest(after=sequence, timeout=settings["camera_poll_timeout_s"])
                inertial = imu.latest(after=last_tick)
                now = time.monotonic()
                camera_age = math.inf if image is None else now-image.time
                imu_age = math.inf if inertial is None else now-inertial.time
                if start is None:
                    if (camera_age > settings["max_camera_age_s"]
                            or imu_age > settings["max_imu_age_s"]):
                        if now >= startup_deadline:
                            raise RuntimeError("camera or stationary IMU startup timed out")
                        continue
                    start = now
                    if not test and not stop.is_set():
                        motors = stack.enter_context(Motors(settings["motor_command_timeout_s"]))
                if seconds is not None and now-start >= seconds:
                    reason = "time_limit"
                    break
                if (imu_age > settings["max_imu_age_s"]
                        or (image is None and now-last_tick > settings["max_camera_age_s"])):
                    raise RuntimeError("sensor stream expired")
                if image is None:
                    continue
                sequence = image.sequence
                frame = SensorFrame(image.time, inertial.time, image.bgr, inertial.yaw_rad,
                                    inertial.gyro_z_rad_s, inertial.acceleration,
                                    inertial.horizontal_rms)
                old_events = len(controller.policy.events)
                vel, pwm, obs = controller.tick(now, frame, skip=skip_pending)
                skip_pending = False
                finished = time.monotonic()
                if motors is not None:
                    if (finished-image.time > settings["max_camera_age_s"]
                            or finished-inertial.time > settings["max_imu_age_s"]):
                        motors.apply(STOP)
                        raise RuntimeError("sensor observation expired during inference")
                    motors.apply(STOP if stop.is_set() else pwm)
                last_tick = now
                state = controller.policy.state
                if obs:
                    if preview:
                        preview.show(controller.preview_image, obs, state)
                    if recorder:
                        recorder.add(image.time, controller.preview_image, obs, state)
                write(log, time=now, elapsed_s=now-start, camera_time=image.time,
                      camera_sequence=sequence, imu_time=inertial.time,
                      processing_ms=(finished-now)*1000, state=state,
                      magnetic_yaw_rad=inertial.magnetic_yaw_rad,
                      observation=asdict(obs) if obs else None, velocity=vel, pi_pwm=pwm,
                      events=controller.policy.events[old_events:])
                if controller.stop_reason:
                    reason = controller.stop_reason
                    break
        except (RuntimeError, OSError, ValueError) as error:
            reason = f"fault: {error}"
        finally:
            write(log, time=time.monotonic(), stop_reason=reason, pi_pwm=STOP)
    failed = reason.startswith("fault:") or reason in ("policy_fault", "invalid_or_stale_sensor")
    if test or failed:
        print(f"Stopped: {reason} | Log: {output}", file=sys.stderr if failed else sys.stdout, flush=True)
    return int(failed)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("test", "run", "record-run"):
        print("Usage: chess test | chess run | chess record-run", file=sys.stderr)
        return 2
    mode = sys.argv[1]
    try:
        settings = load(HERE / "config/pi.local.json")
        seconds = settings["modes"][mode]["seconds"]
        if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("mode seconds must be positive or null")
        drive = Drive(config_path(settings["wheel_profile"]),
                      settings["pwm_55_wheel_speed_m_s_provisional"],
                      settings["yaw_arm_m_provisional"])
        return run(mode, settings, load(config_path(settings["policy_profile"])),
                   load(config_path(settings["camera_profile"])), drive,
                   config_path(settings["model_dir"]))
    except (RuntimeError, OSError, ValueError, TypeError, KeyError, ImportError) as error:
        print(f"Chess {mode}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
