"""Run a 120-second two-car match and optionally swap policies on the same layout."""
import argparse
import base64
from contextlib import ExitStack
import datetime
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread

os.environ.setdefault("MUJOCO_GL","glfw")
import cv2
import numpy as np

SIM = Path(__file__).resolve().parent
sys.path.insert(0,str(SIM/"runtime"))
from core import validate_action
from duel_physics import CAMERA_STEPS, CONTROL_STEPS, HZ, HEADINGS, STARTS, DuelPhysics
from field_spec import FIELD_DEPTH_M, FIELD_WIDTH_M, bounds
from room_appearance import PRESETS


class Worker:
    def __init__(self, policy, log, timeout, checkpoint):
        self.policy=policy
        self.timeout=timeout
        self.checkpoint=checkpoint
        self.log=log
        self.responses=Queue()
        self.requests=Queue()

    def __enter__(self):
        self.proc=subprocess.Popen([sys.executable,"-B",str(SIM/"runtime/duel_worker.py"),str(self.policy)],
                                   cwd=self.policy.parent,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                   stderr=self.log,text=True)
        def drain():
            for line in self.proc.stdout:
                self.responses.put(line)
            self.responses.put(None)
        self.reader=Thread(target=drain,daemon=True)
        self.reader.start()
        def send_pending():
            while (line:=self.requests.get()) is not None:
                try:
                    self.proc.stdin.write(line)
                    self.proc.stdin.flush()
                except (BrokenPipeError,ValueError):
                    self.responses.put(None)
                    break
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        self.writer=Thread(target=send_pending,daemon=True)
        self.writer.start()
        return self

    def __exit__(self,*_):
        self.requests.put(None)
        try:
            self.proc.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.reader.join(timeout=1.)
        self.writer.join(timeout=1.)
        self.proc.stdout.close()

    def send(self,value):
        self.requests.put(json.dumps(value,separators=(",",":"))+"\n")

    def receive(self):
        try:
            reply=self.responses.get(timeout=self.timeout)
        except Empty as error:
            raise RuntimeError(f"Strategy timed out: {self.policy}; see its log") from error
        if reply is None:
            raise RuntimeError(f"Strategy exited: {self.policy}; see its log")
        return json.loads(reply)


def revision(policy):
    def read(*args):
        return subprocess.run(["git","-C",str(policy.parent),*args],capture_output=True,text=True)
    head=read("rev-parse","HEAD")
    if head.returncode:
        return dict(file=str(policy),sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),git_commit=None)
    root=Path(read("rev-parse","--show-toplevel").stdout.strip())
    status=read("status","--short").stdout.splitlines()
    return dict(file=str(policy),sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                git_commit=head.stdout.strip(),git_root=str(root),git_dirty=bool(status))


def encode(image):
    ok,png=cv2.imencode(".png",image[:,:,::-1],[cv2.IMWRITE_PNG_COMPRESSION,1])
    if not ok:
        raise RuntimeError("Could not encode onboard camera frame")
    return base64.b64encode(png).decode("ascii")


def panel(images,overview,t,points):
    top=np.concatenate([cv2.resize(im,(480,360)) for im in images],axis=1)
    canvas=np.zeros((750,960,3),dtype=np.uint8)
    canvas[:360]=top
    canvas[360:720,240:720]=cv2.resize(overview,(480,360))
    for x,label in ((12,"A camera"),(492,"B camera")):
        cv2.putText(canvas,label,(x,26),0,.7,(31,224,255),2,cv2.LINE_AA)
    cv2.putText(canvas,f"{t:05.1f} s    A {points['A']['points']} - {points['B']['points']} B",(300,742),
                0,.65,(242,242,242),1,cv2.LINE_AA)
    return np.ascontiguousarray(canvas[:,:,::-1])


def play(args, policies, name, report, videos, camera):
    directory=report/name
    directory.mkdir()
    env=DuelPhysics(camera,args.seed,args.appearance)
    try:
        with ExitStack() as stack:
            trace=stack.enter_context((directory/"controls.jsonl").open("w"))
            workers=[]
            for side, policy in zip("AB",policies,strict=True):
                log=stack.enter_context((directory/f"strategy_{side}.log").open("w"))
                workers.append(stack.enter_context(Worker(policy,log,args.policy_timeout,revision(policy))))
            for side, worker in zip("AB",workers,strict=True):
                worker.send(dict(protocol_version=1,side=side,starting_xy_m=STARTS[side][:2],
                                 starting_yaw_rad=HEADINGS[side],goal_xyxy_m=bounds("goal_"+side),
                                 field_size_m=[FIELD_WIDTH_M,FIELD_DEPTH_M],camera=camera,
                                 motor_order=["FL","FR","RL","RR"]))
            faults={}
            for side, worker in zip("AB",workers,strict=True):
                try:
                    if worker.receive()!={"ready":True}:
                        raise ValueError("Invalid worker handshake")
                except (ValueError,RuntimeError) as error:
                    faults[side]=dict(time_s=env.time,error=str(error))
            encoder=None
            if args.video:
                videos.mkdir(parents=True,exist_ok=True)
                encoder=stack.enter_context(subprocess.Popen([
                    "ffmpeg","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s","960x750","-r","5",
                    "-i","-","-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p",
                    str(videos/f"{name}.mp4")],stdin=subprocess.PIPE))
                stack.callback(encoder.stdin.close)
            for tick in range(120*HZ//CONTROL_STEPS):
                fresh=env.steps%CAMERA_STEPS==0
                images=[env.render(camera=cam) for cam in ("onboard","onboard_B")] if fresh else [None,None]
                for slot,(side,worker) in enumerate(zip("AB",workers,strict=True)):
                    if side not in faults:
                        worker.send(env.sensor_packet(slot,encode(images[slot]) if fresh else None))
                actions=[]
                for side,worker in zip("AB",workers,strict=True):
                    action={"motor_pwm":[0]*4}
                    if side not in faults:
                        try:
                            candidate=worker.receive()
                            validate_action(candidate)
                            action=candidate
                        except (ValueError,RuntimeError) as error:
                            faults[side]=dict(time_s=env.time,error=str(error))
                    actions.append(action)
                trace.write(json.dumps(dict(t=env.time,reset_id=env.epoch,actions=actions))+"\n")
                if encoder and fresh:
                    encoder.stdin.write(panel(images,env.render(True),env.time,env.scoreboard()).tobytes())
                env.controls(actions)
                if (tick+1)%300==0:
                    print(name,f"{env.time:.0f}s",flush=True)
            assert env.time==120.
            at_horn=env.scoreboard()
            settle_seconds,settled=env.settle()
            terminal=env.scoreboard()
            final_a,final_b=(terminal[s]["points"] for s in "AB")
            result=dict(seed=args.seed,appearance=args.appearance,net_seconds=env.time,
                        strategies={side:w.checkpoint for side,w in zip("AB",workers,strict=True)},
                        at_horn=at_horn,terminal=terminal,settled=settled,settle_seconds=settle_seconds,
                        status="verified" if settled else "unsettled",
                        winner=("A" if final_a>final_b else "B" if final_b>final_a else "draw") if settled else None,
                        collisions=env.collisions,contact_seconds=env.contact_steps/HZ,
                        strategy_faults=faults,
                        referee_resets=env.resets,minimum_car_center_distance_m=env.min_distance,
                        layout_xy_m=env.layout)
        if encoder and encoder.returncode:
            raise RuntimeError("ffmpeg failed")
    finally:
        env.close()
    (directory/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(name,final_a,final_b,"collisions",len(result["collisions"]),result["status"],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a",type=Path,default=SIM/"policies/near_side.py")
    parser.add_argument("--b",type=Path,default=SIM/"policies/behind_pile.py")
    parser.add_argument("--seed",type=int,default=77)
    parser.add_argument("--appearance",choices=list(PRESETS),default="overcast")
    parser.add_argument("--policy-timeout",type=float,default=2.)
    parser.add_argument("--swap",action="store_true")
    parser.add_argument("--video",action="store_true")
    args=parser.parse_args()
    if args.policy_timeout<=0 or not np.isfinite(args.policy_timeout):
        parser.error("policy timeout must be positive and finite")
    policies=tuple(p.resolve(strict=True) for p in (args.a,args.b))
    cv2.setNumThreads(1)
    tag=datetime.datetime.now().strftime(f"duel-{args.seed}-%Y%m%d-%H%M%S-%f")
    artifacts=Path(os.environ.get("SIM_ARTIFACTS_ROOT",SIM.parent.parent/"三维仿真")).resolve()
    if artifacts.is_relative_to(SIM.parent):
        parser.error("simulation outputs must be outside the repository")
    report,video=artifacts/"报告"/tag,artifacts/"视频"/tag
    report.mkdir(parents=True)
    camera=json.loads((SIM/"resources/camera/camera_profile.json").read_text())
    runs=[play(args,policies,"leg1",report,video,camera)]
    if args.swap:
        runs.append(play(args,policies[::-1],"leg2",report,video,camera))
    totals={}
    if all(r["settled"] for r in runs):
        for i,label in enumerate(("first_policy","second_policy")):
            totals[label]=sum(r["terminal"]["AB"[(i+leg)%2]]["points"] for leg,r in enumerate(runs))
    meta=dict(engine=revision(SIM/"duel.py"),mujoco_version=sys.modules["mujoco"].__version__,
              legs=runs,policy_totals=totals)
    (report/"match.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n")
    print(report,totals,flush=True)


if __name__=="__main__":
    main()
