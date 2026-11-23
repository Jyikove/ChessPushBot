"""Pi I²C motor functions from the archived keyboard script, plus command expiry."""
import threading
import time

BOARD_ADDRESS = 0x2B
MOTOR_REGISTER = 0x01
STOPPED = (0, 0, 0, 0)


def write_wheels(bus, speeds):
    for motor, speed in enumerate(speeds):
        bus.write_i2c_block_data(
            BOARD_ADDRESS, MOTOR_REGISTER, [motor, int(speed < 0), abs(speed)]
        )


def stop_wheels(bus):
    pending = list(range(4))
    for _ in range(3):
        failed = []
        for motor in pending:
            try:
                bus.write_i2c_block_data(BOARD_ADDRESS, MOTOR_REGISTER, [motor, 0, 0])
            except OSError:
                failed.append(motor)
        if not failed:
            return
        pending = failed
        time.sleep(0.02)
    raise RuntimeError(f"Could not stop motors {pending}. Switch off the car.")


class Motors:
    def __init__(self, timeout):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.deadline = None
        self.expired = False
        self.error = None
        self.last = STOPPED

    def __enter__(self):
        import smbus

        self.bus = smbus.SMBus(1)
        try:
            stop_wheels(self.bus)
            self.watchdog = threading.Thread(target=self.watch, name="pwm-watchdog", daemon=True)
            self.watchdog.start()
        except BaseException:
            self.bus.close()
            raise
        return self

    def __exit__(self, *_):
        self.done.set()
        self.watchdog.join()
        try:
            stop_wheels(self.bus)
        finally:
            self.bus.close()

    def watch(self):
        while not self.done.wait(.02):
            with self.lock:
                if self.deadline is not None and time.monotonic() >= self.deadline:
                    self.expired = True
                    self.deadline = None
                    try:
                        stop_wheels(self.bus)
                    except Exception as error:
                        self.error = error

    def apply(self, pwm):
        with self.lock:
            if self.error:
                raise RuntimeError(str(self.error)) from self.error
            if self.expired or (self.deadline is not None and time.monotonic() >= self.deadline):
                self.expired = True
                self.deadline = None
                stop_wheels(self.bus)
                raise RuntimeError("motor command expired")
            if pwm != self.last:
                if pwm == STOPPED:
                    stop_wheels(self.bus)
                else:
                    write_wheels(self.bus, pwm)
                self.last = pwm
            self.deadline = None if pwm == STOPPED else time.monotonic() + self.timeout
