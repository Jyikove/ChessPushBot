"""Map a body-frame command to the simulated motor order FL, FR, RL, RR."""
import numpy as np

from core import WHEEL


def motor_action(translation, turn):
    raw = (WHEEL @ np.r_[translation, .11 * turn]) * (237. / .32)
    pwm = np.where(abs(raw) >= 3., raw + 18 * np.sign(raw), 0.)
    return {"motor_pwm": np.rint(np.clip(pwm, -255, 255)).astype(int).tolist()}
