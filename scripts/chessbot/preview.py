"""Shared camera overlay and desktop view for chess test."""
import math
import os
from pathlib import Path

import cv2
import numpy as np

IMAGE_SIZE = (640, 480)
STATUS_HEIGHT = 38
FRAME_SIZE = (IMAGE_SIZE[0], IMAGE_SIZE[1] + STATUS_HEIGHT)


def text(frame, label, x, y, color=(245, 247, 252), scale=.50):
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (18, 22, 28), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def render(bgr, obs, state):
    frame = cv2.resize(bgr, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    width, height = frame.shape[1], frame.shape[0]

    def point(uv):
        return tuple(round(v*s) for v, s in zip(uv, (width-1, height-1)))

    if obs is not None:
        for piece in obs.pieces:
            box = piece.display_box or piece.box
            a, b = point(box[:2]), point(box[2:])
            cv2.rectangle(frame, a, b, (85, 225, 76), 2)
            text(frame, f"{piece.confidence:.2f}", a[0], max(15, a[1]-4), (85, 225, 76))
        if obs.fence is not None:
            a, b = map(point, obs.fence.endpoints_px)
            cv2.line(frame, a, b, (25, 205, 255), 3, cv2.LINE_AA)
            text(frame, "Fence", max(a[0], 5), max(a[1]-6, 20), (25, 205, 255))
        if obs.zone is not None:
            if obs.zone.line_px is not None:
                a, b = map(point, obs.zone.line_px)
                cv2.line(frame, a, b, (255, 198, 40), 3, cv2.LINE_AA)
            x, y = point(obs.zone.entry_px)
            cv2.circle(frame, (x, y), 6, (255, 198, 40), 2)
            text(frame, "Score entry", max(x-70, 5), max(y-10, 20), (255, 198, 40))
        if obs.start_zone is not None:
            x, y = point(obs.start_zone.marker_px)
            cv2.drawMarker(frame, (x, y), (255, 190, 235), cv2.MARKER_CROSS, 14, 2)
            text(frame, "Start", max(x-40, 5), max(y-10, 20), (255, 190, 235))
    panel = np.full((STATUS_HEIGHT, width, 3), (18, 22, 28), dtype=np.uint8)
    yaw = "--" if obs is None else f"{math.degrees(obs.yaw):+.1f} deg"
    text(panel, f"State: {state}", 12, 25, scale=.58)
    text(panel, f"Yaw: {yaw}", 452, 25, scale=.58)
    return np.vstack((frame, panel))


class Preview:
    def __init__(self, settings):
        self.title = settings["title"]
        os.environ.setdefault("DISPLAY", settings["display"])
        os.environ.setdefault("XAUTHORITY", str(Path(settings["xauthority"]).expanduser()))
        os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pygame = pygame
        try:
            pygame.display.init()
            pygame.display.set_mode(settings["window_size"], pygame.RESIZABLE)
            pygame.display.set_caption(self.title)
        except pygame.error as error:
            pygame.display.quit()
            raise RuntimeError(f"cannot open test window: {error}") from error
        self.show(np.zeros((320, 416, 3), dtype=np.uint8), None, "INITIALIZE")

    def show(self, frame, obs, state):
        pygame = self.pygame
        rgb = cv2.cvtColor(render(frame, obs, state), cv2.COLOR_BGR2RGB)
        image = pygame.image.frombuffer(rgb.data, FRAME_SIZE, "RGB")
        screen = pygame.display.get_surface()
        width, height = screen.get_size()
        scale = min(width / FRAME_SIZE[0], height / FRAME_SIZE[1])
        size = tuple(max(1, round(dimension * scale)) for dimension in FRAME_SIZE)
        screen.fill((18, 22, 28))
        screen.blit(pygame.transform.scale(image, size),
                    ((width-size[0])//2, (height-size[1])//2))
        pygame.display.flip()

    def poll(self):
        pygame = self.pygame
        skip = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, "window_closed"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c and event.mod & pygame.KMOD_CTRL:
                    return False, "interrupted"
                skip |= event.key == pygame.K_x
        return skip, None

    def close(self):
        self.pygame.display.quit()
