"""Onboard disk ranks from the tracked detector and score CNN."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from shape_runtime import Detector, postprocess, prepare


class PieceValues:
    def __init__(self):
        assets = Path(__file__).resolve().parents[3] / "sim" / "resources"
        model = assets / "score" / "score_cnn.onnx"
        meta = json.loads(model.with_suffix(".json").read_text(encoding="utf-8"))
        if hashlib.sha256(model.read_bytes()).hexdigest() != meta["onnx_sha256"]:
            raise ValueError("Score CNN does not match its metadata")
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.detector = Detector(assets / "detector", (416, 320), threads=1)
        self.classes = meta["classes"]
        self.threshold = meta["min_score_confidence"]
        self.top = meta["view_top_fractions"][0]

    def observe(self, rgb):
        rows, cols = rgb.shape[:2]
        raw, transform = prepare(rgb[:, :, ::-1].copy(), self.detector.wh)
        detections = postprocess(self.detector.forward(raw)[0], transform, conf=.30, iou=.55)
        # Train and infer on the same top section of the 416-pixel-wide view.
        view = cv2.resize(rgb, (416, 312), interpolation=cv2.INTER_LINEAR)
        crops, boxes = [], []
        for d in detections:
            x0, y0, x1, y1 = d["box"]
            if not (1 < x0 < x1 < cols - 1 and 1 < y0 < y1 < rows - 1 and
                    x1 - x0 > 20 and y1 - y0 > 9):
                continue
            a, c = int(x0 * 416 / cols), int(np.ceil(x1 * 416 / cols))
            b, e = int(y0 * 312 / rows), int(np.ceil(y1 * 312 / rows))
            crop = view[b:b + max(2, round((e-b)*self.top)), a:c]
            crops.append(cv2.resize(crop, (96, 96), interpolation=cv2.INTER_LINEAR))
            boxes.append((x0, y0, x1, y1))
        if not boxes:
            return []
        pixels = np.ascontiguousarray(np.stack(crops).transpose(0, 3, 1, 2), dtype=np.float32)
        self.net.setInput(pixels * (2. / 255.) - 1.)
        logits = self.net.forward()
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        observations = []
        for box, faces in zip(boxes, weights, strict=True):
            team = self.classes[int(faces.argmax())][0]
            probabilities = {}
            for probability, (side, _, value) in zip(faces, self.classes, strict=True):
                if team == side:
                    probabilities[value] = probabilities.get(value, 0.) + float(probability)
            if sum(probabilities.values()) < .90:
                continue
            value = max(probabilities, key=probabilities.get)
            confidence = probabilities[value]
            # The caller retains UNKNOWN (None) until the score is corroborated.
            observations.append((box, team, value if confidence >= self.threshold else None))
        return observations
