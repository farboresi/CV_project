"""Minimal OpenVINO runner for exported YOLO11 detect / pose models.

Lets the tracker run without PyTorch / ultralytics (only openvino + opencv).
Mimics the bits of the ultralytics Results API that skier_tracker_v3 uses:
    r = model(img, imgsz=..., conf=..., classes=[...], verbose=False)[0]
    r.boxes.xyxy / r.boxes.conf / r.boxes.cls   (numpy arrays, .tolist() works)
    r.keypoints.data                             (pose only, [N,17,3])
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_CORE = None


def _get_core():
    global _CORE
    if _CORE is None:
        import openvino as ov
        _CORE = ov.Core()
    return _CORE


class _Arr(np.ndarray):
    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def _a(x):
    return np.asarray(x, dtype=np.float32).view(_Arr)


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _a(xyxy).reshape(-1, 4), _a(conf), _a(cls)

    def __len__(self):
        return len(self.conf)


class _Kp:
    def __init__(self, data):
        self.data = _a(data)


class _Res:
    def __init__(self, boxes, kps=None):
        self.boxes = boxes
        self.keypoints = kps
        self.names = {0: "person", 30: "skis"}


class OVYolo:
    def __init__(self, model_dir, task="detect"):
        d = Path(model_dir)
        xml = next(d.glob("*.xml"))
        core = _get_core()
        m = core.read_model(xml)
        self.size = int(m.inputs[0].get_partial_shape()[2].get_length())
        self.cm = core.compile_model(m, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
        self.task = task

    def __call__(self, img, imgsz=None, conf=0.25, classes=None, iou=0.7, verbose=False):
        S = self.size
        h, w = img.shape[:2]
        r = min(S / h, S / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        px, py = (S - nw) / 2, (S - nh) / 2
        canvas = np.full((S, S, 3), 114, np.uint8)
        x0, y0 = int(round(px - 0.1)), int(round(py - 0.1))
        canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.cm(blob)[0][0]                    # [C, N]
        out = out.T
        bx = out[:, :4]
        if self.task == "pose":
            scores = out[:, 4:5]; kp = out[:, 5:].reshape(-1, 17, 3)
        else:
            scores = out[:, 4:]; kp = None
        cls = scores.argmax(1); sc = scores[np.arange(len(scores)), cls]
        keep = sc >= conf
        if classes is not None and self.task != "pose":
            keep &= np.isin(cls, classes)
        bx, sc, cls = bx[keep], sc[keep], cls[keep]
        if kp is not None:
            kp = kp[keep]
        if not len(sc):
            return [_Res(_Boxes(np.zeros((0, 4)), [], []), _Kp(np.zeros((0, 17, 3))) if kp is not None else None)]
        xyxy = np.stack([bx[:, 0] - bx[:, 2] / 2, bx[:, 1] - bx[:, 3] / 2,
                         bx[:, 0] + bx[:, 2] / 2, bx[:, 1] + bx[:, 3] / 2], 1)
        # class-aware NMS
        sel = []
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            b = xyxy[idx]
            k = cv2.dnn.NMSBoxes([[float(q[0]), float(q[1]), float(q[2] - q[0]), float(q[3] - q[1])] for q in b],
                                 sc[idx].astype(float).tolist(), float(conf), float(iou))
            sel.extend(idx[np.array(k).reshape(-1)].tolist())
        sel = np.array(sorted(sel, key=lambda i: -sc[i]), int)[:300]
        xyxy, sc, cls = xyxy[sel], sc[sel], cls[sel]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - x0) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - y0) / r
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w); xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
        kres = None
        if kp is not None:
            kp = kp[sel].copy()
            kp[:, :, 0] = (kp[:, :, 0] - x0) / r; kp[:, :, 1] = (kp[:, :, 1] - y0) / r
            kres = _Kp(kp)
        return [_Res(_Boxes(xyxy, sc, cls), kres)]
