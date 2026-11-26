"""Outdoor metric Depth Anything V2 with asynchronous, shared-frame inference."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import ceil, floor, sqrt

import cv2
import numpy as np

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


@dataclass
class DistanceReading:
    meters: float | None = None
    time_s: float = -float("inf")

    def update(self, time_s, estimate):
        if estimate is not None and self.meters is not None and time_s - self.time_s <= 0.6:
            estimate = 0.75 * estimate + 0.25 * self.meters
        self.meters = estimate
        self.time_s = time_s


def vehicle_depth(depth, box, frame_shape, limit):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if min(width, height) < 12:
        return None
    depth_h, depth_w = depth.shape
    sy, sx = depth_h / frame_shape[0], depth_w / frame_shape[1]
    left = max(0, ceil((x1 + 0.27 * width) * sx))
    right = min(depth_w, floor((x2 - 0.27 * width) * sx) + 1)
    top = max(0, ceil((y1 + 0.48 * height) * sy))
    bottom = min(depth_h, floor((y1 + 0.82 * height) * sy) + 1)
    if right - left < 2 or bottom - top < 2:
        return None
    values = depth[top:bottom, left:right]
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size < 0.75 * values.size:
        return None
    low, median, high = np.percentile(valid, (25, 50, 75))
    if median >= 0.97 * limit or high - low > 0.25 * median:
        return None
    pixel = ((left + right - 1) / (2 * sx), (top + bottom - 1) / (2 * sy))
    return float(median), pixel


class DepthWorker:
    def __init__(self, device, frequency, model_id=MODEL_ID):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.torch = torch
        self.interval = 1 / frequency
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, use_safetensors=True).eval().to(device)
        if self.model.config.depth_estimation_type != "metric":
            raise ValueError("The distance module requires a metric depth checkpoint.")
        self.max_depth = self.model.config.max_depth
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model.to(dtype=self.dtype)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.last_submit = -float("inf")

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def reset(self):
        self.last_submit = -float("inf")

    def poll(self):
        if self.pending is None or not self.pending.done():
            return None
        result = self.pending.result()
        self.pending = None
        return result

    def submit(self, image, tracks, segment, time_s, intrinsics, reference_scale):
        if not tracks or self.pending is not None or time_s - self.last_submit + 1e-6 < self.interval:
            return
        self.last_submit = time_s
        self.pending = self.executor.submit(
            self._measure, image, tracks, segment, time_s, intrinsics, reference_scale
        )

    def _measure(self, image, tracks, segment, time_s, intrinsics, reference_scale):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(device=self.device, dtype=self.dtype) for key, value in inputs.items()}
        with self.torch.inference_mode():
            depth = self.model(**inputs).predicted_depth[0].float().cpu().numpy()
        readings = {}
        for track in tracks:
            measurement = vehicle_depth(depth, track.box, image.shape, self.max_depth)
            if measurement is None:
                readings[track.id] = None
                continue
            meters, pixel = measurement
            if intrinsics is not None:
                reference_pixel = tuple(x * scale for x, scale in zip(pixel, reference_scale))
                ray_x, ray_y = intrinsics.rays([reference_pixel])[0]
                meters *= sqrt(1 + ray_x ** 2 + ray_y ** 2)
            readings[track.id] = meters
        return segment, time_s, readings
