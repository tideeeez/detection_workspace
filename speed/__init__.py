"""Robust ground-plane trajectory velocities using Theil–Sen slopes."""

from collections import deque
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def point_pairs(count):
    return np.triu_indices(count, 1)


class SpeedEstimator:
    def __init__(self, window_s=0.85, min_span_s=0.40, max_gap_s=0.45, max_kmh=220):
        self.window_s = window_s
        self.min_span_s = min_span_s
        self.max_gap_s = max_gap_s
        self.max_kmh = max_kmh
        self.samples = deque(maxlen=60)
        self.estimate = None
        self.last_fit = -float("inf")

    def update(self, time_s, position_m):
        if position_m is None or (self.samples and time_s - self.samples[-1][0] > self.max_gap_s):
            self.samples.clear()
            self.estimate = None
            self.last_fit = -float("inf")
        if position_m is None:
            return None
        self.samples.append((time_s, *position_m))
        while self.samples and time_s - self.samples[0][0] > self.window_s:
            self.samples.popleft()
        if len(self.samples) < 4 or time_s - self.samples[0][0] < self.min_span_s:
            self.estimate = None
            return None
        if time_s - self.last_fit < 0.10:
            return self.estimate
        self.last_fit = time_s
        data = np.asarray(self.samples)
        start, end = point_pairs(len(data))
        elapsed = data[end, 0] - data[start, 0]
        useful = elapsed >= 0.15
        movement = data[end[useful], 1:] - data[start[useful], 1:]
        component_speed = np.median(movement / elapsed[useful, None], axis=0)
        speed = float(np.linalg.norm(component_speed) * 3.6)
        self.estimate = speed if speed <= self.max_kmh else None
        return self.estimate
