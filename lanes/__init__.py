"""Fixed-camera lane regions with per-track switching hysteresis."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Lane:
    id: str
    polygon: np.ndarray
    length_m: float | None


class LaneMap:
    def __init__(self, configurations):
        self.lanes = {}
        for config in configurations:
            lane_id = str(config["id"])
            polygon = np.asarray(config["polygon"], dtype=np.float32)
            if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                    or not np.isfinite(polygon).all() or abs(cv2.contourArea(polygon)) < 1):
                raise ValueError(f"Invalid polygon for lane {lane_id}.")
            length = config.get("length_m")
            if length is not None and (not np.isfinite(length) or length <= 0):
                raise ValueError(f"Lane {lane_id} length must be positive.")
            if not lane_id or lane_id in self.lanes:
                raise ValueError(f"Missing or duplicate lane ID {lane_id!r}.")
            self.lanes[lane_id] = Lane(lane_id, polygon, length)

    def locate(self, point):
        boundary = None
        for lane in self.lanes.values():
            position = cv2.pointPolygonTest(lane.polygon, point, False)
            if position > 0:
                return lane.id
            if position == 0:
                boundary = lane.id
        return boundary


@dataclass
class LaneAssignment:
    current: str | None = None
    candidate: str | None = None
    consecutive: int = 0

    def update(self, observed):
        if observed is None or self.current is None or observed == self.current:
            self.current = observed
            self.candidate = None
            self.consecutive = 0
        elif observed != self.candidate:
            self.candidate = observed
            self.consecutive = 1
        else:
            self.consecutive += 1
            if self.consecutive >= 3:
                self.current = observed
                self.candidate = None
                self.consecutive = 0
        return self.current
