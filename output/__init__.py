"""Desktop overlays and per-vehicle CSV export."""

import csv
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


CSV_COLUMNS = (
    "时间帧", "车辆ID", "车辆类型", "所在车道", "速度（km/h）", "置信度", "位置坐标",
)
VEHICLE_NAMES = {"car": "小汽车", "motorcycle": "摩托车", "bus": "客车", "truck": "货车"}


@dataclass(frozen=True)
class VehicleInfo:
    track: object
    lane: str | None
    speed: float | None
    distance: float | None


def video_time(time_s):
    seconds, milliseconds = divmod(round(time_s * 1000), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


@dataclass(frozen=True)
class VehicleObservation:
    time_s: float
    vehicle: VehicleInfo

    @property
    def quality(self):
        has_speed = self.vehicle.speed is not None
        has_lane = self.vehicle.lane is not None
        return has_speed + has_lane, has_speed, self.vehicle.track.confidence

    def csv_row(self, vehicle_id):
        track = self.vehicle.track
        box = ", ".join(f"{coordinate:.1f}" for coordinate in track.box)
        return (
            video_time(self.time_s), vehicle_id, VEHICLE_NAMES.get(track.label, track.label),
            self.vehicle.lane, None if self.vehicle.speed is None else round(self.vehicle.speed, 2),
            round(track.confidence, 3), f"({box})",
        )


class Output:
    def __init__(self, root, preview, lane_map, gates, distance_type, window_s):
        self.path = Path(root) / datetime.now().strftime("traffic_results_%Y%m%d_%H%M%S_%f.csv")
        self.preview = preview
        self.lane_map = lane_map
        self.gates = gates
        self.distance_type = distance_type
        self.window_s = window_s
        self.stack = ExitStack()
        self.window = "Highway traffic"
        self.window_open = False
        self.geometry = None
        self.records = []
        self.vehicle_ids = {}
        self.candidates = {}
        self.segment = None
        self.time_offset = self.last_time_s = 0.0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            stream = self.stack.enter_context(self.path.open("x", encoding="utf-8-sig", newline=""))
            self.writer = csv.writer(stream)
            self.writer.writerow(CSV_COLUMNS)
            stream.flush()
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *_):
        try:
            self.writer.writerows(record.csv_row(vehicle_id) for vehicle_id, record in enumerate(self.records, 1))
        finally:
            self.stack.close()
            if self.window_open:
                cv2.destroyAllWindows()

    def observe(self, segment, time_s, vehicles, active_ids):
        """Confirm exports across two consecutive frames and keep one observation per ID."""
        if segment != self.segment:
            self.time_offset = max(self.time_offset, self.last_time_s - time_s)
            self.segment = segment
            self.vehicle_ids.clear()
            self.candidates.clear()
        self.last_time_s = time_s + self.time_offset
        for raw_id in self.vehicle_ids.keys() - active_ids:
            del self.vehicle_ids[raw_id]

        candidates, visible = {}, []
        measured = {vehicle.track.id: vehicle for vehicle in vehicles}
        # Use the prior frame's order to break ties between simultaneous appearances.
        for raw_id, first in self.candidates.items():
            if raw_id in measured:
                self.vehicle_ids[raw_id] = len(self.records) + 1
                self.records.append(first)
        for raw_id, vehicle in measured.items():
            observation = VehicleObservation(self.last_time_s, vehicle)
            vehicle_id = self.vehicle_ids.get(raw_id)
            if vehicle_id is None:
                candidates[raw_id] = observation
                continue
            if observation.quality > self.records[vehicle_id - 1].quality:
                self.records[vehicle_id - 1] = observation
            visible.append(replace(vehicle, track=replace(vehicle.track, id=vehicle_id)))
        self.candidates = candidates
        return visible

    def show(self, frame, scale, vehicles, density, flows, fps, frame_age_ms):
        if not self.preview:
            return False
        image = self.render_frame(frame, scale, vehicles, density, flows, fps, frame_age_ms)
        if not self.window_open:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window, image.shape[1], image.shape[0])
            self.window_open = True
        cv2.imshow(self.window, image)
        return self.quit_requested()

    def render_frame(self, frame, scale, vehicles, density, flows, fps, frame_age_ms):
        """Draw the preview onto an image without opening a window."""
        height, width = frame.shape[:2]
        shrink = min(1.0, 1280 / width, 900 / height)
        if shrink < 1:
            image = cv2.resize(frame, (round(width * shrink), round(height * shrink)), interpolation=cv2.INTER_AREA)
        else:
            image = frame.copy()
        view_x, view_y = image.shape[1] / width, image.shape[0] / height
        sx, sy = scale[0] / view_x, scale[1] / view_y
        if self.geometry is None or self.geometry[0] != (sx, sy):
            polygons = {
                lane.id: np.rint(lane.polygon / [sx, sy]).astype(np.int32)
                for lane in self.lane_map.lanes.values()
            }
            lines = [(tuple(round(p[0] / sx) for p in (gate.a, gate.b)),
                      tuple(round(p[1] / sy) for p in (gate.a, gate.b))) for gate in self.gates.values()]
            self.geometry = (sx, sy), polygons, lines
        for polygon in self.geometry[1].values():
            cv2.polylines(image, [polygon], True, (56, 204, 255), 2, cv2.LINE_AA)
        for (x1, x2), (y1, y2) in self.geometry[2]:
            cv2.line(image, (x1, y1), (x2, y2), (255, 155, 63), 2, cv2.LINE_AA)
        for vehicle in vehicles:
            track = vehicle.track
            left, top, right, bottom = track.box
            x1, y1 = round(left * view_x), round(top * view_y)
            x2, y2 = round(right * view_x), round(bottom * view_y)
            cv2.rectangle(image, (x1, y1), (x2, y2), (70, 220, 100), 2)
            label = f"#{track.id} {track.label}"
            if vehicle.lane:
                label += f" {vehicle.lane}"
            if vehicle.speed is not None:
                label += f" {vehicle.speed:.0f} km/h"
            if vehicle.distance is not None:
                kind = "R" if self.distance_type == "slant_range" else "Z"
                label += f" {kind} {vehicle.distance:.0f} m"
            self._text(image, label, (x1, max(y1 - 6, 15)), 0.47)
        self._text(image, f"{fps:.1f} FPS  |  decode to render {frame_age_ms:.0f} ms", (12, 22), 0.53)
        for row, (lane, (count, k)) in enumerate(density.items(), 1):
            label = f"{lane}: {count} in zone"
            if k is not None:
                label += f"  |  {k:.1f} veh/km/lane"
            if lane in flows:
                total, recent, rate = flows[lane]
                label += f"  |  total {total}, last {self.window_s:g}s {recent}"
                if rate is not None:
                    label += f" ({rate:.0f} veh/h)"
            self._text(image, label, (12, 22 + row * 22), 0.50)
        return image

    def quit_requested(self):
        return self.window_open and (
            cv2.waitKey(1) & 0xFF in (27, ord("q"))
            or cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1
        )

    @staticmethod
    def _text(image, label, point, scale):
        for color, thickness in (((0, 0, 0), 3), ((248, 248, 248), 1)):
            cv2.putText(image, label, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
