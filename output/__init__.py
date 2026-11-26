"""Desktop overlays and streaming CSV export."""

import csv
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


CSV_COLUMNS = (
    "segment", "video_s", "record_type", "track_id", "vehicle_class", "lane",
    "speed_kmh", "density_veh_per_km_per_lane", "vehicles_in_zone",
    "crossings_total", "crossings_in_window", "flow_veh_per_hour", "flow_window_s",
    "distance_m", "distance_type", "confidence", "x1", "y1", "x2", "y2",
)


@dataclass(frozen=True)
class VehicleInfo:
    track: object
    lane: str | None
    speed: float | None
    distance: float | None


def number(value, digits=2):
    return "" if value is None else round(value, digits)


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

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.stream = self.stack.enter_context(self.path.open("x", encoding="utf-8", newline=""))
            self.writer = csv.DictWriter(self.stream, fieldnames=CSV_COLUMNS)
            self.writer.writeheader()
            self.stream.flush()
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *_):
        self.stack.close()
        if self.window_open:
            cv2.destroyAllWindows()

    def crossing(self, segment, event):
        self.writer.writerow({
            "segment": segment, "video_s": number(event.time_s, 3), "record_type": "crossing",
            "track_id": event.track_id, "vehicle_class": event.label, "lane": event.lane,
        })
        self.stream.flush()

    def sample(self, segment, time_s, vehicles, density, flows):
        common = {"segment": segment, "video_s": number(time_s, 3)}
        lane_stats = {}
        for lane, (count, k) in density.items():
            stats = {"vehicles_in_zone": count, "density_veh_per_km_per_lane": number(k)}
            if lane in flows:
                total, recent, rate = flows[lane]
                stats.update(
                    crossings_total=total, crossings_in_window=recent,
                    flow_veh_per_hour=rate, flow_window_s=self.window_s,
                )
            lane_stats[lane] = stats

        occupied = set()
        for vehicle in vehicles:
            track = vehicle.track
            x1, y1, x2, y2 = (round(float(coordinate), 1) for coordinate in track.box)
            self.writer.writerow({
                **common, "record_type": "vehicle", "track_id": track.id, "vehicle_class": track.label,
                "lane": vehicle.lane, "speed_kmh": number(vehicle.speed),
                "distance_m": number(vehicle.distance),
                "distance_type": self.distance_type if vehicle.distance is not None else "",
                "confidence": number(track.confidence, 3), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                **lane_stats.get(vehicle.lane, {}),
            })
            occupied.add(vehicle.lane)
        for lane, stats in lane_stats.items():
            if lane not in occupied:
                self.writer.writerow({**common, "record_type": "lane", "lane": lane, **stats})
        self.stream.flush()

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
