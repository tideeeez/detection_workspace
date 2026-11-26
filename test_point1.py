"""Replay the first 120 seconds of the original Point 1 recording and audit track IDs.

Run python test_point1.py after installing the project requirements. Use --source
if the original 1920x1080, 20 FPS recording has moved.
"""

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from math import ceil, floor, isclose
from pathlib import Path
from tempfile import TemporaryDirectory

import av
import cv2

from output import CSV_COLUMNS, Output
from tracking import VehicleTracker
from vehicle_detection import VehicleDetector
from video import VideoSource
from work import Camera, device_for, process


DEFAULT_SOURCE = Path(r"F:\已恢复\Users\SYS02\Desktop\数据集\实时视频\点位1-2\点位1 8-9点（中间大雾）.mp4")
FPS, SECONDS = 20, 120
SOURCE_SIZE = (1920, 1080)
PREVIEW_SIZE = (1280, 720)


def point1_calibration():
    # Fitted to nominal 6 m dashes, 9 m gaps, and a 3.75 m lane width.
    # The lane coordinates are specific to the original Point 1 framing.
    y_rows = [340, 440, 600, 800, 1040]
    lane_edges = [
        [660, 567, 430, 274, 83], [719, 661, 575, 471, 363], [778, 753, 719, 676, 637],
        [824, 826, 833, 846, 869], [879, 916, 977, 1062, 1164], [949, 1012, 1125, 1276, 1455],
    ]
    lanes, image_points = [], set()
    for lane_id, left_edge, right_edge, downstream in (
        ("L_out", 0, 1, [450, 700]), ("L_in", 1, 2, [612, 700]),
        ("R_in", 3, 4, [880, 500]), ("R_out", 4, 5, [985, 500]),
    ):
        left = list(zip(lane_edges[left_edge], y_rows))
        right = list(zip(lane_edges[right_edge], y_rows))
        polygon = left + right[::-1]
        image_points.update(polygon)
        lanes.append({
            "id": lane_id, "polygon": polygon, "length_m": 90.219,
            "gate": {"line": [left[2], right[2]], "downstream": downstream},
        })

    longitudinal_scale = 21370.418136661512
    horizon_y = 153.05345390742332
    longitudinal_origin = -24.112120299312302
    lateral_scale = 11.599097685053401
    vanishing_x = 806.9600027465451
    points = [{
        "image": [u, v],
        "road_m": [
            lateral_scale * (u - vanishing_x) / (v - horizon_y),
            longitudinal_scale / (v - horizon_y) + longitudinal_origin,
        ],
    } for u, v in sorted(image_points)]
    return {
        "image_size": SOURCE_SIZE, "lanes": lanes,
        "road_calibration": {"points": points, "ransac_threshold_m": 0.10},
    }


class Point1Clip(VideoSource):
    def __iter__(self):
        self.frame_count = 0
        frames = super().__iter__()
        try:
            for frame in frames:
                if frame.time_s >= SECONDS:
                    break
                if frame.image.shape[1::-1] != SOURCE_SIZE:
                    raise ValueError("Point 1 calibration requires the original 1920x1080 framing.")
                if not frame.has_speed_clock or not isclose(frame.time_s, self.frame_count / FPS, abs_tol=0.002):
                    raise ValueError("Point 1 test requires continuous 20 FPS source timestamps.")
                self.frame_count += 1
                if self.frame_count % (30 * FPS) == 0:
                    logging.info("Processed %.0f / %d seconds of source video.", self.frame_count / FPS, SECONDS)
                yield frame
        finally:
            frames.close()


class AuditedTracker(VehicleTracker):
    def __init__(self, names):
        super().__init__(names)
        self.allocated = set()
        self.observations = defaultdict(set)

    def update(self, detections):
        tracks, active = super().update(detections)
        self.allocated.update(active)
        self.allocated.update(track.track_id for track in self.tracker.removed_stracks)
        for track in tracks:
            self.observations[track.id].add(self.tracker.frame_id - 1)
        return tracks, active


class RecordedOutput(Output):
    def __init__(self, root, camera):
        super().__init__(root, False, camera.lane_map, camera.flow.gates, "optical_axis_depth", camera.flow.window_s)
        self.video_path = self.path.with_suffix(".mp4")
        self.frame_count = 0

    def __enter__(self):
        super().__enter__()
        try:
            self.recording = self.stack.enter_context(av.open(str(self.video_path), "w", options={"movflags": "+faststart"}))
            self.encoder = self.recording.add_stream("libx264", rate=FPS, options={
                "preset": "ultrafast", "crf": "22", "tune": "zerolatency",
            })
            self.encoder.width, self.encoder.height = PREVIEW_SIZE
            self.encoder.pix_fmt = "yuv420p"
            self.encoder.codec_context.thread_count = 3
        except BaseException:
            self.stack.close()
            raise
        return self

    def finish(self):
        self.recording.mux(self.encoder.encode(None))

    def show(self, frame, scale, vehicles, density, flows, fps, frame_age_ms):
        image = self.render_frame(frame, scale, vehicles, density, flows, fps, frame_age_ms)
        title = f"+{self.frame_count / FPS:06.2f}s  |  Approximate road scale"
        Output._text(image, title, (781, 91), 0.52)
        encoded = av.VideoFrame.from_ndarray(image, format="bgr24")
        encoded.pts = self.frame_count
        self.recording.mux(self.encoder.encode(encoded))
        self.frame_count += 1
        return False

    @staticmethod
    def _text(image, label, point, scale):
        if point[0] == 12 and ("in zone" in label or "decode to render" in label):
            point = (781, point[1] + 85)
        if " km/h" in label:
            label += " est"
        if " veh/km/lane" in label:
            density, _, traffic = label.partition(" veh/km/lane")
            label = density + " veh/km/lane est" + traffic.split(",")[0]
        Output._text(image, label, point, scale)


def verify(clip, output, tracker, events):
    with av.open(str(output.video_path)) as movie:
        stream = movie.streams.video[0]
        if (clip.frame_count != FPS * SECONDS or stream.frames != clip.frame_count
                or stream.average_rate != FPS or stream.duration * stream.time_base != SECONDS):
            raise ValueError("The test requires an uninterrupted two-minute clip (2,400 frames).")
    with output.path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValueError("CSV does not have the expected vehicle columns.")
    sequential_ids = list(range(1, len(output.records) + 1))
    if [int(row[CSV_COLUMNS[1]]) for row in rows] != sequential_ids:
        raise ValueError("CSV must have one row per exported vehicle and consecutive IDs.")
    samples = {record.vehicle.track.id for record in output.records}
    confirmed = tracker.observations.keys()
    if not samples <= confirmed <= tracker.allocated:
        raise ValueError("The exported track IDs do not agree with the tracker observations.")
    stable = 0
    for event in events:
        nearby = range(max(0, ceil((event.time_s - 0.5) * FPS)),
                       min(clip.frame_count, floor((event.time_s + 0.5) * FPS) + 1))
        stable += set(nearby) <= tracker.observations[event.track_id]
    print("Crossings by lane:", dict(sorted(Counter(event.lane for event in events).items())))
    print(f"Track IDs: {len(tracker.allocated)} allocated, {len(confirmed)} output frame by frame, "
          f"{len(rows)} exported vehicle rows.")
    print(f"Continuous same ID within 0.5 s of the line: {stable}/{len(events)} crossings.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Path to the original Point 1 video")
    parser.add_argument("--weights", default="yolov8s.pt", help="YOLOv8s weights")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs",
                        help="Parent directory for the test CSV and MP4")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cv2.setNumThreads(2)
    clip = Point1Clip(str(args.source))
    detector = VehicleDetector(args.weights, device_for(args.device), image_size=960)
    detector.options.update(agnostic_nms=True, iou=0.60)
    tracker = AuditedTracker(detector.model.names)
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output, prefix=".point1_") as temporary:
        scratch = Path(temporary)
        config = scratch / "camera.json"
        config.write_text(json.dumps(point1_calibration()), encoding="utf-8")
        camera = Camera(config)
        events = []
        with RecordedOutput(scratch / "results", camera) as output, clip:
            process(clip, camera, detector, tracker, None, 5.0, output, on_crossing=events.append)
            output.finish()
        verify(clip, output, tracker, events)
        run_dir = args.output / output.path.stem.replace("traffic_results_", "point1_test_", 1)
        output.path.parent.rename(run_dir)
    print("CSV:", (run_dir / output.path.name).resolve())
    print("MP4:", (run_dir / output.video_path.name).resolve())


if __name__ == "__main__":
    main()
