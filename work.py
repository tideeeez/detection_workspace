"""Entry point for single-camera real-time highway traffic analysis."""

from argparse import ArgumentParser
from dataclasses import dataclass, field
from json import load
from math import isfinite
from pathlib import Path
from time import monotonic
import logging

import cv2

from calibration import Intrinsics, RoadProjection
from density import lane_density
from distance import MODEL_ID, DepthWorker, DistanceReading
from flow import FlowCounter, FlowTrack
from lanes import LaneAssignment, LaneMap
from output import Output, VehicleInfo
from speed import SpeedEstimator
from tracking import VehicleTracker
from vehicle_detection import VehicleDetector


@dataclass
class VehicleState:
    lane: LaneAssignment = field(default_factory=LaneAssignment)
    speed: SpeedEstimator = field(default_factory=SpeedEstimator)
    flow: FlowTrack = field(default_factory=FlowTrack)
    distance: DistanceReading = field(default_factory=DistanceReading)


class Camera:
    def __init__(self, filename):
        config = {}
        if filename:
            with Path(filename).open(encoding="utf-8-sig") as stream:
                config = load(stream)
        self.image_size = config.get("image_size")
        if filename and (
            not isinstance(self.image_size, list) or len(self.image_size) != 2
            or any(not isinstance(n, int) or n <= 0 for n in self.image_size)
        ):
            raise ValueError("Camera config needs image_size [width, height].")
        self.intrinsics = Intrinsics.from_config(config["intrinsics"]) if "intrinsics" in config else None
        self.road = RoadProjection(config["road_calibration"], self.intrinsics) if "road_calibration" in config else None
        if self.road is not None and self.road.check_rmse_m is not None:
            logging.info("Calibration check point RMSE: %.3f m", self.road.check_rmse_m)
        self.lanes = config.get("lanes", [])
        self.lane_map = LaneMap(self.lanes)
        self.flow = FlowCounter(self.lanes)
        for lane_id, gate in self.flow.gates.items():
            polygon = self.lane_map.lanes[lane_id].polygon
            if any(cv2.pointPolygonTest(polygon, point, True) < -2 for point in (gate.a, gate.b)):
                raise ValueError(f"Place the gate endpoints on or inside lane {lane_id}.")

    def scale(self, frame_shape):
        if self.image_size is None:
            return 1.0, 1.0
        width, height = self.image_size
        sx, sy = width / frame_shape[1], height / frame_shape[0]
        if abs(sx / sy - 1) > 0.01:
            raise ValueError("Video aspect ratio differs from the camera calibration image.")
        return sx, sy


def device_for(request):
    import torch

    if request == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if request == "cpu":
        return request
    if request == "cuda":
        request = "cuda:0"
    if request.startswith("cuda:") and request[5:].isdigit() and int(request[5:]) < torch.cuda.device_count():
        return request
    raise ValueError(f"Device {request!r} is unavailable. Use auto, cpu, cuda, or cuda:N.")


def snapshot(source, destination):
    target = Path(destination)
    with source:
        for frame in source:
            if frame is None:
                continue
            success, encoded = cv2.imencode(target.suffix, frame.image)
            if not success:
                raise ValueError("Snapshot could not be encoded as an image.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(encoded.tobytes())
            h, w = frame.image.shape[:2]
            print(f"{target.resolve()}  {w} x {h}")
            return
    raise ValueError("No video frame was decoded.")


def run(source, args):
    camera = Camera(args.camera)
    device = device_for(args.device)
    frequency = args.depth_hz if args.depth_hz is not None else (5.0 if device.startswith("cuda") else 2.0)
    detector = VehicleDetector(args.weights, device, args.imgsz)
    tracker = VehicleTracker(detector.model.names, args.tracker)
    depth = None if args.no_depth else DepthWorker(device, frequency, args.depth_model)
    kind = "slant_range" if camera.intrinsics is not None else "optical_axis_depth"
    logging.info("Inference on %s. Depth update cap: %s.", device, "disabled" if depth is None else f"{frequency:g} Hz")
    try:
        with Output(args.output, not args.no_preview, camera.lane_map, camera.flow.gates,
                    kind, camera.flow.window_s) as output, source:
            logging.info("CSV output: %s", output.path.resolve())
            process(source, camera, detector, tracker, depth, frequency, output)
    finally:
        if depth is not None:
            depth.close()


def process(source, camera, detector, tracker, depth, depth_hz, output):
    states = {}
    source_segment = previous = last_sample = last_frame_at = previous_shape = None
    segment = frames = 0
    fps = 0.0
    depth_fresh_s = max(0.50, 2 / depth_hz)
    started = monotonic()
    for frame in source:
        if frame is None:
            if output.quit_requested():
                break
            continue
        time_s = frame.time_s
        shape = frame.image.shape
        if frame.segment != source_segment or shape != previous_shape or time_s - previous > 0.75:
            if source_segment is not None:
                tracker.reset()
            states.clear()
            camera.flow.reset_window(time_s)
            if depth is not None:
                depth.reset()
            segment += 1
            last_sample = None
            source_segment = frame.segment
        previous = time_s
        previous_shape = shape
        scale = camera.scale(shape)
        detections = detector.detect(frame.image)
        tracks, active = tracker.update(detections)
        for track_id in states.keys() - active:
            del states[track_id]
        if depth is not None:
            result = depth.poll()
            if result is not None and result[0] == segment:
                _, observed_at, readings = result
                for track_id, reading in readings.items():
                    if track_id in states:
                        states[track_id].distance.update(observed_at, reading)
            depth.submit(frame.image, tracks, segment, time_s, camera.intrinsics, scale)
        vehicles = []
        for track in tracks:
            if track.id not in states:
                states[track.id] = VehicleState()
            state = states[track.id]
            pixel = tuple(float(p) * s for p, s in zip(track.ground_pixel, scale))
            lane = state.lane.update(camera.lane_map.locate(pixel))
            ground_m = camera.road.project(pixel) if camera.road is not None and frame.has_speed_clock else None
            speed = state.speed.update(time_s, ground_m)
            distance = state.distance.meters if time_s - state.distance.time_s <= depth_fresh_s else None
            vehicles.append(VehicleInfo(track, lane, speed, distance))
            crossing = camera.flow.observe(state.flow, pixel, time_s, track.id, track.label)
            if crossing is not None:
                output.crossing(segment, crossing)
        density = lane_density(camera.lane_map.lanes, (v.lane for v in vehicles))
        flows = camera.flow.snapshot(time_s)
        if last_sample is None or time_s - last_sample >= 1 - 1e-6:
            output.sample(segment, time_s, vehicles, density, flows)
            last_sample = time_s
        now = monotonic()
        if last_frame_at is not None:
            fps_now = 1 / max(now - last_frame_at, 1e-6)
            fps = fps_now if fps == 0 else 0.15 * fps_now + 0.85 * fps
        last_frame_at = now
        frames += 1
        if output.show(frame.image, scale, vehicles, density, flows, fps, (now - frame.received_at) * 1000):
            break
    if frames == 0:
        raise ValueError("No video frame was processed.")
    if previous - last_sample > 1e-6:
        output.sample(segment, previous, vehicles, density, flows)
    elapsed = monotonic() - started
    logging.info("Processed %d frames in %.1f s (%.1f FPS).", frames, elapsed, frames / max(elapsed, 1e-6))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video path or RTSP URL")
    parser.add_argument("--camera", help="Fixed-camera calibration and lane JSON")
    parser.add_argument("--weights", default="yolov8s.pt", help="YOLOv8s COCO or fine-tuned weights with matching class IDs")
    parser.add_argument("--tracker", help="Override the Ultralytics ByteTrack YAML")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLOv8s input size")
    parser.add_argument("--depth-model", default=MODEL_ID, help="Metric Depth Anything V2 model ID or local safetensors directory")
    parser.add_argument("--depth-hz", type=float, help="Maximum depth refresh rate (default 5 GPU, 2 CPU)")
    parser.add_argument("--no-depth", action="store_true", help="Disable model-based distance measurements")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--output", default="runs", help="Parent directory for run CSV exports")
    parser.add_argument("--no-preview", action="store_true", help="Run with no desktop preview")
    parser.add_argument("--snapshot", help="Save the first source frame for camera annotation and exit")
    args = parser.parse_args()
    if args.imgsz < 32 or (args.depth_hz is not None and (not isfinite(args.depth_hz) or args.depth_hz <= 0)):
        parser.error("--imgsz must be at least 32 and --depth-hz must be positive.")
    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
    try:
        from video import VideoSource

        source = VideoSource(args.source)
        if args.snapshot:
            snapshot(source, args.snapshot)
        else:
            run(source, args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, KeyError, TypeError, OSError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
