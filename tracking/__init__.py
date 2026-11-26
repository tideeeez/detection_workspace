"""ByteTrack adapter and track-wise vehicle classes."""

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class Track:
    id: int
    box: tuple
    confidence: float
    label: str

    @property
    def ground_pixel(self):
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)


class VehicleTracker:
    def __init__(self, names, config=None):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import ROOT, YAML

        settings = YAML.load(config or (Path(ROOT) / "cfg/trackers/bytetrack.yaml"))
        if settings["tracker_type"] != "bytetrack" or settings["track_low_thresh"] < 0.10:
            raise ValueError("The tracker config must use ByteTrack and track_low_thresh >= 0.10.")
        self.tracker = BYTETracker(SimpleNamespace(**settings))
        self.names = names
        self.history = {}

    def reset(self):
        self.tracker.reset()
        self.history.clear()

    def update(self, detections):
        rows = self.tracker.update(detections)
        active = {
            track.track_id
            for track in self.tracker.tracked_stracks + self.tracker.lost_stracks
        }
        for track_id in self.history.keys() - active:
            del self.history[track_id]
        tracks = []
        for x1, y1, x2, y2, track_id, confidence, class_id, _ in rows:
            track_id = int(track_id)
            history = self.history.setdefault(track_id, deque(maxlen=12))
            history.append((int(class_id), float(confidence)))
            votes = Counter()
            for label, score in history:
                votes[label] += score
            label = self.names[votes.most_common(1)[0][0]]
            tracks.append(Track(track_id, (x1, y1, x2, y2), float(confidence), label))
        return tracks, active
