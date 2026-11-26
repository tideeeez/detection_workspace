"""Directional virtual-gate events and rolling per-lane flow."""

from collections import Counter
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import hypot, isfinite


@dataclass(frozen=True)
class Crossing:
    time_s: float
    lane: str
    track_id: int
    label: str


class Gate:
    def __init__(self, lane, config):
        self.lane = lane
        self.a, self.b = (tuple(map(float, p)) for p in config["line"])
        self.margin = float(config.get("margin_px", 3.0))
        dx, dy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        self.length = hypot(dx, dy)
        if not isfinite(self.length) or self.length <= 0 or not isfinite(self.margin) or self.margin <= 0:
            raise ValueError(f"Lane {lane} requires a nonzero gate and a positive margin.")
        self.tangent = dx / self.length, dy / self.length
        self.normal = -self.tangent[1], self.tangent[0]
        downstream = tuple(map(float, config["downstream"]))
        side = self.side(downstream)
        if not isfinite(side) or abs(side) < self.margin:
            raise ValueError(f"Lane {lane} downstream point must be clear of its gate.")
        if side < 0:
            self.normal = -self.normal[0], -self.normal[1]

    def side(self, point):
        return sum((x - origin) * n for x, origin, n in zip(point, self.a, self.normal))

    def segment_contains(self, point):
        offset = sum((x - origin) * t for x, origin, t in zip(point, self.a, self.tangent))
        return 0 <= offset <= self.length


@dataclass
class GateState:
    previous: tuple | None = None
    armed: bool = False
    crossing_time: float | None = None

    def update(self, gate, point, time_s, max_gap_s):
        side = gate.side(point)
        previous = self.previous
        self.previous = (point, time_s, side)
        if previous is None or time_s - previous[1] > max_gap_s:
            self.armed = False
            self.crossing_time = None
            previous = None
        if side <= -gate.margin:
            self.armed = True
        if side <= 0:
            self.crossing_time = None
        if self.armed and previous and previous[2] <= 0 < side:
            last_point, last_time, last_side = previous
            fraction = -last_side / (side - last_side)
            intersection = tuple(a + fraction * (b - a) for a, b in zip(last_point, point))
            if gate.segment_contains(intersection):
                self.crossing_time = last_time + fraction * (time_s - last_time)
        if side < gate.margin:
            return None
        crossing_time = self.crossing_time
        self.armed = False
        self.crossing_time = None
        return crossing_time


@dataclass
class FlowTrack:
    gates: dict = field(default_factory=dict)
    counted: bool = False


class FlowCounter:
    def __init__(self, lanes, window_s=60.0, max_gap_s=0.45):
        self.gates = {
            str(lane["id"]): Gate(str(lane["id"]), lane["gate"]) for lane in lanes if "gate" in lane
        }
        self.window_s = window_s
        self.max_gap_s = max_gap_s
        self.total = Counter()
        self.reset_window(0.0)

    def reset_window(self, time_s):
        self.started_at = time_s
        self.events = []
        self.recent = Counter()

    def observe(self, memory, point, time_s, track_id, label):
        if memory.counted:
            return None
        crossings = []
        for lane, gate in self.gates.items():
            if lane not in memory.gates:
                memory.gates[lane] = GateState()
            crossing_time = memory.gates[lane].update(gate, point, time_s, self.max_gap_s)
            if crossing_time is not None:
                crossings.append((crossing_time, lane))
        if not crossings:
            return None
        crossing_time, lane = min(crossings)
        memory.counted = True
        self.total[lane] += 1
        self.recent[lane] += 1
        heappush(self.events, (crossing_time, lane))
        return Crossing(crossing_time, lane, track_id, label)

    def snapshot(self, time_s):
        boundary = time_s - self.window_s
        while self.events and self.events[0][0] <= boundary:
            _, lane = heappop(self.events)
            self.recent[lane] -= 1
        ready = time_s - self.started_at >= self.window_s
        return {
            lane: (self.total[lane], self.recent[lane], self.recent[lane] * 3600 / self.window_s if ready else None)
            for lane in self.gates
        }
