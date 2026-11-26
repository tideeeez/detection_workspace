"""Instantaneous lane density over a measured visible road segment."""

from collections import Counter


def lane_density(lanes, vehicle_lanes):
    counts = Counter(vehicle_lanes)
    return {
        lane_id: (
            counts[lane_id],
            1000 * counts[lane_id] / lane.length_m if lane.length_m is not None else None,
        )
        for lane_id, lane in lanes.items()
    }
