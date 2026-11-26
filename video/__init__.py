"""PyAV decoding with source timing and a bounded RTSP handoff."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic
import logging

import av


@dataclass(frozen=True)
class Frame:
    image: object
    time_s: float
    received_at: float
    segment: int
    has_speed_clock: bool


class VideoSource:
    def __init__(self, location):
        self.location = location
        self.live = location.lower().startswith(("rtsp://", "rtsps://"))
        if not self.live and not Path(location).is_file():
            raise ValueError(f"Video file does not exist: {location}")
        self.segment = 0
        self.stop = Event()
        self.thread = None
        self.queue = Queue(maxsize=1)

    def __enter__(self):
        if self.live:
            self.thread = Thread(target=self._receive, name="rtsp", daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=6)

    def __iter__(self):
        if not self.live:
            yield from self._decode()
            return
        while self.thread.is_alive() or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.1)
            except Empty:
                yield None
                continue
            if isinstance(item, Exception):
                raise item
            yield item

    def _offer_latest(self, item):
        try:
            self.queue.put_nowait(item)
        except Full:
            try:
                self.queue.get_nowait()
            except Empty:
                pass
            self.queue.put_nowait(item)

    def _receive(self):
        while not self.stop.is_set():
            try:
                for frame in self._decode():
                    self._offer_latest(frame)
                if self.stop.is_set():
                    return
            except (av.FFmpegError, OSError):
                if self.stop.is_set():
                    return
            except Exception as exc:
                self._offer_latest(exc)
                return
            logging.warning("RTSP connection interrupted. Reconnecting in 1 second.")
            self.stop.wait(1)

    def _decode(self):
        options = {"rtsp_transport": "tcp"} if self.live else {}
        with av.open(self.location, options=options, timeout=(5, 5) if self.live else None) as container:
            if not container.streams.video:
                raise ValueError("No video stream found.")
            stream = container.streams.video[0]
            if not self.live:
                stream.thread_type = "AUTO"
            fps = float(stream.average_rate or 0)
            self.segment += 1
            source_clock = None
            started_at = previous = None
            for index, frame in enumerate(container.decode(stream)):
                if self.stop.is_set():
                    return
                received_at = monotonic()
                pts = frame.time
                valid_pts = pts is not None and isfinite(pts)
                if source_clock is None:
                    source_clock = "pts" if valid_pts else ("arrival" if self.live else "fps")
                    if source_clock == "fps" and fps <= 0:
                        raise ValueError("Video has no source timestamps or usable frame rate.")
                    if source_clock != "pts":
                        logging.warning("Video has no initial PTS. Using %s timestamps.", source_clock)
                elif valid_pts and source_clock != "pts":
                    source_clock = "pts"
                    started_at = previous = None
                    self.segment += 1
                if source_clock == "pts":
                    if not valid_pts:
                        continue
                    time_s = pts
                else:
                    time_s = received_at if self.live else index / fps
                if previous is not None and time_s <= previous:
                    if previous - time_s < 0.5:
                        continue
                    started_at = None
                    self.segment += 1
                if started_at is None:
                    started_at = time_s
                previous = time_s
                yield Frame(
                    frame.to_ndarray(format="bgr24"), time_s - started_at, received_at,
                    self.segment, source_clock != "arrival",
                )
