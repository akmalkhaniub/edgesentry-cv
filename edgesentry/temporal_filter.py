"""Temporal debounce that suppresses transient false alarms in edge CV feeds."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from .hazard_detector import Violation


@dataclass
class _Track:
    first_seen: float
    last_seen: float
    alert_fired: bool
    violation: Violation


@dataclass
class VerifiedAlert:
    violation: Violation
    duration_inside_s: float
    status: str = "VERIFIED_PERSISTENT_HAZARD"


class TemporalEventFilter:
    """Only surfaces a violation once it has persisted past a threshold.

    This is the ByteTrack-style "must be present for N seconds" debounce that keeps
    a single flickering detection from firing an alert.
    """

    def __init__(self, persistence_threshold_s: float = 1.5, evict_after_s: float = 2.0) -> None:
        self.persistence_threshold_s = persistence_threshold_s
        self.evict_after_s = evict_after_s
        self._tracks: dict[str, _Track] = {}

    def update(self, violations: Sequence[Violation], now: float | None = None) -> list[VerifiedAlert]:
        now = time.time() if now is None else now
        verified: list[VerifiedAlert] = []
        active: set[str] = set()

        for v in violations:
            key = f"{v.entity_id}_{v.zone_id}"
            active.add(key)
            track = self._tracks.get(key)
            if track is None:
                self._tracks[key] = _Track(first_seen=now, last_seen=now, alert_fired=False, violation=v)
                continue
            track.last_seen = now
            track.violation = v
            duration = now - track.first_seen
            if duration >= self.persistence_threshold_s and not track.alert_fired:
                track.alert_fired = True
                verified.append(VerifiedAlert(violation=v, duration_inside_s=duration))

        # Evict stale tracks that have left all zones.
        for key in list(self._tracks):
            if key not in active and now - self._tracks[key].last_seen > self.evict_after_s:
                del self._tracks[key]

        return verified

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)
