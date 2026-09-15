"""Simulate a GNSS blackout over a chosen window of a loaded route.

MIP Section 9 step 2: masks GNSS input to the fusion core for that
window only - GNSS still feeds the filter normally outside the window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tools.benchmark_replay.route_loader import Route


@dataclass
class BlackoutWindow:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.end_s <= self.start_s:
            raise ValueError(
                f"blackout end_s ({self.end_s}) must be after start_s ({self.start_s})"
            )


def apply_blackout(route: Route, window: BlackoutWindow) -> Route:
    """Return a copy of `route` with `gnss_available` masked False for
    [window.start_s, window.end_s), everything else unchanged.

    Returns a copy rather than mutating in place so the same loaded
    route can be reused across multiple blackout configs (e.g. a
    regression run that sweeps window length) without reloading it.
    """
    if window.start_s < route.t[0] or window.end_s > route.t[-1]:
        raise ValueError(
            f"blackout window [{window.start_s}, {window.end_s}] falls "
            f"outside route duration [{route.t[0]}, {route.t[-1]}]"
        )

    gnss_available = route.gnss_available.copy()
    in_window = (route.t >= window.start_s) & (route.t < window.end_s)
    gnss_available[in_window] = False

    return Route(
        name=route.name,
        dt_s=route.dt_s,
        t=route.t,
        pos=route.pos,
        vel=route.vel,
        heading=route.heading,
        accel_body=route.accel_body,
        gyro_yaw=route.gyro_yaw,
        gnss_available=gnss_available,
        imu_raw=route.imu_raw,  # pass through (Phase 4: new field on Route)
    )


def window_indices(route: Route, window: BlackoutWindow) -> np.ndarray:
    """Indices of `route.t` that fall inside the blackout window -
    used by drift.py to compute path length and end-of-window error."""
    return np.nonzero((route.t >= window.start_s) & (route.t < window.end_s))[0]
