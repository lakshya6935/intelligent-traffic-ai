"""Synthetic traffic generator used by the tests (no YOLO / dataset needed)."""
import numpy as np
import pandas as pd


def make_records(duration=150.0, dt=0.1, stall_at=None, seed=0, noise=1.0):
    """Cars drive left->right in two lanes. If stall_at is set, one car drives in and stops at that time."""
    rng = np.random.default_rng(seed)
    rows, tid = [], 1
    W, H, speed = 40.0, 30.0, 120.0            # box size (px), speed (px/s) -> ~3.5 box-sizes/s
    spawn = 0.0
    cars = []
    while spawn < duration:
        cars.append((tid, spawn, 60.0 + 60.0 * rng.integers(0, 2)))
        tid += 1
        spawn += rng.uniform(1.2, 2.5)
    times = np.arange(0, duration, dt)
    for t in times:
        seen = False
        for cid, t0, y in cars:
            x = (t - t0) * speed
            if 0 <= x <= 640:
                rows.append((t, cid, x + rng.normal(0, noise), y + rng.normal(0, noise), W, H))
                seen = True
        if stall_at is not None and t >= stall_at - 6:
            x = min((t - (stall_at - 6)) * speed, 300.0)          # drives 300 px, then stops
            rows.append((t, 9999, x + rng.normal(0, noise), 200 + rng.normal(0, noise), W, H))
            seen = True
        if not seen:
            rows.append((t, -1, np.nan, np.nan, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["t", "id", "cx", "cy", "w", "h"])


class ScriptedTracker:
    """Stands in for YoloTracker: replays known detections, one processed frame per update()."""

    def __init__(self, records, dt):
        self.by_t = {round(t, 3): g for t, g in records[records["id"] >= 0].groupby("t")}
        self.dt, self.n = dt, 0

    def update(self, frame, want_plot=False):
        t = round(self.n * self.dt, 3)
        self.n += 1
        g = self.by_t.get(t)
        dets = []
        if g is not None:
            for r in g.itertuples():
                dets.append((int(r.id), r.cx - r.w / 2, r.cy - r.h / 2, r.cx + r.w / 2, r.cy + r.h / 2, 2))
        return dets, (frame.copy() if want_plot else None)
