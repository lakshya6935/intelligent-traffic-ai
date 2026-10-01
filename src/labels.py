"""Read AI City Track 4 anomaly annotations and label time windows."""
import re
from pathlib import Path

import numpy as np
import pandas as pd



def parse_anomaly_file(path=None) -> pd.DataFrame:
    """Parse lines of `video_id start_time end_time`. Skips headers/blank/invalid lines."""
    from . import config
    path = Path(path or config.LABEL_FILE)
    rows = []
    for line in path.read_text().splitlines():
        nums = re.findall(r"-?\d+(?:\.\d+)?", line)
        if len(nums) < 3:
            continue
        vid, start, end = int(float(nums[0])), float(nums[1]), float(nums[2])
        if end < start:
            start, end = end, start
        rows.append({"video_id": vid, "start": start, "end": end})
    return pd.DataFrame(rows, columns=["video_id", "start", "end"])


def intervals_for(labels: pd.DataFrame, video_id: int):
    sub = labels[labels["video_id"] == video_id]
    return [(float(r.start), float(r.end)) for r in sub.itertuples()]


def label_window(t0: float, t1: float, intervals, pos_frac: float = 0.5) -> float:
    """1 = anomaly, 0 = normal, NaN = ambiguous (window only partly overlaps an anomaly)."""
    overlap = sum(max(0.0, min(t1, e) - max(t0, s)) for s, e in intervals)
    frac = overlap / max(t1 - t0, 1e-6)
    if frac >= pos_frac:
        return 1.0
    if overlap <= 0:
        return 0.0
    return float("nan")


def label_windows(windows: pd.DataFrame, intervals, pos_frac: float = 0.5) -> pd.Series:
    return pd.Series(
        [label_window(a, b, intervals, pos_frac) for a, b in zip(windows["t_start"], windows["t_end"])],
        index=windows.index, dtype=float)
