"""Turn raw tracking records into per-window movement features.

`records` is a DataFrame with columns: t, id, cx, cy, w, h
  - t  : timestamp in seconds (frame_index / fps)
  - id : tracker ID (-1 marks a processed frame that had no detections)
All movement is normalised by the vehicle's own box size, so the features
depend much less on camera resolution and perspective.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "mean_count", "max_count", "n_tracks",
    "mean_speed", "p90_speed", "std_speed", "moving_frac",
    "stationary_count", "stationary_frac", "max_stationary_sec", "lone_stationary",
    "stopped_after_moving", "max_speed_drop", "max_heading_change",
    "max_iou", "overlap_frames_frac",
]


@dataclass
class FeatureConfig:
    moving_speed: float = 0.30        # box-sizes / second above which a vehicle is "moving"
    stopped_speed: float = 0.10       # box-sizes / second below which it is "stopped"
    stationary_spread: float = 0.35   # 90th pct distance from median position, in box-sizes
    stationary_min_sec: float = 4.0
    min_track_sec: float = 2.0
    min_obs: int = 4
    speed_lag_sec: float = 1.0        # speed measured over ~1 s to ignore detector jitter
    iou_thresh: float = 0.20


def _zero_features() -> dict:
    return {c: 0.0 for c in FEATURE_COLUMNS}


def _smooth(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rolling(3, center=True, min_periods=1).mean().to_numpy()


def _track_stats(g: pd.DataFrame, cfg: FeatureConfig):
    g = g.sort_values("t")
    t = g["t"].to_numpy(float)
    n = len(t)
    if n < cfg.min_obs:
        return None
    duration = t[-1] - t[0]
    if duration < cfg.min_track_sec:
        return None

    x = g["cx"].to_numpy(float)
    y = g["cy"].to_numpy(float)
    size = max(float(np.median(np.sqrt(g["w"].to_numpy(float) * g["h"].to_numpy(float)))), 1.0)
    xs, ys = _smooth(x), _smooth(y)

    med_dt = float(np.median(np.diff(t)))
    lag = int(max(1, min(round(cfg.speed_lag_sec / max(med_dt, 1e-3)), n - 1)))
    dist = np.hypot(xs[lag:] - xs[:-lag], ys[lag:] - ys[:-lag])
    dts = np.maximum(t[lag:] - t[:-lag], 1e-3)
    speed = dist / dts / size

    k = max(1, len(speed) // 3)
    early = float(np.mean(speed[:k]))
    late = float(np.mean(speed[-k:]))

    spread = float(np.percentile(np.hypot(x - np.median(x), y - np.median(y)), 90) / size)
    stationary = spread < cfg.stationary_spread and duration >= cfg.stationary_min_sec

    kk = max(1, n // 3)
    v1 = np.array([xs[kk] - xs[0], ys[kk] - ys[0]])
    v2 = np.array([xs[-1] - xs[-1 - kk], ys[-1] - ys[-1 - kk]])
    heading = 0.0
    if np.linalg.norm(v1) > 0.5 * size and np.linalg.norm(v2) > 0.5 * size:
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = float(v1 @ v2)
        heading = abs(float(np.degrees(np.arctan2(cross, dot))))

    return {
        "speed": speed, "mean_speed": float(np.mean(speed)),
        "early": early, "late": late, "stationary": stationary,
        "duration": duration, "heading": heading,
    }


def _max_iou_stats(dets: pd.DataFrame, thresh: float):
    """Largest IoU between two different vehicles in any frame, and the share of frames above thresh."""
    best, hits, frames = 0.0, 0, 0
    for _, f in dets.groupby("t"):
        frames += 1
        if len(f) < 2:
            continue
        x1 = (f["cx"] - f["w"] / 2).to_numpy(float)
        y1 = (f["cy"] - f["h"] / 2).to_numpy(float)
        x2 = (f["cx"] + f["w"] / 2).to_numpy(float)
        y2 = (f["cy"] + f["h"] / 2).to_numpy(float)
        iw = np.maximum(0, np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]))
        ih = np.maximum(0, np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]))
        inter = iw * ih
        area = (x2 - x1) * (y2 - y1)
        union = area[:, None] + area[None] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        np.fill_diagonal(iou, 0.0)
        m = float(iou.max())
        best = max(best, m)
        hits += m > thresh
    return best, (hits / frames if frames else 0.0)


def window_features(records: pd.DataFrame, t0: float, t1: float, cfg: FeatureConfig = None) -> dict:
    cfg = cfg or FeatureConfig()
    feats = _zero_features()
    if records is None or len(records) == 0:
        return feats
    w = records[(records["t"] >= t0) & (records["t"] < t1)]
    if len(w) == 0:
        return feats
    n_frames = w["t"].nunique()
    dets = w[w["id"] >= 0]
    if len(dets) == 0:
        return feats

    counts = dets.groupby("t").size()
    feats["mean_count"] = float(dets.shape[0] / n_frames)
    feats["max_count"] = float(counts.max())

    stats = []
    for _, g in dets.groupby("id"):
        s = _track_stats(g, cfg)
        if s is not None:
            stats.append(s)
    feats["n_tracks"] = float(len(stats))

    if stats:
        all_speed = np.concatenate([s["speed"] for s in stats])
        feats["mean_speed"] = float(np.mean([s["mean_speed"] for s in stats]))
        feats["p90_speed"] = float(np.percentile(all_speed, 90))
        feats["std_speed"] = float(np.std(all_speed))
        moving = sum(s["mean_speed"] > cfg.moving_speed for s in stats)
        stationary = [s for s in stats if s["stationary"]]
        feats["moving_frac"] = moving / len(stats)
        feats["stationary_count"] = float(len(stationary))
        feats["stationary_frac"] = len(stationary) / len(stats)
        feats["max_stationary_sec"] = float(max((s["duration"] for s in stationary), default=0.0))
        # a stalled vehicle while the rest of the road keeps flowing (not a queue / red light)
        feats["lone_stationary"] = float(len(stationary) >= 1 and feats["moving_frac"] >= 0.5)
        feats["stopped_after_moving"] = float(sum(
            s["early"] > cfg.moving_speed * 1.5 and s["late"] < cfg.stopped_speed for s in stats))
        feats["max_speed_drop"] = float(max(s["early"] - s["late"] for s in stats))
        feats["max_heading_change"] = float(max(s["heading"] for s in stats))

    feats["max_iou"], feats["overlap_frames_frac"] = _max_iou_stats(dets, cfg.iou_thresh)
    return feats


def build_windows(records: pd.DataFrame, seg_start: float, seg_end: float,
                  win_sec: float, hop_sec: float, cfg: FeatureConfig = None) -> pd.DataFrame:
    """Slide a window over [seg_start, seg_end] and compute features for each."""
    cfg = cfg or FeatureConfig()
    rows = []
    t0 = float(seg_start)
    while t0 + win_sec <= seg_end + 1e-6:
        f = window_features(records, t0, t0 + win_sec, cfg)
        f.update(t_start=t0, t_end=t0 + win_sec)
        rows.append(f)
        t0 += hop_sec
    if not rows and seg_end > seg_start:   # clip shorter than one window
        f = window_features(records, seg_start, seg_end, cfg)
        f.update(t_start=float(seg_start), t_end=float(seg_end))
        rows.append(f)
    cols = ["t_start", "t_end"] + FEATURE_COLUMNS
    return pd.DataFrame(rows, columns=cols)
