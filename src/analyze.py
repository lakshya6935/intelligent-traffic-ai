"""Score windows (trained model or rule-based fallback), merge them into incident events."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_COLUMNS, build_windows, FeatureConfig
from .tracking import track_video


# ---------------------------------------------------------------- model / rules
def load_bundle(path=config.MODEL_PATH):
    path = Path(path)
    return joblib.load(path) if path.exists() else None


def rule_signals(w: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "stopped_after_moving": w["stopped_after_moving"] >= 1,
        "stalled_in_flowing_traffic": w["lone_stationary"] >= 1,
        "vehicle_overlap": w["overlap_frames_frac"] >= 0.20,
    }, index=w.index)


def rule_score(w: pd.DataFrame) -> pd.Series:
    s = rule_signals(w)
    return (0.6 * s["stopped_after_moving"] + 0.5 * s["stalled_in_flowing_traffic"]
            + 0.5 * s["vehicle_overlap"]).clip(upper=1.0)


def explain(w: pd.DataFrame) -> pd.Series:
    sig = rule_signals(w)
    return sig.apply(lambda r: ", ".join(k.replace("_", " ") for k, v in r.items() if v) or "-", axis=1)


def score_windows(windows: pd.DataFrame, bundle=None, threshold=None) -> pd.DataFrame:
    out = windows.copy()
    if bundle is not None:
        X = out[bundle["feature_columns"]]
        out["score"] = bundle["model"].predict_proba(X)[:, 1]
        thr = bundle["threshold"] if threshold is None else threshold
        out["method"] = "random_forest"
    else:
        out["score"] = rule_score(out)
        thr = 0.5 if threshold is None else threshold
        out["method"] = "rules"
    out["flag"] = out["score"] >= thr
    out["reason"] = explain(out)
    out.attrs["threshold"] = thr
    return out


def merge_events(windows: pd.DataFrame, min_windows: int = config.MIN_EVENT_WINDOWS) -> list:
    """Merge consecutive flagged windows into events; drop events shorter than min_windows."""
    events, cur = [], None
    for r in windows.sort_values("t_start").itertuples():
        if r.flag:
            if cur is not None and r.t_start <= cur["end"] + 1e-6:
                cur["end"] = r.t_end
                cur["n_windows"] += 1
                cur["peak_score"] = max(cur["peak_score"], r.score)
                cur["reasons"].add(r.reason)
            else:
                if cur:
                    events.append(cur)
                cur = {"start": r.t_start, "end": r.t_end, "n_windows": 1,
                       "peak_score": float(r.score), "reasons": {r.reason}}
        elif cur is not None:
            events.append(cur)
            cur = None
    if cur:
        events.append(cur)
    events = [e for e in events if e["n_windows"] >= min_windows]
    for e in events:
        e["reasons"] = "; ".join(sorted(x for x in e["reasons"] if x != "-")) or "-"
    return events


def fmt_time(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------- end to end
def analyze_video(video_path, bundle=None, tracker=None, stride=None, win_sec=None, hop_sec=None,
                  start_sec=0.0, max_seconds=None, threshold=None,
                  min_windows=config.MIN_EVENT_WINDOWS, annotated_path=None, progress_cb=None):
    """Full pipeline on one video: track -> features -> score -> events."""
    meta_b = bundle or {}
    stride = stride or meta_b.get("stride", config.STRIDE)
    win_sec = win_sec or meta_b.get("win_sec", config.WIN_SEC)
    hop_sec = hop_sec or meta_b.get("hop_sec", config.HOP_SEC)

    records, meta = track_video(video_path, tracker=tracker, stride=stride, start_sec=start_sec,
                                max_seconds=max_seconds, annotated_path=annotated_path,
                                progress_cb=progress_cb)
    windows = build_windows(records, meta["seg_start"], meta["seg_end"], win_sec, hop_sec, FeatureConfig())
    scored = score_windows(windows, bundle, threshold) if len(windows) else windows.assign(
        score=[], flag=[], reason=[], method=[])
    events = merge_events(scored, min_windows) if len(scored) else []
    return {
        "records": records, "windows": scored, "events": events, "meta": meta,
        "n_tracks": int(records.loc[records["id"] >= 0, "id"].nunique()),
        "method": scored["method"].iloc[0] if len(scored) else "none",
        "threshold": scored.attrs.get("threshold") if len(scored) else None,
    }
