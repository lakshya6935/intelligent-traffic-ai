"""Run YOLO + ByteTrack on labelled AI City videos and build dataset/features.csv.

Examples
    python -m src.extract_features --videos 1 2 3            # quick start, 3 videos
    python -m src.extract_features --limit 20                # first 20 labelled videos
    python -m src.extract_features --videos 1 --mode full    # whole 15-minute video (slow)

Raw tracking records are cached in dataset/cache, so YOLO never re-runs on a video you already
processed; changing window sizes only re-computes features.
"""
import argparse
import json

import pandas as pd

from . import config
from .check_dataset import find_video
from .features import FeatureConfig, build_windows
from .labels import intervals_for, label_windows, parse_anomaly_file
from .tracking import YoloTracker, track_video


def segment_for(mode, intervals, video_duration, pre=120.0, post=180.0, normal_head=0.0):
    """'sample' = a continuous slice around the first anomaly start (pre-anomaly normal + anomaly onset).
    'full'   = the entire video."""
    if mode == "full" or not intervals:
        if mode == "sample" and not intervals:
            return 0.0, min(video_duration or 300.0, 300.0)
        return 0.0, None
    first = min(s for s, _ in intervals)
    start = max(0.0, first - pre)
    end = first + post
    if video_duration:
        end = min(end, video_duration)
    return start, end - start


def process_video(vid, labels, args):
    path = find_video(vid)
    if path is None:
        print(f"  [skip] {vid}.mp4 not found under {config.RAW_DIR}")
        return None
    intervals = intervals_for(labels, vid)
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps or None
    cap.release()

    start, length = segment_for(args.mode, intervals, dur, args.pre, args.post)
    tag = f"{vid}_{args.mode}_s{args.stride}_z{args.imgsz}_{int(start)}"
    rec_file, meta_file = config.CACHE_DIR / f"{tag}.csv", config.CACHE_DIR / f"{tag}.json"
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if rec_file.exists() and meta_file.exists() and not args.force:
        print(f"  [cache] {vid}.mp4")
        records = pd.read_csv(rec_file)
        meta = json.loads(meta_file.read_text())
    else:
        print(f"  [track] {vid}.mp4  from {start:.0f}s, length {'full' if length is None else f'{length:.0f}s'}")
        tracker = YoloTracker(args.weights, imgsz=args.imgsz, conf=args.conf, device=args.device)
        last = {"p": -1}

        def cb(p):
            pct = int(p * 100)
            if pct // 10 != last["p"] // 10:
                print(f"      {pct}%")
                last["p"] = pct

        records, meta = track_video(path, tracker=tracker, stride=args.stride, start_sec=start,
                                    max_seconds=length, progress_cb=cb)
        records.to_csv(rec_file, index=False)
        meta_file.write_text(json.dumps(meta))

    windows = build_windows(records, meta["seg_start"], meta["seg_end"], args.win, args.hop, FeatureConfig())
    windows.insert(0, "video_id", vid)
    windows["label"] = label_windows(windows, intervals)
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, nargs="*", help="video ids, e.g. 1 2 3")
    ap.add_argument("--limit", type=int, help="use the first N labelled videos")
    ap.add_argument("--mode", choices=["sample", "full"], default="sample")
    ap.add_argument("--pre", type=float, default=120.0, help="seconds of normal traffic before the anomaly (sample mode)")
    ap.add_argument("--post", type=float, default=180.0, help="seconds after anomaly start (sample mode)")
    ap.add_argument("--stride", type=int, default=config.STRIDE)
    ap.add_argument("--imgsz", type=int, default=config.IMGSZ)
    ap.add_argument("--conf", type=float, default=config.CONF)
    ap.add_argument("--weights", default=config.DEFAULT_YOLO)
    ap.add_argument("--device", default=None, help="cpu, mps (Apple GPU) or 0 (NVIDIA)")
    ap.add_argument("--win", type=float, default=config.WIN_SEC)
    ap.add_argument("--hop", type=float, default=config.HOP_SEC)
    ap.add_argument("--force", action="store_true", help="ignore cache")
    args = ap.parse_args()

    if not config.LABEL_FILE.exists():
        raise SystemExit(f"Label file not found: {config.LABEL_FILE}\nRun: python -m src.check_dataset")
    labels = parse_anomaly_file()
    ids = args.videos or sorted(labels["video_id"].unique().tolist())
    if args.limit:
        ids = ids[: args.limit]

    frames = []
    for vid in ids:
        w = process_video(int(vid), labels, args)
        if w is not None and len(w):
            frames.append(w)
    if not frames:
        raise SystemExit("No features produced. Run: python -m src.check_dataset")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(config.FEATURES_CSV, index=False)
    (config.FEATURES_CSV.with_suffix(".json")).write_text(json.dumps({
        "win_sec": args.win, "hop_sec": args.hop, "stride": args.stride,
        "imgsz": args.imgsz, "conf": args.conf, "yolo": args.weights}))
    print(f"\nSaved {len(df)} windows from {df['video_id'].nunique()} video(s) -> {config.FEATURES_CSV}")
    print("Labels: " + ", ".join(f"{k}={v}" for k, v in df['label'].value_counts(dropna=False).items())
          + "   (NaN = ambiguous, ignored in training)")


if __name__ == "__main__":
    main()
