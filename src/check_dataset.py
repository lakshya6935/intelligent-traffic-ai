"""Sanity-check the downloaded AI City Track 4 files before doing anything expensive.

    python -m src.check_dataset
"""
from pathlib import Path

import cv2

from . import config
from .labels import parse_anomaly_file


def find_video(video_id: int, root: Path = config.RAW_DIR):
    hits = sorted(root.rglob(f"{video_id}.mp4"))
    return hits[0] if hits else None


def main():
    print(f"Looking in: {config.RAW_DIR}")
    if not config.LABEL_FILE.exists():
        print(f"\nMISSING label file: {config.LABEL_FILE}\n"
              "Copy train-anomaly-results.txt there (see README step 3).")
        return
    labels = parse_anomaly_file()
    print(f"\nParsed {len(labels)} anomaly rows from {len(labels['video_id'].unique())} videos.")
    print("First rows (check the columns mean video_id, start, end in SECONDS):")
    print(labels.head(10).to_string(index=False))

    n_found = 0
    print("\nVideo check:")
    for vid in sorted(labels["video_id"].unique()):
        p = find_video(int(vid))
        if p is None:
            continue
        n_found += 1
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        dur = frames / fps if fps else 0
        rows = labels[labels["video_id"] == vid]
        notes = []
        for r in rows.itertuples():
            if dur and r.end > dur + 1:
                notes.append(f"anomaly end {r.end:.0f}s is AFTER video end -> label units may differ")
        print(f"  {vid}.mp4  {w}x{h}  {fps:.1f} fps  {dur/60:.1f} min  anomalies: "
              + ", ".join(f"{r.start:.0f}-{r.end:.0f}s" for r in rows.itertuples())
              + (("   !! " + "; ".join(notes)) if notes else ""))
    print(f"\n{n_found} labelled video(s) found under {config.RAW_DIR}.")
    if n_found == 0:
        print("No videos found. Put 1.mp4, 2.mp4, ... inside dataset/raw/train-data/")


if __name__ == "__main__":
    main()
