"""Run YOLO + ByteTrack on a video and return per-frame tracking records."""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config


class YoloTracker:
    """Thin wrapper around Ultralytics so the rest of the code never touches YOLO directly.

    `update(frame)` returns (detections, annotated_frame_or_None), where detections is a list of
    (track_id, x1, y1, x2, y2, class_id). A fresh instance means fresh tracker state, so always
    create a new one per video.
    """

    def __init__(self, weights=config.DEFAULT_YOLO, tracker=config.TRACKER,
                 imgsz=config.IMGSZ, conf=config.CONF, device=None):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("ultralytics is not installed. Run: pip install ultralytics") from e
        self.model = YOLO(weights)
        self.tracker, self.imgsz, self.conf, self.device = tracker, imgsz, conf, device

    def update(self, frame, want_plot=False):
        kwargs = dict(persist=True, tracker=self.tracker, classes=config.VEHICLE_CLASSES,
                      conf=self.conf, imgsz=self.imgsz, verbose=False)
        if self.device:
            kwargs["device"] = self.device
        r = self.model.track(frame, **kwargs)[0]
        dets = []
        b = r.boxes
        if b is not None and b.id is not None:
            ids = b.id.int().cpu().tolist()
            xyxy = b.xyxy.cpu().numpy()
            cls = b.cls.int().cpu().tolist()
            for i, box, c in zip(ids, xyxy, cls):
                dets.append((int(i), float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(c)))
        return dets, (r.plot() if want_plot else None)


def _to_h264(src: Path) -> Path:
    """Browsers cannot play OpenCV's mp4v files; re-encode with ffmpeg when available."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src
    dst = src.with_name(src.stem + "_h264.mp4")
    try:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", str(dst)], check=True)
        src.unlink(missing_ok=True)
        return dst
    except Exception:
        return src


def track_video(video_path, tracker=None, stride=config.STRIDE, start_sec=0.0, max_seconds=None,
                annotated_path=None, progress_cb=None):
    """Process a video (or a segment of it). Returns (records DataFrame, meta dict).

    Timestamps are absolute video time, computed from the frame index and FPS
    (never wall-clock time, so slow hardware does not distort durations).
    """
    if tracker is None:
        tracker = YoloTracker()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total / fps if total > 0 else None

    first = int(start_sec * fps)
    if first > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    last = total if total > 0 else 10**12
    if max_seconds is not None:
        last = min(last, first + int(max_seconds * fps))
    to_do = max(1, (last - first) // max(stride, 1)) if last < 10**12 else None

    writer, tmp_out = None, None
    if annotated_path:
        tmp_out = Path(annotated_path)
        tmp_out.parent.mkdir(parents=True, exist_ok=True)

    rows, done, i = [], 0, first
    last_i = first
    while i < last:
        if not cap.grab():
            break
        if (i - first) % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = i / fps
            dets, plot = tracker.update(frame, want_plot=bool(tmp_out))
            if dets:
                for tid, x1, y1, x2, y2, _ in dets:
                    rows.append((t, tid, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
            else:
                rows.append((t, -1, np.nan, np.nan, np.nan, np.nan))
            if tmp_out is not None and plot is not None:
                if writer is None:
                    h, w = plot.shape[:2]
                    writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(fps / stride, 1.0), (w, h))
                writer.write(plot)
            done += 1
            if progress_cb and to_do:
                progress_cb(min(done / to_do, 1.0))
        last_i = i
        i += 1
    cap.release()
    if writer is not None:
        writer.release()
        annotated_path = _to_h264(tmp_out)

    records = pd.DataFrame(rows, columns=["t", "id", "cx", "cy", "w", "h"])
    seg_end = (last_i + 1) / fps
    meta = {
        "fps": fps, "width": width, "height": height, "stride": stride,
        "video_duration": video_duration, "seg_start": first / fps, "seg_end": seg_end,
        "frames_processed": done, "annotated_path": str(annotated_path) if annotated_path else None,
    }
    if progress_cb:
        progress_cb(1.0)
    return records, meta
