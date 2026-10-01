import numpy as np
import pandas as pd
import cv2
import pytest

from src import config
from src.analyze import analyze_video, merge_events, score_windows, load_bundle
from src.features import FEATURE_COLUMNS, build_windows, window_features
from src.labels import label_window, parse_anomaly_file
from tests.synthetic import ScriptedTracker, make_records


def test_label_window_rules():
    iv = [(100.0, 200.0)]
    assert label_window(0, 10, iv) == 0.0
    assert label_window(120, 130, iv) == 1.0
    assert np.isnan(label_window(92, 102, iv))      # 20% overlap -> ambiguous, dropped from training
    assert label_window(95, 105, iv) == 1.0         # 50% overlap -> anomaly


def test_parse_anomaly_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("video_id start end\n2 587 894\n5 10.5 40\n\nbad line\n")
    df = parse_anomaly_file(f)
    assert df.to_dict("records") == [
        {"video_id": 2, "start": 587.0, "end": 894.0}, {"video_id": 5, "start": 10.5, "end": 40.0}]


def test_normal_traffic_has_no_stationary_vehicle():
    rec = make_records(duration=60, seed=1)
    f = window_features(rec, 20, 30)
    assert f["stationary_count"] == 0 and f["lone_stationary"] == 0
    assert f["moving_frac"] > 0.9 and f["mean_speed"] > 1.0


def test_stalled_vehicle_is_detected():
    rec = make_records(duration=90, stall_at=40, seed=2)
    before = window_features(rec, 10, 20)
    after = window_features(rec, 50, 60)
    assert before["lone_stationary"] == 0
    assert after["lone_stationary"] == 1 and after["max_stationary_sec"] >= 8
    onset = window_features(rec, 34, 44)      # the car drives in, then stops at ~36.5 s
    assert onset["stopped_after_moving"] >= 1


def test_overlap_feature():
    rows = [(t, i, 100 + i * 5, 100, 40, 30) for t in np.arange(0, 6, 0.5) for i in (1, 2)]
    rec = pd.DataFrame(rows, columns=["t", "id", "cx", "cy", "w", "h"])
    f = window_features(rec, 0, 6)
    assert f["max_iou"] > 0.5 and f["overlap_frames_frac"] == 1.0


def test_merge_events_needs_consecutive_windows():
    w = pd.DataFrame({"t_start": [0, 5, 10, 15, 20, 25], "t_end": [10, 15, 20, 25, 30, 35],
                      "flag": [False, True, True, False, True, False],
                      "score": [0, .9, .8, 0, .9, 0], "reason": ["-"] * 6})
    ev = merge_events(w, min_windows=2)
    assert len(ev) == 1 and ev[0]["start"] == 5 and ev[0]["end"] == 20


def test_rule_based_flags_only_the_stalled_video():
    normal = build_windows(make_records(120, seed=3), 0, 120, 10, 5)
    stalled = build_windows(make_records(120, stall_at=50, seed=4), 0, 120, 10, 5)
    assert len(merge_events(score_windows(normal))) == 0
    ev = merge_events(score_windows(stalled))
    assert len(ev) >= 1 and ev[0]["start"] >= 35


def test_track_video_end_to_end(tmp_path):
    fps, stride, dur = 10, 2, 120
    rec = make_records(duration=dur, dt=stride / fps, stall_at=60, seed=5)
    vp = tmp_path / "v.mp4"
    wr = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for _ in range(fps * dur):
        wr.write(np.zeros((120, 160, 3), np.uint8))
    wr.release()
    res = analyze_video(vp, tracker=ScriptedTracker(rec, stride / fps), stride=stride, win_sec=10, hop_sec=5)
    assert abs(np.median(np.diff(np.unique(res["records"]["t"]))) - stride / fps) < 1e-6
    assert res["method"] == "rules" and res["n_tracks"] > 10
    assert len(res["events"]) >= 1
    assert 45 <= res["events"][0]["start"] <= 75


def test_extract_train_and_reload(tmp_path, monkeypatch):
    from src import train_model
    labels, rows = [], []
    for vid in range(1, 9):                                   # 8 videos, 6 with a stalled vehicle
        stall = 40 + 5 * vid if vid <= 6 else None
        rec = make_records(150, stall_at=stall, seed=vid)
        w = build_windows(rec, 0, 150, 10, 5)
        w.insert(0, "video_id", vid)
        iv = [(stall, 150.0)] if stall else []
        from src.labels import label_windows
        w["label"] = label_windows(w, iv)
        rows.append(w)
        if stall:
            labels.append(f"{vid} {stall} 150")
    lf = tmp_path / "labels.txt"
    lf.write_text("\n".join(labels))
    feats = tmp_path / "features.csv"
    pd.concat(rows).to_csv(feats, index=False)
    monkeypatch.setattr(config, "LABEL_FILE", lf)
    monkeypatch.setattr(config, "FEATURES_CSV", feats)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "m.pkl")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    train_model.main()
    b = load_bundle(tmp_path / "m.pkl")
    assert b["feature_columns"] == FEATURE_COLUMNS
    assert b["metrics"]["f1"] > 0.8 and b["metrics"]["event_recall"] >= 0.8
    scored = score_windows(build_windows(make_records(120, stall_at=50, seed=77), 0, 120, 10, 5), b)
    assert scored["flag"].any()
    clean = score_windows(build_windows(make_records(120, seed=78), 0, 120, 10, 5), b)
    assert len(merge_events(clean)) == 0


def test_streamlit_app_starts():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception
