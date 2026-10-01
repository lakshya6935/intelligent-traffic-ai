# Intelligent Traffic & Possible-Accident Detection
project live link - https://intelligent-traffic-ai-bpbpkqzchrnd36sfcbt9ef.streamlit.app/

A computer-vision pipeline for the **AI City Challenge 2021 – Track 4** (crashes and stalled vehicles on highway cameras).

```
video -> OpenCV -> YOLO (vehicles) -> ByteTrack (IDs) -> movement features per 10 s window
      -> Random Forest (or transparent rules) -> alert timeline -> Streamlit dashboard
```

Alerts mean **"possible incident, please review"** - not a confirmed accident.
A stopped vehicle can be a crash, a breakdown, or a parked car.

## 1. Setup (Mac)

```bash
cd intelligent_traffic_ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg        # optional: makes annotated preview videos playable in the browser
```

## 2. Get the dataset

Download AIC21 Track 4 from the link on <https://www.aicitychallenge.org/2021-track4-download/> (Google Drive) and
check the dataset terms. **Do not commit or redistribute the videos.**

## 3. Put the files in place

```
dataset/raw/train-data/1.mp4, 2.mp4, 3.mp4 ...
dataset/raw/train-anomaly-results.txt
```

Then verify (this prints the parsed labels and each video's FPS / length):

```bash
python -m src.check_dataset
```

**Read its output.** Confirm the three columns really are `video_id start end` in *seconds* and that the
anomaly end times do not exceed the video length. If the units differ, fix `src/labels.py` before going on.

## 4. Extract features (YOLO + ByteTrack runs here - the slow step)

```bash
python -m src.extract_features --videos 1 2 3          # first test
python -m src.extract_features --limit 20              # then scale up (aim for 20+ videos)
```

* Default `--mode sample` processes only a slice around the first anomaly (120 s before, 180 s after) so a
  MacBook can cope; `--mode full` processes whole 15-minute videos.
* Every 3rd frame is processed (`--stride`). Raise `--imgsz 960` if vehicles are small/far; add `--device mps`
  to try the Apple GPU.
* Raw tracking is cached in `dataset/cache/`, so re-runs are instant.

## 5. Train + honest evaluation

```bash
python -m src.train_model
```

* Whole **videos** are held out (GroupKFold). Random window splits would leak and inflate scores.
* Prints window-level accuracy / precision / recall / F1, event-level precision / recall / F1 and
  false alarms per hour, and saves `models/accident_model.pkl`, `outputs/metrics.json`,
  `outputs/confusion_matrix.png`.
* With fewer than 3 videos it warns that the numbers are meaningless. Report only numbers you obtained
  from 10+ held-out videos.

## 6. Run the app

```bash
streamlit run app.py
```

Upload a clip, press **Analyse video**. Without a trained model the app uses the rule-based fallback
(stopped-after-moving, stalled car in flowing traffic, overlapping vehicles). Analyse a slice
(default 120 s) - a full 15-minute video takes a long time on a laptop.

## 7. Tests

```bash
python -m pytest -q
```

The tests use synthetic traffic and a scripted tracker, so they need no dataset and no YOLO weights.

## Features (per 10 s window, see `src/features.py`)

Vehicle count; mean / p90 / std speed; share of moving vehicles; stationary vehicle count, fraction and
duration; **stalled vehicle while the rest of traffic flows** (separates a breakdown from a red light or queue);
vehicles that **stopped after moving**; largest speed drop; heading change; maximum box overlap between two vehicles
(a crash proxy). Speeds are in *box-sizes per second*, so they are less sensitive to resolution and perspective.

## Known limitations (put these in your report)

* Rules/RF work on tracker output, so ID switches, occlusion, night and rain reduce accuracy.
* Thresholds in `FeatureConfig` were chosen on synthetic data; tune them on your footage.
* Ordinary congestion must not count as an anomaly, but dense slow traffic can still produce false alarms.
* This is not the official AI City evaluation (which also measures detection-time error).
* Real-time processing is not claimed: measure your own FPS before saying anything about speed.

## Viva cheat-sheet

* **YOLO** = detection (what/where). **ByteTrack** = tracking (which vehicle is which over time; IoU + Kalman
  filter, uses low-confidence boxes to recover tracks). **Random Forest** = decides normal vs anomaly from features.
* **Why not raw-pixel deep learning?** 100 videos are too few; features from tracks are small, explainable, trainable on a laptop.
* **Why group split?** Neighbouring windows of one video are near-duplicates; a random split leaks them.
* **Why F1, not accuracy?** Anomalies are rare; predicting "normal" always gives high accuracy and zero recall.

## Layout

```
app.py                 Streamlit dashboard
src/config.py          paths and defaults
src/check_dataset.py   verify labels + videos
src/tracking.py        YOLO + ByteTrack wrapper, video loop, annotated output
src/features.py        window features
src/labels.py          label parsing and window labelling
src/extract_features.py  videos -> dataset/features.csv (cached)
src/train_model.py     Random Forest, video-level CV, metrics, model file
src/analyze.py         scoring, rules fallback, event merging, end-to-end pipeline
tests/                 synthetic-data tests
```
