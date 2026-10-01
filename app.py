"""Intelligent Traffic & Accident Detection - Streamlit dashboard.

    streamlit run app.py
"""
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config
from src.analyze import analyze_video, fmt_time, load_bundle

st.set_page_config(page_title="Intelligent Traffic AI", page_icon="🚗", layout="wide")
st.title("🚗 Intelligent Traffic & Possible-Accident Detection")
st.caption("YOLO detects vehicles, ByteTrack follows them, movement features feed a Random Forest "
           "(or simple rules if no model is trained yet). Alerts are **possible incidents for human review**, "
           "not confirmed accidents.")


@st.cache_resource
def get_bundle():
    return load_bundle()


bundle = get_bundle()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    if bundle is not None:
        m = bundle.get("metrics", {})
        st.success("Trained Random Forest loaded")
        if "f1" in m:
            st.write(f"Held-out window F1: **{m['f1']:.2f}** "
                     f"(P {m['precision']:.2f} / R {m['recall']:.2f})")
        else:
            st.warning("Model was trained without validation; treat its output as a demo.")
    else:
        st.info("No trained model found - using transparent rule-based alerts.\n\n"
                "Train one with `python -m src.train_model`.")

    start_sec = st.number_input("Start at (seconds)", min_value=0, value=0, step=10)
    max_sec = st.slider("Analyse up to (seconds)", 20, 600, 120, step=10,
                        help="Long videos are slow on a laptop. Analyse a slice.")
    default_thr = float(bundle["threshold"]) if bundle else 0.5
    threshold = st.slider("Alert threshold", 0.1, 0.95, default_thr, 0.05,
                          help="Lower = more alerts (more false alarms); higher = fewer alerts.")
    min_windows = st.slider("Consecutive windows required", 1, 5, config.MIN_EVENT_WINDOWS,
                            help="Suppresses one-off blips.")
    annotate = st.checkbox("Create annotated preview video (slower)", value=False)
    device = st.selectbox("Device", ["auto", "cpu", "mps"], help="mps = Apple-silicon GPU")

# ------------------------------------------------------------------ input
uploaded = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])

if uploaded is not None:
    tmp_in = Path(tempfile.gettempdir()) / f"upload_{uploaded.name}"
    tmp_in.write_bytes(uploaded.getbuffer())
    with st.expander("Original video", expanded=False):
        st.video(str(tmp_in))

    if st.button("Analyse video", type="primary"):
        bar = st.progress(0.0, text="Loading model...")
        try:
            from src.tracking import YoloTracker
            tracker = YoloTracker(
                weights=(bundle or {}).get("yolo", config.DEFAULT_YOLO),
                imgsz=(bundle or {}).get("imgsz", config.IMGSZ),
                conf=(bundle or {}).get("conf", config.CONF),
                device=None if device == "auto" else device)
            ann = Path(tempfile.gettempdir()) / "traffic_annotated.mp4" if annotate else None
            result = analyze_video(
                tmp_in, bundle=bundle, tracker=tracker, start_sec=float(start_sec), max_seconds=float(max_sec),
                threshold=threshold, min_windows=min_windows, annotated_path=ann,
                progress_cb=lambda p: bar.progress(min(p, 1.0), text=f"Processing video... {int(p * 100)}%"))
            st.session_state["result"] = result
            bar.empty()
        except Exception as e:  # show a friendly error instead of a traceback wall
            bar.empty()
            st.error(f"Analysis failed: {e}")

# ------------------------------------------------------------------ results
res = st.session_state.get("result")
if res is not None:
    meta, win, events = res["meta"], res["windows"], res["events"]
    st.subheader("Traffic analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analysed", f"{fmt_time(meta['seg_end'] - meta['seg_start'])} min")
    c2.metric("Unique vehicle tracks", res["n_tracks"],
              help="Distinct tracker IDs. ID switches can inflate this, so it is an estimate.")
    c3.metric("Windows analysed", len(win))
    c4.metric("Possible incidents", len(events))

    if events:
        st.error(f"⚠ {len(events)} possible incident(s) flagged for review")
        st.dataframe(pd.DataFrame([{
            "From": fmt_time(e["start"]), "To": fmt_time(e["end"]),
            "Peak score": round(e["peak_score"], 2), "Signals": e["reasons"]} for e in events]),
            hide_index=True, use_container_width=True)
    else:
        st.success("No incidents flagged in the analysed segment.")

    if len(win):
        fig, ax = plt.subplots(figsize=(10, 2.8))
        mid = (win["t_start"] + win["t_end"]) / 2
        ax.plot(mid, win["score"], marker="o", ms=3, label="anomaly score")
        ax.axhline(res["threshold"], color="red", ls="--", lw=1, label="threshold")
        for e in events:
            ax.axvspan(e["start"], e["end"], color="red", alpha=0.15)
        ax.set_xlabel("video time (s)")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right")
        st.pyplot(fig)

        with st.expander("Per-window details"):
            st.dataframe(win.round(3), hide_index=True, use_container_width=True)
        st.download_button("Download window table (CSV)", win.to_csv(index=False), "windows.csv", "text/csv")

    if meta.get("annotated_path") and Path(meta["annotated_path"]).exists():
        st.subheader("Annotated preview")
        st.video(meta["annotated_path"])
        st.caption("If the preview is blank in your browser, install ffmpeg (`brew install ffmpeg`) so it can be "
                   "re-encoded to H.264, or download the file.")
        with open(meta["annotated_path"], "rb") as f:
            st.download_button("Download annotated video", f, "annotated.mp4")

    st.caption(f"Method: {res['method']} · threshold {res['threshold']:.2f} · "
               "A stopped vehicle is not automatically an accident - always verify visually.")
