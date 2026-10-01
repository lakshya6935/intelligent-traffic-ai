
# Intelligent Traffic & Possible-Accident Detection

A computer vision project that detects possible traffic incidents from highway videos using YOLO, ByteTrack and Machine Learning.

**Live Demo:**  
https://intelligent-traffic-ai-bpbpkqzchrnd36sfcbt9ef.streamlit.app/

## How It Works

```text
Video
  ↓
OpenCV
  ↓
YOLO - Vehicle Detection
  ↓
ByteTrack - Vehicle Tracking
  ↓
Feature Extraction
  ↓
Random Forest
  ↓
Possible Incident Alert
````

The system detects unusual vehicle movements and gives a **possible incident alert** for further review. It does not confirm that an accident has happened.

## Technologies

* Python
* OpenCV
* YOLO
* ByteTrack
* Random Forest
* Scikit-learn
* Streamlit
* Pandas
* NumPy

## Dataset

**AI City Challenge 2021 - Track 4**

The dataset contains highway traffic videos with traffic anomalies such as crashes and stalled vehicles.

Dataset:
[https://www.aicitychallenge.org/2021-track4-download/](https://www.aicitychallenge.org/2021-track4-download/)

The dataset videos are not included in this repository.

## Main Features

* Vehicle detection using YOLO
* Vehicle tracking using ByteTrack
* Vehicle movement analysis
* Speed and movement features
* Possible stopped/stalled vehicle detection
* Possible collision detection
* Random Forest based classification
* Alert timeline
* Video upload through Streamlit

## Run Locally

```bash
git clone https://github.com/lakshya6935/intelligent-traffic-ai.git
cd intelligent-traffic-ai

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

Then open the local Streamlit URL in your browser.

## Project Structure

```text
intelligent_traffic_ai/
│
├── app.py
├── requirements.txt
├── models/
│   └── accident_model.pkl
│
├── src/
│   ├── tracking.py
│   ├── features.py
│   ├── labels.py
│   ├── analyze.py
│   ├── train_model.py
│   └── extract_features.py
│
└── tests/
```

## Limitations

* Detection depends on video quality and tracking accuracy.
* Rain, night scenes and occlusion can affect results.
* Traffic congestion can sometimes create false alerts.
* The system reports possible incidents, not confirmed accidents.
* Processing long videos can take time.

## Future Improvements

* Improve accident detection accuracy
* Add real-time camera support
* Improve tracking in crowded scenes
* Add more training videos
* Deploy a faster version for real-time detection

```

