<div align="center">

# ⚽ Football Possession Analytics

**AI-Powered Team Color Clustering & Ball Possession Tracking for Football Match Video**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-CV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Analytics-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-Not%20Specified-lightgrey)](#-license)
[![Status](https://img.shields.io/badge/Status-Active-success)](#)

</div>

An end-to-end computer vision pipeline that analyzes raw football match footage and produces a live **broadcast-style possession scoreboard**, automatically detects the two competing teams by jersey color, tracks which player/team controls the ball frame-by-frame, and generates an interactive analytics dashboard with downloadable video, JSON, and CSV reports — all through a **Streamlit** web app.

---

## 📑 Table of Contents

- [Features](#-features)
- [Demo Preview](#️-demo-preview)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Usage](#️-usage)
- [How It Works](#-how-it-works)
- [Project Team](#-project-team)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Features

- **Custom-trained YOLOv8 detector** for players, referees, and the ball, tuned for small/fast-moving objects (`imgsz=1280`, low confidence threshold to avoid losing the ball).
- **Automatic team identification** — no manual jersey-color tagging. Player torso crops are sampled from the first N frames and clustered into two teams using a **Gaussian Mixture Model (GMM)** in **CIELAB color space** (robust to lighting variation).
- **Ball possession tracking** based on player-to-ball proximity, with confidence scoring per frame.
- **Broadcast-style overlay** — a permanent TV-style possession bar rendered on top of the output video, highlighting the current ball possessor.
- **Interactive Streamlit dashboard**:
  - Adjustable possession distance threshold, frame sampling step, and clustering sample size.
  - Upload your own match video or use a bundled sample.
  - Live pie chart of overall possession share and a cumulative possession-over-time line chart (Plotly).
- **Exportable results** — annotated match video (`.mp4`), full possession event log (`.json`), and tabular report (`.csv`).
- **CLI pipeline script** for running possession analysis outside the web UI.

---

## 🖥️ Demo Preview

Once a video is analyzed, the app renders the annotated video, team possession cards, a possession pie chart, and a possession-over-time trend chart, with one-click downloads for the video/JSON/CSV outputs.

<div align="center">

| Live Broadcast Scoreboard | Analytics Dashboard |
|:---:|:---:|
| *`docs/screenshot-video.png`* | *`docs/screenshot-dashboard.png`* |
| <sub>Add a screenshot or GIF of the annotated video output here</sub> | <sub>Add a screenshot of the Plotly possession dashboard here</sub> |

</div>

> 💡 Tip: create a `docs/` folder, drop your screenshots/GIFs there, and update the paths above — GitHub will render them automatically.

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **Web App / UI** | Streamlit |
| **Object Detection** | YOLOv8 (Ultralytics), PyTorch / TorchVision |
| **Computer Vision** | OpenCV (`opencv-python-headless`) |
| **Team Clustering** | scikit-learn (Gaussian Mixture Model, CIELAB color space) |
| **Data Handling** | pandas, NumPy |
| **Video Encoding** | imageio + imageio-ffmpeg (H.264) |
| **Object Tracking** | lapx |
| **Visualization** | Plotly (pie chart, cumulative trend chart) |

---

## 📁 Project Structure

```
football-possession-analytics/
├── Model/                        # Trained YOLOv8 model weights (best.pt)
├── NoteBooks/                    # Jupyter notebooks (training / experimentation)
├── Test Data/                    # Sample match videos for quick testing
├── src/
│   ├── config.py                 # Paths & pipeline configuration constants
│   ├── detector.py                # FootballDetector — YOLOv8 wrapper (players/ball/referees)
│   ├── team_clusterer.py          # TeamClusterer — GMM-based jersey color clustering
│   └── possession_tracker.py      # PossessionTracker — ball-possession logic
├── app.py                        # Streamlit web application (main entry point)
├── run_team_possession_pipeline.py  # CLI script to run the full pipeline
├── test_detector.py               # Unit test for the detector (image)
├── test_detector_video.py         # Unit test for the detector (video)
├── test_team_classifier.py        # Unit test for the team clustering module
├── requirements.txt               # Python dependencies
└── requirement.txt
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- (Recommended) A CUDA-capable GPU for faster YOLOv8 inference — CPU also works, just slower.

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/ahmedkamal14/football-possession-analytics.git
cd football-possession-analytics

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Key dependencies:** `streamlit`, `ultralytics` (YOLOv8), `torch`/`torchvision`, `opencv-python-headless`, `scikit-learn`, `pandas`, `numpy`, `plotly`, `imageio` + `imageio-ffmpeg`, `lapx` (for object tracking).

> Make sure the trained model weights (`best.pt`) are available under the `Model/` directory — the app expects `MODEL_PATH` (defined in `src/config.py`) to point to it.

---

## ▶️ Usage

### Option 1 — Web App (recommended)

```bash
streamlit run app.py
```

Then, in the browser tab that opens:
1. Set team names in the sidebar.
2. Tune analysis parameters (possession distance, frame step, clustering sample size).
3. Upload a match video or pick one of the bundled sample videos.
4. Click **🚀 Analyze Video & Track Possession**.
5. Review the annotated video and analytics dashboard, then download the video/JSON/CSV outputs.

### Option 2 — Command Line Pipeline

```bash
python run_team_possession_pipeline.py --video path/to/match.mp4
```

*(Run `python run_team_possession_pipeline.py --help` to see all available arguments.)*

### Quick Model Sanity Check

To verify the YOLOv8 model detects players/ball/referees correctly on a single test image:

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict('test.jpg', imgsz=1280, conf=0.15, agnostic_nms=True)

annotated_frame = results[0].plot()
```

---

## 🧠 How It Works

1. **Detection** — YOLOv8 locates all players, the referee(s), and the ball in each frame.
2. **Team Clustering** — Player jersey crops are sampled from the opening frames, converted to CIELAB color space, and clustered into two groups via GMM to auto-assign team identity/color.
3. **Possession Tracking** — For each frame, the player closest to the ball (within a configurable distance threshold) is marked as the current possessor; their team is credited with that frame.
4. **Rendering & Reporting** — A broadcast-style scoreboard is overlaid live on the output video, and cumulative possession stats are exported to JSON/CSV for further analysis.

---

## 👨‍💻 Project Team

**Developed by:**
- Ahmed Kamal
- Georgios Marqus
- Omar Hamdy
- Abdallah Abdelrahim
- Mohamed Hamed

**Under the supervision of:**
- Eng. Mohamed Samir
- Eng. Maram Khalid
- Eng. Ahmed Essam

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## 📄 License

No license file is currently included in this repository. Add a `LICENSE` file to clarify usage terms if you plan to share or open-source this project.
