import cv2
import os
import json
import time
import base64
import tempfile
import numpy as np
import pandas as pd
import imageio
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple

# Import pipeline modules
from src.config import (
    MODEL_PATH,
    TEST_DATA_DIR,
    POSSESSION_DIST_THRESHOLD,
    SAMPLE_MAX_FRAMES,
    SAMPLE_FRAME_STEP
)
from src.detector import FootballDetector
from src.team_clusterer import TeamClusterer
from src.possession_tracker import PossessionTracker, PossessionInfo

# Page Configuration
st.set_page_config(
    page_title="Football Possession Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dashboard Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .team-card {
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stDownloadButton>button {
        width: 100%;
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        font-size: 1rem;
        border-radius: 8px;
        padding: 0.6rem;
    }
    .dev-card {
        background-color: #1E222A;
        border: 1px solid #2D3139;
        border-radius: 12px;
        padding: 1rem 0.5rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .dev-card:hover {
        transform: translateY(-3px);
        border-color: #1E88E5;
    }
    .dev-avatar {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        object-fit: cover;
        margin: 0 auto 0.6rem auto;
        display: block;
        border: 2px solid #1E88E5;
    }
    .dev-name {
        font-weight: 700;
        font-size: 1.05rem;
        color: #FFFFFF;
        margin-bottom: 0.4rem;
    }
    .dev-link-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        background-color: #262C36;
        color: #90CAF9 !important;
        text-decoration: none !important;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        border: 1px solid #3A404D;
        transition: background-color 0.2s ease;
    }
    .dev-link-btn:hover {
        background-color: #1E88E5;
        color: #FFFFFF !important;
        border-color: #1E88E5;
    }
    .platform-icon {
        width: 16px;
        height: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚽ Football Match Possession Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Team Color Clustering & Ball Possession Tracking</div>', unsafe_allow_html=True)

def draw_broadcast_scoreboard(
    frame: np.ndarray,
    team1_name: str,
    team2_name: str,
    t1_bgr: Tuple[int, int, int],
    t2_bgr: Tuple[int, int, int],
    pct_t1: float,
    pct_t2: float,
    pct_none: float,
    poss_info: Optional[PossessionInfo] = None
):
    """Draws a permanent TV broadcast style possession scoreboard at the top of the frame."""
    h, w = frame.shape[:2]
    
    # Broadcast bar dimensions
    bar_w = min(820, w - 40)
    bar_h = 45
    x1, y1 = (w - bar_w) // 2, 15
    x2, y2 = x1 + bar_w, y1 + bar_h

    # Dark broadcast container background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

    # 1. Team 1 Block
    t1_active = (poss_info is not None and poss_info.team_id == 0)
    cv2.rectangle(frame, (x1 + 8, y1 + 8), (x1 + 26, y2 - 8), t1_bgr, -1)
    t1_text = f"{team1_name.upper()[:12]}: {pct_t1:.1f}%"
    if t1_active:
        cv2.rectangle(frame, (x1 + 30, y1 + 4), (x1 + 255, y2 - 4), t1_bgr, 2)
        t1_text = f"> {t1_text}"
    cv2.putText(frame, t1_text, (x1 + 35, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Divider 1
    div1_x = x1 + 265
    cv2.line(frame, (div1_x, y1 + 8), (div1_x, y2 - 8), (100, 100, 100), 1)

    # 2. Team 2 Block
    t2_active = (poss_info is not None and poss_info.team_id == 1)
    t2_start_x = div1_x + 15
    cv2.rectangle(frame, (t2_start_x, y1 + 8), (t2_start_x + 18, y2 - 8), t2_bgr, -1)
    t2_text = f"{team2_name.upper()[:12]}: {pct_t2:.1f}%"
    if t2_active:
        cv2.rectangle(frame, (t2_start_x + 22, y1 + 4), (t2_start_x + 245, y2 - 4), t2_bgr, 2)
        t2_text = f"> {t2_text}"
    cv2.putText(frame, t2_text, (t2_start_x + 26, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Divider 2
    div2_x = t2_start_x + 255
    cv2.line(frame, (div2_x, y1 + 8), (div2_x, y2 - 8), (100, 100, 100), 1)

    # 3. Loose Ball Block
    loose_text = f"LOOSE: {pct_none:.1f}%"
    cv2.putText(frame, loose_text, (div2_x + 15, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

def render_developers_section():
    """Renders the Project Developers section with avatar and platform image icons."""
    st.markdown("---")
    st.subheader("👨‍💻 Project Developers")
    st.markdown("""
    <div style="text-align: center; margin-top: -0.5rem; margin-bottom: 1.5rem; color: #B0BEC5; font-size: 1.05rem;">
        🎓 <strong>Under the supervision of:</strong> 
        <span style="color: #64B5F6; font-weight: 600;">Eng. Mohamed Samir</span> &bull; 
        <span style="color: #64B5F6; font-weight: 600;">Eng. Maram Khalid</span> &bull; 
        <span style="color: #64B5F6; font-weight: 600;">Eng. Ahmed Essam</span>
    </div>
    """, unsafe_allow_html=True)

    developers = [
        {
            "name": "Ahmed Kamal",
            "avatar": "https://github.com/ahmedkamal14.png",
            "link": "https://github.com/ahmedkamal14",
            "platform": "GitHub",
            "icon": "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='white'><path d='M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z'/></svg>"
        },
        {
            "name": "Georgios Marqus",
            "avatar": "https://drive.google.com/file/d/1jElRKGnWEOdRSqYA4q_ROj1cGxF3bsvK/view?usp=drive_link",
            "link": "https://www.linkedin.com/in/georgios-marqus",
            "platform": "LinkedIn",
            "icon": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iIzBBNjZDMiIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0Ij48cGF0aCBkPSJNMjAuNDQ3IDIwLjQ1MmgtMy41NTR2LTUuNTY5YzAtMS4zMjgtLjAyNy0zLjAzNy0xLjg1Mi0zLjAzNy0xLjg1MyAwLTIuMTM2IDEuNDQ1LTIuMTM2IDIuOTM5djUuNjY3SDkuMzUxVjloMy40MTR2MS41NjFoLjA0NmMuNDc3LS45IDEuNjM3LTEuODUgMy4zNy0xLjg1IDMuNjAxIDAgNC4yNjcgMi4zNyA0LjI2NyA1LjQ1NXY2LjI4NnpNNS4zMzcgNy40MzNjLTEuMTQ4IDAtMi4wNjMtLjkyNi0yLjA2My0yLjA2NSAwLTEuMTM4LjkyLTIuMDYzIDIuMDYzLTIuMDYzIDEuMTQgMCAyLjA2NC45MjUgMi4wNjQgMi4wNjMgMCAxLjEzOS0uOTI1IDIuMDY1LTIuMDY0IDIuMDY5em0xLjc4MiAxMy4wMTlIMy41NTVWOWgzLjU2NHYxMS40NTJ6TTIyLjIyNSAwSDEuNzcxQy43OTIgMCAwIC43NzQgMCAxLjcyOXYyMC41NDJDMCAyMy4yMjcuNzkyIDI0IDEuNzcxIDI0aDIwLjQ1MUMyMy4yIDI0IDI0IDIzLjIyNyAyNCAyMi4yNzFWMS43MjlDMjQgLjc3NCAyMy4yIDAgMjIuMjIyIDBoLjAwM3oiLz48L3N2Zz4="
        },
        {
            "name": "Omar Hamdy",
            "avatar": "https://drive.google.com/file/d/1IOK9jo-yy31kSbtsysQJ1eYe2Pnh2MwY/view?usp=drive_link",
            "link": "https://www.linkedin.com/in/omar-hamdy-883579343?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app",
            "platform": "LinkedIn",
            "icon": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iIzBBNjZDMiIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0Ij48cGF0aCBkPSJNMjAuNDQ3IDIwLjQ1MmgtMy41NTR2LTUuNTY5YzAtMS4zMjgtLjAyNy0zLjAzNy0xLjg1Mi0zLjAzNy0xLjg1MyAwLTIuMTM2IDEuNDQ1LTIuMTM2IDIuOTM5djUuNjY3SDkuMzUxVjloMy40MTR2MS41NjFoLjA0NmMuNDc3LS45IDEuNjM3LTEuODUgMy4zNy0xLjg1IDMuNjAxIDAgNC4yNjcgMi4zNyA0LjI2NyA1LjQ1NXY2LjI4NnpNNS4zMzcgNy40MzNjLTEuMTQ4IDAtMi4wNjMtLjkyNi0yLjA2My0yLjA2NSAwLTEuMTM4LjkyLTIuMDYzIDIuMDYzLTIuMDYzIDEuMTQgMCAyLjA2NC45MjUgMi4wNjQgMi4wNjMgMCAxLjEzOS0uOTI1IDIuMDY1LTIuMDY4IDIuMDY1em0xLjc4MiAxMy4wMTlIMy41NTVWOWgzLjU2NHYxMS40NTJ6TTIyLjIyNSAwSDEuNzcxQy43OTIgMCAwIC43NzQgMCAxLjcyOXYyMC41NDJDMCAyMy4yMjcuNzkyIDI0IDEuNzcxIDI0aDIwLjQ1MUMyMy4yIDI0IDI0IDIzLjIyNyAyNCAyMi4yNzFWMS43MjlDMjQgLjc3NCAyMy4yIDAgMjIuMjIyIDBoLjAwM3oiLz48L3N2Zz4="
        },
        {
            "name": "Abdallah Abdelrahim",
            "avatar": "https://drive.google.com/file/d/1kaulcuEoSXsfoHy3xoINlHE7qqp_rnhE/view?usp=drive_link",
            "link": "https://www.linkedin.com/in/abdullah-abdulrahem-abdulaty",
            "platform": "LinkedIn",
            "icon": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iIzBBNjZDMiIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0Ij48cGF0aCBkPSJNMjAuNDQ3IDIwLjQ1MmgtMy41NTR2LTUuNTY5YzAtMS4zMjgtLjAyNy0zLjAzNy0xLjg1Mi0zLjAzNy0xLjg1MyAwLTIuMTM2IDEuNDQ1LTIuMTM2IDIuOTM5djUuNjY3SDkuMzUxVjloMy40MTR2MS41NjFoLjA0NmMuNDc3LS45IDEuNjM3LTEuODUgMy4zNy0xLjg1IDMuNjAxIDAgNC4yNjcgMi4zNyA0LjI2NyA1LjQ1NXY2LjI4NnpNNS4zMzcgNy40MzNjLTEuMTQ4IDAtMi4wNjMtLjkyNi0yLjA2My0yLjA2NSAwLTEuMTM4LjkyLTIuMDYzIDIuMDYzLTIuMDYzIDEuMTQgMCAyLjA2NC45MjUgMi4wNjQgMi4wNjMgMCAxLjEzOS0uOTI1IDIuMDY1LTIuMDY0IDIuMDY1em0xLjc4MiAxMy4wMTlIMy41NTVWOWgzLjU2NHYxMS40NTJ6TTIyLjIyNSAwSDEuNzcxQy43OTIgMCAwIC43NzQgMCAxLjcyOXYyMC41NDJDMCAyMy4yMjcuNzkyIDI0IDEuNzcxIDI0aDIwLjQ1MUMyMy4yIDI0IDI0IDIzLjIyNyAyNCAyMi4yNzFWMS43MjlDMjQgLjc3NCAyMy4yIDAgMjIuMjIyIDBoLjAwM3oiLz48L3N2Zz4="
        },
        {
            "name": "Mohamed Hamed",
            "avatar": "https://drive.google.com/file/d/1l_oVe5JGMpwwTKgA7Qs8lP2lTAsRVJ_H/view?usp=drive_link",
            "link": "https://www.linkedin.com/in/mohamed-hamed-035a07340?utm_source=share_via&utm_content=profile&utm_medium=member_ios",
            "platform": "LinkedIn",
            "icon": "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0iIzBBNjZDMiIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0Ij48cGF0aCBkPSJNMjAuNDQ3IDIwLjQ1MmgtMy41NTR2LTUuNTY5YzAtMS4zMjgtLjAyNy0zLjAzNy0xLjg1Mi0zLjAzNy0xLjg1MyAwLTIuMTM2IDEuNDQ1LTIuMTM2IDIuOTM5djUuNjY3SDkuMzUxVjloMy40MTR2MS41NjFoLjA0NmMuNDc3LS45IDEuNjM3LTEuODUgMy4zNy0xLjg1IDMuNjAxIDAgNC4yNjcgMi4zNyA0LjI2NyA1LjQ1NXY2LjI4NnpNNS4zMzcgNy40MzNjLTEuMTQ4IDAtMi4wNjMtLjkyNi0yLjA2My0yLjA2NSAwLTEuMTM4LjkyLTIuMDYzIDIuMDYzLTIuMDYzIDEuMTQgMCAyLjA2NC45MjUgMi4wNjQgMi4wNjMgMCAxLjEzOS0uOTI1IDIuMDY1LTIuMDY4IDIuMDY1em0xLjc4MiAxMy4wMTlIMy41NTVWOWgzLjU2NHYxMS40NTJ6TTIyLjIyNSAwSDEuNzcxQy43OTIgMCAwIC43NzQgMCAxLjcyOXYyMC41NDJDMCAyMy4yMjcuNzkyIDI0IDEuNzcxIDI0aDIwLjQ1MUMyMy4yIDI0IDI0IDIzLjIyNyAyNCAyMi4yNzFWMS43MjlDMjQgLjc3NCAyMy4yIDAgMjIuMjIyIDBoLjAwM3oiLz48L3N2Zz4="
        }
    ]

    dev_cols = st.columns(len(developers))

    for col, dev in zip(dev_cols, developers):
        avatar_url = dev['avatar']
        fallback_avatar = f"https://ui-avatars.com/api/?name={dev['name'].replace(' ', '+')}&background=0E66C2&color=fff&size=128&bold=true"
        
        if os.path.exists(avatar_url):
            ext = Path(avatar_url).suffix.lower().replace(".", "")
            mime = "jpeg" if ext in ["jpg", "jpeg"] else ext
            with open(avatar_url, "rb") as f:
                data = f.read()
            if data.startswith(b"\xff\xd8") or data.startswith(b"\x89PNG"):
                b64 = base64.b64encode(data).decode()
                avatar_url = f"data:image/{mime};base64,{b64}"
            else:
                avatar_url = fallback_avatar
        elif "drive.google.com" in avatar_url and "/file/d/" in avatar_url:
            file_id = avatar_url.split("/file/d/")[1].split("/")[0]
            avatar_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w400"

        with col:
            st.markdown(
                f'''
                <div class="dev-card">
                    <img src="{avatar_url}" class="dev-avatar" alt="{dev['name']}" referrerpolicy="no-referrer" onerror="this.onerror=null;this.src='{fallback_avatar}';">
                    <div class="dev-name">{dev['name']}</div>
                    <a href="{dev['link']}" target="_blank" class="dev-link-btn">
                        <img src="{dev['icon']}" class="platform-icon" alt="{dev['platform']}"> {dev['platform']}
                    </a>
                </div>
                ''',
                unsafe_allow_html=True
            )

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("⚽ Team Configuration")

team1_name = st.sidebar.text_input("Team 1 Name", value="Home Team")
team2_name = st.sidebar.text_input("Team 2 Name", value="Away Team")

st.sidebar.markdown("---")
st.sidebar.header("🎛 Analysis Parameters")

dist_threshold = st.sidebar.slider(
    "Ball Possession Distance (px)",
    min_value=40.0,
    max_value=150.0,
    value=POSSESSION_DIST_THRESHOLD,
    step=5.0
)

process_step = st.sidebar.slider(
    "Frame Processing Step (Drop Frames)",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="Process 1 frame every N frames (e.g., 3 = process every 3rd frame for 3x faster speed)."
)

sample_max_frames = st.sidebar.slider(
    "Clustering Sample Frames",
    min_value=60,
    max_value=300,
    value=SAMPLE_MAX_FRAMES,
    step=30
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Pipeline Workflow:**\n1. Samples initial video frames.\n2. Masks pitch grass & extracts jersey colors.\n3. Fits GMM ($k=2$) in CIELAB color space.\n4. Tracks ball proximity & renders broadcast possession scoreboard.")

# ---------------------------------------------------------
# Input Video Selection
# ---------------------------------------------------------
st.subheader("📹 Input Video Source")

video_source = st.radio("Select Video Source:", ["Upload Video", "Use Sample Video"], horizontal=True)

temp_video_path = None

if video_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload Match Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, f"upload_{uploaded_file.name}")
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"✅ Uploaded: `{uploaded_file.name}`")
else:
    sample_videos = list(TEST_DATA_DIR.glob("*.mp4"))
    sample_options = [v.name for v in sample_videos if not v.name.endswith("_output.mp4")]
    if sample_options:
        selected_sample = st.selectbox("Select Sample Video:", sample_options)
        temp_video_path = str(TEST_DATA_DIR / selected_sample)
        st.info(f"Selected sample video: `{selected_sample}`")
    else:
        st.warning("No sample videos found in 'Test Data' directory.")

analyze_button = st.button("🚀 Analyze Video & Track Possession", use_container_width=True, type="primary")

# ---------------------------------------------------------
# Pipeline Execution & Processing
# ---------------------------------------------------------
if analyze_button:
    if not temp_video_path or not os.path.exists(temp_video_path):
        st.error("Please upload or select a valid video file before starting analysis.")
        st.stop()

    if not Path(MODEL_PATH).exists():
        st.error(f"Model weights not found at {MODEL_PATH}")
        st.stop()

    st.markdown("---")

    # 1. Initialize Pipeline Components
    detector = FootballDetector(model_path=str(MODEL_PATH))
    clusterer = TeamClusterer(n_teams=2)
    clusterer.team_names = [team1_name, team2_name]
    possession_tracker = PossessionTracker(dist_threshold=dist_threshold)

    # ---------------------------------------------------------
    # PHASE 1: Harvesting Crops & Fitting GMM Team Clusterer
    # ---------------------------------------------------------
    with st.spinner(f"Phase 1: Harvesting player crops from first {sample_max_frames} frames..."):
        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sampled_crops = []
        frame_idx = 0
        pbar_harvest = st.progress(0, text="Extracting player jersey colors...")

        while cap.isOpened() and frame_idx < sample_max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % SAMPLE_FRAME_STEP == 0:
                frame_state = detector.process_frame(frame, frame_index=frame_idx, use_tracking=False)
                for player in frame_state.players.values():
                    crop = clusterer.crop_torso(frame, player.bbox)
                    if crop.size > 0:
                        sampled_crops.append(crop)

            frame_idx += 1
            pbar_harvest.progress(min(1.0, frame_idx / sample_max_frames), text=f"Harvested {len(sampled_crops)} crops across {frame_idx} frames...")

        cap.release()
        pbar_harvest.empty()

    if len(sampled_crops) < 4:
        st.error("Not enough players detected in initial frames to perform team clustering.")
        st.stop()

    # Fit GMM
    fit_success = clusterer.fit_gmm(sampled_crops)
    if not fit_success:
        st.error("Team clustering GMM fitting failed.")
        st.stop()

    detector.reset_tracker()

    # Display Identified Team Color Cards
    col_t1, col_t2 = st.columns(2)
    t1_bgr = clusterer.team_colors_bgr[0]
    t2_bgr = clusterer.team_colors_bgr[1]

    t1_hex = f"#{t1_bgr[2]:02x}{t1_bgr[1]:02x}{t1_bgr[0]:02x}"
    t2_hex = f"#{t2_bgr[2]:02x}{t2_bgr[1]:02x}{t2_bgr[0]:02x}"

    with col_t1:
        st.markdown(f'<div class="team-card" style="background-color: {t1_hex};">👕 {team1_name} (Cluster 1)</div>', unsafe_allow_html=True)
    with col_t2:
        st.markdown(f'<div class="team-card" style="background-color: {t2_hex};">👕 {team2_name} (Cluster 2)</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # PHASE 2: Video Processing & Broadcast Banner Rendering
    # ---------------------------------------------------------
    st.subheader(f"⚙️ Processing Video (Step = {process_step} frames)")

    output_filename = f"possession_annotated_{int(time.time())}.mp4"
    output_path = os.path.join(tempfile.gettempdir(), output_filename)
    json_path = os.path.join(tempfile.gettempdir(), f"possession_data_{int(time.time())}.json")
    csv_path = os.path.join(tempfile.gettempdir(), f"possession_data_{int(time.time())}.csv")

    output_fps = max(1.0, fps / process_step)

    # Use ImageIO with libx264 for HTML5 web browser video compatibility
    try:
        video_writer = imageio.get_writer(output_path, fps=output_fps, codec='libx264', macro_block_size=1)
        use_imageio = True
    except Exception as e:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        use_imageio = False

    progress_bar = st.progress(0)
    status_text = st.empty()

    cap = cv2.VideoCapture(temp_video_path)
    frame_idx = 0
    start_time = time.time()
    possession_counts = {0: 0, 1: 0, "none": 0}
    
    # Possession Event Logger
    frame_logs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame
        if frame_idx % process_step == 0:
            frame_state = detector.process_frame(frame, frame_index=frame_idx, use_tracking=True)
            poss_info = possession_tracker.get_possession(frame, frame_state, clusterer)

            annotated = frame.copy()

            # 1. Draw Referees (White)
            for r_id, ref in frame_state.referees.items():
                b = ref.bbox
                cv2.rectangle(annotated, (b.x1, b.y1), (b.x2, b.y2), (255, 255, 255), 1)

            # 2. Draw Players (Cyan)
            for p_id, player in frame_state.players.items():
                b = player.bbox
                cv2.rectangle(annotated, (b.x1, b.y1), (b.x2, b.y2), (255, 255, 0), 2)

            # 3. Draw Ball (Green)
            if frame_state.ball:
                bb = frame_state.ball.bbox
                cv2.rectangle(annotated, (bb.x1, bb.y1), (bb.x2, bb.y2), (0, 255, 0), 2)
                cv2.putText(annotated, "BALL", (bb.x1, max(15, bb.y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # 4. Possessor Highlight
            time_sec = round(frame_idx / fps, 2) if fps > 0 else 0.0

            if poss_info is not None:
                possession_counts[poss_info.team_id] += 1
                pb = poss_info.player_obj.bbox
                team_col = poss_info.team_color_bgr

                cv2.rectangle(annotated, (pb.x1, pb.y1), (pb.x2, pb.y2), team_col, 4)
                if frame_state.ball:
                    cv2.line(annotated, frame_state.ball.bbox.center, pb.bottom_center, team_col, 2)

                frame_logs.append({
                    "frame": frame_idx,
                    "timestamp_sec": time_sec,
                    "possession": poss_info.team_name,
                    "player_id": poss_info.player_id,
                    "confidence": round(poss_info.confidence, 4)
                })
            else:
                possession_counts["none"] += 1
                frame_logs.append({
                    "frame": frame_idx,
                    "timestamp_sec": time_sec,
                    "possession": "Loose Ball",
                    "player_id": None,
                    "confidence": 0.0
                })

            # Calculate Live Cumulative Percentages for Broadcast Scoreboard
            curr_proc = max(1, len(frame_logs))
            curr_pct_t1 = (possession_counts[0] / curr_proc) * 100
            curr_pct_t2 = (possession_counts[1] / curr_proc) * 100
            curr_pct_none = (possession_counts["none"] / curr_proc) * 100

            # 5. Draw Permanent Broadcast Scoreboard Banner Overlay
            draw_broadcast_scoreboard(
                frame=annotated,
                team1_name=team1_name,
                team2_name=team2_name,
                t1_bgr=t1_bgr,
                t2_bgr=t2_bgr,
                pct_t1=curr_pct_t1,
                pct_t2=curr_pct_t2,
                pct_none=curr_pct_none,
                poss_info=poss_info
            )

            # Write frame to video file (ImageIO RGB or OpenCV BGR)
            if use_imageio:
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_writer.append_data(frame_rgb)
            else:
                video_writer.write(annotated)

        # Progress Update
        total_proc = max(1, frame_idx + 1)
        progress_pct = min(1.0, total_proc / total_frames) if total_frames > 0 else 0
        progress_bar.progress(progress_pct)
        status_text.text(f"Processing frame {frame_idx + 1}/{total_frames} (Sampling every {process_step} frames)...")

        frame_idx += 1

    cap.release()
    if use_imageio:
        video_writer.close()
    else:
        video_writer.release()

    progress_bar.progress(1.0)
    status_text.success("✅ Processing & H.264 Video Encoding Complete!")

    # ---------------------------------------------------------
    # PHASE 3: Save Data File (JSON & CSV)
    # ---------------------------------------------------------
    processed_frames_count = max(1, len(frame_logs))
    pct_t1 = (possession_counts[0] / processed_frames_count) * 100
    pct_t2 = (possession_counts[1] / processed_frames_count) * 100
    pct_none = (possession_counts["none"] / processed_frames_count) * 100

    match_report = {
        "match_info": {
            "team_1": team1_name,
            "team_2": team2_name,
            "total_frames_in_video": frame_idx,
            "processed_frames": processed_frames_count,
            "step": process_step,
            "fps": fps,
            "duration_sec": round(frame_idx / fps, 2) if fps > 0 else 0
        },
        "possession_summary": {
            f"{team1_name}_possession_pct": round(pct_t1, 2),
            f"{team2_name}_possession_pct": round(pct_t2, 2),
            "loose_ball_pct": round(pct_none, 2),
            f"{team1_name}_frames": possession_counts[0],
            f"{team2_name}_frames": possession_counts[1],
            "loose_ball_frames": possession_counts["none"]
        },
        "frame_events": frame_logs
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(match_report, f, indent=2)

    df_logs = pd.DataFrame(frame_logs)
    df_logs.to_csv(csv_path, index=False)

    # ---------------------------------------------------------
    # PHASE 4: Video Player & Dynamic Plotly Analytics Dashboard
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📺 Final Processed Video")
    if os.path.exists(output_path):
        st.video(output_path)

    st.markdown("---")
    st.subheader("📊 Dynamic Match Possession Analytics (Plotly)")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(f"⚽ {team1_name}", f"{pct_t1:.1f}%", delta=f"{possession_counts[0]} processed frames")
    with col_m2:
        st.metric(f"⚽ {team2_name}", f"{pct_t2:.1f}%", delta=f"{possession_counts[1]} processed frames")
    with col_m3:
        st.metric("⚪ Loose Ball / Unassigned", f"{pct_none:.1f}%", delta=f"{possession_counts['none']} processed frames")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 🥧 Overall Possession Share")
        fig_pie = px.pie(
            names=[team1_name, team2_name, "Loose Ball"],
            values=[pct_t1, pct_t2, pct_none],
            hole=0.45,
            color=[team1_name, team2_name, "Loose Ball"],
            color_discrete_map={
                team1_name: t1_hex,
                team2_name: t2_hex,
                "Loose Ball": "#777777"
            }
        )
        fig_pie.update_traces(textinfo="percent+label", hoverinfo="label+percent+value")
        fig_pie.update_layout(template="plotly_dark", margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.markdown("#### 📈 Possession Control Shift Over Match Duration")
        if not df_logs.empty:
            df_logs['t1_cum'] = (df_logs['possession'] == team1_name).cumsum()
            df_logs['t2_cum'] = (df_logs['possession'] == team2_name).cumsum()
            df_logs['loose_cum'] = (df_logs['possession'] == "Loose Ball").cumsum()
            df_logs['total_cum'] = np.arange(1, len(df_logs) + 1)

            df_logs[f"{team1_name} (%)"] = (df_logs['t1_cum'] / df_logs['total_cum']) * 100
            df_logs[f"{team2_name} (%)"] = (df_logs['t2_cum'] / df_logs['total_cum']) * 100
            df_logs["Loose Ball (%)"] = (df_logs['loose_cum'] / df_logs['total_cum']) * 100

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df_logs['timestamp_sec'], y=df_logs[f"{team1_name} (%)"],
                mode='lines', name=team1_name, line=dict(color=t1_hex, width=3)
            ))
            fig_line.add_trace(go.Scatter(
                x=df_logs['timestamp_sec'], y=df_logs[f"{team2_name} (%)"],
                mode='lines', name=team2_name, line=dict(color=t2_hex, width=3)
            ))
            fig_line.add_trace(go.Scatter(
                x=df_logs['timestamp_sec'], y=df_logs["Loose Ball (%)"],
                mode='lines', name="Loose Ball", line=dict(color="#777777", width=2, dash='dash')
            ))
            fig_line.update_layout(
                xaxis_title="Match Time (Seconds)",
                yaxis_title="Cumulative Possession (%)",
                template="plotly_dark",
                hovermode="x unified",
                margin=dict(t=30, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_line, use_container_width=True)

    # ---------------------------------------------------------
    # Downloads Section
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Export & Download Results")

    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        if os.path.exists(output_path):
            with open(output_path, "rb") as vf:
                st.download_button(
                    label="⬇️ Download Match Video (.mp4)",
                    data=vf.read(),
                    file_name=f"{team1_name}_vs_{team2_name}_possession.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

    with dl_col2:
        if os.path.exists(json_path):
            with open(json_path, "rb") as jf:
                st.download_button(
                    label="⬇️ Download Possession Data (.json)",
                    data=jf.read(),
                    file_name=f"{team1_name}_vs_{team2_name}_possession.json",
                    mime="application/json",
                    use_container_width=True
                )

    with dl_col3:
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as cf:
                st.download_button(
                    label="⬇️ Download Possession Report (.csv)",
                    data=cf.read(),
                    file_name=f"{team1_name}_vs_{team2_name}_possession.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# Always render Developers Section at the bottom of the main page
render_developers_section()
