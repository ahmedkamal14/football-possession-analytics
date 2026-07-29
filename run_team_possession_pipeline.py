import argparse
import cv2
import time
import imageio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from src.config import (
    INPUT_VIDEO_PATH,
    OUTPUT_VIDEO_PATH,
    MODEL_PATH,
    SAMPLE_MAX_FRAMES,
    SAMPLE_FRAME_STEP,
    POSSESSION_DIST_THRESHOLD
)
from src.detector import FootballDetector
from src.team_clusterer import TeamClusterer
from src.possession_tracker import PossessionTracker, PossessionInfo

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

def run_pipeline(
    input_video: Path,
    output_video: Path,
    sample_max_frames: int = SAMPLE_MAX_FRAMES,
    sample_step: int = SAMPLE_FRAME_STEP,
    process_step: int = 3,
    dist_threshold: float = POSSESSION_DIST_THRESHOLD
):
    if not input_video.exists():
        print(f"❌ Error: Input video not found at {input_video}")
        return

    print("==================================================")
    print(" ⚽ FOOTBALL POSSESSION & TEAM CLUSTERING PIPELINE")
    print("==================================================")
    print(f"📹 Input Video:   {input_video.name}")
    print(f"🤖 YOLO Model:    {MODEL_PATH}")
    print(f"⏩ Process Step:  Every {process_step} frames (Drop {process_step - 1} frames)")
    print(f"🎯 Dist Threshold: {dist_threshold} px")

    # Initialize components
    detector = FootballDetector(model_path=str(MODEL_PATH))
    clusterer = TeamClusterer(n_teams=2)
    possession_tracker = PossessionTracker(dist_threshold=dist_threshold)

    # -------------------------------------------------------------
    # PHASE 1: Capture first N frames (default 180), drop each 3 frames
    # -------------------------------------------------------------
    print(f"\n--- PHASE 1: Sampling first {sample_max_frames} frames (step = {sample_step}) ---")
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        print("❌ Error: Failed to open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sampled_crops = []
    frame_index = 0

    while cap.isOpened() and frame_index < sample_max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % sample_step == 0:
            frame_state = detector.process_frame(frame, frame_index=frame_index, use_tracking=False)
            for player in frame_state.players.values():
                crop = clusterer.crop_torso(frame, player.bbox)
                if crop.size > 0:
                    sampled_crops.append(crop)

        frame_index += 1

    cap.release()
    print(f"✅ Captured {len(sampled_crops)} player jersey crops across first {frame_index} frames.")

    # Fit Gaussian Mixture Model (k=2) in CIELAB color space
    fit_success = clusterer.fit_gmm(sampled_crops)
    if not fit_success:
        print("❌ Error: Team clustering failed.")
        return

    detector.reset_tracker()
    t1_bgr = clusterer.team_colors_bgr[0]
    t2_bgr = clusterer.team_colors_bgr[1]

    # -------------------------------------------------------------
    # PHASE 2: Process Video (Sampling Every Nth Frame)
    # -------------------------------------------------------------
    output_fps = max(1.0, fps / process_step)
    print(f"\n--- PHASE 2: Processing video ({total_frames} frames, process every {process_step} frames at {output_fps:.1f} FPS) ---")
    cap = cv2.VideoCapture(str(input_video))
    
    try:
        out_writer = imageio.get_writer(str(output_video), fps=output_fps, codec='libx264', macro_block_size=1)
        use_imageio = True
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_video), fourcc, output_fps, (width, height))
        use_imageio = False

    frame_index = 0
    processed_count = 0
    start_time = time.time()
    possession_counts = {0: 0, 1: 0, "none": 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % process_step == 0:
            processed_count += 1
            frame_state = detector.process_frame(frame, frame_index=frame_index, use_tracking=True)
            poss_info = possession_tracker.get_possession(frame, frame_state, clusterer)

            annotated = frame.copy()

            # 1. Draw Referees (White)
            for r_id, ref in frame_state.referees.items():
                b = ref.bbox
                cv2.rectangle(annotated, (b.x1, b.y1), (b.x2, b.y2), (255, 255, 255), 1)

            # 2. Draw Players (Cyan default)
            for p_id, player in frame_state.players.items():
                b = player.bbox
                cv2.rectangle(annotated, (b.x1, b.y1), (b.x2, b.y2), (255, 255, 0), 2)

            # 3. Draw Ball (Green)
            if frame_state.ball:
                bb = frame_state.ball.bbox
                cv2.rectangle(annotated, (bb.x1, bb.y1), (bb.x2, bb.y2), (0, 255, 0), 2)
                cv2.putText(annotated, "BALL", (bb.x1, max(15, bb.y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            # 4. Highlight Ball Possessor
            if poss_info is not None:
                possession_counts[poss_info.team_id] += 1
                pb = poss_info.player_obj.bbox
                team_col = poss_info.team_color_bgr
                
                cv2.rectangle(annotated, (pb.x1, pb.y1), (pb.x2, pb.y2), team_col, 4)
                if frame_state.ball:
                    cv2.line(annotated, frame_state.ball.bbox.center, pb.bottom_center, team_col, 2)
            else:
                possession_counts["none"] += 1

            # Live cumulative percentages
            curr_proc = max(1, processed_count)
            curr_pct_t1 = (possession_counts[0] / curr_proc) * 100
            curr_pct_t2 = (possession_counts[1] / curr_proc) * 100
            curr_pct_none = (possession_counts["none"] / curr_proc) * 100

            # 5. Draw Permanent Broadcast Scoreboard Banner Overlay
            draw_broadcast_scoreboard(
                frame=annotated,
                team1_name=clusterer.team_names[0],
                team2_name=clusterer.team_names[1],
                t1_bgr=t1_bgr,
                t2_bgr=t2_bgr,
                pct_t1=curr_pct_t1,
                pct_t2=curr_pct_t2,
                pct_none=curr_pct_none,
                poss_info=poss_info
            )

            if use_imageio:
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                out_writer.append_data(frame_rgb)
            else:
                out_writer.write(annotated)

        if frame_index > 0 and frame_index % 30 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_index / elapsed
            print(f"  -> Video frame {frame_index}/{total_frames} ({fps_proc:.1f} FPS) | Elapsed: {elapsed:.1f}s")

        frame_index += 1

    cap.release()
    if use_imageio:
        out_writer.close()
    else:
        out_writer.release()

    total_time = time.time() - start_time
    total_proc = max(1, processed_count)
    print("\n==================================================")
    print(" ✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================")
    print(f"⏱ Total Time:        {total_time:.2f} seconds")
    print(f"📊 Processed Frames: {processed_count} (Every {process_step} frames)")
    print(f"📊 Possession Breakdown:")
    print(f"   - Team 1 Possession: {possession_counts[0]} frames ({possession_counts[0]/total_proc*100:.1f}%)")
    print(f"   - Team 2 Possession: {possession_counts[1]} frames ({possession_counts[1]/total_proc*100:.1f}%)")
    print(f"   - Loose Ball:        {possession_counts['none']} frames ({possession_counts['none']/total_proc*100:.1f}%)")
    print(f"🎥 Output Saved To:   {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Team Clustering & Possession Tracker")
    parser.add_argument("--video", type=str, default=None, help="Input video filename or path inside 'Test Data'")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--sample-frames", type=int, default=SAMPLE_MAX_FRAMES, help="Number of initial frames to sample for clustering")
    parser.add_argument("--step", type=int, default=SAMPLE_FRAME_STEP, help="Sampling frame step for Phase 1")
    parser.add_argument("--process-step", type=int, default=3, help="Processing frame step for Phase 2 (default: 3 = process 1 frame every 3 frames)")
    parser.add_argument("--dist-threshold", type=float, default=POSSESSION_DIST_THRESHOLD, help="Max distance threshold for ball possession (px)")

    args = parser.parse_args()

    if args.video is not None:
        video_p = Path(args.video)
        if not video_p.is_absolute():
            video_p = INPUT_VIDEO_PATH.parent / args.video
    else:
        video_p = INPUT_VIDEO_PATH

    if args.output is not None:
        out_p = Path(args.output)
    else:
        out_p = video_p.parent / f"{video_p.stem}_possession_output.mp4"

    run_pipeline(
        input_video=video_p,
        output_video=out_p,
        sample_max_frames=args.sample_frames,
        sample_step=args.step,
        process_step=args.process_step,
        dist_threshold=args.dist_threshold
    )
