import argparse
import cv2
import time
from pathlib import Path
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
from src.possession_tracker import PossessionTracker

def run_pipeline(
    input_video: Path,
    output_video: Path,
    sample_max_frames: int = SAMPLE_MAX_FRAMES,
    sample_step: int = SAMPLE_FRAME_STEP,
    dist_threshold: float = POSSESSION_DIST_THRESHOLD
):
    if not input_video.exists():
        print(f"❌ Error: Input video not found at {input_video}")
        return

    print("==================================================")
    print(" ⚽ FOOTBALL POSSESSION & TEAM CLUSTERING PIPELINE")
    print("==================================================")
    print(f"📹 Input Video:  {input_video.name}")
    print(f"🤖 YOLO Model:   {MODEL_PATH}")
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
            # Use detection (use_tracking=False) for Phase 1 crop harvesting
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

    # Reset tracker state before starting Phase 2 full video processing
    detector.reset_tracker()

    # -------------------------------------------------------------
    # PHASE 2: Process Full Video & Classify Ball Possessor Team
    # -------------------------------------------------------------
    print(f"\n--- PHASE 2: Processing full video ({total_frames} frames) ---")
    cap = cv2.VideoCapture(str(input_video))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    frame_index = 0
    start_time = time.time()
    possession_counts = {0: 0, 1: 0, "none": 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with ByteTrack active
        frame_state = detector.process_frame(frame, frame_index=frame_index, use_tracking=True)
        poss_info = possession_tracker.get_possession(frame, frame_state, clusterer)

        annotated = frame.copy()

        # 1. Draw Referees (White)
        for r_id, ref in frame_state.referees.items():
            b = ref.bbox
            cv2.rectangle(annotated, (b.x1, b.y1), (b.x2, b.y2), (255, 255, 255), 1)
            cv2.putText(annotated, f"REF:{r_id}", (b.x1, max(15, b.y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 2. Draw Players (Cyan default)
        for p_id, player in frame_state.players.items():
            b = player.bbox
            cv2.rectangle(annotated, (b.x1, b.y1), (b.x2, b.y2), (255, 255, 0), 2)
            cv2.putText(annotated, f"P:{p_id}", (b.x1, max(15, b.y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # 3. Draw Ball (Green)
        if frame_state.ball:
            bb = frame_state.ball.bbox
            cv2.rectangle(annotated, (bb.x1, bb.y1), (bb.x2, bb.y2), (0, 255, 0), 2)
            cv2.putText(annotated, "BALL", (bb.x1, max(15, bb.y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # 4. Highlight Ball Possessor & Display Team Possession Banner
        if poss_info is not None:
            possession_counts[poss_info.team_id] += 1
            pb = poss_info.player_obj.bbox
            team_col = poss_info.team_color_bgr
            
            # Thick glowing highlight box around possessor in team color
            cv2.rectangle(annotated, (pb.x1, pb.y1), (pb.x2, pb.y2), team_col, 4)
            
            # Connecting line from ball center to player feet
            if frame_state.ball:
                cv2.line(annotated, frame_state.ball.bbox.center, pb.bottom_center, team_col, 2)

            # Top Possession Callout Banner
            banner_text = f"POSSESSION: {poss_info.team_name.upper()} (Player #{poss_info.player_id} | {poss_info.confidence * 100:.1f}%)"
            cv2.rectangle(annotated, (20, 20), (580, 65), (0, 0, 0), -1)
            cv2.rectangle(annotated, (20, 20), (580, 65), team_col, 2)
            cv2.putText(annotated, banner_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, team_col, 2)
        else:
            possession_counts["none"] += 1
            cv2.rectangle(annotated, (20, 20), (320, 65), (0, 0, 0), -1)
            cv2.putText(annotated, "POSSESSION: LOOSE BALL", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        out.write(annotated)

        if frame_index > 0 and frame_index % 30 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_index / elapsed
            print(f"  -> Processed frame {frame_index}/{total_frames} ({fps_proc:.1f} FPS) | Elapsed: {elapsed:.1f}s")

        frame_index += 1

    cap.release()
    out.release()

    total_time = time.time() - start_time
    print("\n==================================================")
    print(" ✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("==================================================")
    print(f"⏱ Total Time:  {total_time:.2f} seconds")
    print(f"📊 Possession Stats:")
    print(f"   - Team 1 Possession: {possession_counts[0]} frames ({possession_counts[0]/total_frames*100:.1f}%)")
    print(f"   - Team 2 Possession: {possession_counts[1]} frames ({possession_counts[1]/total_frames*100:.1f}%)")
    print(f"   - Loose Ball:        {possession_counts['none']} frames ({possession_counts['none']/total_frames*100:.1f}%)")
    print(f"🎥 Output Saved To: {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Team Clustering & Possession Tracker")
    parser.add_argument("--video", type=str, default=None, help="Input video filename or path inside 'Test Data'")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--sample-frames", type=int, default=SAMPLE_MAX_FRAMES, help="Number of initial frames to sample for clustering")
    parser.add_argument("--step", type=int, default=SAMPLE_FRAME_STEP, help="Sampling frame step")
    parser.add_argument("--dist-threshold", type=float, default=POSSESSION_DIST_THRESHOLD, help="Max distance threshold for ball possession (px)")

    args = parser.parse_args()

    # Determine input video path
    if args.video is not None:
        video_p = Path(args.video)
        if not video_p.is_absolute():
            video_p = INPUT_VIDEO_PATH.parent / args.video
    else:
        video_p = INPUT_VIDEO_PATH

    # Determine output video path
    if args.output is not None:
        out_p = Path(args.output)
    else:
        out_p = video_p.parent / f"{video_p.stem}_possession_output.mp4"

    run_pipeline(
        input_video=video_p,
        output_video=out_p,
        sample_max_frames=args.sample_frames,
        sample_step=args.step,
        dist_threshold=args.dist_threshold
    )
