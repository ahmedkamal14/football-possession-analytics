import argparse
import cv2
import numpy as np
from pathlib import Path
from src.config import (
    INPUT_VIDEO_PATH,
    MODEL_PATH,
    SAMPLE_MAX_FRAMES,
    SAMPLE_FRAME_STEP
)
from src.detector import FootballDetector
from src.team_clusterer import TeamClusterer

def test_team_classifier(
    input_video: Path,
    num_sample_frames: int = 5,
    sample_max_frames: int = SAMPLE_MAX_FRAMES,
    sample_step: int = SAMPLE_FRAME_STEP
):
    if not input_video.exists():
        print(f"❌ Error: Video not found at {input_video}")
        return

    print("==================================================")
    print(" 🎨 TEAM CLASSIFIER UNIT TEST")
    print("==================================================")
    print(f"📹 Testing Video: {input_video.name}")

    detector = FootballDetector(model_path=str(MODEL_PATH))
    clusterer = TeamClusterer(n_teams=2)

    # 1. PHASE 1: Collect crops from first 180 frames to fit GMM
    print(f"\n1. Harvesting player crops (first {sample_max_frames} frames, step = {sample_step})...")
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        print("❌ Error opening video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_crops = []
    frame_idx = 0

    while cap.isOpened() and frame_idx < sample_max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            frame_state = detector.process_frame(frame, frame_index=frame_idx, use_tracking=False)
            for player in frame_state.players.values():
                crop = clusterer.crop_torso(frame, player.bbox)
                if crop.size > 0:
                    sampled_crops.append(crop)

        frame_idx += 1

    cap.release()
    print(f"   -> Collected {len(sampled_crops)} player torso crops.")

    # 2. Fit Gaussian Mixture Model
    print("\n2. Fitting GMM ($k=2$) team clusterer...")
    if not clusterer.fit_gmm(sampled_crops):
        print("❌ Team clustering failed.")
        return

    # 3. Classify players on sample test frames
    print(f"\n3. Running team classification on {num_sample_frames} sample frames across the video...")
    cap = cv2.VideoCapture(str(input_video))
    
    # Pick frame indices evenly distributed across video
    sample_indices = np.linspace(0, max(0, total_frames - 1), num=num_sample_frames, dtype=int)
    
    frame_count = 0
    saved_outputs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in sample_indices:
            frame_state = detector.process_frame(frame, frame_index=frame_count, use_tracking=False)
            annotated = frame.copy()

            player_classifications = {0: 0, 1: 0}

            # Classify all detected players
            for p_id, player in frame_state.players.items():
                torso_crop = clusterer.crop_torso(frame, player.bbox)
                team_id, confidence, color_bgr = clusterer.predict_team(torso_crop)
                player_classifications[team_id] += 1

                b = player.bbox
                # Draw player box in predicted team color
                cv2.rectangle(annotated, (b.x1, b.y1), (b.x2, b.y2), color_bgr, 3)
                
                label = f"Team {team_id + 1} ({confidence * 100:.0f}%)"
                cv2.rectangle(annotated, (b.x1, max(0, b.y1 - 20)), (b.x1 + 130, b.y1), color_bgr, -1)
                cv2.putText(annotated, label, (b.x1 + 3, max(12, b.y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Draw Referees (White)
            for r_id, ref in frame_state.referees.items():
                b = ref.bbox
                cv2.rectangle(annotated, (b.x1, b.y1), (b.x2, b.y2), (255, 255, 255), 1)
                cv2.putText(annotated, "REF", (b.x1, max(12, b.y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Summary banner on top
            summary_txt = f"Frame {frame_count} | Team 1: {player_classifications[0]} players | Team 2: {player_classifications[1]} players"
            cv2.rectangle(annotated, (10, 10), (600, 45), (0, 0, 0), -1)
            cv2.putText(annotated, summary_txt, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Save annotated test image
            output_img_path = input_video.parent / f"team_classifier_frame_{frame_count}.jpg"
            cv2.imwrite(str(output_img_path), annotated)
            saved_outputs.append(output_img_path)
            print(f"   -> Frame {frame_count:4d}: Classified {len(frame_state.players)} players "
                  f"(Team 1: {player_classifications[0]}, Team 2: {player_classifications[1]}) | Saved: {output_img_path.name}")

        frame_count += 1

    cap.release()

    print("\n==================================================")
    print(" ✅ TEAM CLASSIFIER TEST COMPLETE!")
    print("==================================================")
    print("📸 Annotated Test Frames Saved:")
    for out_img in saved_outputs:
        print(f"   - {out_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Team Classifier Only")
    parser.add_argument("--video", type=str, default="input_01.mp4", help="Input video filename in 'Test Data'")
    parser.add_argument("--frames", type=int, default=5, help="Number of test frames to classify and save")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = INPUT_VIDEO_PATH.parent / args.video

    test_team_classifier(input_video=video_path, num_sample_frames=args.frames)
