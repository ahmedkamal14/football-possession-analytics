import cv2
import time
from pathlib import Path
from src.config import ROOT_DIR, MODEL_PATH
from src.detector import FootballDetector

def run_video_detector_test():
    # 1. Update this filename to match your 40-second test video in Test Data
    input_video_path = ROOT_DIR / "Test Data" / "test_video.mp4" 
    output_video_path = ROOT_DIR / "Test Data" / "detector_test_output.mp4"

    if not input_video_path.exists():
        print(f"❌ Error: Video not found at {input_video_path}")
        print("Please place your 40s video in 'Test Data' or update 'input_video_path'.")
        return

    print("--- STARTING VIDEO TRACKING UNIT TEST ---")
    print(f"Loading video: {input_video_path}")

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print("❌ Error: Could not open input video.")
        return

    # Extract video metadata
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    print("Initializing FootballDetector (with ByteTrack)...")
    detector = FootballDetector(model_path=str(MODEL_PATH))

    frame_index = 0
    start_time = time.time()

    print(f"Processing {total_frames} frames... This may take a moment.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Process frame with tracking
        frame_state = detector.process_frame(frame, frame_index=frame_index)

        # 3. Draw lightweight debug annotations
        annotated = frame.copy()

        # Draw Players (Cyan) + Track IDs
        for t_id, player in frame_state.players.items():
            bbox = player.bbox
            cv2.rectangle(annotated, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (255, 255, 0), 2)
            cv2.putText(annotated, f"ID:{t_id}", (bbox.x1, max(15, bbox.y1 - 8)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw Referees (White)
        for t_id, ref in frame_state.referees.items():
            bbox = ref.bbox
            cv2.rectangle(annotated, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (255, 255, 255), 2)
            cv2.putText(annotated, f"REF:{t_id}", (bbox.x1, max(15, bbox.y1 - 8)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw Ball (Green)
        if frame_state.ball:
            bbox = frame_state.ball.bbox
            cv2.rectangle(annotated, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"BALL ({frame_state.ball.confidence:.2f})", 
                        (bbox.x1, max(15, bbox.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame to output video
        out.write(annotated)

        # Print progress log every 30 frames
        if frame_index % 30 == 0:
            elapsed = time.time() - start_time
            print(f"  -> Processed frame {frame_index}/{total_frames} | Elapsed: {elapsed:.1f}s")

        frame_index += 1

    cap.release()
    out.release()
    total_time = time.time() - start_time
    print(f"\n✅ SUCCESS: Processed {frame_index} frames in {total_time:.2f} seconds.")
    print(f"🎥 Output video saved to: {output_video_path}")

if __name__ == "__main__":
    run_video_detector_test()