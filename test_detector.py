import cv2
from pathlib import Path
from src.config import TEST_DATA_DIR, MODEL_PATH
from src.detector import FootballDetector

def run_detector_test():
    # 1. Verify files exist
    test_image_path = TEST_DATA_DIR / "im.jpg"  # or test.jpg / gg.jpg
    
    if not test_image_path.exists():
        print(f"❌ Error: Test image not found at {test_image_path}")
        return

    if not Path(MODEL_PATH).exists():
        print(f"❌ Error: Model weights not found at {MODEL_PATH}")
        return

    print("--- STARTING DETECTOR UNIT TEST ---")
    print(f"Loading image from: {test_image_path}")
    
    frame = cv2.imread(str(test_image_path))
    if frame is None:
        print("❌ Error: Failed to read image with OpenCV.")
        return

    # 2. Instantiate detector
    print("Initializing FootballDetector...")
    detector = FootballDetector(model_path=str(MODEL_PATH))

    # 3. Process frame
    print("Running detection and tracking...")
    frame_state = detector.process_frame(frame, frame_index=0)

    # 4. Print inspection summary
    print("\n--- DETECTION SUMMARY ---")
    print(f"Frame Index: {frame_state.frame_index}")
    print(f"Players Detected: {len(frame_state.players)}")
    print(f"Referees Detected: {len(frame_state.referees)}")
    
    if frame_state.ball:
        ball = frame_state.ball
        print(f"Ball Detected: YES | Confidence: {ball.confidence:.2f} | Center: {ball.bbox.center}")
    else:
        print("Ball Detected: NO")

    # 5. Draw bounding boxes on raw frame for visual confirmation
    annotated = frame.copy()

    # Draw Players (Cyan)
    for t_id, player in frame_state.players.items():
        bbox = player.bbox
        cv2.rectangle(annotated, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (255, 255, 0), 2)
        label = f"P:{t_id} ({player.confidence:.2f})"
        cv2.putText(annotated, label, (bbox.x1, max(15, bbox.y1 - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Draw Referees (White)
    for t_id, ref in frame_state.referees.items():
        bbox = ref.bbox
        cv2.rectangle(annotated, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (255, 255, 255), 2)
        label = f"Ref:{t_id}"
        cv2.putText(annotated, label, (bbox.x1, max(15, bbox.y1 - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw Ball (Green)
    if frame_state.ball:
        bbox = frame_state.ball.bbox
        cv2.rectangle(annotated, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"Ball ({frame_state.ball.confidence:.2f})", 
                    (bbox.x1, max(15, bbox.y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save output for inspection
    output_path = TEST_DATA_DIR / "detector_test_output.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"\n✅ SUCCESS: Annotated output saved to {output_path}")

if __name__ == "__main__":
    run_detector_test()