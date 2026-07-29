import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
from src.config import MODEL_PATH, CLASS_BALL, CLASS_PLAYER, CLASS_REFEREE, CLASS_NAMES

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> Tuple[int, int]:
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))

    @property
    def bottom_center(self) -> Tuple[int, int]:
        return (int((self.x1 + self.x2) / 2), self.y2)

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

@dataclass
class DetectedObject:
    id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox

@dataclass
class FrameState:
    frame_index: int
    players: Dict[int, DetectedObject]
    referees: Dict[int, DetectedObject]
    ball: Optional[DetectedObject]

class FootballDetector:
    def __init__(self, model_path: str = str(MODEL_PATH), conf: float = 0.20):
        self.model = YOLO(model_path)
        self.conf = conf

    def reset_tracker(self):
        """Resets tracking state if active."""
        if hasattr(self.model, 'predictor') and self.model.predictor is not None:
            self.model.predictor.trackers = []

    def process_frame(self, frame: np.ndarray, frame_index: int = 0, use_tracking: bool = True) -> FrameState:
        h, w = frame.shape[:2]
        if use_tracking:
            results = self.model.track(frame, persist=True, conf=self.conf, verbose=False)
        else:
            results = self.model.predict(frame, conf=self.conf, verbose=False)
        
        players: Dict[int, DetectedObject] = {}
        referees: Dict[int, DetectedObject] = {}
        ball: Optional[DetectedObject] = None

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    
                    # Track ID if ByteTrack tracking is active, otherwise fallback to box index
                    track_id = int(box.id[0].item()) if (box.id is not None and len(box.id) > 0) else i + 1
                    
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1 = max(0, min(w - 1, int(xyxy[0])))
                    y1 = max(0, min(h - 1, int(xyxy[1])))
                    x2 = max(x1 + 1, min(w, int(xyxy[2])))
                    y2 = max(y1 + 1, min(h, int(xyxy[3])))
                    
                    bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
                    model_cls_name = result.names.get(cls_id, str(cls_id)) if hasattr(result, 'names') else CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                    name_lower = str(model_cls_name).lower()
                    obj = DetectedObject(id=track_id, class_id=cls_id, class_name=model_cls_name, confidence=confidence, bbox=bbox)
                    
                    if "player" in name_lower:
                        players[track_id] = obj
                    elif "ref" in name_lower:
                        referees[track_id] = obj
                    elif "ball" in name_lower:
                        if ball is None or confidence > ball.confidence:
                            ball = obj

        return FrameState(frame_index=frame_index, players=players, referees=referees, ball=ball)
