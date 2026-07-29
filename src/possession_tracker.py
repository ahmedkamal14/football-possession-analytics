import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from src.config import POSSESSION_DIST_THRESHOLD
from src.detector import FrameState, DetectedObject
from src.team_clusterer import TeamClusterer

@dataclass
class PossessionInfo:
    player_id: int
    distance: float
    team_id: int
    team_name: str
    confidence: float
    team_color_bgr: Tuple[int, int, int]
    player_obj: DetectedObject

class PossessionTracker:
    def __init__(self, dist_threshold: float = POSSESSION_DIST_THRESHOLD):
        self.dist_threshold = dist_threshold

    def get_possession(self, frame: np.ndarray, frame_state: FrameState, clusterer: TeamClusterer) -> Optional[PossessionInfo]:
        """
        Identifies the player closest to the ball. If within distance threshold,
        classifies their team membership using the fitted TeamClusterer.
        """
        if frame_state.ball is None or len(frame_state.players) == 0:
            return None

        ball_center = frame_state.ball.bbox.center
        bx, by = ball_center

        closest_player: Optional[DetectedObject] = None
        min_dist = float('inf')

        for player_id, player in frame_state.players.items():
            # Use bottom center of player bounding box (feet location) for field proximity
            px, py = player.bbox.bottom_center
            dist = math.hypot(bx - px, by - py)
            
            if dist < min_dist:
                min_dist = dist
                closest_player = player

        if closest_player is not None and min_dist <= self.dist_threshold:
            torso_crop = clusterer.crop_torso(frame, closest_player.bbox)
            team_id, conf, team_color = clusterer.predict_team(torso_crop)
            team_name = clusterer.team_names[team_id] if (hasattr(clusterer, 'team_names') and len(clusterer.team_names) > team_id) else f"Team {team_id + 1}"
            
            return PossessionInfo(
                player_id=closest_player.id,
                distance=min_dist,
                team_id=team_id,
                team_name=team_name,
                confidence=conf,
                team_color_bgr=team_color,
                player_obj=closest_player
            )

        return None
