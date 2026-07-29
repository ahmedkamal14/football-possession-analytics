import cv2
import numpy as np
from typing import List, Tuple, Optional
from sklearn.mixture import GaussianMixture
from src.detector import BBox

class TeamClusterer:
    def __init__(self, n_teams: int = 2):
        self.n_teams = n_teams
        self.gmm: Optional[GaussianMixture] = None
        self.team_colors_bgr: List[Tuple[int, int, int]] = [(255, 0, 0), (0, 0, 255)]  # Default Blue & Red
        self.team_names: List[str] = ["Team A", "Team B"]
        self.is_fitted: bool = False

    def crop_torso(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Crops the upper torso region of a player to capture jersey color."""
        h = bbox.height
        w = bbox.width
        
        # Take vertical range: 15% to 65% of bbox height (torso)
        y1 = max(0, bbox.y1 + int(h * 0.15))
        y2 = min(frame.shape[0], bbox.y1 + int(h * 0.65))
        
        # Take horizontal range: 20% to 80% of bbox width (center of body)
        x1 = max(0, bbox.x1 + int(w * 0.20))
        x2 = min(frame.shape[1], bbox.x1 + int(w * 0.80))
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        return crop

    def mask_grass(self, torso_bgr: np.ndarray) -> np.ndarray:
        """Masks out green pitch/grass background pixels from the torso crop."""
        if torso_bgr is None or torso_bgr.size == 0:
            return torso_bgr

        hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
        
        # HSV range for football field grass
        lower_green = np.array([30, 35, 35])
        upper_green = np.array([85, 255, 255])
        
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)
        non_grass_mask = cv2.bitwise_not(grass_mask)
        
        non_grass_pixels = torso_bgr[non_grass_mask > 0]
        
        # If too many pixels were masked out, return original torso pixels
        if len(non_grass_pixels) < 15:
            return torso_bgr.reshape(-1, 3)
            
        return non_grass_pixels

    def extract_dominant_color_lab(self, torso_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts dominant jersey color vector in CIELAB space from unmasked pixels.
        CIELAB separates lightness (L) from chromaticity (a, b) for robust color clustering.
        """
        pixels = self.mask_grass(torso_bgr)
        if len(pixels) == 0:
            return np.array([128.0, 128.0, 128.0])

        # Convert BGR pixels (N, 1, 3) to CIELAB
        pixels_bgr_2d = pixels.reshape(-1, 1, 3).astype(np.uint8)
        pixels_lab_2d = cv2.cvtColor(pixels_bgr_2d, cv2.COLOR_BGR2LAB)
        pixels_lab = pixels_lab_2d.reshape(-1, 3).astype(np.float32)

        # Compute median LAB color vector for robust representation against noise
        dominant_lab = np.median(pixels_lab, axis=0)
        return dominant_lab

    def fit_gmm(self, player_crops: List[np.ndarray]) -> bool:
        """Fits Gaussian Mixture Model (k=2) on collected player jersey LAB color features."""
        lab_features = []
        for crop in player_crops:
            lab_vec = self.extract_dominant_color_lab(crop)
            lab_features.append(lab_vec)

        if len(lab_features) < self.n_teams * 2:
            print("⚠️ Warning: Not enough player crops to fit team clustering GMM.")
            return False

        X = np.array(lab_features, dtype=np.float32)
        
        # Fit Gaussian Mixture Model with 2 components
        self.gmm = GaussianMixture(n_components=self.n_teams, covariance_type='full', random_state=42)
        self.gmm.fit(X)
        self.is_fitted = True

        # Convert GMM cluster centroids back from CIELAB to BGR for visualization
        self.team_colors_bgr = []
        for mean_lab in self.gmm.means_:
            lab_pixel = np.clip(mean_lab, 0, 255).astype(np.uint8).reshape(1, 1, 3)
            bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)[0, 0]
            bgr_color = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))
            self.team_colors_bgr.append(bgr_color)

        print("✅ GMM Team Clustering Successfully Fitted!")
        for idx, col in enumerate(self.team_colors_bgr):
            print(f"   Team {idx + 1} Centroid BGR Color: {col}")

        return True

    def predict_team(self, torso_bgr: np.ndarray) -> Tuple[int, float, Tuple[int, int, int]]:
        """
        Predicts team index (0 or 1), confidence score (0.0 - 1.0), and team BGR color.
        """
        if not self.is_fitted or self.gmm is None:
            return 0, 0.5, (255, 255, 255)

        lab_vec = self.extract_dominant_color_lab(torso_bgr).reshape(1, -1)
        probs = self.gmm.predict_proba(lab_vec)[0]
        
        team_id = int(np.argmax(probs))
        confidence = float(probs[team_id])
        color_bgr = self.team_colors_bgr[team_id]
        
        return team_id, confidence, color_bgr
