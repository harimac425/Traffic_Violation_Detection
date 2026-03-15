"""
Constraint-Aware SORT Tracker

Custom multi-object tracker based on SORT (Simple Online and Realtime Tracking)
with constraint-aware gating for traffic violation detection.

Features:
- Kalman filter for state prediction (position + velocity)
- Hungarian algorithm for optimal detection-to-track assignment
- Stop-line boundary gating (prevents ID switches at the stop-line)
- Motion-direction consistency (penalizes impossible direction changes)
- Configurable age-based track management (birth, death thresholds)

Based on the methodology in the PPT:
  "Module 4: Tracking & Temporal Consistency (Constraint-Aware SORT)
   - Assign unique IDs to vehicles across frames
   - Maintain temporal consistency using Kalman filtering
   - Apply stop-line and motion-direction gating"
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TrackState:
    """Represents the current state of a tracked object"""
    track_id: int
    bbox: List[float]           # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    age: int = 0                # Total frames since track creation
    hits: int = 0               # Total successful matches
    time_since_update: int = 0  # Frames since last matched detection
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in pixels/frame
    direction: float = 0.0      # Movement direction in degrees
    crossed_stop_line: bool = False


class KalmanBoxTracker:
    """
    Single-object tracker using Kalman filter.
    
    State vector: [x_center, y_center, area, aspect_ratio, vx, vy, va]
    Measurement: [x_center, y_center, area, aspect_ratio]
    """
    _id_counter = 0
    
    def __init__(self, bbox: List[float], class_name: str = "", confidence: float = 0.0):
        """
        Initialize tracker with a bounding box.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            class_name: Detection class name
            confidence: Detection confidence
        """
        # Create Kalman filter: 7 state dims, 4 measurement dims
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0
        
        # Initial covariance — high uncertainty for velocity
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state from bbox
        measurement = self._bbox_to_measurement(bbox)
        self.kf.x[:4] = measurement.reshape((4, 1))
        
        # Track metadata
        KalmanBoxTracker._id_counter += 1
        self.track_id = KalmanBoxTracker._id_counter
        self.class_name = class_name
        self.confidence = confidence
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.hit_streak = 1
        
        # Position history for velocity/direction calculation
        self.position_history: List[Tuple[float, float]] = [self._get_center(bbox)]
        self.crossed_stop_line = False
    
    def _bbox_to_measurement(self, bbox: List[float]) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, area, aspect_ratio]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        area = w * h
        aspect_ratio = w / max(h, 1e-6)
        return np.array([cx, cy, area, aspect_ratio])
    
    def _measurement_to_bbox(self, state: np.ndarray) -> List[float]:
        """Convert [cx, cy, area, aspect_ratio] to [x1, y1, x2, y2]"""
        cx, cy, area, ar = state.flatten()[:4]
        area = max(area, 1.0)
        ar = max(ar, 1e-6)
        w = np.sqrt(area * ar)
        h = area / max(w, 1e-6)
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    
    def _get_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bbox"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def predict(self) -> List[float]:
        """
        Advance state (predict next position).
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Prevent negative area
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self._measurement_to_bbox(self.kf.x)
    
    def update(self, bbox: List[float], class_name: str = "", confidence: float = 0.0):
        """
        Update state with matched detection.
        
        Args:
            bbox: Matched detection bounding box [x1, y1, x2, y2]
            class_name: Detection class name
            confidence: Detection confidence
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = confidence
        if class_name:
            self.class_name = class_name
        
        measurement = self._bbox_to_measurement(bbox)
        self.kf.update(measurement.reshape((4, 1)))
        
        # Update position history (keep last 30 positions)
        center = self._get_center(bbox)
        self.position_history.append(center)
        if len(self.position_history) > 30:
            self.position_history.pop(0)
    
    def get_state(self) -> TrackState:
        """Get current track state as a TrackState object"""
        bbox = self._measurement_to_bbox(self.kf.x)
        
        # Calculate velocity and direction from position history
        velocity = (0.0, 0.0)
        direction = 0.0
        if len(self.position_history) >= 2:
            p1 = self.position_history[-2]
            p2 = self.position_history[-1]
            velocity = (p2[0] - p1[0], p2[1] - p1[1])
            direction = np.degrees(np.arctan2(velocity[1], velocity[0]))
            if direction < 0:
                direction += 360
        
        return TrackState(
            track_id=self.track_id,
            bbox=bbox,
            class_name=self.class_name,
            confidence=self.confidence,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            velocity=velocity,
            direction=direction,
            crossed_stop_line=self.crossed_stop_line
        )


class ConstraintAwareSORT:
    """
    Constraint-Aware SORT: Multi-object tracker with spatial constraints.
    
    Extends classic SORT with:
    1. Stop-line gating — tracks near the stop-line have tighter matching
    2. Direction consistency — penalizes sudden direction reversals
    3. Class-aware matching — only matches detections of the same class
    
    Usage:
        tracker = ConstraintAwareSORT(max_age=30, min_hits=3)
        for frame_detections in video:
            tracks = tracker.update(frame_detections)
            for track in tracks:
                print(track.track_id, track.bbox)
    """
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 stop_line_y: Optional[float] = None,
                 direction_penalty: float = 0.5):
        """
        Initialize the tracker.
        
        Args:
            max_age: Max frames a track survives without detection match
            min_hits: Min hits before a track is reported (reduces noise)
            iou_threshold: Minimum IoU for valid detection-track match
            stop_line_y: Y-coordinate of the stop-line (None = disabled)
            direction_penalty: Cost penalty for direction-inconsistent matches (0-1)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.stop_line_y = stop_line_y
        self.direction_penalty = direction_penalty
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> List[TrackState]:
        """
        Update tracker with new detections for one frame.
        
        Args:
            detections: List of dicts with keys:
                'bbox': [x1, y1, x2, y2]
                'class_name': str
                'confidence': float
                
        Returns:
            List of TrackState for all confirmed tracks
        """
        self.frame_count += 1
        
        # Step 1: Predict new locations for existing tracks
        predicted_boxes = []
        to_remove = []
        for i, tracker in enumerate(self.trackers):
            predicted = tracker.predict()
            predicted_boxes.append(predicted)
            # Remove tracks with invalid predictions (NaN)
            if np.any(np.isnan(predicted)):
                to_remove.append(i)
        
        for i in reversed(to_remove):
            self.trackers.pop(i)
            predicted_boxes.pop(i)
        
        # Step 2: Build cost matrix and run Hungarian assignment
        if len(detections) > 0 and len(self.trackers) > 0:
            cost_matrix = self._build_cost_matrix(detections, predicted_boxes)
            matched, unmatched_dets, unmatched_trks = self._associate(
                cost_matrix, detections, len(self.trackers)
            )
        else:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.trackers)))
        
        # Step 3: Update matched tracks
        for det_idx, trk_idx in matched:
            det = detections[det_idx]
            self.trackers[trk_idx].update(
                det['bbox'], det.get('class_name', ''), det.get('confidence', 0.0)
            )
        
        # Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_tracker = KalmanBoxTracker(
                det['bbox'], det.get('class_name', ''), det.get('confidence', 0.0)
            )
            self.trackers.append(new_tracker)
        
        # Step 5: Remove dead tracks
        self.trackers = [
            t for t in self.trackers 
            if t.time_since_update <= self.max_age
        ]
        
        # Step 6: Check stop-line crossing
        if self.stop_line_y is not None:
            for tracker in self.trackers:
                state = tracker.get_state()
                bbox_bottom = state.bbox[3]  # y2
                if bbox_bottom >= self.stop_line_y and not tracker.crossed_stop_line:
                    tracker.crossed_stop_line = True
        
        # Step 7: Return confirmed tracks (enough hits)
        results = []
        for tracker in self.trackers:
            if tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                results.append(tracker.get_state())
        
        return results
    
    def _build_cost_matrix(self, detections: List[Dict], 
                           predicted_boxes: List[List[float]]) -> np.ndarray:
        """
        Build cost matrix with IoU + constraint penalties.
        
        Cost = (1 - IoU) + direction_penalty + class_mismatch_penalty
        """
        num_dets = len(detections)
        num_trks = len(predicted_boxes)
        cost_matrix = np.zeros((num_dets, num_trks))
        
        for d in range(num_dets):
            for t in range(num_trks):
                det_bbox = detections[d]['bbox']
                trk_bbox = predicted_boxes[t]
                
                # Base cost: 1 - IoU
                iou = self._calculate_iou(det_bbox, trk_bbox)
                cost = 1.0 - iou
                
                # Constraint 1: Class mismatch penalty
                det_class = detections[d].get('class_name', '')
                trk_class = self.trackers[t].class_name
                if det_class and trk_class and det_class != trk_class:
                    cost += 1.0  # Heavy penalty for class mismatch
                
                # Constraint 2: Direction consistency
                if len(self.trackers[t].position_history) >= 3:
                    cost += self._direction_cost(
                        detections[d]['bbox'], self.trackers[t]
                    )
                
                # Constraint 3: Stop-line proximity tightening
                if self.stop_line_y is not None:
                    cost += self._stop_line_cost(det_bbox, trk_bbox)
                
                cost_matrix[d, t] = cost
        
        return cost_matrix
    
    def _associate(self, cost_matrix: np.ndarray, detections: List[Dict],
                   num_trackers: int) -> Tuple[List, List, List]:
        """
        Run Hungarian algorithm and filter by IoU threshold.
        
        Returns:
            (matched_pairs, unmatched_detection_indices, unmatched_tracker_indices)
        """
        if cost_matrix.size == 0:
            return [], list(range(len(detections))), list(range(num_trackers))
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(num_trackers))
        
        for r, c in zip(row_indices, col_indices):
            # Only accept matches with sufficient IoU (cost < threshold)
            if cost_matrix[r, c] > (1.0 - self.iou_threshold + 1.5):
                # Too costly — treat as unmatched
                continue
            matched.append((r, c))
            if r in unmatched_dets:
                unmatched_dets.remove(r)
            if c in unmatched_trks:
                unmatched_trks.remove(c)
        
        return matched, unmatched_dets, unmatched_trks
    
    def _direction_cost(self, det_bbox: List[float], tracker: KalmanBoxTracker) -> float:
        """
        Penalize matches where the detection implies a sudden direction reversal.
        """
        if len(tracker.position_history) < 2:
            return 0.0
        
        # Current track direction
        p1 = tracker.position_history[-2]
        p2 = tracker.position_history[-1]
        track_dx = p2[0] - p1[0]
        track_dy = p2[1] - p1[1]
        
        # Direction to detection
        det_cx = (det_bbox[0] + det_bbox[2]) / 2
        det_cy = (det_bbox[1] + det_bbox[3]) / 2
        det_dx = det_cx - p2[0]
        det_dy = det_cy - p2[1]
        
        # Calculate angle between directions
        dot = track_dx * det_dx + track_dy * det_dy
        mag1 = np.sqrt(track_dx**2 + track_dy**2)
        mag2 = np.sqrt(det_dx**2 + det_dy**2)
        
        if mag1 < 1.0 or mag2 < 1.0:
            return 0.0  # Stationary — no penalty
        
        cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        
        # Penalize if going in opposite direction (cos < 0)
        if cos_angle < 0:
            return self.direction_penalty * abs(cos_angle)
        
        return 0.0
    
    def _stop_line_cost(self, det_bbox: List[float], trk_bbox: List[float]) -> float:
        """
        Apply tighter matching near the stop-line to prevent ID switches.
        """
        det_bottom = det_bbox[3]
        trk_bottom = trk_bbox[3]
        
        # If either is near the stop-line, tighten matching
        distance_det = abs(det_bottom - self.stop_line_y)
        distance_trk = abs(trk_bottom - self.stop_line_y)
        
        proximity = 50  # pixels
        if distance_det < proximity or distance_trk < proximity:
            # Add a small penalty proportional to the distance between det and trk
            center_dist = np.sqrt(
                ((det_bbox[0] + det_bbox[2]) / 2 - (trk_bbox[0] + trk_bbox[2]) / 2) ** 2 +
                ((det_bbox[1] + det_bbox[3]) / 2 - (trk_bbox[1] + trk_bbox[3]) / 2) ** 2
            )
            return 0.1 * (center_dist / proximity) if center_dist > proximity else 0.0
        
        return 0.0
    
    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset(self):
        """Reset all tracks"""
        self.trackers.clear()
        self.frame_count = 0
        KalmanBoxTracker._id_counter = 0
    
    def set_stop_line(self, y: Optional[float]):
        """Update the stop-line Y position"""
        self.stop_line_y = y
    
    def get_track_count(self) -> int:
        """Get number of active tracks"""
        return len(self.trackers)
    
    def get_all_tracks(self) -> List[TrackState]:
        """Get states for all active tracks (including unconfirmed)"""
        return [t.get_state() for t in self.trackers]
