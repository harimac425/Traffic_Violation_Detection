"""
Red Signal Violation Detection Module

Detects vehicles that cross the stop-line while the traffic signal is red.

How it works:
1. User defines a Signal ROI (Region of Interest) on the frame where the 
   traffic light is visible.
2. User defines a Stop-Line Y position (horizontal line across the road).
3. Each frame: the signal ROI is analyzed for dominant color (red/green/yellow)
   using HSV color space thresholding.
4. Tracked vehicles (from the SORT tracker) are checked for crossing the 
   stop-line while the signal is RED.
5. A violation is recorded only once per track ID (cooldown-based).

Based on PPT Module 4-5:
  "Define stop-line and traffic signal ROI"
  "Red signal crossing detection"
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum


class SignalState(Enum):
    """Traffic signal states"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


@dataclass
class SignalROI:
    """
    Region of Interest for the traffic signal in the frame.
    Coordinates are in pixels (absolute, not normalized).
    """
    x1: int
    y1: int
    x2: int
    y2: int
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)
    
    def is_valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1


class SignalDetector:
    """
    Detects the traffic signal state (red/green/yellow) from a defined ROI
    using HSV color space analysis.
    """
    
    # HSV ranges for traffic light colors
    # Red has two ranges in HSV (wraps around 0/180)
    RED_LOWER_1 = np.array([0, 100, 100])
    RED_UPPER_1 = np.array([10, 255, 255])
    RED_LOWER_2 = np.array([160, 100, 100])
    RED_UPPER_2 = np.array([180, 255, 255])
    
    YELLOW_LOWER = np.array([15, 100, 100])
    YELLOW_UPPER = np.array([35, 255, 255])
    
    GREEN_LOWER = np.array([40, 80, 80])
    GREEN_UPPER = np.array([90, 255, 255])
    
    def __init__(self, min_pixel_ratio: float = 0.05):
        """
        Args:
            min_pixel_ratio: Minimum ratio of colored pixels to total ROI pixels
                             to consider a color as "active". Default 5%.
        """
        self.min_pixel_ratio = min_pixel_ratio
    
    def detect_signal(self, frame: np.ndarray, roi: SignalROI) -> Tuple[SignalState, Dict[str, float]]:
        """
        Detect the traffic signal state from the ROI region of the frame.
        
        Args:
            frame: Full BGR video frame
            roi: Signal region of interest
            
        Returns:
            Tuple of (SignalState, color_ratios_dict)
            color_ratios_dict has keys 'red', 'yellow', 'green' with pixel ratios
        """
        if not roi.is_valid():
            return SignalState.UNKNOWN, {}
        
        # Crop the ROI
        x1, y1, x2, y2 = roi.to_tuple()
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        roi_crop = frame[y1:y2, x1:x2]
        if roi_crop.size == 0:
            return SignalState.UNKNOWN, {}
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]
        
        if total_pixels == 0:
            return SignalState.UNKNOWN, {}
        
        # Detect each color
        red_mask_1 = cv2.inRange(hsv, self.RED_LOWER_1, self.RED_UPPER_1)
        red_mask_2 = cv2.inRange(hsv, self.RED_LOWER_2, self.RED_UPPER_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        
        yellow_mask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
        green_mask = cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)
        
        # Calculate ratios
        red_ratio = np.count_nonzero(red_mask) / total_pixels
        yellow_ratio = np.count_nonzero(yellow_mask) / total_pixels
        green_ratio = np.count_nonzero(green_mask) / total_pixels
        
        ratios = {
            'red': round(red_ratio, 4),
            'yellow': round(yellow_ratio, 4),
            'green': round(green_ratio, 4)
        }
        
        # Determine state: pick the color with highest ratio above threshold
        max_color = max(ratios, key=ratios.get)
        max_ratio = ratios[max_color]
        
        if max_ratio < self.min_pixel_ratio:
            return SignalState.UNKNOWN, ratios
        
        state_map = {
            'red': SignalState.RED,
            'yellow': SignalState.YELLOW,
            'green': SignalState.GREEN
        }
        
        return state_map[max_color], ratios


class RedSignalViolationDetector:
    """
    Detects red-signal jumping violations by combining:
    1. Signal state detection (from SignalDetector)
    2. Stop-line crossing detection (from vehicle tracking)
    
    A violation occurs when a tracked vehicle crosses the stop-line
    while the signal is RED.
    """
    
    def __init__(self, stop_line_y: int = None, signal_roi: SignalROI = None, 
                 cooldown_frames: int = 90):
        """
        Args:
            stop_line_y: Y-coordinate of the stop-line (pixels from top)
            signal_roi: Region of interest for the traffic signal
            cooldown_frames: Min frames between violations for same track ID
        """
        self.stop_line_y = stop_line_y
        self.signal_roi = signal_roi
        self.signal_detector = SignalDetector()
        self.cooldown_frames = cooldown_frames
        
        # State tracking
        self.current_signal = SignalState.UNKNOWN
        self.current_ratios = {}
        
        # Track previous positions: track_id -> last_y_bottom
        self.prev_positions: Dict[int, float] = {}
        
        # Cooldown: track_id -> frames remaining
        self.violation_cooldowns: Dict[int, int] = {}
        
        # Frame counter
        self.frame_count = 0
    
    def set_stop_line(self, y: int):
        """Set the stop-line Y position"""
        self.stop_line_y = y
    
    def set_signal_roi(self, x1: int, y1: int, x2: int, y2: int):
        """Set the signal ROI"""
        self.signal_roi = SignalROI(x1, y1, x2, y2)
    
    def is_configured(self) -> bool:
        """Check if both stop-line and signal ROI are configured"""
        return (self.stop_line_y is not None and 
                self.signal_roi is not None and 
                self.signal_roi.is_valid())
    
    def check_violations(self, frame: np.ndarray, 
                         tracked_vehicles: List[Dict]) -> List[Dict]:
        """
        Check for red-signal violations in the current frame.
        
        Args:
            frame: Current BGR video frame
            tracked_vehicles: List of tracked vehicle dicts with keys:
                'track_id': int, 'bbox': [x1,y1,x2,y2], 'class_name': str
                
        Returns:
            List of violation dicts with keys:
                'track_id', 'bbox', 'class_name', 'signal_state', 'confidence'
        """
        self.frame_count += 1
        violations = []
        
        if not self.is_configured():
            return violations
        
        # Step 1: Detect current signal state
        self.current_signal, self.current_ratios = \
            self.signal_detector.detect_signal(frame, self.signal_roi)
        
        # Step 2: Decrement cooldowns
        expired = []
        for tid, remaining in self.violation_cooldowns.items():
            self.violation_cooldowns[tid] = remaining - 1
            if self.violation_cooldowns[tid] <= 0:
                expired.append(tid)
        for tid in expired:
            del self.violation_cooldowns[tid]
        
        # Step 3: Only check crossings if signal is RED
        if self.current_signal != SignalState.RED:
            # Update positions and return (no violations possible)
            for vehicle in tracked_vehicles:
                tid = vehicle.get('track_id')
                bbox = vehicle.get('bbox', [0, 0, 0, 0])
                if tid is not None:
                    self.prev_positions[tid] = bbox[3]  # y2 = bottom
            return violations
        
        # Step 4: Check each vehicle for stop-line crossing
        for vehicle in tracked_vehicles:
            tid = vehicle.get('track_id')
            bbox = vehicle.get('bbox', [0, 0, 0, 0])
            
            if tid is None:
                continue
            
            # Skip if on cooldown
            if tid in self.violation_cooldowns:
                self.prev_positions[tid] = bbox[3]
                continue
            
            current_bottom = bbox[3]  # y2
            prev_bottom = self.prev_positions.get(tid)
            
            # Check if the vehicle crossed the stop-line in this frame
            # (was above the line, now below it)
            if prev_bottom is not None and prev_bottom < self.stop_line_y <= current_bottom:
                # Red signal crossing detected!
                violations.append({
                    'track_id': tid,
                    'bbox': bbox,
                    'class_name': vehicle.get('class_name', 'vehicle'),
                    'signal_state': 'red',
                    'confidence': self.current_ratios.get('red', 0.0),
                    'details': f"Vehicle T{tid} crossed stop-line during RED signal"
                })
                # Set cooldown
                self.violation_cooldowns[tid] = self.cooldown_frames
            
            # Update position
            self.prev_positions[tid] = current_bottom
        
        return violations
    
    def draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw stop-line and signal ROI overlays on the frame for visualization.
        
        Args:
            frame: BGR frame to draw on (modified in-place)
            
        Returns:
            Frame with overlays drawn
        """
        h, w = frame.shape[:2]
        
        # Draw stop-line
        if self.stop_line_y is not None:
            color = (0, 0, 255) if self.current_signal == SignalState.RED else (0, 255, 0)
            cv2.line(frame, (0, self.stop_line_y), (w, self.stop_line_y), color, 2)
            label = f"STOP LINE (Signal: {self.current_signal.value.upper()})"
            cv2.putText(frame, label, (10, self.stop_line_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw signal ROI box
        if self.signal_roi is not None and self.signal_roi.is_valid():
            roi = self.signal_roi
            # Color the ROI border based on detected signal
            roi_color = {
                SignalState.RED: (0, 0, 255),
                SignalState.YELLOW: (0, 255, 255),
                SignalState.GREEN: (0, 255, 0),
                SignalState.UNKNOWN: (128, 128, 128)
            }.get(self.current_signal, (128, 128, 128))
            
            cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), roi_color, 2)
            cv2.putText(frame, f"Signal: {self.current_signal.value}",
                        (roi.x1, roi.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, roi_color, 1)
        
        return frame
    
    def get_signal_state(self) -> SignalState:
        """Get current detected signal state"""
        return self.current_signal
    
    def reset(self):
        """Reset all tracking state"""
        self.prev_positions.clear()
        self.violation_cooldowns.clear()
        self.current_signal = SignalState.UNKNOWN
        self.current_ratios = {}
        self.frame_count = 0
