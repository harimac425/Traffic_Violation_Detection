"""
Phone Usage Detection Module (MediaPipe + YOLO)

Detects mobile phone usage while riding using a dual-evidence approach:
1. MediaPipe Pose — detects hand-near-ear/face gestures
2. YOLO — detects "cell phone" objects

A phone violation is triggered when EITHER:
- MediaPipe detects a calling gesture (hand raised near head)
- YOLO detects a phone object near a rider on a motorcycle

Combining both reduces false positives significantly.

Based on PPT:
  "MediaPipe — Pose estimation for phone detection"
  "Mobile phone usage detection"
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PhoneViolation:
    """Represents a detected phone usage violation"""
    bbox: List[float]           # Person bounding box
    confidence: float           # Detection confidence
    method: str                 # "pose", "yolo", or "both"
    track_id: Optional[int]     # Person track ID
    details: str                # Human-readable details


class MediaPipePhoneDetector:
    """
    Detects phone usage while riding using MediaPipe Pose landmarks.
    
    Detection logic:
    - Extract upper body pose landmarks for persons on motorcycles
    - Check if either hand (wrist) is raised near the head/ear region
    - A 'calling gesture' = wrist Y is above shoulder Y AND 
      wrist is near the head X position
    
    This is combined with YOLO's cell phone detection for dual confirmation.
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Pose detector using Tasks API.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self._available = False
        self._landmark_indices = None
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                PoseLandmarker, PoseLandmarkerOptions, RunningMode
            )
            from pathlib import Path
            
            # Model path
            model_path = Path(__file__).parent.parent / "models" / "pose_landmarker_lite.task"
            
            if not model_path.exists():
                print("[PhoneDetection] Downloading pose landmark model...")
                import urllib.request
                model_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(
                    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
                    str(model_path)
                )
                print("[PhoneDetection] Model downloaded.")
            
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.IMAGE,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                num_poses=1
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            
            # Store landmark index references
            # 0: nose, 7: left_ear, 8: right_ear
            # 11: left_shoulder, 12: right_shoulder  
            # 15: left_wrist, 16: right_wrist
            self._available = True
            print("[PhoneDetection] MediaPipe PoseLandmarker initialized successfully.")
            
        except Exception as e:
            print(f"[PhoneDetection] MediaPipe not available: {e}")
            print("[PhoneDetection] Falling back to YOLO-only phone detection.")
    
    @property
    def is_available(self) -> bool:
        """Check if MediaPipe is available"""
        return self._available
    
    def detect_phone_gesture(self, frame: np.ndarray, 
                              person_bbox: List[float]) -> Tuple[bool, float, str]:
        """
        Check if a person (cropped from frame) is in a phone-calling gesture.
        
        Args:
            frame: Full BGR video frame
            person_bbox: Person bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple of (is_calling, confidence, details_string)
        """
        if not self._available:
            return False, 0.0, ""
        
        try:
            import mediapipe as mp
            
            # Crop person region from frame
            x1, y1, x2, y2 = map(int, person_bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return False, 0.0, ""
            
            # Convert BGR to RGB for MediaPipe
            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            # Run pose detection
            result = self.landmarker.detect(mp_image)
            
            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return False, 0.0, "no_pose_detected"
            
            landmarks = result.pose_landmarks[0]  # First person's landmarks
            
            # Key landmarks for phone detection
            # MediaPipe landmark indices:
            # 0: nose, 7: left_ear, 8: right_ear
            # 11: left_shoulder, 12: right_shoulder
            # 15: left_wrist, 16: right_wrist
            nose = landmarks[0]
            left_ear = landmarks[7]
            right_ear = landmarks[8]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # Check calling gesture for each hand
            left_calling = self._is_calling_gesture(
                left_wrist, left_shoulder, left_ear, nose
            )
            right_calling = self._is_calling_gesture(
                right_wrist, right_shoulder, right_ear, nose
            )
            
            if left_calling or right_calling:
                side = "left" if left_calling else "right"
                confidence = max(
                    left_wrist.visibility if left_calling else 0,
                    right_wrist.visibility if right_calling else 0
                )
                return True, confidence, f"{side}_hand_near_ear"
            
            return False, 0.0, "no_calling_gesture"
            
        except Exception as e:
            return False, 0.0, f"error: {str(e)}"
    
    def _is_calling_gesture(self, wrist, shoulder, ear, nose) -> bool:
        """
        Check if a wrist-shoulder-ear arrangement indicates phone usage.
        
        Relaxed criteria for better detection:
        1. Wrist is near head/face region (horizontally)
        2. Wrist is at or above shoulder level OR near face level
        3. Landmarks have minimum visibility
        """
        min_visibility = 0.25  # Lowered from 0.4 — CCTV crops have poor visibility
        
        # Check visibility
        if (wrist.visibility < min_visibility or 
            shoulder.visibility < min_visibility):
            return False
        
        # Head reference point
        head_x = ear.x if ear.visibility > min_visibility else nose.x
        head_y = ear.y if ear.visibility > min_visibility else nose.y
        
        # Criterion 1: Wrist horizontally close to head
        horizontal_distance = abs(wrist.x - head_x)
        hand_near_head_x = horizontal_distance < 0.35  # Widened from 0.25
        
        # Criterion 2: Wrist near head level (relaxed — doesn't need to be above shoulder)
        vertical_distance = abs(wrist.y - head_y)
        hand_near_head_y = vertical_distance < 0.40  # Widened from 0.3
        
        # Criterion 3: Hand is at least somewhat raised (above hip level, i.e. above midpoint)
        hand_raised = wrist.y < (shoulder.y + 0.15)  # Allow slightly below shoulder
        
        return hand_near_head_x and hand_near_head_y and hand_raised
    
    def check_phone_violations(self, 
                                frame: np.ndarray,
                                persons: List[Dict],
                                motorcycles: List[Dict],
                                phone_detections: List[Dict] = None,
                                iou_threshold: float = 0.05) -> List[PhoneViolation]:
        """
        Check for phone usage violations among motorcycle riders.
        
        Combines:
        - MediaPipe pose gesture detection
        - YOLO cell phone object detection
        
        Uses relaxed association (IoU 0.05 or overlap check) for better
        detection in side-view and close-up shots.
        """
        violations = []
        phone_detections = phone_detections or []
        
        for person in persons:
            person_bbox = person.get('bbox', [0, 0, 0, 0])
            track_id = person.get('track_id')
            
            # Check if this person is on a motorcycle (relaxed association)
            on_motorcycle = False
            for moto in motorcycles:
                moto_bbox = moto.get('bbox', [0, 0, 0, 0])
                iou = self._calculate_iou(person_bbox, moto_bbox)
                if iou > iou_threshold:
                    on_motorcycle = True
                    break
                # Fallback: check if person overlaps motorcycle at all
                if self._boxes_overlap_any(person_bbox, moto_bbox):
                    on_motorcycle = True
                    break
            
            if not on_motorcycle:
                continue
            
            # Evidence 1: MediaPipe gesture
            pose_detected = False
            pose_confidence = 0.0
            pose_details = ""
            if self._available:
                pose_detected, pose_confidence, pose_details = \
                    self.detect_phone_gesture(frame, person_bbox)
            
            # Evidence 2: YOLO phone object near person (wider search)
            yolo_detected = False
            yolo_confidence = 0.0
            for phone in phone_detections:
                phone_bbox = phone.get('bbox', [0, 0, 0, 0])
                phone_conf = phone.get('confidence', 0.0)
                # Check if phone is near this person (wider margin)
                if self._is_near(person_bbox, phone_bbox, margin=100):
                    yolo_detected = True
                    yolo_confidence = phone_conf
                    break
            
            # Determine violation
            if pose_detected and yolo_detected:
                method = "both"
                confidence = max(pose_confidence, yolo_confidence)
                details = f"Phone detected by pose ({pose_details}) AND YOLO"
            elif pose_detected:
                method = "pose"
                confidence = pose_confidence
                details = f"Calling gesture detected ({pose_details})"
            elif yolo_detected:
                method = "yolo"
                confidence = yolo_confidence
                details = "Phone object detected near rider (YOLO)"
            else:
                continue  # No evidence
            
            violations.append(PhoneViolation(
                bbox=person_bbox,
                confidence=confidence,
                method=method,
                track_id=track_id,
                details=details
            ))
        
        return violations
    
    def _boxes_overlap_any(self, box1: List[float], box2: List[float]) -> bool:
        """Check if two boxes have ANY overlap at all (even 1 pixel)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return x1 < x2 and y1 < y2
    
    def _is_near(self, box1: List[float], box2: List[float], margin: int = 100) -> bool:
        """Check if two boxes are near each other (expanded IoU check)"""
        # Expand box1 by margin
        expanded = [
            box1[0] - margin, box1[1] - margin,
            box1[2] + margin, box1[3] + margin
        ]
        return self._calculate_iou(expanded, box2) > 0
    
    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def close(self):
        """Release MediaPipe resources"""
        if self._available and hasattr(self, 'landmarker'):
            self.landmarker.close()
