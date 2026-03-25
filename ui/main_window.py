"""
Main Window - Traffic Violation Detection Application

PyQt5-based GUI with live video feed and violation detection.
"""
import sys
import os
import cv2
import numpy as np
from datetime import datetime
import time
from typing import Optional, List
from pathlib import Path
import json
from collections import defaultdict

from PyQt5.QtWidgets import (QMenu, 
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QListWidget,
    QListWidgetItem, QGroupBox, QComboBox, QFrame,
    QSplitter, QMessageBox, QApplication, QTableWidget,
    QTableWidgetItem, QHeaderView, QSlider, QStyle,
    QTabWidget, QSpinBox, QInputDialog, QScrollArea,
    QButtonGroup
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.logger import get_logger

logger = get_logger("UI")

class PersistentMenu(QMenu):
    """A QMenu that stays open when checkable actions are toggled."""
    def mouseReleaseEvent(self, event):
        action = self.activeAction()
        if action and action.isCheckable():
            action.trigger()
            return  # Don't close
        super().mouseReleaseEvent(event)

from src.detector import Detector, Detection
from src.violations import ViolationDetector, Violation
from src.tracker import ConstraintAwareSORT
from src.ocr import get_ocr_engine, PlateReadingAccumulator
from src.utils import (
    draw_detection, 
    draw_violation_alert, 
    boxes_overlap, 
    RateLimiter, 
    BoxSmoother, 
    get_lower_region,
    is_person_on_motorcycle
)
from ui.llm_widgets import ModelSelector
from ui.camera_manager import CameraManagerDialog, CameraEditDialog, CameraItemWidget


class DetectionThread(QThread):
    """Background thread for detection processing"""
    frame_ready = pyqtSignal(np.ndarray, dict, list, list)  # frame, detections dict, violations, vehicle_info
    error = pyqtSignal(str)
    models_loaded = pyqtSignal(bool, bool)  # has_helmet_model, has_plate_model
    device_info = pyqtSignal(str)  # device name
    video_duration = pyqtSignal(int) # Total frames
    video_position = pyqtSignal(int) # Current frame
    
    def __init__(self):
        super().__init__()
        self.detector: Optional[Detector] = None
        self.violation_detector: Optional[ViolationDetector] = None
        self.plate_ocr = None  # OCR for license plates
        self.llm_provider = None # LLM Provider
        self.llm_attempted_plates = set() # Track IDs where LLM was already used
        self.llm_attempted_helmets = set() # Track IDs where LLM was already used for helmets
        self.llm_attempted_triples = set() # Track IDs for triples
        self.llm_attempted_phones = set()  # Track IDs for phone check
        self.llm_voting_buffer = defaultdict(list) # {track_id: [crops]} for triples
        self.llm_voting_buffer_phones = defaultdict(list) # {track_id: [crops]} for phones
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.source = None
        self.use_helmet_model = False
        self.use_plate_model = False
        self.device_type = "auto"
        self.vehicle_plates = {}  # Cache for plate numbers: {track_id: plate_text}
        self.paused = False
        self.seek_target = -1 # Frame to seek to
        self.llm_rate_limiter = RateLimiter(config.LLM_MAX_RPM)
        
        # Stability Components
        self.plate_smoothers = defaultdict(BoxSmoother) # {track_id: BoxSmoother}
        self.plate_accumulator = PlateReadingAccumulator()
        self.plate_relative_offsets = {} # {track_id: [rel_x1, rel_y1, rel_x2, rel_y2]}
        self.plate_ghost_count = defaultdict(int) # {track_id: frames_since_det}
        
        # Wiring Parameters from UI
        self.stop_line_y = config.STOP_LINE_Y
        self.signal_roi = config.SIGNAL_ROI
        self.ocr_engine_type = config.OCR_ENGINE
        
        # Tracker
        self.tracker = None

    
    def pause(self):
        """Pause detection"""
        self.paused = True
    
    def resume(self):
        """Resume detection"""
        self.paused = False
        
    def seek(self, frame_index):
        """Seek to specific frame"""
        self.seek_target = frame_index
    
    def initialize_models(self, device_type: str = "auto"):
        """Load detection models with specified device"""
        try:
            self.device_type = device_type
            self.detector = Detector(device_type=device_type)
            self.violation_detector = ViolationDetector()
            
            # Check which models are available
            self.use_helmet_model = self.detector.has_helmet_model()
            self.use_plate_model = self.detector.has_plate_model()
            
            # Initialize OCR using factory
            try:
                self.plate_ocr = get_ocr_engine(self.ocr_engine_type, use_gpu=(self.device_type != "cpu"))
                
                # Initialize LLM
                from src.llm import get_llm_provider
                self.llm_provider = get_llm_provider()
                if self.llm_provider:
                    print(f"LLM Provider initialized: {self.llm_provider.model}")
                else:
                    print("LLM Provider not configured (no API key).")
                    
            except Exception as e:
                print(f"OCR/LLM initialization failed: {e}")
            
            # Initialize Tracker
            self.tracker = ConstraintAwareSORT(
                max_age=config.TRACKER_MAX_AGE,
                min_hits=config.TRACKER_MIN_HITS,
                iou_threshold=config.TRACKER_IOU_THRESHOLD
            )

            
            # Emit model availability and device info
            self.models_loaded.emit(self.use_helmet_model, self.use_plate_model)
            self.device_info.emit(self.detector.device_name)
            
            return True
        except Exception as e:
            self.error.emit(f"Failed to load models: {e}")
            return False
            
    def set_source(self, source):
        """Set video source (file path or camera index)"""
        self.source = source

    def clear_caches(self, hard: bool = True):
        """Reset all temporal caches and accumulators. If hard=False, persist plates."""
        if hard:
            self.vehicle_plates.clear()
        else:
            # Persist plates that match the Indian pattern
            from src.ocr import validate_indian_plate
            to_keep = {tid: p for tid, p in self.vehicle_plates.items() if validate_indian_plate(p)}
            self.vehicle_plates = to_keep

        self.plate_smoothers.clear()
        
        # Reset stateful modules
        if hasattr(self.plate_accumulator, 'reset'):
            self.plate_accumulator.reset(hard=hard)
        
        if hasattr(self.violation_detector, 'reset') and hard:
            self.violation_detector.reset()

        self.plate_relative_offsets.clear()
        self.plate_ghost_count.clear()
        self.llm_attempted_plates.clear()
        self.llm_attempted_helmets.clear()
        self.llm_attempted_triples.clear()
        self.llm_attempted_phones.clear()
        self.llm_voting_buffer.clear()
        self.llm_voting_buffer_phones.clear()
        
        if self.tracker and hard:
            self.tracker = ConstraintAwareSORT(
                max_age=config.TRACKER_MAX_AGE,
                min_hits=config.TRACKER_MIN_HITS,
                iou_threshold=config.TRACKER_IOU_THRESHOLD
            )
        logger.info(f"Session Reset ({'HARD' if hard else 'SOFT'}): Caches cleared.")

    def run(self):
        """Main detection loop"""
        if self.source is None:
            self.error.emit("No video source specified")
            return
        
        # Fresh start for every run
        self.clear_caches()
        
        # Open video source
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(str(self.source))
        
        if not self.cap.isOpened():
            self.error.emit(f"Failed to open video source: {self.source}")
            return
        
        self.running = True
        
        # Emit total duration if file
        if isinstance(self.source, str):
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration.emit(total_frames)
        
        while self.running:
            # Handle Seek
            if self.seek_target >= 0:
                # Clear tracking cache for large seeks
                if abs(self.cap.get(cv2.CAP_PROP_POS_FRAMES) - self.seek_target) > 50:
                    self.clear_caches(hard=True)

                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_target)
                self.seek_target = -1
                
                # If paused, we still want to show the new frame
                if self.paused:
                    # Robust fetch: Sometimes one read is empty right after seek
                    for _ in range(5):
                        ret, frame = self.cap.read()
                        if ret:
                            self.frame_ready.emit(frame, {}, [], [])
                            break
                        self.msleep(10)
                    continue

            # Handle Pause
            if self.paused:
                self.msleep(100)
                continue
                
            ret, frame = self.cap.read()
            
            if not ret:
                # Loop video file
                if isinstance(self.source, str):
                    logger.info("Video Looping: Resetting Session (Persisting Plates)...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.clear_caches(hard=False)
                    continue
                else:
                    break
            
            try:
                # Run multi-model detection
                detections = self.detector.detect(frame, track=True)
                
                # --- LOGGING ACTIONS ---
                main_count = len(detections.get("main", []))
                helmet_count = len(detections.get("helmets", []))
                plate_count = len(detections.get("plates", []))
                if main_count > 0:
                    logger.info(f"Frame {int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))}: Detected {main_count} vehicles, {helmet_count} helmets, {plate_count} plates.")
                
                # --- Vehicle & Plate Association Logic ---
                vehicle_info = []
                main_dets = detections.get("main", [])
                plate_dets = detections.get("plates", [])
                
                for det in main_dets:
                    if det.class_name in ["car", "motorcycle", "bus", "truck"] and det.track_id is not None:
                        # Check for associated plate
                        plate_text = self.vehicle_plates.get(det.track_id)
                        
                        # If no plate cached, try to find one and OCR it
                        if not plate_text or self.plate_ghost_count[det.track_id] > 0:
                            # Use Plausibility Filter: Plate must be in the lower half of vehicle
                            lower_region = get_lower_region(det.box, ratio=0.6)
                            
                            for plate in plate_dets:
                                # Strict association: Box must overlap with lower region
                                if boxes_overlap(lower_region, plate.box, threshold=0.1):
                                    # Smooth the plate box for more stable cropping/drawing
                                    # Since plates aren't tracked, we associate smoother with vehicle ID
                                    smoother_key = f"P_{det.track_id}"
                                    plate.box = self.plate_smoothers[smoother_key].update(plate.box)
                                    
                                    # Try standard OCR first
                                    if self.plate_ocr:
                                        cropped = self.plate_ocr.crop_plate_from_frame(frame, plate.box)
                                        text, conf = self.plate_ocr.read_plate(cropped)
                                        
                                        # Accumulate for consensus (now pattern-aware via module constants)
                                        self.plate_accumulator.add_reading(det.track_id, text, cropped)
                                        consensus_text = self.plate_accumulator.get_consensus(det.track_id)
                                        
                                        # Use consensus if available (now pattern-aware), fallback to single reading
                                        text = consensus_text or text
                                        
                                        # RELOCK LOGIC: If we find a higher confidence real detection, 
                                        # update the tracking offset even if we had a ghost
                                        is_new_best = False
                                        if det.track_id not in self.plate_relative_offsets or conf > 0.6:
                                            is_new_best = True
                                        
                                        # Fallback to LLM if standard OCR is weak and consensus isn't reached
                                        if (not text or len(text) < 5) and self.llm_provider:
                                            # ... rest of LLM logic stays same ...
                                            if det.track_id not in self.llm_attempted_plates:
                                                if self.llm_rate_limiter.can_request():
                                                    # Use the BEST crop from accumulator for LLM call
                                                    best_crops = self.plate_accumulator.best_crops.get(det.track_id, [])
                                                    llm_crop = best_crops[0][1] if best_crops else cropped
                                                    
                                                    logger.info(f"AI Action: Requesting LLM Plate Reading for ID {det.track_id}")
                                                    llm_text, llm_conf = self.llm_provider.read_plate(llm_crop)
                                                    logger.info(f"AI Reaction: LLM read plate {det.track_id} as '{llm_text}'")
                                                    if llm_text:
                                                        text = llm_text
                                                        conf = llm_conf
                                                        self.llm_attempted_plates.add(det.track_id)
                                                        det.llm_verified = True
                                                        # Put LLM result back into accumulator
                                                        self.plate_accumulator.confirmed_plates[det.track_id] = llm_text
                                        
                                        # Update cache with best pattern/consensus (including comma-separated candidates)
                                        if text:
                                            plate_text = text
                                            self.vehicle_plates[det.track_id] = text
                                        
                                        # --- RE-LOCK RELATIVE POSITION ---
                                        if is_new_best:
                                            v_w = det.box[2] - det.box[0]
                                            v_h = det.box[3] - det.box[1]
                                            if v_w > 0 and v_h > 0:
                                                self.plate_relative_offsets[det.track_id] = [
                                                    (plate.box[0] - det.box[0]) / v_w,
                                                    (plate.box[1] - det.box[1]) / v_h,
                                                    (plate.box[2] - det.box[0]) / v_w,
                                                    (plate.box[3] - det.box[1]) / v_h
                                                ]
                                                self.plate_ghost_count[det.track_id] = 0
                                        break
                                        
                        # --- NEW: GHOST TRACKING (If detector missed this frame) ---
                        if not any(boxes_overlap(det.box, p.box, threshold=0.01) for p in plate_dets):
                            if det.track_id in self.plate_relative_offsets:
                                if self.plate_ghost_count[det.track_id] < 15: # Increased persistence window for smoother highlighting
                                    v_box = det.box
                                    v_w = v_box[2] - v_box[0]
                                    v_h = v_box[3] - v_box[1]
                                    rel = self.plate_relative_offsets[det.track_id]
                                    
                                    # More robust calculation
                                    ghost_box = [
                                        v_box[0] + rel[0] * v_w,
                                        v_box[1] + rel[1] * v_h,
                                        v_box[0] + rel[2] * v_w,
                                        v_box[1] + rel[3] * v_h
                                    ]
                                    
                                    # Create a virtual plate detection for the UI
                                    ghost_det = Detection(
                                        box=ghost_box,
                                        class_id=-1,
                                        class_name="plate",
                                        confidence=0.5,
                                        track_id=None,
                                        source_model="ghost_tracker"
                                    )
                                    detections.setdefault("plates", []).append(ghost_det)
                        
                        # Deduplication Logic
                        # Do not add to info list if this plate was already seen on another vehicle this frame
                        if plate_text and any(v.get("plate") == plate_text for v in vehicle_info):
                            continue
                        
                        # Format candidate list for the UI (Patterns from Accumulator)
                        consensus = self.plate_accumulator.get_consensus(det.track_id)
                        raw_readings = self.plate_accumulator.readings.get(det.track_id, [])
                        
                        if consensus:
                            # Primary display is the consensus/best pattern
                            display_plate = consensus
                            
                            # Add secondary candidates if they exist and are different from consensus
                            others = [r for r in set(raw_readings) if r != consensus]
                            if others:
                                display_plate += f" ({', '.join(others[:2])})"
                        else:
                            display_plate = ", ".join(list(set(raw_readings))[:3]) if raw_readings else "No Plate detected"

                        # Add to info list
                        vehicle_info.append({
                            "id": det.track_id,
                            "type": det.class_name,
                            "plate": display_plate,
                            "violation": ""  # Placeholder
                        })

                # --- EXCESS RIDER HEURISTIC COUNTING ---
                for det in main_dets:
                    if det.class_name == "motorcycle":
                        # Count persons overlapping with this motorcycle
                        rider_count = 0
                        for person in main_dets:
                            if person.class_name == "person":
                                if is_person_on_motorcycle(person.box, det.box):
                                    rider_count += 1
                        
                        # Store count as attribute, DO NOT modify class_name yet
                        det.rider_count = rider_count

                # Check for violations with helmet/plate model flags
                violations = self.violation_detector.detect_all(
                    frame,
                    detections,
                    use_helmet_model=self.use_helmet_model,
                    use_plate_model=self.use_plate_model,
                    stop_line_y=self.stop_line_y,
                    signal_roi=self.signal_roi
                )
                
                # --- UNIVERSAL LLM HELMET VERIFICATION (Guarded by toggles) ---
                if self.llm_provider:
                    self.llm_rate_limiter.max_rpm = config.LLM_MAX_RPM
                    
                    # Check EVERY motorcycle for helmet status (not just violations)
                    for det in main_dets:
                        if det.class_name == "motorcycle":
                            # Use separate tracking sets for each LLM check type
                            needs_helmet = det.track_id not in self.llm_attempted_helmets
                            needs_triple = det.track_id not in self.llm_attempted_triples
                            needs_phone = det.track_id not in self.llm_attempted_phones
                            
                            if (needs_helmet or needs_triple or needs_phone) and self.llm_rate_limiter.can_request():
                                # Take a high-res crop with enough padding for riders/passengers
                                x1, y1, x2, y2 = map(int, det.box)
                                h, w = frame.shape[:2]
                                pad = 80
                                x1 = max(0, x1 - pad); y1 = max(0, y1 - (pad*3)) # Headroom for riders
                                x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
                                crop = frame[y1:y2, x1:x2]
                                
                                if crop.size > 0:
                                    # --- HELMET CHECK ---
                                    if needs_helmet and config.ENABLED_VIOLATIONS.get("NO_HELMET", True):
                                        if self.llm_rate_limiter.can_request():
                                            logger.info(f"AI Action: Requesting Helmet Verification for ID {det.track_id}")
                                            is_wearing, reasoning = self.llm_provider.verify_helmet(crop)
                                            logger.info(f"AI Reaction: LLM says ID {det.track_id} is wearing helmet: {is_wearing} ({reasoning})")
                                            
                                            if not is_wearing:
                                                # Delete valid helmet boxes & Force NO_HELMET violation
                                                if "helmets" in detections:
                                                    riders = [d for d in main_dets if d.track_id is not None and 
                                                             d.class_name == "person" and 
                                                             boxes_overlap(d.box, det.box, threshold=0.1)]
                                                    rider_ids = [r.track_id for r in riders]
                                                    detections["helmets"] = [h for h in detections["helmets"] 
                                                                           if h.track_id not in rider_ids]
                                                
                                                if not any(v.track_id == det.track_id and v.type == "NO_HELMET" for v in violations):
                                                    new_v = Violation(
                                                        type="NO_HELMET", vehicle_box=det.box, track_id=det.track_id,
                                                        confidence=0.99, timestamp=time.time(),
                                                        details=f"AI VERIFIED: {reasoning}"
                                                    )
                                                    new_v.llm_verified = True; new_v.llm_reasoning = reasoning
                                                    violations.append(new_v)
                                            else:
                                                # If LLM confirms they ARE wearing a helmet, drop any false-positive YOLO NO_HELMET violations
                                                violations = [v for v in violations if not (v.track_id == det.track_id and v.type == "NO_HELMET")]
                                        # If rate limited, we do nothing and let the YOLO base heuristic stand (No else block dropping them!)

                                    # --- EXCESS RIDING CHECK (SMART VOTING) ---
                                    if needs_triple and config.ENABLED_VIOLATIONS.get("TRIPLE_RIDING", True):
                                        # Add to buffer
                                        self.llm_voting_buffer[det.track_id].append(crop)
                                        
                                        # Only call LLM once we have 3 crops (Voting)
                                        if len(self.llm_voting_buffer[det.track_id]) >= 3:
                                            logger.info(f"AI Action: Starting Multi-Frame Voting for ID {det.track_id}")
                                            crops = self.llm_voting_buffer[det.track_id]
                                            count, triple_reason = self.llm_provider.check_passengers_voting(crops)
                                            logger.info(f"AI Reaction: LLM Voting for ID {det.track_id} resulted in {count} riders ({triple_reason})")
                                            
                                            if "Error:" in triple_reason:
                                                print(f"CRITICAL: AI Brain Error in Voting: {triple_reason}")
                                                det.llm_verified = False 
                                            else:
                                                self.llm_attempted_triples.add(det.track_id)
                                                det.llm_verified = True
                                                
                                                # If confirmed, update/add violation
                                                if count >= getattr(config, 'TRIPLE_RIDING_MIN_PERSONS', 3):
                                                    new_v = Violation(
                                                        type="TRIPLE_RIDING", 
                                                        vehicle_box=det.box, 
                                                        track_id=det.track_id,
                                                        confidence=0.99, 
                                                        timestamp=time.time(),
                                                        details=f"AI VOTED ({count} riders): {triple_reason}"
                                                    )
                                                    new_v.llm_verified = True
                                                    new_v.llm_reasoning = triple_reason
                                                    violations.append(new_v)
                                                
                                                # Clear buffer for this track
                                                del self.llm_voting_buffer[det.track_id]
                                            
                                            # Store count as attribute for UI, DO NOT modify class_name
                                            det.rider_count = count
                                        else:
                                            # If LLM says < 3, remove any heuristic-based triple riding violation
                                            violations = [v for v in violations if not (v.track_id == det.track_id and v.type == "TRIPLE_RIDING")]
                                    
                                    # --- PHONE USAGE CHECK (SMART VOTING) ---
                                    if needs_phone and config.ENABLED_VIOLATIONS.get("PHONE_USAGE", True):
                                        # Add to phone buffer
                                        self.llm_voting_buffer_phones[det.track_id].append(crop)
                                        
                                        # Only call LLM once we have 3 crops (Voting)
                                        if len(self.llm_voting_buffer_phones[det.track_id]) >= 3:
                                            logger.info(f"AI Action: Starting Multi-Frame Phone Voting for ID {det.track_id}")
                                            crops = self.llm_voting_buffer_phones[det.track_id]
                                            is_using_phone, phone_reason = self.llm_provider.verify_phone_usage_voting(crops)
                                            logger.info(f"AI Reaction: LLM Phone Voting for ID {det.track_id} resulted in: {is_using_phone} ({phone_reason})")
                                            
                                            if "Error:" in phone_reason:
                                                print(f"CRITICAL: AI Brain Error in Phone Voting: {phone_reason}")
                                            else:
                                                self.llm_attempted_phones.add(det.track_id)
                                                
                                                if is_using_phone:
                                                    # Associate phone usage with the rider (person) track ID if possible
                                                    target_track_id = det.track_id
                                                    target_box = det.box
                                                    
                                                    # Find the rider (person) associated with this motorcycle
                                                    riders = [r for r in main_dets if r.class_name == "person" and is_person_on_motorcycle(r.box, det.box)]
                                                    if riders:
                                                        target_track_id = riders[0].track_id
                                                        target_box = riders[0].box
                                                        
                                                    # Add phone usage violation
                                                    if not any(v.track_id == target_track_id and v.type == "PHONE_USAGE" for v in violations):
                                                        new_v = Violation(
                                                            type="PHONE_USAGE",
                                                            vehicle_box=target_box,
                                                            track_id=target_track_id,
                                                            confidence=0.99,
                                                            timestamp=time.time(),
                                                            details=f"AI VOTED: {phone_reason}"
                                                        )
                                                        new_v.llm_verified = True
                                                        new_v.llm_reasoning = phone_reason
                                                        violations.append(new_v)
                                                
                                                # Clear buffer for this track
                                                del self.llm_voting_buffer_phones[det.track_id]
                                        else:
                                            # If we don't have enough frames yet, ensure any pending heuristic-only violation is suppressed
                                            # (Wait for LLM consensus)
                                            violations = [v for v in violations if not (v.track_id == det.track_id and v.type == "PHONE_USAGE")]
                                
                
                # Filter out LLM-overridden violations (internal utility type)
                violations = [v for v in violations if v.type != "HELMET_VERIFIED"]
                
                # Auto-save violations to database
                from src.database import ViolationDatabase
                db = ViolationDatabase() # Get singleton/instance
                for v in violations:
                    # Enrich violation with plate if available
                    plate_text = self.vehicle_plates.get(v.track_id)
                    v.plate_number = plate_text
                    
                    # Save to DB
                    db.insert_violation(
                        v.type, v.confidence, track_id=v.track_id, 
                        plate_number=plate_text, details=v.details, 
                        frame=frame, llm_reasoning=v.llm_reasoning or ""
                    )

                
                # Map violations to vehicle_info
                from collections import defaultdict
                violation_map = defaultdict(list)
                for v in violations:
                    if v.track_id is not None and v.type not in violation_map[v.track_id]:
                        violation_map[v.track_id].append(v.type)
                        
                for v_info in vehicle_info:
                    if v_info["id"] in violation_map:
                        v_info["violation"] = ", ".join(violation_map[v_info["id"]])
                
                # Emit results
                self.frame_ready.emit(frame, detections, violations, vehicle_info)
                
                # Emit current progress
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.video_position.emit(current_frame)
                
            except Exception as e:
                print(f"Detection error: {e}")
                import traceback
                traceback.print_exc()
        
        if self.cap:
            self.cap.release()
    
    def stop(self):
        """Stop the detection loop"""
        self.running = False
        self.wait()



class KpiCard(QFrame):
    """Modern KPI metric card"""
    def __init__(self, title, value="0", parent=None):
        super().__init__(parent)
        self.setObjectName("kpiCard")
        layout = QVBoxLayout(self)
        
        self.title_label = QLabel(title)
        self.title_label.setObjectName("kpiTitle")
        
        self.value_label = QLabel(value)
        self.value_label.setObjectName("kpiValue")
        self.value_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
    
    def set_value(self, value):
        self.value_label.setText(str(value))


class JumpSlider(QSlider):
    """Custom QSlider that jumps to click position"""
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            val = self.minimum() + ((self.maximum() - self.minimum()) * event.x()) / self.width()
            self.setValue(int(val))
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    """Main application window - Dashboard Redesign"""
    
    def __init__(self):
        super().__init__()
        self.detection_thread: Optional[DetectionThread] = None
        self.is_running = False
        self.violation_counts = {
            "TRIPLE_RIDING": 0,
            "OVERSPEED": 0,
            "WRONG_WAY": 0,
            "PHONE_USAGE": 0,
            "NO_HELMET": 0,
            "MISSING_PLATE": 0,
            "RED_SIGNAL": 0,
        }
        self.seen_violations = set() # Track (track_id, type) to avoid double counting
        self.slider_pressed = False
        self.stop_line_y = None      # Stop-line Y position
        self.signal_roi = None       # Signal ROI (x1, y1, x2, y2)
        self.db = None               # ViolationDatabase reference
        self.report_gen = None       # ReportGenerator reference
        
        # Initialize database and report generator
        try:
            from src.database import ViolationDatabase
            from src.report import ReportGenerator
            self.db = ViolationDatabase()
            self.report_gen = ReportGenerator()
        except Exception as e:
            print(f"DB/Report init: {e}")
        
        self.init_ui()
        self.load_stylesheet()
        self.init_detection_thread()
    
    def init_ui(self):
        """Initialize the dashboard user interface"""
        self.setWindowTitle(config.WINDOW_TITLE)
        self.setMinimumSize(1300, 850)
        
        # Set App Icon
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Base horizontal layout (Sidebar | Content)
        base_layout = QHBoxLayout(central)
        base_layout.setContentsMargins(0, 0, 0, 0)
        base_layout.setSpacing(0)
        
        # --- SIDEBAR ---
        sidebar = QFrame()
        sidebar.setObjectName("sidePanel")
        sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 30, 20, 30)
        sidebar_layout.setSpacing(15)
        
        # App Title in Sidebar
        logo_label = QLabel("TVDS")
        logo_label.setStyleSheet("font-size: 28px; font-weight: 600; color: #D0BCFF; letter-spacing: -0.5px;")
        logo_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(logo_label)
        
        slogan_label = QLabel("V I O L A T I O N   D E T E C T O R")
        slogan_label.setStyleSheet("font-size: 8px; font-weight: 700; color: #94A3B8; letter-spacing: 2px; margin-top: -8px;")
        slogan_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(slogan_label)
        
        sidebar_layout.addSpacing(30)
        
        # Navigation Tabs Button Style (using Sidebar buttons to control QTabWidget)
        nav_label = QLabel("NAVIGATION")
        nav_label.setStyleSheet("color: #C4C6D0; font-size: 11px; font-weight: 700; letter-spacing: 1.2px; margin-left: 10px;")
        sidebar_layout.addWidget(nav_label)
        
        # Navigation Button Group for Material Tonal state
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        self.btn_nav_live = QPushButton("  ○  Live View")
        self.btn_nav_live.setCheckable(True)
        self.btn_nav_live.setChecked(True)
        self.btn_nav_live.clicked.connect(lambda: self._switch_tab(0))
        self.nav_group.addButton(self.btn_nav_live)
        sidebar_layout.addWidget(self.btn_nav_live)
        
        self.btn_nav_history = QPushButton("  ⟳  History")
        self.btn_nav_history.setCheckable(True)
        self.btn_nav_history.clicked.connect(lambda: self._switch_tab(1))
        self.nav_group.addButton(self.btn_nav_history)
        sidebar_layout.addWidget(self.btn_nav_history)
        
        self.btn_nav_settings = QPushButton("  ⚙  Settings")
        self.btn_nav_settings.setCheckable(True)
        self.btn_nav_settings.clicked.connect(lambda: self._switch_tab(2))
        self.nav_group.addButton(self.btn_nav_settings)
        sidebar_layout.addWidget(self.btn_nav_settings)
        
        sidebar_layout.addSpacing(15)
        det_label = QLabel("DETECTION ENGINE")
        det_label.setStyleSheet("color: #74777F; font-size: 11px; font-weight: 700; letter-spacing: 0.8px; margin-left: 10px;")
        sidebar_layout.addWidget(det_label)
        
        self.btn_violation_toggle = QPushButton("Detection Toggles (7/8)")
        self.btn_violation_toggle.setObjectName("violationToggleBtn")
        self.violation_menu = PersistentMenu(self)
        self.violation_menu.setObjectName("violationMenu")
        self._build_violation_menu()
        self.btn_violation_toggle.setMenu(self.violation_menu)
        sidebar_layout.addWidget(self.btn_violation_toggle)
        
        # --- WEB CAM MANAGEMENT ---
        sidebar_layout.addSpacing(25)
        
        cam_header = QHBoxLayout()
        cam_label = QLabel("WEB CAMS")
        cam_label.setStyleSheet("color: #74777F; font-size: 11px; font-weight: 700; letter-spacing: 0.8px; margin-left: 10px;")
        cam_header.addWidget(cam_label)
        cam_header.addStretch()
        
        self.cam_add_btn = QPushButton("+")
        self.cam_add_btn.setFixedSize(28, 28)
        self.cam_add_btn.setCursor(Qt.PointingHandCursor)
        self.cam_add_btn.setObjectName("galleryBrowseBtn") # Reusing add button style
        self.cam_add_btn.setToolTip("Add New Web Cam")
        self.cam_add_btn.clicked.connect(self._add_web_cam)
        cam_header.addWidget(self.cam_add_btn)
        cam_header.addSpacing(8)
        sidebar_layout.addLayout(cam_header)
        
        self.camera_list = QListWidget()
        self.camera_list.setObjectName("cameraList")
        self.camera_list.setSpacing(4)
        self.camera_list.setMaximumHeight(350)
        self.camera_list.setMinimumHeight(150)
        sidebar_layout.addWidget(self.camera_list)
        
        # Load cameras immediately
        self._load_web_cams()
        
        sidebar_layout.addStretch()
        
        # System Health/Status
        sidebar_layout.addWidget(QLabel("<b>SYSTEM</b>"))
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Device Status:"))
        self.device_status = QLabel("⚪")
        self.device_status.setAlignment(Qt.AlignCenter)
        status_row.addWidget(self.device_status)
        sidebar_layout.addLayout(status_row)
        
        base_layout.addWidget(sidebar)
        
        # --- MAIN CONTENT AREA ---
        content_area = QVBoxLayout()
        content_area.setContentsMargins(30, 30, 30, 30)
        content_area.setSpacing(20)
        
        # Header Row (Brain Selector)
        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("<h2>Dashboard Overview</h2>"))
        header_row.addStretch()
        
        self.model_selector = ModelSelector()
        self.model_selector.model_changed.connect(self.switch_brain)
        self.model_selector.device_changed.connect(self.switch_device)
        header_row.addWidget(self.model_selector)
        content_area.addLayout(header_row)
        
        # KPI Row (Stats Cards)
        kpi_row = QHBoxLayout()
        self.stat_cards = {}
        violation_types = [
            ("RED_SIGNAL", "Red Signal"),
            ("TRIPLE_RIDING", "Triple Riding"),
            ("PHONE_USAGE", "Phone Use"),
            ("NO_HELMET", "No Helmet"),
            ("MISSING_PLATE", "No Plate"),
            ("OVERSPEED", "Speeding")
        ]
        
        for key, title in violation_types:
            card = KpiCard(title)
            kpi_row.addWidget(card)
            self.stat_cards[key] = card
        
        content_area.addLayout(kpi_row)
        
        # Tab Widget (Hidden Header)
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("mainTabs")
        self.tab_widget.tabBar().hide() # Hide default tabs, use sidebar
        
        # --- TAB 1: LIVE DASHBOARD ---
        live_tab = QWidget()
        live_layout = QHBoxLayout(live_tab)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setSpacing(20)
        
        # Video Feed Column
        video_col = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setObjectName("videoPanel")
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_col.addWidget(self.video_label, stretch=1)
        
        # Control Bar
        control_bar = QFrame()
        control_bar.setObjectName("controlBar")
        control_layout = QHBoxLayout(control_bar)
        
        self.btn_open = QPushButton("Open File")
        self.btn_open.setObjectName("openBtn")
        self.btn_open.clicked.connect(self.open_video)
        control_layout.addWidget(self.btn_open)
        
        self.btn_camera = QPushButton("Camera")
        self.btn_camera.clicked.connect(self.start_camera)
        control_layout.addWidget(self.btn_camera)
        
        # Seek Slider
        self.seek_slider = JumpSlider(Qt.Horizontal)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderPressed.connect(self.on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self.on_slider_released)
        self.seek_slider.valueChanged.connect(self.on_slider_changed)
        control_layout.addWidget(self.seek_slider, stretch=1)
        
        self.btn_play_pause = QPushButton("Pause")
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        control_layout.addWidget(self.btn_play_pause)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_detection)
        control_layout.addWidget(self.btn_stop)
        
        video_col.addWidget(control_bar)
        live_layout.addLayout(video_col, stretch=2)
        
        # Real-time Logs Column
        log_col = QVBoxLayout()
        
        # Vehicle Table Mini
        self.vehicle_table = QTableWidget()
        self.vehicle_table.setColumnCount(4)
        self.vehicle_table.setHorizontalHeaderLabels(["ID", "TYPE", "PLATE", "ERR"])
        self.vehicle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        log_col.addWidget(self.vehicle_table, stretch=1)
        
        # Violation Log
        self.violation_list = QListWidget()
        log_col.addWidget(self.violation_list, stretch=1)
        
        self.btn_clear_log = QPushButton("Clear Activity")
        self.btn_clear_log.clicked.connect(self.clear_log)
        log_col.addWidget(self.btn_clear_log)
        
        live_layout.addLayout(log_col, stretch=1)
        
        self.tab_widget.addTab(live_tab, "Live")
        
        # --- TAB 2: HISTORY ---
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(["ID", "TIME", "TYPE", "ID", "PLATE", "CONF"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.history_table, stretch=1)
        
        h_ctrls = QHBoxLayout()
        self.btn_refresh_history = QPushButton("Refresh Data")
        self.btn_refresh_history.clicked.connect(self.refresh_history)
        h_ctrls.addWidget(self.btn_refresh_history)
        
        self.btn_export_csv = QPushButton("CSV Report")
        self.btn_export_csv.clicked.connect(self.export_csv)
        h_ctrls.addWidget(self.btn_export_csv)
        
        self.btn_export_html = QPushButton("HTML Report")
        self.btn_export_html.clicked.connect(self.export_html)
        h_ctrls.addWidget(self.btn_export_html)
        history_layout.addLayout(h_ctrls)
        
        self.tab_widget.addTab(history_tab, "History")
        
        # --- TAB 3: SETTINGS ---
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        # Config Grid
        cfg_frame = QFrame()
        cfg_layout = QVBoxLayout(cfg_frame)
        
        # Stopline
        sl_box = QGroupBox("Calibration: Stop-Line")
        sl_l = QHBoxLayout(sl_box)
        self.stop_line_spin = QSpinBox()
        self.stop_line_spin.setRange(0, 2000)
        self.stop_line_spin.setValue(config.STOP_LINE_Y)
        self.stop_line_spin.valueChanged.connect(self.set_stop_line)
        sl_l.addWidget(QLabel("Vertical Y (px):"))
        sl_l.addWidget(self.stop_line_spin)
        self.stopline_status = QLabel("")
        sl_l.addWidget(self.stopline_status)
        cfg_layout.addWidget(sl_box)
        
        # Signal ROI
        roi_box = QGroupBox("Calibration: Signal ROI")
        roi_l = QHBoxLayout(roi_box)
        self.roi_x1 = QSpinBox()
        self.roi_y1 = QSpinBox()
        self.roi_x2 = QSpinBox()
        self.roi_y2 = QSpinBox()
        for s in [self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2]:
            s.setRange(0, 3000)
            roi_l.addWidget(s)
        self.btn_set_signal_roi = QPushButton("Update ROI")
        self.btn_set_signal_roi.clicked.connect(self.set_signal_roi)
        roi_l.addWidget(self.btn_set_signal_roi)
        cfg_layout.addWidget(roi_box)
        
        # OCR
        ocr_box = QGroupBox("Intelligence: OCR Engine")
        ocr_l = QHBoxLayout(ocr_box)
        self.ocr_combo = QComboBox()
        self.ocr_combo.addItems(["EasyOCR", "TrOCR", "PaddleOCR"])
        current_lower = config.OCR_ENGINE.lower()
        if current_lower == "trocr": self.ocr_combo.setCurrentText("TrOCR")
        elif current_lower == "paddleocr": self.ocr_combo.setCurrentText("PaddleOCR")
        self.ocr_combo.currentIndexChanged.connect(self.on_ocr_changed)
        ocr_l.addWidget(QLabel("Primary Engine:"))
        ocr_l.addWidget(self.ocr_combo)
        cfg_layout.addWidget(ocr_box)

        # LLM RPM Control
        llm_box = QGroupBox("Intelligence: AI Brain Speed")
        llm_l = QHBoxLayout(llm_box)
        self.rpm_spin = QSpinBox()
        self.rpm_spin.setRange(1, 1000)
        self.rpm_spin.setValue(config.LLM_MAX_RPM)
        self.rpm_spin.setSuffix(" RPM")
        self.rpm_spin.valueChanged.connect(self.on_rpm_changed)
        llm_l.addWidget(QLabel("Max Requests/Min:"))
        llm_l.addWidget(self.rpm_spin)
        cfg_layout.addWidget(llm_box)
        
        settings_layout.addWidget(cfg_frame)
        settings_layout.addStretch()
        
        self.tab_widget.addTab(settings_tab, "Settings")
        
        content_area.addWidget(self.tab_widget)
        base_layout.addLayout(content_area)
    
    def load_stylesheet(self):
        """Load the QSS stylesheet"""
        style_path = Path(__file__).parent / "styles.qss"
        if style_path.exists():
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
    
    def _build_violation_menu(self):
        """Build the checkable popup menu for enabling/disabling violations."""
        self.violation_menu.clear()
        self.violation_actions = {}
        
        # Clean text labels (no emojis) for each violation type
        violation_labels = [
            ("NO_HELMET", "No Helmet Detection"),
            ("TRIPLE_RIDING", "Triple / Excess Riding"),
            ("OVERSPEED", "Over Speed Detection"),
            ("WRONG_WAY", "Wrong Way Detection"),
            ("PHONE_USAGE", "Phone Usage Detection"),
            ("RED_SIGNAL", "Red Signal Jumping"),
            ("MISSING_PLATE", "Missing Number Plate"),
            ("ZEBRA_CROSSING", "Zebra Crossing Violation"),
        ]
        
        for key, label in violation_labels:
            action = self.violation_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(config.ENABLED_VIOLATIONS.get(key, False))
            action.toggled.connect(lambda checked, k=key: self._on_violation_toggled(k, checked))
            self.violation_actions[key] = action
        
        self._update_violation_badge()
    
    def _on_violation_toggled(self, violation_key: str, enabled: bool):
        """Handle a violation checkbox being toggled. Updates config immediately."""
        config.ENABLED_VIOLATIONS[violation_key] = enabled
        self._update_violation_badge()
        
        status = "ENABLED" if enabled else "DISABLED"
        label = violation_key.replace("_", " ").title()
        
        # Add to violation log for operator awareness
        item = QListWidgetItem(f"⚙ {label} detection {status}")
        if enabled:
            item.setForeground(QColor(80, 200, 120))  # Green
        else:
            item.setForeground(QColor(200, 80, 80))    # Red
        self.violation_list.insertItem(0, item)
    
    def _update_violation_badge(self):
        """Update the button text to show how many violations are active."""
        active = sum(1 for v in config.ENABLED_VIOLATIONS.values() if v)
        total = len(config.ENABLED_VIOLATIONS)
        self.btn_violation_toggle.setText(f"Detection Toggles ({active}/{total})")

    # --- WEB CAM LOGIC ---
    def _get_cam_config(self):
        """Helper to get camera list from file."""
        config_path = Path(__file__).parent.parent / "data" / "cameras.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except: pass
        return [{"name": "Default Webcam", "url": "0"}]

    def _save_cam_config(self, cameras):
        """Helper to save camera list to file."""
        config_path = Path(__file__).parent.parent / "data" / "cameras.json"
        config_path.parent.mkdir(exist_ok=True)
        try:
            with open(config_path, "w") as f:
                json.dump(cameras, f, indent=4)
        except: pass

    def _load_web_cams(self):
        """Populate the sidebar camera list."""
        self.camera_list.clear()
        cameras = self._get_cam_config()
        
        for idx, cam in enumerate(cameras):
            item = QListWidgetItem(self.camera_list)
            widget = CameraItemWidget(idx, cam['name'], cam['url'])
            
            # Connect signals
            widget.connect_requested.connect(self._connect_web_cam)
            widget.edit_requested.connect(self._edit_web_cam)
            widget.delete_requested.connect(self._delete_web_cam)
            widget.duplicate_requested.connect(self._duplicate_web_cam)
            
            item.setSizeHint(widget.sizeHint())
            self.camera_list.addItem(item)
            self.camera_list.setItemWidget(item, widget)

    def _add_web_cam(self):
        diag = CameraEditDialog(self)
        if diag.exec_():
            new_cam = diag.get_data()
            if new_cam['name'] and new_cam['url']:
                cameras = self._get_cam_config()
                cameras.append(new_cam)
                self._save_cam_config(cameras)
                self._load_web_cams()

    def _connect_web_cam(self, idx):
        cameras = self._get_cam_config()
        if 0 <= idx < len(cameras):
            self.start_detection(cameras[idx]['url'])

    def _edit_web_cam(self, idx):
        cameras = self._get_cam_config()
        if 0 <= idx < len(cameras):
            cam = cameras[idx]
            diag = CameraEditDialog(self, cam['name'], cam['url'])
            if diag.exec_():
                cameras[idx] = diag.get_data()
                self._save_cam_config(cameras)
                self._load_web_cams()

    def _delete_web_cam(self, idx):
        cameras = self._get_cam_config()
        if 0 <= idx < len(cameras):
            reply = QMessageBox.question(self, "Delete Camera", f"Delete '{cameras[idx]['name']}'?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                cameras.pop(idx)
                self._save_cam_config(cameras)
                self._load_web_cams()

    def _duplicate_web_cam(self, idx):
        cameras = self._get_cam_config()
        if 0 <= idx < len(cameras):
            new_cam = cameras[idx].copy()
            new_cam['name'] += " (Copy)"
            cameras.insert(idx + 1, new_cam)
            self._save_cam_config(cameras)
            self._load_web_cams()

    def init_detection_thread(self, device_type: str = "auto"):
        """Initialize the detection thread with specified device"""
        self.detection_thread = DetectionThread()
        self.detection_thread.frame_ready.connect(self.update_frame)
        self.detection_thread.error.connect(self.show_error)
        self.detection_thread.device_info.connect(self.update_device_status)
        self.detection_thread.video_duration.connect(self.set_seek_bar_range)
        self.detection_thread.video_position.connect(self.update_seek_bar)
        
        # Load models with selected device
        self.device_status.setText("...")
        self.device_status.setStyleSheet("color: #F59E0B;")
        self.device_status.setToolTip("Loading models...")
        
        if self.detection_thread.initialize_models(device_type):
            self.device_status.setText("ON")
            self.device_status.setStyleSheet("color: #10B981; font-weight: 700;")
            self.device_status.setToolTip(f"Active: {self.detection_thread.detector.device_name}")
        else:
            self.device_status.setText("ERR")
            self.device_status.setStyleSheet("color: #EF4444; font-weight: 700;")
            self.device_status.setToolTip("Failed to load models")
            QMessageBox.warning(
                self,
                "Model Loading",
                "Detection models failed to load.\\nPlease check console for errors."
            )
    
    def open_video(self):
        """Open a video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            getattr(self, "last_opened_dir", ""),
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        
        if file_path:
            self.last_opened_dir = os.path.dirname(file_path)
            self.start_detection(file_path)
    
    def start_camera(self):
        """Open the Camera Manager dialog to select an IP camera or webcam"""
        dialog = CameraManagerDialog(self)
        if dialog.exec_():
            url = dialog.selected_camera_url
            if url:
                # Cast numeric strings (like '0' or '1') to int for OpenCV local webcams
                if url.isdigit():
                    url = int(url)
                self.start_detection(url)
    
    def start_detection(self, source):
        """Start detection on the given source. Auto-stops any running detection first."""
        if self.is_running:
            self.stop_detection()
            # Wait briefly for thread to fully stop
            if self.detection_thread:
                self.detection_thread.wait(500)
        
        # Reset seek bar for the new source
        self.seek_slider.setValue(0)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        
        self.detection_thread.set_source(source)
        self.detection_thread.start()
        
        self.is_running = True
        self.btn_open.setEnabled(True)    # Keep enabled — allow loading new source anytime
        self.btn_camera.setEnabled(True)  # Keep enabled — allow switching to camera anytime
        self.btn_stop.setEnabled(True)
        self.btn_play_pause.setEnabled(True)
        self.btn_play_pause.setText("Pause")
        self.btn_play_pause.setToolTip("Pause Video")
        
    def toggle_play_pause(self):
        """Toggle between Play and Pause"""
        if not self.detection_thread:
            return
            
        if self.detection_thread.paused:
            self.resume_video()
        else:
            self.pause_video()
    
    def pause_video(self):
        """Pause the video"""
        if self.detection_thread:
            self.detection_thread.pause()
            self.btn_play_pause.setText("Resume")
            self.btn_play_pause.setToolTip("Resume Video")
            self.video_label.setText("Paused")
            
    def _switch_tab(self, index):
        """Switches tab and updates button checked state"""
        self.tab_widget.setCurrentIndex(index)
        
    def _setup_live_view(self):
        """Setup the live view tab, ensuring the video is resumed if paused."""
        # Ensure the Live View button is checked
        self.btn_live_view.setChecked(True)
        self.tab_widget.setCurrentIndex(0) # Switch to Live View tab
        if self.detection_thread and self.detection_thread.paused:
            self.resume_video()

    def _setup_gallery_view(self):
        """Setup the gallery view tab, pausing video if running."""
        # Ensure the Gallery button is checked
        self.btn_gallery.setChecked(True)
        self.tab_widget.setCurrentIndex(1) # Switch to Gallery tab
        if self.is_running and not self.detection_thread.paused:
            self.pause_video()
        
    def resume_video(self):
        """Resume the video"""
        if self.detection_thread:
            self.detection_thread.resume()
            self.btn_play_pause.setText("Pause")
            self.btn_play_pause.setToolTip("Pause Video")
    
    def stop_detection(self):
        """Stop the current detection and reset player state."""
        if self.detection_thread:
            self.detection_thread.stop()
        
        self.is_running = False
        self.btn_open.setEnabled(True)
        self.btn_camera.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.setText("Pause")
        self.btn_play_pause.setToolTip("Pause Video")
        
        # Reset seek bar to zero
        self.seek_slider.setValue(0)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        self.slider_pressed = False
        
        self.video_label.setText("Detection stopped  |  Click 'Open File' or 'Camera' to begin")

    # --- Seek Bar Methods ---
    def set_seek_bar_range(self, duration: int):
        self.seek_slider.setRange(0, duration)
        self.seek_slider.setEnabled(True)

    def update_seek_bar(self, position: int):
        if not self.slider_pressed:
            self.seek_slider.setValue(position)
            
    def on_slider_pressed(self):
        self.slider_pressed = True
        
    def on_slider_released(self):
        self.slider_pressed = False
        pos = self.seek_slider.value()
        if self.detection_thread:
            self.detection_thread.seek(pos)
            
    def on_slider_changed(self, value):
        # We could implement live seeking here but it might be laggy with heavy models
        pass

    
    def update_frame(self, frame: np.ndarray, detections: dict, violations: List[Violation], vehicle_info: list):
        """Update the video display with new frame and detections"""
        
        # Create a set of track_ids that have violations
        violator_ids = {v.track_id for v in violations if v.track_id is not None}
        
        # Draw main detections
        for det in detections.get("main", []):
            cls_name = det.class_name.lower().strip()
            if cls_name == "cell phone":
                continue  # Skip drawing phone boxes as requested
                
            display_name = det.class_name
            # Only show rider count if TRIPLE_RIDING toggle is enabled
            if det.class_name == "motorcycle" and hasattr(det, "rider_count") and config.ENABLED_VIOLATIONS.get("TRIPLE_RIDING", True):
                display_name = f"Motorcycle ({det.rider_count} Riders)"
                
            label = f"{display_name} ID:{det.track_id}" if det.track_id is not None else display_name
            
            # Default color: Green if okay, Orange if questionable, Red if violating
            color = (0, 255, 0) # Green
            
            # Specific color logic for Phone Usage Violations (Highlight the person red)
            is_phone_violator = False
            for v in violations:
                if v.type == "PHONE_USAGE" and v.track_id == det.track_id:
                    is_phone_violator = True
                    break
            
            if det.track_id in violator_ids:
                color = (0, 0, 255) # Red for violators
                # Find specific violation for label
                v_type = next((v.type for v in violations if v.track_id == det.track_id), det.class_name)
                label = f"VIOLATION! {v_type.replace('_', ' ').upper()}"
            elif "Riders" in det.class_name:
                # Highlight motorcycles with riders to show association is working
                color = (255, 165, 0) # Orange
            
            # Draw plate in label if cached
            if det.track_id in self.detection_thread.vehicle_plates:
                label += f" [{self.detection_thread.vehicle_plates[det.track_id]}]"
            
            llm_v = getattr(det, "llm_verified", False)
            frame = draw_detection(frame, det.box, label, color, llm_verified=llm_v)
        
        # Draw violations (These typically draw over the vehicle box with specific info like 'No Helmet')
        # We keep this as it adds the specific red box logic for the violation type
        
        # Update Vehicle Table
        self.vehicle_table.setSortingEnabled(False)  # Disable sorting while updating
        self.vehicle_table.setRowCount(len(vehicle_info))
        
        for i, v in enumerate(vehicle_info):
            # ID Column (Number)
            id_item = QTableWidgetItem()
            id_item.setData(Qt.DisplayRole, int(v['id'])) # Ensure correct numeric sorting
            self.vehicle_table.setItem(i, 0, id_item)
            
            # Type Column (String)
            type_item = QTableWidgetItem(v['type'].upper())
            self.vehicle_table.setItem(i, 1, type_item)
            
            # Plate Column (String)
            plate_val = v['plate']
            if plate_val == "No Plate detected":
                plate_val = "-"
            plate_item = QTableWidgetItem(plate_val)
            self.vehicle_table.setItem(i, 2, plate_item)
            
            # Violation Column (String) & Red Background
            violation_text = v['violation']
            violation_item = QTableWidgetItem(violation_text)
            self.vehicle_table.setItem(i, 3, violation_item)
            
            if violation_text:
                # Set soft translucent red background for entire row if violation exists
                red_color = QColor(239, 68, 68, 40)  # Soft Red with Alpha
                for col in range(4):
                    item = self.vehicle_table.item(i, col)
                    if item:
                        item.setBackground(red_color)
            
        self.vehicle_table.setSortingEnabled(True)  # Re-enable sorting
            
        # Draw helmet detections — only if NO_HELMET toggle is enabled
        if config.ENABLED_VIOLATIONS.get("NO_HELMET", True):
            for det in detections.get("helmets", []):
                if det.class_name.lower() == "helmet":
                    color = (0, 255, 0)  # Green for helmet
                else:
                    color = (0, 165, 255)  # Orange for no helmet
                label = f"{det.class_name}"
                llm_v = getattr(det, "llm_verified", False)
                frame = draw_detection(frame, det.box, label, color, llm_verified=llm_v)
        
        # Draw plate detections — only if MISSING_PLATE toggle is enabled
        if config.ENABLED_VIOLATIONS.get("MISSING_PLATE", True):
            for det in detections.get("plates", []):
                color = (255, 255, 0)  # Cyan for plates
                label = "PLATE"
                llm_v = getattr(det, "llm_verified", False)
                frame = draw_detection(frame, det.box, label, color, llm_verified=llm_v)
        
        # Draw violations and update stats — only for enabled violation types
        for violation in violations:
            # Skip drawing if this violation type is disabled
            if not config.ENABLED_VIOLATIONS.get(violation.type, True):
                continue
            llm_v = getattr(violation, "llm_verified", False)
            frame = draw_violation_alert(frame, violation.vehicle_box, violation.type, llm_verified=llm_v)
            self.add_violation_to_log(violation)
            
            # UNIQUE VIOLATION COUNTING
            v_key = (violation.track_id or 0, violation.type)
            if v_key not in self.seen_violations:
                self.seen_violations.add(v_key)
                self.violation_counts[violation.type] = self.violation_counts.get(violation.type, 0) + 1
        
        # Update stats
        self.update_stats()
        
        # Convert frame to Qt format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        scaled = qt_image.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(QPixmap.fromImage(scaled))
        
    def switch_brain(self, model_id):
        """Switch the AI model (Brain) used for LLM vision"""
        if self.detection_thread:
            if model_id == "local":
                # Local only - disable LLM provider
                self.detection_thread.llm_provider = None
                if config:
                    config.LLM_PROVIDER = "local"
                print(f"Switched to Local Processing Only")
            else:
                # LLM based - re-init to pick up potential new keys
                if config:
                    config.LLM_PROVIDER = model_id
                from src.llm import get_llm_provider
                self.detection_thread.llm_provider = get_llm_provider()
                print(f"Switched Brain to: {model_id}")

    def switch_device(self, device_key):
        """Switch the hardware device (CPU/GPU) used for detection"""
        if self.detection_thread:
            # Re-initialize models on the new device
            # This will also emit device_info which updates the status indicator
            self.detection_thread.initialize_models(device_type=device_key)
            print(f"Switched Vision Engine to: {device_key}")
    
    def update_device_status(self, device_name):
        """Update the device status label"""
        self.device_status.setToolTip(f"Active Device: {device_name}")
        if "GPU" in device_name:
            self.device_status.setText("🚀")
        else:
            self.device_status.setText("🟢")
    
    def add_violation_to_log(self, violation: Violation):
        """Add a violation to the log list"""
        timestamp = datetime.fromtimestamp(violation.timestamp).strftime("%H:%M:%S")
        text = f"[{timestamp}] {violation.type}"
        if violation.track_id:
            text += f" (ID: {violation.track_id})"
        
        # Show plate number if available
        if violation.plate_number:
            text += f" | Plate: {violation.plate_number}"
        
        # Show helmet status if relevant
        if violation.helmet_status and violation.type == "NO_HELMET":
            text += f" | {violation.helmet_status}"
        
        if violation.details:
            text += f" - {violation.details}"
            
        if violation.llm_reasoning:
            text += f" | AI: {violation.llm_reasoning}"
        
        item = QListWidgetItem(text)
        
        # Color code by severity
        if violation.type in ["WRONG_WAY", "OVERSPEED"]:
            item.setForeground(QColor("#F87171")) # Soft Red
        elif violation.type in ["NO_HELMET", "TRIPLE_RIDING", "EXCESS_RIDING"]:
            item.setForeground(QColor("#FACC15")) # Amber/Yellow
        else:
            item.setForeground(QColor("#E2E8F0")) # Ghost White
        
        self.violation_list.insertItem(0, item)
        
        # Limit log size
        while self.violation_list.count() > 100:
            self.violation_list.takeItem(self.violation_list.count() - 1)
    
    def update_stats(self):
        """Update the statistics display using KPI cards"""
        for key, card in self.stat_cards.items():
            count = self.violation_counts.get(key, 0)
            card.set_value(count)
    
    def clear_log(self):
        """Clear the violation log and reset stats"""
        self.violation_list.clear()
        self.seen_violations.clear() # Reset unique tracking
        for key in self.violation_counts:
            self.violation_counts[key] = 0
        self.update_stats()
        # Clear database for this session? No, just the UI view
        self.vehicle_table.setRowCount(0)
    
    def show_error(self, message: str):
        """Show an error message"""
        QMessageBox.critical(self, "Error", message)
    
    def closeEvent(self, event):
        """Handle window close"""
        self.stop_detection()
        if self.db:
            self.db.close()
        event.accept()
    
    # --- NEW: Stop-line & Signal ROI ---
    def set_stop_line(self):
        """Set the stop-line Y from the spinbox"""
        y = self.stop_line_spin.value()
        if y > 0:
            self.stop_line_y = y
            self.stopline_status.setText(f"Stop-line: Y = {y} px")
            if self.detection_thread:
                self.detection_thread.stop_line_y = y
        else:
            self.stop_line_y = None
            self.stopline_status.setText("Stop-line: Not set")

    
    def set_signal_roi(self):
        """Set the signal ROI from spinboxes"""
        x1 = self.roi_x1.value()
        y1 = self.roi_y1.value()
        x2 = self.roi_x2.value()
        y2 = self.roi_y2.value()
        if x2 > x1 and y2 > y1:
            self.signal_roi = [x1, y1, x2, y2]
            self.signal_status.setText(f"Signal ROI: ({x1},{y1}) to ({x2},{y2})")
            if self.detection_thread:
                self.detection_thread.signal_roi = self.signal_roi
        else:
            self.signal_roi = None
            self.signal_status.setText("Signal ROI: Invalid (x2 > x1, y2 > y1 required)")
    
    def on_ocr_changed(self, index):
        """Handle OCR engine change"""
        engine = self.ocr_combo.currentText().lower()
        if self.detection_thread:
            self.detection_thread.ocr_engine_type = engine
            # Re-init OCR in thread
            self.detection_thread.plate_ocr = get_ocr_engine(engine)

    
    # --- NEW: History Tab ---
    def on_rpm_changed(self, value):
        """Handle LLM RPM change"""
        config.LLM_MAX_RPM = value
        # Sync with settings.json
        try:
            settings_path = "settings.json"
            data = {}
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    data = json.load(f)
            data["LLM_MAX_RPM"] = value
            with open(settings_path, 'w') as f:
                json.dump(data, f, indent=4)
            # Update active thread for real-time response
            if self.detection_thread:
                self.detection_thread.llm_rate_limiter.max_rpm = value
        except: pass

    def refresh_history(self):
        """Load violation history from database into the history table"""
        if not self.db:
            QMessageBox.information(self, "Info", "Database not available.")
            return
        
        records = self.db.get_violations(limit=500)
        self.history_table.setSortingEnabled(False)
        self.history_table.setRowCount(len(records))
        
        for i, r in enumerate(records):
            self.history_table.setItem(i, 0, QTableWidgetItem(str(r.id)))
            self.history_table.setItem(i, 1, QTableWidgetItem(r.timestamp))
            vtype = r.violation_type.replace('_', ' ').title()
            self.history_table.setItem(i, 2, QTableWidgetItem(vtype))
            self.history_table.setItem(i, 3, QTableWidgetItem(str(r.track_id or '-')))
            self.history_table.setItem(i, 4, QTableWidgetItem(r.plate_number or '-'))
            self.history_table.setItem(i, 5, QTableWidgetItem(f"{r.confidence*100:.0f}%"))
            
            # Color rows by violation type using translucent dark shades
            color_map = {
                'no_helmet': QColor(239, 68, 68, 30),
                'triple_riding': QColor(245, 158, 11, 30),
                'excess_riding': QColor(245, 158, 11, 30),
                'phone_usage': QColor(139, 92, 246, 30),
                'red_signal': QColor(239, 68, 68, 40),
            }
            bg = color_map.get(r.violation_type, QColor(30, 41, 59, 100))
            for col in range(6):
                item = self.history_table.item(i, col)
                if item:
                    item.setBackground(bg)
        
        self.history_table.setSortingEnabled(True)
    
    # --- NEW: Export ---
    def export_csv(self):
        """Export violations to CSV"""
        if not self.db or not self.report_gen:
            QMessageBox.information(self, "Info", "Database/Report not available.")
            return
        
        records = self.db.get_violations()
        if not records:
            QMessageBox.information(self, "Info", "No violations to export.")
            return
        
        filepath = self.report_gen.generate_csv(records)
        QMessageBox.information(self, "CSV Exported", f"Saved to:\n{filepath}")
    
    def export_html(self):
        """Export violations to HTML report"""
        if not self.db or not self.report_gen:
            QMessageBox.information(self, "Info", "Database/Report not available.")
            return
        
        records = self.db.get_violations()
        stats = self.db.get_stats()
        if not records:
            QMessageBox.information(self, "Info", "No violations to export.")
            return
        
        filepath = self.report_gen.generate_html(records, stats)
        QMessageBox.information(self, "HTML Report Generated", f"Saved to:\n{filepath}")
        
        # Try to open in browser
        try:
            import webbrowser
            webbrowser.open(f"file:///{filepath}")
        except:
            pass


def run_app():
    """Run the application"""
    # Hide console on Windows for "Stealth" launch
    if os.name == 'nt':
        try:
            import ctypes
            # 0 = SW_HIDE
            kernel32 = ctypes.WinDLL('kernel32')
            user32 = ctypes.WinDLL('user32')
            hWnd = kernel32.GetConsoleWindow()
            if hWnd:
                user32.ShowWindow(hWnd, 0)
        except:
            pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
