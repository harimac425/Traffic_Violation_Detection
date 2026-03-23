"""
YOLO11 Object Detection Wrapper with Multi-Model Support

Supports:
- Main model (COCO classes: person, vehicle, phone)
- Helmet detection model (helmet, no_helmet)
- License plate detection model (plate)
"""
from ultralytics import YOLO
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import config
from src.utils import non_max_suppression, calculate_iou, calculate_containment


@dataclass
class Detection:
    """Represents a single detection"""
    box: List[float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    source_model: str = "main"  # Which model detected this


class MultiModelDetector:
    """
    Multi-model detector that combines:
    - Main YOLO model for general detection
    - Helmet detection model (optional)
    - License plate detection model (optional)
    """
    
    def __init__(self, device_type: str = "auto"):
        """
        Initialize all available models.
        
        Args:
            device_type: "cpu", "cuda", or "auto"
        """
        config.print_config_status()
        
        # Get device
        self.device, self.device_name = config.get_device(device_type)
        print(f"Using device: {self.device_name}")
        
        # --- SMART DETECTION ENHANCEMENTS ---
        self.prev_frame = None
        self.roi_cache = None
        self.roi_persistence = 0 # How many frames to keep the same ROI
        self.enable_dynamic_roi = getattr(config, 'ENABLE_DYNAMIC_ROI', False)
        
        # Load main model
        print(f"Loading main model: {config.MODEL_PATH}")
        self.main_model = YOLO(config.MODEL_PATH)
        self.main_model.to(self.device)
        self.main_class_names = self.main_model.names
        print(f"  Main model classes: {len(self.main_class_names)}")
        
        # Load helmet model if available
        self.helmet_model = None
        self.helmet_class_names = {}
        if config.USE_HELMET_MODEL:
            print(f"Loading helmet model: {config.HELMET_MODEL_PATH}")
            try:
                self.helmet_model = YOLO(str(config.HELMET_MODEL_PATH))
                self.helmet_model.to(self.device)
                self.helmet_class_names = self.helmet_model.names
                print(f"  Helmet model classes: {self.helmet_class_names}")
            except Exception as e:
                print(f"  ✗ Failed to load helmet model: {e}")
        else:
            print("  Helmet model not found - helmet detection disabled")
        
        # Load plate model if available
        self.plate_model = None
        self.plate_class_names = {}
        if config.USE_PLATE_MODEL:
            print(f"Loading plate model: {config.PLATE_MODEL_PATH}")
            try:
                self.plate_model = YOLO(str(config.PLATE_MODEL_PATH))
                self.plate_model.to(self.device)
                self.plate_class_names = self.plate_model.names
                print(f"  Plate model classes: {self.plate_class_names}")
            except Exception as e:
                print(f"  ✗ Failed to load plate model: {e}")
        else:
            print("  Plate model not found - plate detection disabled")
        
        print("\nModels loaded successfully!")
    
    def detect(self, frame: np.ndarray, track: bool = True) -> Dict[str, List[Detection]]:
        """
        Run detection on a frame using all available models.
        
        Args:
            frame: BGR image as numpy array
            track: Whether to use object tracking
        
        Returns:
            Dictionary with keys: 'main', 'helmets', 'plates'
            Each contains a list of Detection objects
        """
        results = {
            "main": [],
            "helmets": [],
            "plates": []
        }
        
        # Apply CLAHE if enabled (for CCTV enhancement)
        full_enhanced_frame = frame
        if getattr(config, 'ENABLE_CLAHE', False):
            full_enhanced_frame = self._apply_clahe(frame)
        
        inference_frame = full_enhanced_frame
        
        # --- DYNAMIC ROI OPTIMIZATION ---
        inference_box = None
        current_imgsz = 960 # Default base resolution
        
        if self.enable_dynamic_roi:
            inference_box = self._get_dynamic_roi(frame)
            if inference_box:
                x1, y1, x2, y2 = inference_box
                roi_w = x2 - x1
                h, w = frame.shape[:2]
                
                # If ROI is small (distant/concentrated motion), use higher imgsz for detail
                if roi_w < w * 0.4:
                    current_imgsz = 1600 # Super-resolution for small crops
                elif roi_w < w * 0.6:
                    current_imgsz = 1280 
                
                inference_frame = inference_frame[y1:y2, x1:x2]
        
        # --- PRIMARY DETECTION ---
        # Run main model with tracking and adaptive resolution
        main_detections = self._run_model(
            self.main_model,
            inference_frame,
            track=track,
            source="main",
            imgsz=current_imgsz
        )
        
        # Adjust detection boxes back to full frame if ROI was used
        if inference_box:
            ox, oy = inference_box[0], inference_box[1]
            for det in main_detections:
                det.box[0] += ox; det.box[1] += oy
                det.box[2] += ox; det.box[3] += oy
        
        # --- GLOBAL NMS ON MAIN DETECTIONS ---
        if main_detections:
            # Separate persons to use a more relaxed NMS (allow more overlap on bikes)
            persons = [d for d in main_detections if d.class_name == "person"]
            others = [d for d in main_detections if d.class_name != "person"]
            
            # 1. NMS for Others (Cars, Motos, etc.)
            if others:
                o_boxes = np.array([d.box for d in others])
                o_scores = np.array([d.confidence for d in others])
                o_keep = non_max_suppression(o_boxes, o_scores, iou_threshold=0.45)
                others = [others[i] for i in o_keep]
            
            # 2. NMS for Persons (Relaxed to allow riders to overlap)
            if persons:
                p_boxes = np.array([d.box for d in persons])
                p_scores = np.array([d.confidence for d in persons])
                # Increase IOU threshold to 0.85 to allow riders to stay distinct even when packed
                p_keep = non_max_suppression(p_boxes, p_scores, iou_threshold=0.85)
                persons = [persons[i] for i in p_keep]
            
            main_detections = others + persons
            results["main"] = main_detections

        # --- Cascade Detection for Helmets ---
        if self.helmet_model is not None:
            persons = [d for d in main_detections if d.class_name == "person"]
            motorcycles = [d for d in main_detections if d.class_name == "motorcycle"]
            
            if persons:
                for person in persons:
                    # --- HIGH PRECISION RIDER IDENTIFICATION ---
                    # Only consider person a rider if they overlap significantly with a motorcycle
                    is_rider = any(self._boxes_overlap(person.box, moto.box, threshold=0.15) for moto in motorcycles)
                    
                    # Also check if the person box is "contained" or sitting on top of the motorcycle
                    # (Precision enhancement from aneesarom logic)
                    if not is_rider:
                        # Fallback: Check if person center is near motorcycle upper region
                        px_mid = (person.box[0] + person.box[2]) / 2
                        py_mid = (person.box[1] + person.box[3]) / 2
                        for moto in motorcycles:
                            # If person is in the top-half vicinity of motorcycle center
                            moto_w = moto.box[2] - moto.box[0]
                            dist_x = abs(px_mid - (moto.box[0] + moto.box[2])/2)
                            dist_y = abs(person.box[3] - moto.box[1]) # feet near moto top
                            if dist_x < moto_w * 0.4 and dist_y < moto_w * 0.3:
                                is_rider = True
                                break

                    if not is_rider: continue

                    x1, y1, x2, y2 = map(int, person.box)
                    h, w = frame.shape[:2]
                    margin_y = int((y2 - y1) * 0.2); margin_x = int((x2 - x1) * 0.2)
                    x1 = max(0, x1 - margin_x); y1 = max(0, y1 - margin_y)
                    x2 = min(w, x2 + margin_x); y2 = min(h, y2 + margin_y)
                    
                    crop = full_enhanced_frame[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    crop_detections = self._run_model(
                        self.helmet_model, crop, track=False, source="helmet",
                        confidence=config.HELMET_CONFIDENCE
                    )
                    
                    for d in crop_detections:
                        dx1, dy1, dx2, dy2 = d.box
                        absolute_box = [x1 + dx1, y1 + dy1, x1 + dx2, y1 + dy2]
                        
                        # --- SPATIAL CONTAINMENT CHECK ---
                        # Ensure helmet is actually in the upper region of the person
                        # (This is the "aneesarom" high-precision containment rule)
                        containment = calculate_containment(absolute_box, person.box)
                        if containment < 0.6: continue # Must be at least 60% inside person box
                        
                        d.box = absolute_box
                        # Clean label for UI (Robust string matching)
                        c_name = d.class_name.lower().strip()
                        is_helmet = ("helmet" in c_name and "without" not in c_name and "no" not in c_name)
                        d.class_name = "Helmet" if is_helmet else "No Helmet"
                        d.track_id = person.track_id # Associate with person's track ID
                        results["helmets"].append(d)

        # --- Plate Detection (Cascade on Vehicle Crops) ---
        if self.plate_model is not None:
            vehicles = [d for d in main_detections if d.class_name in ["car", "motorcycle", "bus", "truck"]]
            
            for vehicle in vehicles:
                x1, y1, x2, y2 = map(int, vehicle.box)
                h, w = frame.shape[:2]
                margin = 20
                x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)
                
                crop = full_enhanced_frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                crop_plates = self._run_model(
                    self.plate_model, crop, track=False, source="plate",
                    confidence=config.PLATE_CONFIDENCE
                )
                
                if crop_plates:
                    best_plate = max(crop_plates, key=lambda x: x.confidence)
                    dx1, dy1, dx2, dy2 = best_plate.box
                    absolute_plate_box = [x1 + dx1, y1 + dy1, x1 + dx2, y1 + dy2]
                    
                    # --- SPATIAL CONTAINMENT CHECK (Plates) ---
                    # Ensure plate is contained within or adjacent to the specific vehicle
                    plate_containment = calculate_containment(absolute_plate_box, vehicle.box)
                    if plate_containment < 0.5: continue # Ignore plates that wander out of vehicle crop ROI
                    
                    best_plate.box = absolute_plate_box
                    best_plate.class_name = "Number Plate"
                    results["plates"].append(best_plate)

        # --- GLOBAL NMS (Deduplicate overlapping cascade detections) ---
        for key in ["helmets", "plates"]:
            if not results[key]: continue
            boxes = np.array([d.box for d in results[key]])
            scores = np.array([d.confidence for d in results[key]])
            keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.25)
            results[key] = [results[key][i] for i in keep_indices]
        
        return results
    
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance contrast for CCTV footage"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(
            clipLimit=getattr(config, 'CLAHE_CLIP_LIMIT', 2.0),
            tileGridSize=getattr(config, 'CLAHE_GRID_SIZE', (8, 8))
        )
        cl = clahe.apply(l)
        
        # Merge and convert back
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final
    
    def _run_model(
        self,
        model: YOLO,
        frame: np.ndarray,
        track: bool = False,
        source: str = "main",
        confidence: float = None,
        imgsz: int = 1280
    ) -> List[Detection]:
        """Run a single model on the frame"""
        conf = confidence or config.CONFIDENCE_THRESHOLD
        
        # Use a lower global threshold to catch more candidates, then filter strictly
        global_conf = min(conf, 0.25)
        
        try:
            if track:
                model_results = model.track(
                    frame,
                    conf=global_conf,
                    iou=config.IOU_THRESHOLD,
                    persist=True,
                    verbose=False,
                    imgsz=imgsz
                )
            else:
                model_results = model(
                    frame,
                    conf=global_conf,
                    iou=config.IOU_THRESHOLD,
                    verbose=False,
                    imgsz=imgsz
                )
        except Exception as e:
            print(f"Detection error ({source}): {e}")
            return []
        
        detections = []
        
        for result in model_results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy().tolist()
                class_id = int(boxes.cls[i].cpu().numpy())
                conf_score = float(boxes.conf[i].cpu().numpy())
                
                # Get track ID if available
                track_id = None
                if track and boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                
                # Get class name
                class_name = model.names.get(class_id, f"class_{class_id}")
                
                # Normalize helmet class names
                if source == "helmet":
                    class_name = self._normalize_helmet_class(class_name)
                
                # Normalize plate class names
                if source == "plate":
                    class_name = self._normalize_plate_class(class_name)

                # --- PER-CLASS CONFIDENCE FILTERING (ByteTrack Support) ---
                # We return everything >= 0.1 so the tracker can perform secondary association.
                # However, we still use the 'target_conf' logic for logging/filtering IF needed.
                if conf_score < 0.1:
                    continue
                
                detections.append(Detection(
                    box=box,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf_score,
                    track_id=track_id,
                    source_model=source
                ))
        
        return detections
    
    def _normalize_helmet_class(self, class_name: str) -> str:
        """Normalize helmet class names to standard format"""
        class_lower = class_name.lower()
        
        # Debug print for rare/new classes
        # print(f"DEBUG: Normalizing class '{class_name}'")
        
        for standard_name, variants in config.HELMET_CLASSES.items():
            for variant in variants:
                if variant.lower() in class_lower:
                    # print(f"  -> Match found: {standard_name} (variant: {variant})")
                    return standard_name
        
        print(f"WARNING: Unknown helmet class '{class_name}' - please add to attributes in config.py")
        return class_name
    
    def _normalize_plate_class(self, class_name: str) -> str:
        """Normalize plate class names to 'license_plate'"""
        class_lower = class_name.lower()
        
        for variant in config.PLATE_CLASSES:
            if variant.lower() in class_lower:
                return "license_plate"
        
        return class_name
    
    # Convenience methods
    def get_persons(self, detections: Dict[str, List[Detection]]) -> List[Detection]:
        """Get all person detections from main model"""
        return [d for d in detections["main"] if d.class_name == "person"]
    
    def get_vehicles(self, detections: Dict[str, List[Detection]]) -> List[Detection]:
        """Get all vehicle detections"""
        vehicle_classes = ["motorcycle", "car", "bus", "truck", "bicycle"]
        return [d for d in detections["main"] if d.class_name in vehicle_classes]
    
    def get_motorcycles(self, detections: Dict[str, List[Detection]]) -> List[Detection]:
        """Get motorcycle detections"""
        return [d for d in detections["main"] if d.class_name == "motorcycle"]
    
    def get_phones(self, detections: Dict[str, List[Detection]]) -> List[Detection]:
        """Get cell phone detections"""
        return [d for d in detections["main"] if d.class_name == "cell phone"]
    
    def get_helmets(self, detections: Dict[str, List[Detection]]) -> List[Detection]:
        """Get helmet detections (wearing helmet)"""
        return [d for d in detections["helmets"] if d.class_name == "helmet"]
    
    def get_no_helmets(self, detections: Dict[str, List[Detection]]) -> List[Detection]:
        """Get no-helmet detections (head without helmet)"""
        return [d for d in detections["helmets"] if d.class_name == "no_helmet"]
    
    def get_plates(self, detections: Dict[str, List[Detection]]) -> List[Detection]:
        """Get license plate detections"""
        return detections["plates"]
    
    def has_helmet_model(self) -> bool:
        """Check if helmet model is loaded"""
        return self.helmet_model is not None
    
    def has_plate_model(self) -> bool:
        """Check if plate model is loaded"""
        return self.plate_model is not None
    
    # === Association Methods (based on reference repos) ===
    
    def associate_helmet_to_rider(
        self, 
        detections: Dict[str, List[Detection]],
        iou_threshold: float = 0.3
    ) -> Dict[int, str]:
        """
        Associate helmet status with motorcycle riders.
        
        Based on: github.com/aneesarom/Real-Time-Detection-of-Helmet-Violations
        
        Returns:
            Dict mapping track_id -> helmet_status ("helmet", "no_helmet", "unknown")
        """
        from src.utils import calculate_iou
        
        rider_helmet_status = {}
        motorcycles = self.get_motorcycles(detections)
        persons = self.get_persons(detections)
        helmets = self.get_helmets(detections)
        no_helmets = self.get_no_helmets(detections)
        
        # For each motorcycle, find persons on it
        for moto in motorcycles:
            riders_on_moto = []
            
            for person in persons:
                # Check if person overlaps with motorcycle (riding it)
                iou = calculate_iou(person.box, moto.box)
                if iou > iou_threshold:
                    riders_on_moto.append(person)
            
            # For each rider, check helmet status
            for rider in riders_on_moto:
                rider_id = rider.track_id if rider.track_id else id(rider)
                helmet_status = "unknown"
                
                # Check for helmet near rider's upper body
                rider_head_box = self._get_head_region(rider.box)
                
                # Check for helmet
                for helmet in helmets:
                    if self._boxes_overlap(helmet.box, rider_head_box, threshold=0.2):
                        helmet_status = "helmet"
                        break
                
                # Check for no helmet
                if helmet_status == "unknown":
                    for no_helmet in no_helmets:
                        if self._boxes_overlap(no_helmet.box, rider_head_box, threshold=0.2):
                            helmet_status = "no_helmet"
                            break
                
                rider_helmet_status[rider_id] = helmet_status
        
        return rider_helmet_status
    
    def associate_plate_to_vehicle(
        self,
        detections: Dict[str, List[Detection]],
        frame: np.ndarray = None
    ) -> Dict[int, Detection]:
        """
        Associate license plates with vehicles.
        
        Based on: github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8
        
        Returns:
            Dict mapping vehicle track_id -> plate Detection
        """
        vehicle_plates = {}
        vehicles = self.get_vehicles(detections)
        plates = self.get_plates(detections)
        
        for vehicle in vehicles:
            vehicle_id = vehicle.track_id if vehicle.track_id else id(vehicle)
            
            # Find plate within/near vehicle bounding box
            for plate in plates:
                if self._is_plate_in_vehicle(plate.box, vehicle.box):
                    vehicle_plates[vehicle_id] = plate
                    break
        
        return vehicle_plates
    
    def _get_head_region(self, person_box: tuple) -> tuple:
        """Get approximate head region from person bounding box"""
        x1, y1, x2, y2 = person_box
        height = y2 - y1
        # Head is approximately top 25% of person box
        head_y2 = y1 + height * 0.25
        return (x1, y1, x2, head_y2)
    
    def _boxes_overlap(self, box1: tuple, box2: tuple, threshold: float = 0.1) -> bool:
        """Check if two boxes overlap above threshold"""
        from src.utils import calculate_iou
        return calculate_iou(box1, box2) > threshold
    
    def _is_plate_in_vehicle(self, plate_box: tuple, vehicle_box: tuple) -> bool:
        """Check if plate is within or near vehicle bounding box"""
        px1, py1, px2, py2 = plate_box
        vx1, vy1, vx2, vy2 = vehicle_box
        
        # Plate center
        plate_cx = (px1 + px2) / 2
        plate_cy = (py1 + py2) / 2
        
        # Expand vehicle box slightly
        margin = 20
        return (vx1 - margin <= plate_cx <= vx2 + margin and 
                vy1 - margin <= plate_cy <= vy2 + margin)

    def _get_dynamic_roi(self, frame: np.ndarray) -> Optional[List[int]]:
        """
        Calculate a dynamic ROI based on motion to focus YOLO inference.
        
        Returns:
            [x1, y1, x2, y2] of the motion-active area, or None if whole frame.
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            return None
        
        # 1. Prepare low-res grayscale frames for fast differencing
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        
        # 2. Calculate frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray
        
        # 3. Threshold and dilate to get motion blobs
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # 4. Find bounding box of all motion
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 10: # Very little motion
            self.roi_persistence -= 1
            return self.roi_cache if self.roi_persistence > 0 else None
            
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        
        # 5. Scale back to original resolution and add padding
        h, w = frame.shape[:2]
        fx, fy = w / 320, h / 180
        
        # Add 15% margin for context
        pad_x, pad_y = int(w * 0.15), int(h * 0.15)
        
        nx1 = max(0, int(x1 * fx) - pad_x)
        ny1 = max(0, int(y1 * fy) - pad_y)
        nx2 = min(w, int(x2 * fx) + pad_x)
        ny2 = min(h, int(y2 * fy) + pad_y)
        
        # Limit ROI min size to avoid tiny crops that confuse YOLO
        if (nx2 - nx1) < w * 0.4 or (ny2 - ny1) < h * 0.4:
            self.roi_persistence -= 1
            return self.roi_cache if self.roi_persistence > 0 else None
            
        # 6. Smooth the ROI (Persistence)
        current_roi = [nx1, ny1, nx2, ny2]
        self.roi_cache = current_roi
        self.roi_persistence = 10 # Keep for 10 frames even if motion stops/slows
        
        return current_roi


# Backward compatibility alias
class Detector(MultiModelDetector):
    """Alias for MultiModelDetector for backward compatibility"""
    pass

