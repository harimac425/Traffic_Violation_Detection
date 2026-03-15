"""
Geometry, visualization, and rate-limiting utility functions
"""
import time
import cv2
import numpy as np
from typing import List, Tuple, Optional


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_containment(box_child: List[float], box_parent: List[float]) -> float:
    """
    Calculate what percentage of the child box is contained within the parent box.
    Useful for ensuring sub-detections (helmets, plates) are actually on the person/vehicle.
    """
    x1 = max(box_child[0], box_parent[0])
    y1 = max(box_child[1], box_parent[1])
    x2 = min(box_child[2], box_parent[2])
    y2 = min(box_child[3], box_parent[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    child_area = (box_child[2] - box_child[0]) * (box_child[3] - box_child[1])
    
    return intersection_area / child_area if child_area > 0 else 0


def box_contains_point(box: List[float], point: Tuple[float, float]) -> bool:
    """Check if a point is inside a bounding box"""
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def get_box_center(box: List[float]) -> Tuple[float, float]:
    """Get the center point of a bounding box"""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def calculate_speed(
    pos1: Tuple[float, float],
    pos2: Tuple[float, float],
    time_delta: float,
    pixels_per_meter: float
) -> float:
    """
    Calculate speed in km/h given two positions and time delta.
    
    Args:
        pos1: Previous position (x, y) in pixels
        pos2: Current position (x, y) in pixels
        time_delta: Time between positions in seconds
        pixels_per_meter: Calibration factor
    
    Returns:
        Speed in km/h
    """
    if time_delta <= 0:
        return 0
    
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    distance_pixels = np.sqrt(dx**2 + dy**2)
    distance_meters = distance_pixels / pixels_per_meter
    speed_mps = distance_meters / time_delta
    speed_kmh = speed_mps * 3.6
    
    return speed_kmh


def calculate_direction(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """
    Calculate movement direction in degrees.
    0 = right, 90 = down, 180 = left, 270 = up
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 360
    
    return angle


def boxes_overlap(box1: List[float], box2: List[float], threshold: float = 0.3) -> bool:
    """Check if two boxes overlap with IoU above threshold"""
    return calculate_iou(box1, box2) > threshold


def get_upper_region(box: List[float], ratio: float = 0.4) -> List[float]:
    """
    Get the upper region of a bounding box (for head/helmet detection).
    
    Args:
        box: Bounding box [x1, y1, x2, y2]
        ratio: How much of the top to consider (0.4 = top 40%)
    
    Returns:
        Upper region bounding box
    """
    height = box[3] - box[1]
    return [box[0], box[1], box[2], box[1] + height * ratio]


def draw_detection(
    frame: np.ndarray,
    box: List[float],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    llm_verified: bool = False
) -> np.ndarray:
    """Draw a detection box with label on the frame"""
    if llm_verified:
        color = (255, 255, 0) # Cyan (BGR)
        label = f"✨ {label}"
        thickness += 1
    
    x1, y1, x2, y2 = map(int, box)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(
        frame,
        (x1, y1 - label_size[1] - 10),
        (x1 + label_size[0], y1),
        color,
        -1
    )
    
    # Label text
    cv2.putText(
        frame,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )
    
    return frame


def draw_violation_alert(
    frame: np.ndarray,
    box: List[float],
    violation_type: str,
    llm_verified: bool = False
) -> np.ndarray:
    """Draw a violation alert with red styling"""
    return draw_detection(frame, box, f"⚠ {violation_type}", color=(0, 0, 255), thickness=3, llm_verified=llm_verified)


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.
    
    Args:
        boxes: Numpy array of shape (N, 4) in [x1, y1, x2, y2]
        scores: Numpy array of shape (N,)
        iou_threshold: Overlap threshold
        
    Returns:
        Indices of the boxes to keep
    """
    if len(boxes) == 0:
        return []

    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by scores in descending order
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
            
        # Find overlap region
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # Correct intersection calculation
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        # Calculate IoU
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Find boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep


class RateLimiter:
    """
    Simple rate limiter to manage API requests per minute (RPM).
    Uses a simple timestamp history to enforce limits.
    """
    def __init__(self, max_rpm: int = 15):
        self.max_rpm = max_rpm
        self.request_times = []
        
    def can_request(self) -> bool:
        """Check if a request can be made based on RPM"""
        if self.max_rpm <= 0:
            return True
            
        current_time = time.time()
        # Remove timestamps older than 60 seconds
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        if len(self.request_times) < self.max_rpm:
            self.request_times.append(current_time)
            return True
            
        return False
    
    def wait_time(self) -> float:
        """Get approximate wait time in seconds if throttled"""
        if not self.request_times or len(self.request_times) < self.max_rpm:
            return 0
        
        # Time until the oldest request in the window expires
        oldest = self.request_times[0]
        return max(0, 60 - (time.time() - oldest))
