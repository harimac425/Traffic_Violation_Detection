
import sys
import os
import time
sys.path.append(os.getcwd())

from src.detector import Detection
from src.violations import ViolationDetector, Violation

def test_single_rider_no_helmet():
    print("Testing single rider with NO helmet...")
    
    # Mock Detection objects
    # 1. Motorcycle
    moto = Detection(
        box=(100, 100, 300, 300),
        confidence=0.9,
        class_id=0,
        class_name="motorcycle",
        track_id=1,
        source_model="main"
    )
    
    # 2. Rider (Person) on the motorcycle
    # Overlapping with motorcycle
    rider = Detection(
        box=(150, 80, 250, 250), # Center of moto
        confidence=0.9,
        class_id=1,
        class_name="person",
        track_id=2,
        source_model="main"
    )
    
    # 3. Helmets list is EMPTY (this was the failure case)
    helmets = []
    
    # 4. No Helmets list (direct detection) - let's say model missed the "no_helmet" class direct detection
    # but detected the person. 
    # Or let's test BOTH cases.
    
    detector = ViolationDetector()
    
    # Case A: Model detects 'no_helmet' class
    print("\nCase A: Model detects 'no_helmet' class directly")
    no_helmet_det = Detection(
        box=(160, 80, 240, 120), # Head region
        confidence=0.85, 
        class_id=2,
        class_name="no_helmet",
        source_model="helmet"
    )
    
    violations_a = detector.check_no_helmet_with_model(
        persons=[rider],
        motorcycles=[moto],
        helmets=[],
        no_helmets=[no_helmet_det]
    )
    
    print(f"Violations found: {len(violations_a)}")
    if len(violations_a) > 0:
        print(f"SUCCESS: Detected {violations_a[0].type}: {violations_a[0].details}")
    else:
        print("FAILURE: Did not detect violation")

    # Case B: Model detects rider, NO helmet detection, NO no_helmet detection (just missing helmet)
    # This relies on the 'head region' logic
    print("\nCase B: Indirect detection (Rider present, no helmet objects detected at all)")
    
    # Reset cooldown
    detector.violation_cooldown = {}
    
    violations_b = detector.check_no_helmet_with_model(
        persons=[rider],
        motorcycles=[moto],
        helmets=[], # Empty
        no_helmets=[] # Empty
    )
    
    print(f"Violations found: {len(violations_b)}")
    if len(violations_b) > 0:
        print(f"SUCCESS: Detected {violations_b[0].type}: {violations_b[0].details}")
    else:
        print("FAILURE: Did not detect violation (This was the bug)")

if __name__ == "__main__":
    test_single_rider_no_helmet()
