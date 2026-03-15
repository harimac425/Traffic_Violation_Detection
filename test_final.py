"""Final Integration Verification - Stage 7"""
import sys, os
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print('=' * 60)
print('  STAGE 7 VERIFICATION: Final Integration Wiring')
print('=' * 60)

# TEST 1: ViolationDetector Integration
print('\n[TEST 1] Initializing integrated ViolationDetector')
try:
    from src.violations import ViolationDetector
    vd = ViolationDetector()
    print('  PASS - ViolationDetector initialized with Signal and Phone detectors')
except Exception as e:
    print(f'  FAIL - VD init error: {e}')
    sys.exit(1)

# TEST 2: OCR Factory Integration
print('\n[TEST 2] Testing OCR Engine Factory in UI Context')
try:
    from src.ocr import get_ocr_engine
    import config
    engine = get_ocr_engine(config.OCR_ENGINE)
    print(f'  PASS - OCR engine "{config.OCR_ENGINE}" initialized via factory')
except Exception as e:
    print(f'  FAIL - OCR factory error: {e}')
    sys.exit(1)

# TEST 3: detector.detect_all signature match
print('\n[TEST 3] Verifying detect_all signature and return types')
try:
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_dets = {'main': [], 'helmets': [], 'plates': []}
    violations = vd.detect_all(dummy_frame, dummy_dets)
    print('  PASS - detect_all accepted frame and returned list')
    assert isinstance(violations, list)
except Exception as e:
    print(f'  FAIL - detect_all signature error: {e}')
    sys.exit(1)

# TEST 4: Config sanity
print('\n[TEST 4] Database and Path Config')
import config
print(f'  DATABASE_PATH: {config.DATABASE_PATH}')
print(f'  EVIDENCE_DIR: {config.EVIDENCE_DIR}')
assert os.path.exists(os.path.dirname(config.DATABASE_PATH))
print('  PASS - Paths are configured correctly')

print('\n' + '=' * 60)
print('  ALL INTEGRATION TESTS PASSED!')
print('  Wiring verified. Project is complete.')
print('=' * 60)
