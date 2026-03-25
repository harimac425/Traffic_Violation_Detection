"""
License Plate OCR Module

Supports two OCR backends:
1. EasyOCR (default) — fast, lightweight
2. TrOCR (optional) — Microsoft Transformer-based OCR for higher accuracy on printed text

Switch between engines using config.OCR_ENGINE = "easyocr" or "trocr"

Based on: 
- github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8
- PPT Module 6: "Extract plate number (TrOCR)"
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple
import re


class PlateOCR:
    """License plate text extraction using EasyOCR with preprocessing"""
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = True):
        """
        Initialize OCR reader.
        
        Args:
            languages: List of language codes. Default: ['en']
            use_gpu: Whether to use GPU for OCR
        """
        import easyocr
        languages = languages or ['en']
        print("Initializing EasyOCR engine (this may take a moment)...")
        try:
            self.reader = easyocr.Reader(languages, gpu=use_gpu)
        except:
            print("GPU not available for OCR, using CPU")
            self.reader = easyocr.Reader(languages, gpu=False)
        print("EasyOCR engine ready.")
        
        # Character mapping for common OCR mistakes
        self.char_map = {
            'O': '0', 'I': '1', 'J': '1', 'A': '4', 'G': '6',
            'S': '5', 'B': '8', 'Z': '2', 'Q': '0'
        }
    
    def read_plate(
        self, 
        plate_image: np.ndarray,
        preprocess: bool = True
    ) -> Tuple[Optional[str], float]:
        """
        Extract text from a license plate image.
        """
        if plate_image is None or plate_image.size == 0:
            return None, 0.0
        
        try:
            # Preprocess image for better OCR
            if preprocess:
                processed = self._preprocess_plate(plate_image)
            else:
                processed = plate_image
            
            # Run OCR
            results = self.reader.readtext(processed)
            
            if not results:
                return None, 0.0
            
            # Extract text and confidence
            texts = []
            confidences = []
            for bbox, text, conf in results:
                texts.append(text)
                confidences.append(conf)
            
            combined = "".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Clean up and Format
            cleaned = self._clean_plate_text(combined)
            formatted = self.format_indian_plate(cleaned)
            
            return formatted if formatted else None, avg_confidence
            
        except Exception as e:
            print(f"OCR error: {e}")
            return None, 0.0
    
    def _preprocess_plate(self, image: np.ndarray) -> np.ndarray:
        """
        Refined preprocessing for license plates with adaptive scaling.
        """
        if image is None or image.size == 0:
            return image
            
        # 1. Adaptive Scaling: Ensure enough pixels for OCR
        height, width = image.shape[:2]
        if width < 300: # Increased from 200 for better detail
            scale = 300 / width
            # Use INTER_CUBIC for sharper edges on upscaling
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        elif width > 800: # Scale down if too large
            scale = 800 / width
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # 2. Convert to Gray and Enhance Contrast (Tuned CLAHE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Increase clipLimit for more aggressive edge definition
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3. Subtle Denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=3, templateWindowSize=7, searchWindowSize=21)
        
        return denoised
    
    def _clean_plate_text(self, text: str) -> str:
        """Clean and normalize plate text"""
        # Convert to uppercase
        text = text.upper()
        
        # Remove special characters except alphanumeric
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        return text
    
    def format_indian_plate(self, text: str) -> str:
        """
        Format text as Indian license plate.
        Format: SS DD AA DDDD (State, District, Series, Number)
        Example: KA 01 AB 1234
        """
        if not text or len(text) < 9:
            return text
        
        # Remove spaces
        text = text.replace(" ", "")
        
        # Apply character corrections based on position
        corrected = ""
        for i, char in enumerate(text):
            if i < 2:  # State code - letters
                if char in '01':
                    corrected += 'O' if char == '0' else 'I'
                else:
                    corrected += char
            elif i < 4:  # District - numbers
                if char in self.char_map and char.isalpha():
                    corrected += self.char_map[char]
                else:
                    corrected += char
            elif i < 6:  # Series - letters
                corrected += char
            else:  # Number - digits
                if char in self.char_map and char.isalpha():
                    corrected += self.char_map[char]
                else:
                    corrected += char
        
        # Format with spaces
        if len(corrected) >= 10:
            return f"{corrected[:2]} {corrected[2:4]} {corrected[4:6]} {corrected[6:10]}"
        
        return corrected
    
    def validate_indian_plate(self, text: str) -> bool:
        """
        Validate if text matches Indian license plate format.
        Format: SS-DD-AA-DDDD (State, District, Series, Number)
        Example: KA01AB1234
        """
        if not text:
            return False
        
        # Remove spaces for pattern matching
        text = text.replace(" ", "")
        
        # Indian plate patterns
        patterns = [
            r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',  # Standard: KA01AB1234
            r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,4}$',  # Flexible: KA01A1234
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def crop_plate_from_frame(
        self, 
        frame: np.ndarray, 
        box: Tuple[float, float, float, float],
        padding: int = 5
    ) -> np.ndarray:
        """
        Crop license plate region from frame.
        
        Args:
            frame: Full video frame
            box: Bounding box (x1, y1, x2, y2)
            padding: Extra pixels around the box
        
        Returns:
            Cropped plate image
        """
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        
        # Add padding and clamp to image bounds
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return frame[y1:y2, x1:x2].copy()


class TrOCRPlateReader:
    """
    License plate text extraction using Microsoft TrOCR (Transformer-based OCR).
    
    Uses the pre-trained 'microsoft/trocr-base-printed' model for reading 
    printed text — well-suited for license plates.
    
    The model auto-downloads on first use (~500MB, cached after that).
    """
    
    def __init__(self, model_name: str = "microsoft/trocr-base-printed", use_gpu: bool = True):
        """
        Initialize TrOCR reader.
        
        Args:
            model_name: HuggingFace model ID for TrOCR
            use_gpu: Whether to use GPU for inference
        """
        print("Initializing TrOCR engine (first run will download the model)...")
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from PIL import Image
            import torch
            
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            self.device = "cpu"
            if use_gpu and torch.cuda.is_available():
                self.device = "cuda"
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._Image = Image  # Store PIL reference
            self._torch = torch
            self._available = True
            print(f"TrOCR engine ready (device: {self.device}).")
            
        except Exception as e:
            print(f"TrOCR initialization failed: {e}")
            print("Falling back to EasyOCR if available.")
            self._available = False
        
        # Character mapping for common OCR mistakes
        self.char_map = {
            'O': '0', 'I': '1', 'J': '1', 'A': '4', 'G': '6',
            'S': '5', 'B': '8', 'Z': '2', 'Q': '0'
        }
    
    @property
    def is_available(self) -> bool:
        """Check if TrOCR loaded successfully"""
        return self._available
    
    def read_plate(
        self,
        plate_image: np.ndarray,
        preprocess: bool = True
    ) -> Tuple[Optional[str], float]:
        """
        Extract text from a license plate image using TrOCR.
        """
        if not self._available:
            return None, 0.0
        
        if plate_image is None or plate_image.size == 0:
            return None, 0.0
        
        try:
            # Preprocess
            if preprocess:
                processed = self._preprocess_plate(plate_image)
            else:
                processed = plate_image
            
            # Convert BGR to RGB PIL Image
            if len(processed.shape) == 2:  # Grayscale
                rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            pil_image = self._Image.fromarray(rgb)
            
            # Run TrOCR
            pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            with self._torch.no_grad():
                generated = self.model.generate(
                    pixel_values,
                    max_length=20,
                    num_beams=4,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode text
            text = self.processor.batch_decode(
                generated.sequences, skip_special_tokens=True
            )[0]
            
            # Estimate confidence from sequence scores
            if hasattr(generated, 'sequences_scores') and generated.sequences_scores is not None:
                confidence = float(self._torch.sigmoid(generated.sequences_scores[0]))
            else:
                confidence = 0.8  # Default confidence if scores unavailable
            
            # Clean and Format
            cleaned = self._clean_plate_text(text)
            formatted = self.format_indian_plate(cleaned)
            
            return formatted if formatted else None, confidence
            
        except Exception as e:
            print(f"TrOCR error: {e}")
            return None, 0.0
    
    def _preprocess_plate(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate image for TrOCR.
        TrOCR works better with clean color images (less aggressive than EasyOCR preprocessing).
        """
        height, width = image.shape[:2]
        
        # Resize if too small
        if width < 100:
            scale = 100 / width
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Light denoising (bilateral preserves edges)
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Slight contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _clean_plate_text(self, text: str) -> str:
        """Clean and normalize plate text"""
        text = text.upper()
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text
    
    def format_indian_plate(self, text: str) -> str:
        """Format text as Indian license plate (delegates to same logic as EasyOCR)"""
        if not text or len(text) < 9:
            return text
        text = text.replace(" ", "")
        corrected = ""
        for i, char in enumerate(text):
            if i < 2:
                if char in '01':
                    corrected += 'O' if char == '0' else 'I'
                else:
                    corrected += char
            elif i < 4:
                if char in self.char_map and char.isalpha():
                    corrected += self.char_map[char]
                else:
                    corrected += char
            elif i < 6:
                corrected += char
            else:
                if char in self.char_map and char.isalpha():
                    corrected += self.char_map[char]
                else:
                    corrected += char
        if len(corrected) >= 10:
            return f"{corrected[:2]} {corrected[2:4]} {corrected[4:6]} {corrected[6:10]}"
        return corrected
    
    def validate_indian_plate(self, text: str) -> bool:
        """Validate Indian plate format"""
        if not text:
            return False
        text = text.replace(" ", "")
        patterns = [
            r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$',
            r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,4}$',
        ]
        return any(re.match(p, text) for p in patterns)
    
    def crop_plate_from_frame(
        self,
        frame: np.ndarray,
        box: Tuple[float, float, float, float],
        padding: int = 5
    ) -> np.ndarray:
        """Crop license plate region from frame"""
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return frame[y1:y2, x1:x2].copy()


class PaddleOCREngine:
    """
    License plate text extraction using PaddleOCR.
    This provides highly robust, production-level text extraction aligned with the 
    aneesarom high-precision repository standards.
    """
    def __init__(self, use_gpu: bool = True):
        print("Initializing PaddleOCR engine (this may take a moment)...")
        try:
            from paddleocr import PaddleOCR
            import logging
            # Suppress excessive PaddleOCR logging
            logging.getLogger("ppocr").setLevel(logging.ERROR)
            
            self.reader = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=use_gpu, show_log=False)
            self._available = True
            print("PaddleOCR engine ready.")
        except Exception as e:
            print(f"PaddleOCR initialization failed: {e}")
            self._available = False
            
        # Character mapping for common OCR mistakes
        self.char_map = {
            'O': '0', 'I': '1', 'J': '1', 'A': '4', 'G': '6',
            'S': '5', 'B': '8', 'Z': '2', 'Q': '0'
        }
        
    @property
    def is_available(self) -> bool:
        return self._available
        
    def read_plate(self, plate_image: np.ndarray, preprocess: bool = True) -> Tuple[Optional[str], float]:
        if not self._available or plate_image is None or plate_image.size == 0:
            return None, 0.0
            
        try:
            if preprocess:
                processed = self._preprocess_plate(plate_image)
            else:
                processed = plate_image
                
            results = self.reader.ocr(processed, cls=False)
            
            if not results or not results[0]:
                return None, 0.0
                
            texts = []
            confidences = []
            
            # PaddleOCR returns [[[box_coords], (text, confidence)], ...]
            for line in results[0]:
                _, (text, conf) = line
                texts.append(text)
                confidences.append(conf)
                
            combined = "".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            cleaned = self._clean_plate_text(combined)
            formatted = self.format_indian_plate(cleaned)
            
            return formatted if formatted else None, avg_confidence
            
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return None, 0.0
            
    def _preprocess_plate(self, image: np.ndarray) -> np.ndarray:
        """PaddleOCR works exceptionally well with simple grayscale contrast enhancement."""
        height, width = image.shape[:2]
        if width < 150:
            scale = 150 / width
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
        # Convert to grayscale to remove color noise
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply simple grayscale CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
        
    def _clean_plate_text(self, text: str) -> str:
        text = text.upper()
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text
        
    def format_indian_plate(self, text: str) -> str:
        if not text or len(text) < 9: return text
        text = text.replace(" ", "")
        corrected = ""
        for i, char in enumerate(text):
            if i < 2: corrected += 'O' if char == '0' else ('1' if char in 'IJ' else char)
            elif i < 4: corrected += self.char_map.get(char, char) if char.isalpha() else char
            elif i < 6: corrected += char
            else: corrected += self.char_map.get(char, char) if char.isalpha() else char
        if len(corrected) >= 10: return f"{corrected[:2]} {corrected[2:4]} {corrected[4:6]} {corrected[6:10]}"
        return corrected
        
def validate_indian_plate(text: str) -> bool:
    """
    Validate Indian plate format strictly. 
    Pattern: [State code][District numbers][Series letters][Unique digits]
    Format: AA 11 AA 1111
    """
    if not text: return False
    # Remove all non-alphanumeric for matching
    t = re.sub(r'[^A-Z0-9]', '', text.upper())
    # [State: 2 chars][Dist: 2 digits][Series: 1-2 chars][Num: 1-4 digits]
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{1,4}$'
    return bool(re.match(pattern, t))

def score_plate_match(text: str) -> int:
    """
    Score how well a string matches the standard Indian plate pattern.
    Higher score = better match. 
    """
    if not text: return 0
    t = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    score = 0
    # State: KL, KA, MH...
    if re.match(r'^[A-Z]{2}', t): score += 20         
    # District: 01, 12...
    if re.match(r'^[A-Z]{2}[0-9]{2}', t): score += 20   
    # Series: A, AQ, B...
    if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}', t): score += 30 
    # Full Number: 3439...
    if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{1,4}$', t): 
        score += 30 
        if len(t) >= 9: score += 10 # Ideal length
        
    return score

class PlateReadingAccumulator:
    """
    Accumulates multiple OCR readings for a vehicle to find the most accurate consensus.
    Uses temporal voting, pattern-priority matching, and clarity ranking.
    """
    def __init__(self, max_samples: int = 15, consensus_threshold: float = 0.6):
        self.max_samples = max_samples
        self.threshold = consensus_threshold
        self.readings = {} # track_id -> List[str]
        self.best_crops = {} # track_id -> List[Tuple[float, np.ndarray]] (clarity, crop)
        self.confirmed_plates = {} # track_id -> str
        self.pattern_scores = {} # track_id -> int (highest score seen)

    def reset(self, hard: bool = True):
        """
        Clear accumulated state. 
        If hard=False, retain confirmed Indian plates and their scores.
        """
        if hard:
            self.readings.clear()
            self.best_crops.clear()
            self.confirmed_plates.clear()
            self.pattern_scores.clear()
        else:
            # Selective clearing: only keep confirmed valid patterns
            # This allows IDs starting back at 1 after a loop to inherit their plate
            to_keep_plates = {}
            to_keep_scores = {}
            for tid, plate in self.confirmed_plates.items():
                if validate_indian_plate(plate):
                    to_keep_plates[tid] = plate
                    to_keep_scores[tid] = self.pattern_scores.get(tid, 100)
            
            self.readings.clear()
            self.best_crops.clear()
            self.confirmed_plates = to_keep_plates
            self.pattern_scores = to_keep_scores

    def calculate_clarity(self, image: np.ndarray) -> float:
        """Estimate image clarity using Laplacian variance (blur detection)"""
        if image is None or image.size == 0: return 0
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except: return 0

    def add_reading(self, track_id: int, plate_text: str, crop: np.ndarray, **kwargs):
        """Add a candidate reading and crop for a specific track"""
        if track_id is None: return
        
        # 1. Update text history if valid
        if plate_text and len(plate_text) > 3:
            if track_id not in self.readings: self.readings[track_id] = []
            self.readings[track_id].append(plate_text)
            if len(self.readings[track_id]) > self.max_samples: 
                self.readings[track_id].pop(0)

            # 2. Pattern Scoring & Locking
            score = score_plate_match(plate_text)
            current_best = self.pattern_scores.get(track_id, 0)
            if score >= 90: # Near-perfect pattern
                self.confirmed_plates[track_id] = plate_text
                self.pattern_scores[track_id] = 100
            elif score > current_best:
                self.pattern_scores[track_id] = score

        # 3. Store high-resolution best crops
        if crop is not None and crop.size > 0:
            clarity = self.calculate_clarity(crop)
            if track_id not in self.best_crops: self.best_crops[track_id] = []
            self.best_crops[track_id].append((clarity, crop))
            self.best_crops[track_id].sort(key=lambda x: x[0], reverse=True)
            self.best_crops[track_id] = self.best_crops[track_id][:10]

    def get_consensus(self, track_id: int, **kwargs) -> Optional[str]:
        """
        Determine candidate plate(s) based on history and pattern match.
        Returns comma-separated candidates, prioritizing the best pattern match.
        """
        if track_id in self.confirmed_plates:
            return self.confirmed_plates[track_id]
            
        history = self.readings.get(track_id, [])
        if not history: return None
        
        # 1. Group candidates and apply pattern scores
        from collections import Counter
        counts = Counter(history)
        
        # 2. Get high-scoring pattern matches
        scored_candidates = []
        for text in counts.keys():
            score = score_plate_match(text)
            scored_candidates.append((text, score, counts[text]))
        
        # Sort by: Pattern Score (Primary), Frequency (Secondary), Length (Tertiary)
        scored_candidates.sort(key=lambda x: (x[1], x[2], len(x[0])), reverse=True)

        # 3. If we have a clear pattern match, return it
        best_text, best_score, _ = scored_candidates[0]
        if best_score >= 80:
            # Optionally "lock" it if it has been seen multiple times
            if counts[best_text] >= 2:
                self.confirmed_plates[track_id] = best_text
            return best_text

        # 4. Otherwise return top 2-3 candidates separated by comma
        top_candidates = [t for t, s, c in scored_candidates[:3]]
        return ", ".join(top_candidates)

def get_ocr_engine(engine: str = "easyocr", use_gpu: bool = True):
    """
    Factory function to get the configured OCR engine.
    
    Args:
        engine: "easyocr" or "trocr"
        use_gpu: Whether to use GPU
        
    Returns:
        PlateOCR or TrOCRPlateReader instance
    """
    if engine == "trocr":
        reader = TrOCRPlateReader(use_gpu=use_gpu)
        if reader.is_available:
            return reader
        print("[OCR] TrOCR not available, falling back to EasyOCR")
        
    if engine == "paddleocr":
        reader = PaddleOCREngine(use_gpu=use_gpu)
        if reader.is_available:
            return reader
        print("[OCR] PaddleOCR not available, falling back to EasyOCR")
    
    return PlateOCR(use_gpu=use_gpu)

