"""
LLM Provider Backend for Traffic Violation Detection System.
Supports Gemini and OpenAI models for image analysis and OCR.
"""
import os
import base64
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import cv2
import requests
import numpy as np
import config
from src.logger import get_logger

logger = get_logger("LLM")

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def analyze_frame(self, frame, prompt: str) -> str:
        """Analyze a full frame with a text prompt"""
        pass
    
    @abstractmethod
    def analyze_frames(self, frames: List[np.ndarray], prompt: str) -> str:
        """Analyze multiple frames with a single prompt"""
        pass
        
    @abstractmethod
    def read_plate(self, plate_crop) -> Tuple[str, float]:
        """Specialized method to read license plates"""
        pass
        
    @abstractmethod
    def verify_helmet(self, rider_crop) -> Tuple[bool, str]:
        """Verify if the rider in the crop is wearing a helmet. Returns (is_wearing, reasoning)"""
        pass
    
    @abstractmethod
    def verify_phone_usage(self, rider_crop) -> Tuple[bool, str]:
        """Detect if the rider in the crop is using a phone. Returns (is_using_phone, reasoning)"""
        pass
        
    @abstractmethod
    def check_passengers(self, rider_crop) -> Tuple[int, str]:
        """Count people on the motorcycle. Returns (count, reasoning)"""
        pass

    @abstractmethod
    def check_passengers_voting(self, frames: List[np.ndarray]) -> Tuple[int, str]:
        """Count people on the motorcycle using multiple frames. Returns (count, reasoning)"""
        pass
        
    @abstractmethod
    def test_connectivity(self) -> Tuple[bool, str]:
        """Test if the API key and connection are working"""
        pass
        
    def encode_image(self, image):
        """Convert cv2 image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

class GeminiProvider(LLMProvider):
    """Provider for Google Gemini Models (1.5 Flash/Pro)"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        # Ensure model has the 'models/' prefix
        if not model.startswith("models/"):
            model = f"models/{model}"
        self.model = model
        # Try v1beta as fallback if v1 fails, but v1beta is standard for latest models
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:generateContent?key={api_key}"
        
    def analyze_frame(self, frame, prompt: str) -> str:
        # Gemini REST API implementation
        b64_image = self.encode_image(frame)
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_image
                    }}
                ]
            }]
        }
        
        try:
            logger.info(f"Gemini Request: {prompt[:50]}...")
            response = requests.post(self.api_url, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"Gemini API HTTP {response.status_code}: {response.text}")
            response.raise_for_status()
            data = response.json()
            result = data["candidates"][0]["content"]["parts"][0]["text"]
            logger.info(f"Gemini Response: {result[:50]}...")
            return result
        except Exception as e:
            print(f"DEBUG: Gemini Provider Error: {str(e)}")
            return f"Error: {str(e)}"

    def analyze_frames(self, frames: List[np.ndarray], prompt: str) -> str:
        """Analyze multiple frames in a single prompt for voting/persistence"""
        parts = [{"text": prompt}]
        for frame in frames:
            b64_image = self.encode_image(frame)
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64_image
                }
            })
            
        payload = {
            "contents": [{"parts": parts}]
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"DEBUG: Gemini Multi-Frame Error: {str(e)}")
            return f"Error: {str(e)}"

    def test_connectivity(self) -> Tuple[bool, str]:
        """Test Gemini connection with a simple text prompt"""
        test_url = self.api_url.replace("generateContent", "generateContent") # Same endpoint
        payload = {
            "contents": [{"parts": [{"text": "Say 'OK'"}]}]
        }
        try:
            response = requests.post(test_url, json=payload, timeout=10)
            if response.status_code == 200:
                return True, "Connection Successful (Gemini)"
            else:
                err = response.json().get("error", {}).get("message", "Unknown Error")
                return False, f"Error {response.status_code}: {err}"
        except Exception as e:
            return False, f"Connection Failed: {str(e)}"

    def read_plate(self, plate_crop) -> Tuple[str, float]:
        # Specialized OCR prompt
        prompt = "Read the license plate text in this image. Return ONLY the text, no other words. If unreadable, return 'UNREADABLE'."
        text = self.analyze_frame(plate_crop, prompt).strip()
        
        if "UNREADABLE" in text or "Error" in text:
            return "", 0.0
            
        return text, 0.9 # Return high confidence if LLM reads it

    def verify_helmet(self, rider_crop) -> Tuple[bool, str]:
        """Gemini-based helmet verification"""
        prompt = (
            "Look at the riders on this motorcycle. Is EVERY person wearing a safety helmet? "
            "WARNING: Do NOT confuse a hand near the head with a helmet. Look for the straps and hard shell. "
            "Reply with 'YES' or 'NO' followed by a short reason. "
            "Example: 'NO - Driver is bareheaded, hand is just near ear' or 'YES - Both wearing helmets'."
        )
        response = self.analyze_frame(rider_crop, prompt).strip()
        
        is_wearing = response.upper().startswith("YES")
        return is_wearing, response
    
    def verify_phone_usage(self, rider_crop) -> Tuple[bool, str]:
        """Gemini-based phone usage detection"""
        prompt = (
            "Look at this motorcycle rider carefully. Is the rider holding or using a mobile phone? "
            "Check for: phone held to ear, phone in hand while riding, phone near face, texting while riding. "
            "A phone is a small rectangular device. Do NOT confuse gloves, mirrors, or handlebar grips with a phone. "
            "Reply with 'YES' or 'NO' followed by a short reason. "
            "Example: 'YES - Rider is holding phone to right ear' or 'NO - Hands are on handlebars'."
        )
        response = self.analyze_frame(rider_crop, prompt).strip()
        
        is_using = response.upper().startswith("YES")
        return is_using, response

    def check_passengers(self, rider_crop) -> Tuple[int, str]:
        """Gemini-based passenger counting"""
        prompt = (
            "Look very closely at this motorcycle. How many individual people are sitting on it? "
            "Count EVERY person, including children, infants, or partially hidden passengers. "
            "Search for extra hands, legs, or heads. If you see 3, 4, or more people, it's a critical violation. "
            "Return the NUMBER first, then a brief list of who you see. "
            "Example: '4 - Driver, two adults behind, and a child in front of the driver'."
        )
        response = self.analyze_frame(rider_crop, prompt).strip()
        
        # Robust Error Check: Never parse errors as counts
        if response.startswith("Error:"):
            return 1, response
            
        import re
        # Try to find digit first
        match = re.search(r'\d+', response)
        count = int(match.group()) if match else 1
        return count, response

    def check_passengers_voting(self, frames: List[np.ndarray]) -> Tuple[int, str]:
        """Multi-frame voting for passenger counting"""
        prompt = (
            "I'm showing you multiple chronological snapshots of the same motorcycle. "
            "Examine ALL of them to find hidden passengers. A passenger might be hidden in frame 1 but visible in frame 3. "
            "How many TOTAL individual people are on this bike? "
            "Return the NUMBER first, then the reason. Example: '3 - Visible behind driver in 2nd frame'."
        )
        response = self.analyze_frames(frames, prompt).strip()
        
        if response.startswith("Error:"):
            return 1, response
            
        import re
        match = re.search(r'\d+', response)
        count = int(match.group()) if match else 1
        return count, response
        match = re.search(r'\d+', response)
        if match:
            # Basic range check: Impossible to have 100+ people on a bike. 
            # If so, it's likely a misparsed error code (like 404).
            val = int(match.group())
            count = val if val < 20 else 1 
        else:
            # Fallback to common text numbers if digits are missing
            res_lower = response.lower()
            if "three" in res_lower: count = 3
            elif "four" in res_lower: count = 4
            elif "five" in res_lower: count = 5
            elif "two" in res_lower: count = 2
            elif "one" in res_lower: count = 1
            else: count = 1
        return count, response

class OpenAIProvider(LLMProvider):
    """Provider for OpenAI Models (GPT-4o)"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
    def analyze_frame(self, frame, prompt: str) -> str:
        b64_image = self.encode_image(frame)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        try:
            logger.info(f"OpenAI Request: {prompt[:50]}...")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()['choices'][0]['message']['content']
            logger.info(f"OpenAI Response: {result[:50]}...")
            return result
        except Exception as e:
            return f"Error: {str(e)}"

    def read_plate(self, plate_crop) -> Tuple[str, float]:
        prompt = "Read the license plate text. Return ONLY the text. No markdown, no explanations."
        text = self.analyze_frame(plate_crop, prompt).strip()
        return text, 0.9
        
    def verify_helmet(self, rider_crop) -> Tuple[bool, str]:
        """OpenAI-based helmet verification"""
        prompt = (
            "Is the person on the motorcycle wearing a helmet? "
            "Be careful: don't mistake a hand near the head for a helmet. "
            "Reply with YES or NO and a brief reason."
        )
        response = self.analyze_frame(rider_crop, prompt).strip()
        is_wearing = response.upper().startswith("YES")
        return is_wearing, response
    
    def verify_phone_usage(self, rider_crop) -> Tuple[bool, str]:
        """OpenAI-based phone usage detection"""
        prompt = (
            "Is this motorcycle rider holding or using a mobile phone while riding? "
            "Look for a phone held to the ear, in the hand, or near the face. "
            "Do NOT confuse gloves, mirrors, or handlebar grips with a phone. "
            "Reply with YES or NO and a brief reason."
        )
        response = self.analyze_frame(rider_crop, prompt).strip()
        is_using = response.upper().startswith("YES")
        return is_using, response

    def check_passengers(self, rider_crop) -> Tuple[int, str]:
        """OpenAI-based passenger counting"""
        prompt = "How many people are on this motorcycle? Return the number followed by reasoning."
        response = self.analyze_frame(rider_crop, prompt).strip()
        if response.startswith("Error:"): return 1, response
        
        import re
        match = re.search(r'\d+', response)
        if match:
            val = int(match.group())
            count = val if val < 20 else 1
        else:
            res_lower = response.lower()
            if "three" in res_lower: count = 3
            elif "two" in res_lower: count = 2
            else: count = 1
        return count, response

    def test_connectivity(self) -> Tuple[bool, str]:
        """Test OpenAI connection with a simple text prompt"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Say 'OK'"}],
            "max_tokens": 5
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                return True, "Connection Successful (OpenAI)"
            else:
                err = response.json().get("error", {}).get("message", "Unknown Error")
                return False, f"Error {response.status_code}: {err}"
        except Exception as e:
            logger.error(f"OpenAI test error: {str(e)}")
            return False, f"Connection Failed: {str(e)}"

    def analyze_frames(self, frames: List[np.ndarray], prompt: str) -> str:
        """Analyze multiple frames for OpenAI (GPT-4o Vision)"""
        content = [{"type": "text", "text": prompt}]
        for frame in frames:
            b64 = self.encode_image(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
            
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 512
        }
        
        try:
            logger.info(f"OpenAI Multi-Frame Request ({len(frames)} frames)")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=20
            )
            response.raise_for_status()
            result = response.json()['choices'][0]['message']['content']
            logger.info(f"OpenAI Multi-Frame Response: {result[:50]}...")
            return result
        except Exception as e:
            logger.error(f"OpenAI Multi-Frame Error: {str(e)}")
            return f"Error: {str(e)}"

    def check_passengers_voting(self, frames: List[np.ndarray]) -> Tuple[int, str]:
        """Multi-frame voting for OpenAI counting"""
        prompt = (
            "I'm showing you multiple chronological snapshots of the same motorcycle. "
            "Examine ALL of them to find hidden passengers. Count TOTAL individual people. "
            "Return the NUMBER first, then the reason."
        )
        response = self.analyze_frames(frames, prompt).strip()
        if response.startswith("Error:"): return 1, response
        import re
        match = re.search(r'\d+', response)
        count = int(match.group()) if match else 1
        return count, response

class CustomLLMProvider(LLMProvider):
    """Provider for Custom/Local OpenAI-compatible Models (Ollama, LM Studio)"""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
    def analyze_frame(self, frame, prompt: str) -> str:
        b64_image = self.encode_image(frame)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]}],
            "max_tokens": 512
        }
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

    def test_connectivity(self) -> Tuple[bool, str]:
        """Test custom API connection"""
        try:
            logger.info(f"Custom Provider Test: {self.base_url}")
            # Simple models endpoint check
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                return True, "Connection Successful (Custom/Local)"
            else:
                return False, f"HTTP Error {response.status_code}"
        except Exception as e:
            logger.error(f"Custom Test Error: {str(e)}")
            return False, f"Connection Failed: {str(e)}"

    def analyze_frames(self, frames: List[np.ndarray], prompt: str) -> str:
        """Analyze multiple frames for Custom Providers (assumes OpenAI compatibility)"""
        content = [{"type": "text", "text": prompt}]
        for frame in frames:
            b64 = self.encode_image(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        payload = {"model": self.model, "messages": [{"role": "user", "content": content}], "max_tokens": 512}
        try:
            logger.info(f"Custom Multi-Frame Request ({len(frames)} frames)")
            response = requests.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()['choices'][0]['message']['content']
            logger.info(f"Custom Multi-Frame Response: {result[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Custom Multi-Frame Error: {str(e)}")
            return f"Error: {str(e)}"

    def check_passengers_voting(self, frames: List[np.ndarray]) -> Tuple[int, str]:
        """Custom multi-frame voting"""
        prompt = "Look at these frames of a motorcycle. How many people in total?"
        response = self.analyze_frames(frames, prompt).strip()
        if response.startswith("Error:"): return 1, response
        import re
        match = re.search(r'\d+', response)
        count = int(match.group()) if match else 1
        return count, response

    def read_plate(self, plate_crop) -> Tuple[str, float]:
        text = self.analyze_frame(plate_crop, "Read the license plate.").strip()
        return text, 0.8
        
    def verify_helmet(self, rider_crop) -> Tuple[bool, str]:
        response = self.analyze_frame(rider_crop, "Is the rider wearing a helmet? YES/NO").strip()
        return response.upper().startswith("YES"), response

    def check_passengers(self, rider_crop) -> Tuple[int, str]:
        """Custom-based passenger counting"""
        response = self.analyze_frame(rider_crop, "How many people are on this motorcycle?").strip()
        if response.startswith("Error:"): return 1, response
        
        import re
        match = re.search(r'\d+', response)
        if match:
            val = int(match.group())
            count = val if val < 20 else 1
        else:
            res_lower = response.lower()
            if "three" in res_lower: count = 3
            elif "two" in res_lower: count = 2
            else: count = 1
        return count, response

def get_llm_provider():
    """Factory to get the configured LLM provider based on LLM_PROVIDER setting"""
    provider_type = getattr(config, 'LLM_PROVIDER', 'gemini')
    selected_model = getattr(config, 'SELECTED_MODEL', 'gemini-2.0-flash')
    
    api_key_gemini = getattr(config, 'GEMINI_API_KEY', "")
    api_key_openai = getattr(config, 'OPENAI_API_KEY', "")
    
    if provider_type == "gemini" and api_key_gemini:
        return GeminiProvider(api_key_gemini, selected_model)
    elif provider_type == "openai" and api_key_openai:
        # If model doesn't look like GPT, use a sensible default
        if "gpt" not in selected_model.lower():
            selected_model = "gpt-4o"
        return OpenAIProvider(api_key_openai, selected_model)
    elif provider_type == "custom":
        custom_name = getattr(config, 'CUSTOM_MODEL_NAME', "Local Model")
        custom_url = getattr(config, 'CUSTOM_BASE_URL', "")
        custom_key = getattr(config, 'CUSTOM_API_KEY', "")
        if custom_url:
            return CustomLLMProvider(custom_key, custom_url, custom_name)
    
    return None
