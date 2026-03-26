import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.llm import get_llm_provider

def test_triple_riding_parsing():
    print("Testing LLM Triple Riding Parsing...")
    
    # Initialize provider
    provider = get_llm_provider()
    if not provider:
        print("ERROR: LLM Provider not initialized. Check your API keys in settings.json.")
        return

    print(f"Using Provider: {provider.model}")
    
    # Create a dummy "black" frame for testing (since we can't easily load the user's video here)
    dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "TEST", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    try:
        print("Sending dummy frame to LLM for counting...")
        count, reason = provider.check_passengers(dummy_frame)
        print(f"RESULT: Count={count}")
        print(f"REASON: {reason}")
        
        if isinstance(count, int):
            print("✅ SUCCESS: Count is a valid integer.")
        else:
            print("❌ FAILURE: Count is not an integer.")
            
    except Exception as e:
        print(f"❌ ERROR calling LLM: {e}")

if __name__ == "__main__":
    test_triple_riding_parsing()
