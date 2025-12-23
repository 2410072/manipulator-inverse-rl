import gymnasium as gym
import panda_gym
import numpy as np
import cv2
from pathlib import Path
import time

def verify_codec(codec_name):
    output_path = Path(f"test_output/test_{codec_name}.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Testing codec: {codec_name} ---")
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        w, h = 640, 480
        writer = cv2.VideoWriter(str(output_path), fourcc, 10.0, (w, h))
        
        if writer.isOpened():
            print(f"Writer opened for {codec_name}")
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Write some frames
            for _ in range(10):
                writer.write(frame)
            writer.release()
            
            size = output_path.stat().st_size
            print(f"File created: {size} bytes")
            if size > 500:
                print("SUCCESS: File seems valid.")
                return True
            else:
                print("FAILURE: File too small (probably header only).")
                return False
        else:
            print(f"Writer FAILED to open for {codec_name}")
            return False
            
    except Exception as e:
        print(f"Exception testing {codec_name}: {e}")
        return False

if __name__ == "__main__":
    test_avc1 = verify_codec('avc1')
    test_mp4v = verify_codec('mp4v')
    
    # Test VP8 (WebM)
    print("\n--- Testing codec: vp80 (.webm) ---")
    try:
        output_path = Path("test_output/test_vp80.webm")
        fourcc = cv2.VideoWriter_fourcc(*'vp80')
        w, h = 640, 480
        writer = cv2.VideoWriter(str(output_path), fourcc, 10.0, (w, h))
        if writer.isOpened():
             print("Writer opened for vp80")
             frame = np.zeros((h, w, 3), dtype=np.uint8)
             for _ in range(10): writer.write(frame)
             writer.release()
             if output_path.stat().st_size > 500:
                 print("SUCCESS: VP8 seems valid.")
                 test_vp80 = True
             else:
                 test_vp80 = False
        else:
             print("Writer FAILED for vp80")
             test_vp80 = False
    except Exception as e:
        print(f"VP8 error: {e}")
        test_vp80 = False

    print("\nSummary:")
    print(f"avc1: {test_avc1}")
    print(f"mp4v: {test_mp4v}")
    print(f"vp80: {test_vp80}")
