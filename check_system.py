import torch
import sys

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

try:
    import pyautogui
    print("pyautogui: INSTALLED")
except ImportError:
    print("pyautogui: NOT INSTALLED")

try:
    from PIL import ImageGrab
    print("PIL.ImageGrab: INSTALLED")
except ImportError:
    print("PIL.ImageGrab: NOT INSTALLED")

import cv2
print(f"OpenCV: {cv2.__version__}")
