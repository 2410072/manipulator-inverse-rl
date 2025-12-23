from PIL import ImageGrab
import numpy as np

try:
    print("Attempting ImageGrab.grab()...")
    img = ImageGrab.grab()
    print(f"Grab success! Size: {img.size}")
    img.save("test_output/grab_test.png")
except Exception as e:
    print(f"ImageGrab failed: {e}")
