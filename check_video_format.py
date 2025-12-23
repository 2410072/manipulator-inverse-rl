import cv2
import struct

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

video_path = "assets/TD3 Training.mp4"
cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    codec = decode_fourcc(fourcc)
    print(f"Video: {video_path}")
    print(f"FourCC code: {fourcc}")
    print(f"Codec: {codec}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    cap.release()
else:
    print(f"Failed to open {video_path}")
