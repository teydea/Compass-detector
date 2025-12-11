import cv2
import argparse
import torch
from ultralytics import YOLO
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
args = parser.parse_args()

model = YOLO("../models/best.pt")
names = ["N", "S", "W", "E"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Инференс на {device.upper()}")

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    exit("Не удалось открыть видео")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device=device, conf=0.5, verbose=False)
    label, conf = "??", 0.0
    p1 = p2 = None

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]
            p1 = (x1, y1)
            p2 = (x2, y2)
            break

    if p1 and p2:
        cv2.line(frame, p1, p2, (0, 255, 0), 3)
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = (dx**2 + dy**2)**0.5
        
        if length > 0:
            ux, uy = dx / length, dy / length
            px, py = -uy, ux
            arrow_len = 15
            
            tip = p2
            left_wing = (
                int(tip[0] - arrow_len * (ux - px)),
                int(tip[1] - arrow_len * (uy - py))
            )
            right_wing = (
                int(tip[0] - arrow_len * (ux + px)),
                int(tip[1] - arrow_len * (uy + py))
            )
            
            cv2.line(frame, tip, left_wing, (0, 255, 0), 3)
            cv2.line(frame, tip, right_wing, (0, 255, 0), 3)

    text = f"{label} ({conf:.2f})"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    cv2.imshow("Compass Direction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()