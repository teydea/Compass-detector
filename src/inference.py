import cv2
import argparse
import torch
import os
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
args = parser.parse_args()

if not os.path.exists("../models/best.pt"):
    print("Ошибка: Модель не найдена")
    exit(1)

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

    for r in results:
        if len(r.boxes) == 0:
            continue
        best_box = max(r.boxes, key=lambda b: float(b.conf))
        cls_id = int(best_box.cls[0])
        conf = float(best_box.conf[0])
        label = names[cls_id]
        break

    text = f"{names[label]} ({conf:.2f})"
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    text_x = frame.shape[1] - text_size[0] - 20
    text_y = text_size[1] + 20
    
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    
    cv2.imshow("Compass Direction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()