import cv2
import numpy as np
import csv
from pathlib import Path

VIDEOS_DIR = Path("../dataset/videos")
CSV_PATH = Path("../data.csv")

IMG_DIR = Path("../dataset/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label", "x1", "y1", "x2", "y2"])

clicks = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Annotate: center -> tip", param)

for video_path in sorted(VIDEOS_DIR.glob("*.mp4")):
    print(f"Обрабатываю: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Видео не открылось")
        continue

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        cv2.imshow("Annotate: center -> tip", display_frame)
        cv2.setMouseCallback("Annotate: center -> tip", lambda e, x, y, f, p: mouse_callback(e, x, y, f, display_frame))

        while len(clicks) < 2:
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        center = clicks[0]
        tip = clicks[1]
        clicks.clear()

        x1 = min(center[0], tip[0])
        y1 = min(center[1], tip[1])
        x2 = max(center[0], tip[0])
        y2 = max(center[1], tip[1])

        dx = tip[0] - center[0]
        dy = tip[1] - center[1]
        angle_deg = np.degrees(np.arctan2(-dy, dx)) % 360

        if 45 <= angle_deg < 135:
            label = "N"
        elif 135 <= angle_deg < 225:
            label = "W"
        elif 225 <= angle_deg < 315:
            label = "S"
        else:
            label = "E"

        filename = IMG_DIR / f"{video_path.stem}_{frame_id:05d}.jpg"
        cv2.imwrite(str(filename), frame)

        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"dataset/images/{filename.name}", label, x1, y1, x2, y2])

        print(f"  {filename.name} -> {label}")
        frame_id += 1

    cap.release()

cv2.destroyAllWindows()