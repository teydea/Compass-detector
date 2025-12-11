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

def make_callback(frame_ref):
    def callback(event, x, y, flags, param):
        mouse_callback(event, x, y, flags, frame_ref)
    return callback

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
        cv2.setMouseCallback("Annotate: center -> tip", make_callback(display_frame))

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

        # ЖДЕМ ВВОДА МЕТКИ ОТ ЮЗЕРА
        print("Введи направление (N/S/W/E): ", end='')
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n') or key == ord('N'):
                label = "N"
                break
            elif key == ord('s') or key == ord('S'):
                label = "S"
                break
            elif key == ord('w') or key == ord('W'):
                label = "W"
                break
            elif key == ord('e') or key == ord('E'):
                label = "E"
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
            else:
                print("Неверный ввод. Введи N/S/W/E: ", end='')
        print(f" -> {label}")

        filename = IMG_DIR / f"{video_path.stem}_{frame_id:05d}.jpg"
        cv2.imwrite(str(filename), frame)

        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"dataset/images/{filename.name}", label, x1, y1, x2, y2])

        print(f"  {filename.name} -> {label}")
        frame_id += 1

    cap.release()

cv2.destroyAllWindows()