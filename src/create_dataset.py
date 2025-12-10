import cv2
from pathlib import Path

Path("data/images").mkdir(parents=True, exist_ok=True)
Path("data/labels").mkdir(parents=True, exist_ok=True)

videos = list(Path("videos").glob("*.mp4"))
if not videos:
    raise FileNotFoundError("Поместите видео в папку `videos/`")

for vid in videos:
    cap = cv2.VideoCapture(str(vid))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_id % (fps * 2) == 0:
            img_path = f"data/images/{vid.stem}_{frame_id}.jpg"
            cv2.imwrite(img_path, frame)
        frame_id += 1
    cap.release()
print("Кадры сохранены в data/images/. Разметьте вручную в data/labels/")