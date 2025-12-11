import csv
import os
import yaml
from ultralytics import YOLO
import torch
import cv2
import random
from pathlib import Path

df = []
with open("../data.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        df.append(row)

random.shuffle(df)
split = int(0.8 * len(df))
train_df = df[:split]
val_df = df[split:]

print(f"Всего данных: {len(df)}")
print(f"Train: {len(train_df)}")
print(f"Val: {len(val_df)}")

for split_name, data in [("train", train_df), ("val", val_df)]:
    os.makedirs(f"yolo_data/{split_name}/images", exist_ok=True)
    os.makedirs(f"yolo_data/{split_name}/labels", exist_ok=True)

    for row in data:
        img_src = Path(f"../{row['filename']}")
        img_dst = f"yolo_data/{split_name}/images/{img_src.name}"
        os.system(f"cp {img_src} {img_dst}")
        
        img = cv2.imread(str(img_src))
        h, w = img.shape[:2]
        
        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])
        
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        
        label_id = {"N": 0, "S": 1, "W": 2, "E": 3}[row["label"]]
        
        txt_name = img_src.stem + ".txt"
        with open(f"yolo_data/{split_name}/labels/{txt_name}", "w") as f:
            f.write(f"{label_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

data_yaml = {
    "path": str(Path.cwd() / "yolo_data"),
    "train": "train/images",
    "val": "val/images",
    "nc": 4,
    "names": ["N", "S", "W", "E"]
}

with open("yolo_data/data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Обучение на {device.upper()}")

model = YOLO("yolov8n.pt")
model.train(
    data="yolo_data/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    name="compass",
    device=device,
    augment=True
)

os.makedirs("../models", exist_ok=True)
os.system("cp runs/detect/compass/weights/best.pt ../models/best.pt")

os.system("rm -rf yolo_data runs")
print("Модель сохранена в ../models/best.pt")