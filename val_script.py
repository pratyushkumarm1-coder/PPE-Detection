import os, sys, json, shutil
from pathlib import Path
from ultralytics import YOLO

# Import download_dataset from train.py
from train import download_dataset

if __name__ == "__main__":
    download_dataset()
    print("Loading best.pt for validation...")
    model = YOLO("best.pt")
    
    print("Running validation...")
    metrics = model.val(data="dataset/data.yaml")
    
    print("\n─── Validation Results ───")
    print(f"  mAP50        : {metrics.box.map50:.4f}")
    print(f"  mAP50-95     : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print("──────────────────────────")
