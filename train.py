"""
PPE Detection — Local Training Script
Trains YOLOv8s on Helmet + Gloves dataset (downloaded via Roboflow).
Run: python train.py
"""

import os, sys, json, shutil
from pathlib import Path

# ── 1. Install deps if missing ────────────────────────────────────────────────
def ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {pkg} -q")

ensure("ultralytics")
ensure("roboflow")
ensure("opencv-python-headless", "cv2")

# ── 2. Imports ────────────────────────────────────────────────────────────────
from ultralytics import YOLO

DATA_YAML = Path("dataset/data.yaml")
RUNS_DIR  = Path("runs/detect/ppe_train")
WEIGHTS   = Path("best.pt")

# ── 3. Download dataset from Roboflow ─────────────────────────────────────────
# Known public PPE datasets on Roboflow Universe (tried in order)
ROBOFLOW_CANDIDATES = []

def download_dataset():
    if DATA_YAML.exists():
        print("Dataset already exists — skipping download.")
        return

    print("\n[1/4] Downloading PPE dataset from Roboflow Universe …")
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key="xOyWivXC9d9oRfxRBj5V")
        for ws, proj, ver in ROBOFLOW_CANDIDATES:
            try:
                print(f"  Trying {ws}/{proj} v{ver} …")
                dataset = rf.workspace(ws).project(proj).version(ver).download(
                    "yolov8", location="dataset"
                )
                print(f"  Downloaded: {ws}/{proj} v{ver} ✓")
                _fix_data_yaml()   # ensure class names match our 4 classes
                return
            except Exception as inner:
                print(f"  Skipped ({inner})")
        print("All Roboflow candidates failed. Falling back to synthetic dataset.")
    except Exception as e:
        print(f"Roboflow init failed ({e}). Falling back to synthetic dataset.")

    # ── Fallback: synthetic 4-class dataset for smoke-test ──────────────────
    _make_synthetic_dataset()


def _fix_data_yaml():
    """
    After Roboflow download, rewrite data.yaml so class names are exactly
    [Helmet, No-Helmet, Gloves, No-Gloves] regardless of the source dataset's
    original labels. If the downloaded dataset already has 4 classes we keep
    them; otherwise we pad / remap as best we can.
    """
    import yaml, re
    yaml_path = Path("dataset/data.yaml")
    if not yaml_path.exists():
        return
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    nc = cfg.get("nc", 0)
    if nc == 4:
        # Already 4 classes — force our canonical names
        cfg["names"] = ["Helmet", "No-Helmet", "Gloves", "No-Gloves"]
    elif nc == 2:
        cfg["names"] = ["Helmet", "No-Helmet"]
    # else keep whatever came down

    # Fix relative paths so training can find images
    cfg["path"] = str(Path("dataset").resolve())
    for key in ("train", "val", "test"):
        if key in cfg and cfg[key] and not Path(cfg[key]).is_absolute():
            # leave as-is — ultralytics resolves relative to `path`
            pass

    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"  data.yaml updated: nc={cfg['nc']}, names={cfg.get('names')}")

def _make_synthetic_dataset():
    """
    Creates a tiny synthetic dataset with solid-colour rectangles so the
    pipeline (train → val → export) runs end-to-end without real images.
    Real detection quality requires a real dataset — replace this with
    your Roboflow download or any YOLOv8-format dataset.
    """
    import cv2, numpy as np, random

    CLASSES = ["Helmet", "No-Helmet", "Gloves", "No-Gloves"]
    COLORS  = [(255,255,0),(0,0,255),(0,255,0),(0,165,255)]
    IMG_SZ  = 640

    for split, n in [("train", 80), ("valid", 20), ("test", 10)]:
        img_dir = Path(f"dataset/images/{split}")
        lbl_dir = Path(f"dataset/labels/{split}")
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            img = np.full((IMG_SZ, IMG_SZ, 3), 40, dtype=np.uint8)
            labels = []
            n_objs = random.randint(1, 3)
            for _ in range(n_objs):
                cls = random.randint(0, 3)
                x1  = random.randint(0, IMG_SZ - 80)
                y1  = random.randint(0, IMG_SZ - 80)
                w   = random.randint(40, 120)
                h   = random.randint(40, 120)
                x2, y2 = min(x1+w, IMG_SZ), min(y1+h, IMG_SZ)
                cv2.rectangle(img, (x1,y1), (x2,y2), COLORS[cls], -1)
                # YOLO format: cls cx cy w h (normalised)
                cx = ((x1+x2)/2) / IMG_SZ
                cy = ((y1+y2)/2) / IMG_SZ
                nw = (x2-x1) / IMG_SZ
                nh = (y2-y1) / IMG_SZ
                labels.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            cv2.imwrite(str(img_dir / f"img_{i:04d}.jpg"), img)
            (lbl_dir / f"img_{i:04d}.txt").write_text("\n".join(labels))

        print(f"  synthetic {split}: {n} images created")

    # Write data.yaml
    Path("dataset").mkdir(exist_ok=True)
    yaml_content = (
        "path: dataset\n"
        "train: images/train\n"
        "val:   images/valid\n"
        "test:  images/test\n\n"
        "nc: 4\n"
        "names: [Helmet, No-Helmet, Gloves, No-Gloves]\n"
    )
    Path("dataset/data.yaml").write_text(yaml_content)
    print("  data.yaml written")


# ── 4. Train ──────────────────────────────────────────────────────────────────
def train():
    print("\n[2/4] Starting YOLOv8s training …")
    model = YOLO("yolov8s.pt")
    results = model.train(
        data    = str(DATA_YAML),
        epochs  = 30,
        imgsz   = 640,
        batch   = 8,           # lower than 16 for CPU comfort
        workers = 2,
        project = "runs/detect",
        name    = "ppe_train",
        exist_ok= True,
        verbose = True,
    )
    return model


# ── 5. Validate ───────────────────────────────────────────────────────────────
def validate(model):
    print("\n[3/4] Running validation …")
    metrics = model.val()
    print("\n─── Validation Results ───")
    print(f"  mAP50        : {metrics.box.map50:.4f}")
    print(f"  mAP50-95     : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print("──────────────────────────")
    return metrics


# ── 6. Export weights ─────────────────────────────────────────────────────────
def export_weights():
    print("\n[4/4] Copying best.pt to project root …")
    src = RUNS_DIR / "weights" / "best.pt"
    if src.exists():
        shutil.copy(src, WEIGHTS)
        print(f"  Saved → {WEIGHTS.resolve()}")
    else:
        print(f"  WARNING: {src} not found. Check training output.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_dataset()
    model = train()
    validate(model)
    export_weights()
    print("\nDone! best.pt is ready for the FastAPI server.")
    print("Start the server with: uvicorn main:app --reload")
