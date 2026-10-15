# PPE Detection — Helmet & Gloves MVP

YOLOv8s-based prototype. Detects **Helmet / No-Helmet / Gloves / No-Gloves** in images.

## Files

| File | Purpose |
|------|---------|
| `train.py` | Download dataset + train + export `best.pt` |
| `main.py` | FastAPI server (3 endpoints) |
| `requirements.txt` | Python deps |

## Step 1 — Install deps

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Step 2 — Train (creates best.pt)

```powershell
python train.py
```

> First run auto-downloads the dataset. If Roboflow key is not set it uses a synthetic dataset so the full pipeline still runs.  
> To use a real dataset: open `train.py`, replace `YOUR_API_KEY_HERE` with your free Roboflow API key.

## Step 3 — Start the API

```powershell
uvicorn main:app --reload
```

Open **http://localhost:8000/docs** — Swagger UI lets you upload images directly.

## Endpoints

| Method | URL | Returns |
|--------|-----|---------|
| GET | `/health` | JSON status |
| POST | `/detect` | JSON list of detections |
| POST | `/detect/annotated` | Annotated JPEG |

### Test with curl

```bash
# JSON detections
curl -X POST http://localhost:8000/detect -F "file=@worker.jpg"

# Annotated image saved to out.jpg
curl -X POST http://localhost:8000/detect/annotated -F "file=@worker.jpg" -o out.jpg
```

### Example JSON response

```json
{
  "image_width": 1280, "image_height": 720,
  "total_detections": 2,
  "detections": [
    {"class_name": "Helmet",    "confidence": 0.87, "bbox": [112.3, 45.1, 310.8, 198.5]},
    {"class_name": "No-Gloves","confidence": 0.63, "bbox": [402.0, 310.2, 580.4, 490.0]}
  ]
}
```

## Classes

| ID | Name | Meaning |
|----|------|---------|
| 0 | Helmet | Hard hat ON — compliant |
| 1 | No-Helmet | Hard hat missing — violation |
| 2 | Gloves | Gloves ON — compliant |
| 3 | No-Gloves | Gloves missing — violation |

## Config (top of main.py)

| Variable | Default | Notes |
|----------|---------|-------|
| `CONF_THRESHOLD` | `0.4` | Lower → more detections, more FP |
| `IMGSZ` | `640` | Must match training size |
| `WEIGHTS_PATH` | `best.pt` | Swap weights any time, restart server |
