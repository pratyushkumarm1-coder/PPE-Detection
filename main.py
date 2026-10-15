"""
PPE Detection API  —  Helmet & Gloves only
FastAPI backend powered by fine-tuned YOLOv8s (best.pt)

Start:  uvicorn main:app --reload
Test:   http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

# Restrict threads to drastically reduce RAM usage on free servers (prevents crashes)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

# Enforce single-threading for libraries
cv2.setNumThreads(1)
torch.set_num_threads(1)
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH   = Path("best.pt")
CONF_THRESHOLD = 0.25
IMGSZ          = 320
ALLOWED_MIME   = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("ppe")

# ── Shared model state ────────────────────────────────────────────────────────
class _State:
    model: YOLO | None = None

_state = _State()


@asynccontextmanager
async def lifespan(_: FastAPI):
    weights = str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else "yolov8s.pt"
    log.info(f"Loading model: {weights}")
    _state.model = YOLO(weights)
    log.info("Model ready ✓")
    yield
    _state.model = None


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PPE Detection API",
    description="Detect **Helmet / No-Helmet / Gloves / No-Gloves** in images.\n\nUse `/docs` to test via Swagger UI.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]   # [x_min, y_min, x_max, y_max]

class DetectResponse(BaseModel):
    image_width: int
    image_height: int
    total_detections: int
    detections: List[Detection]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _guard_mime(f: UploadFile):
    if f.content_type not in ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported type: {f.content_type}")

async def _decode(f: UploadFile) -> np.ndarray:
    raw = await f.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "Cannot decode image")
    return img

def _infer(img: np.ndarray):
    if _state.model is None:
        raise HTTPException(503, "Model not loaded")
    # classes=[0, 1] limits detection to Helmet (0) and No-Helmet (1)
    return _state.model.predict(source=img, conf=CONF_THRESHOLD, classes=[0, 1], imgsz=IMGSZ, verbose=False)[0]

def _parse(r) -> tuple[List[Detection], int, int]:
    h, w = r.orig_shape
    dets = []
    if r.boxes:
        for b in r.boxes:
            dets.append(Detection(
                class_name = r.names[int(b.cls[0])],
                confidence = round(float(b.conf[0]), 4),
                bbox       = [round(v, 2) for v in b.xyxy[0].tolist()],
            ))
    return dets, int(w), int(h)

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>PPE Detection - Image Upload</title>
    <style>
        body { font-family: system-ui, sans-serif; text-align: center; background: #111; color: #fff; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: #222; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .image-wrapper { margin: 20px auto; max-width: 100%; border-radius: 8px; overflow: hidden; min-height: 200px; background: #1a1a1a; display: flex; align-items: center; justify-content: center; }
        img#result { max-width: 100%; height: auto; object-fit: contain; }
        .file-upload { display: inline-block; padding: 12px 24px; cursor: pointer; border-radius: 6px; font-weight: bold; background: #333; color: white; border: 2px dashed #555; transition: 0.3s; width: 80%; max-width: 400px; margin-bottom: 20px; }
        .file-upload:hover { background: #444; border-color: #10a37f; }
        input[type="file"] { display: none; }
        button { padding: 14px 28px; font-size: 16px; margin: 10px; cursor: pointer; border-radius: 6px; border: none; font-weight: bold; background: #10a37f; color: white; transition: 0.2s; }
        button:hover { background: #0e8c6d; }
        button:disabled { background: #555; cursor: not-allowed; }
        .hidden { display: none !important; }
        .status { margin-top: 15px; font-family: monospace; padding: 10px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>PPE Detection 🪖</h1>
        <p>Upload a photo to detect helmets and gloves.</p>
        
        <label class="file-upload">
            📸 Select an Image
            <input type="file" id="imageInput" accept="image/*">
        </label>
        <p id="fileName" style="color: #aaa; font-size: 0.9em;"></p>

        <div>
            <button id="detectBtn" disabled>Detect PPE</button>
        </div>
        
        <div id="statusMsg" class="status hidden"></div>

        <div class="image-wrapper hidden" id="imageWrapper">
            <img id="result" />
        </div>
    </div>

    <script>
        const imageInput = document.getElementById('imageInput');
        const fileName = document.getElementById('fileName');
        const detectBtn = document.getElementById('detectBtn');
        const resultImg = document.getElementById('result');
        const imageWrapper = document.getElementById('imageWrapper');
        const statusMsg = document.getElementById('statusMsg');

        // When user selects a file
        imageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                // Enforce 2MB limit on the original file
                if (file.size > 2 * 1024 * 1024) {
                    alert("Please select an image under 2MB.");
                    imageInput.value = ""; // Clear selection
                    fileName.innerText = "";
                    detectBtn.disabled = true;
                    return;
                }
                
                fileName.innerText = "Selected: " + file.name;
                detectBtn.disabled = false;
                
                // Show a preview immediately
                const objUrl = URL.createObjectURL(file);
                resultImg.src = objUrl;
                imageWrapper.classList.remove('hidden');
                statusMsg.classList.add('hidden');
            }
        });

        // Resize image before uploading to make processing blazing fast
        async function resizeImageToBlob(file, maxDimension = 800) {
            return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => {
                    let width = img.width;
                    let height = img.height;
                    
                    if (width > maxDimension || height > maxDimension) {
                        const ratio = Math.min(maxDimension / width, maxDimension / height);
                        width = Math.round(width * ratio);
                        height = Math.round(height * ratio);
                    }

                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, width, height);
                    
                    canvas.toBlob(resolve, 'image/jpeg', 0.8);
                };
                img.src = URL.createObjectURL(file);
            });
        }

        // When user clicks Detect
        detectBtn.addEventListener('click', async () => {
            const file = imageInput.files[0];
            if (!file) return;

            detectBtn.disabled = true;
            detectBtn.innerText = "Analyzing...";
            statusMsg.classList.add('hidden');
            
            // Resize the image to make upload and server-side OpenCV decoding extremely fast
            const fastBlob = await resizeImageToBlob(file, 800);
            
            const formData = new FormData();
            formData.append('file', fastBlob, 'upload.jpg');

            try {
                const response = await fetch('/detect/annotated', {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const imageBlob = await response.blob();
                    const imageUrl = URL.createObjectURL(imageBlob);
                    
                    if (resultImg.src && resultImg.src.startsWith('blob:')) {
                        URL.revokeObjectURL(resultImg.src);
                    }
                    
                    resultImg.src = imageUrl;
                    
                    statusMsg.innerText = "✅ Detection Complete!";
                    statusMsg.style.background = "rgba(16, 163, 127, 0.2)";
                    statusMsg.style.color = "#10a37f";
                } else {
                    const errText = await response.text();
                    statusMsg.innerText = `❌ API Error ${response.status}: ${errText}`;
                    statusMsg.style.background = "rgba(224, 40, 70, 0.2)";
                    statusMsg.style.color = "#e02846";
                }
            } catch (e) {
                statusMsg.innerText = `❌ Network Error: ${e.message}`;
                statusMsg.style.background = "rgba(224, 40, 70, 0.2)";
                statusMsg.style.color = "#e02846";
            }
            
            statusMsg.classList.remove('hidden');
            detectBtn.disabled = false;
            detectBtn.innerText = "Detect PPE";
        });
    </script>
</body>
</html>
"""



# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse(content=INDEX_HTML)

@app.get("/health", tags=["util"])
def health():
    return {
        "status":       "ok",
        "model_loaded": _state.model is not None,
        "weights":      str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else "yolov8s.pt (fallback)",
        "conf":         CONF_THRESHOLD,
    }


@app.post("/detect", response_model=DetectResponse, tags=["detection"],
          summary="Upload image → JSON detections")
async def detect(file: UploadFile = File(...)):
    _guard_mime(file)
    img   = await _decode(file)
    r     = _infer(img)
    dets, w, h = _parse(r)
    log.info(f"/detect → {len(dets)} objects  [{file.filename}]")
    return DetectResponse(image_width=w, image_height=h,
                          total_detections=len(dets), detections=dets)


@app.post("/detect/annotated", tags=["detection"],
          summary="Upload image → annotated JPEG",
          responses={200: {"content": {"image/jpeg": {}}}})
async def detect_annotated(file: UploadFile = File(...)):
    _guard_mime(file)
    img  = await _decode(file)
    r    = _infer(img)
    ok, buf = cv2.imencode(".jpg", r.plot(), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(500, "Encoding failed")
    log.info(f"/detect/annotated → annotated [{file.filename}]")
    return Response(buf.tobytes(), media_type="image/jpeg")
