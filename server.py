"""FastAPI application exposing the portrait framing service."""

from __future__ import annotations

import io
import math
import json
import zipfile
from typing import List, Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from portrait_framer import PortraitFramer, RunParameters

# Frontend (React) build paths
PROJECT_ROOT = Path(__file__).parent
FRONTEND_DIR = PROJECT_ROOT / "build" / "client"  # expected by React Router SPA build
LEGACY_STATIC_DIR = PROJECT_ROOT / "static"        # fallback (existing static folder)

BUILD_ROOT = PROJECT_ROOT / "build"

# Decide where assets live: prefer SPA client assets, else raw build/assets, else legacy static/assets
if (FRONTEND_DIR / "assets").exists():
    ASSETS_DIR = FRONTEND_DIR / "assets"
elif (BUILD_ROOT / "assets").exists():
    ASSETS_DIR = BUILD_ROOT / "assets"
else:
    ASSETS_DIR = LEGACY_STATIC_DIR / "assets"

# Favicon resolution preference: build/client -> build root -> legacy static
if (FRONTEND_DIR / "favicon.ico").exists():
    FAVICON_PATH = str(FRONTEND_DIR / "favicon.ico")
elif (BUILD_ROOT / "favicon.ico").exists():
    FAVICON_PATH = str(BUILD_ROOT / "favicon.ico")
else:
    FAVICON_PATH = str(LEGACY_STATIC_DIR / "favicon.ico")

# Prefer SPA build index if present, else fallback to legacy static index
if (FRONTEND_DIR / "index.html").exists():
    CLIENT_INDEX = str(FRONTEND_DIR / "index.html")
else:
    CLIENT_INDEX = str(LEGACY_STATIC_DIR / "index.html")

app = FastAPI(title="Portrait Framing Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
framer = PortraitFramer()

# Serve assets from whichever resolved directory exists
if ASSETS_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(ASSETS_DIR), html=False),
        name="assets",
    )

# Keep legacy /static available if present (useful for old index or images)
if LEGACY_STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(LEGACY_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root_page() -> FileResponse:
    if not Path(CLIENT_INDEX).exists():
        raise HTTPException(status_code=500, detail="Frontend index.html not found. Build the client or provide static/index.html.")
    return FileResponse(CLIENT_INDEX)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    if Path(FAVICON_PATH).exists():
        return FileResponse(FAVICON_PATH)
    raise HTTPException(status_code=404, detail="favicon not found")


def _read_image_from_upload(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    return image


def _clamp_resize_scaling(value: Optional[float]) -> float:
    """Normalise user-provided scaling factor to [0, 1]."""

    if value is None:
        return 0.0
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value_f):
        return 0.0
    return max(0.0, min(1.0, value_f))


# Helper: encode JPEG bytes with DPI for physical size (e.g. 35x45mm)
def _encode_jpeg_with_dpi(bgr_np: np.ndarray, width_mm: float, height_mm: float, quality: int = 95) -> bytes:
    """Return JPEG bytes encoded with DPI so editors show correct 35×45 mm size.

    width_mm / height_mm are physical target sizes (e.g., 35×45).
    """
    # Convert BGR (OpenCV) to RGB (PIL)
    rgb = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    h_px, w_px = bgr_np.shape[:2]
    # Compute DPI dynamically
    dpi_x = (w_px / max(1e-6, width_mm)) * 25.4
    dpi_y = (h_px / max(1e-6, height_mm)) * 25.4
    buf = io.BytesIO()
    im.save(buf, format="JPEG", dpi=(dpi_x, dpi_y), quality=quality, subsampling=0)
    return buf.getvalue()


@app.post("/process", summary="Process a portrait image", response_description="ZIP archive with crops and metadata")
async def process_image(
    file: UploadFile = File(...),
    save_debug: bool = Form(False),
    target_w_over_h: float = Form(7.0 / 9.0),
    top_margin_ratio: float = Form(0.10),
    bottom_upper_ratio: float = Form(0.80),
    target_height_mm: float = Form(45.0),
    min_height_px: int = Form(540),
    min_width_px: int = Form(420),
    resize_scaling: float = Form(0.0),
    max_crown_to_chin_mm: float = Form(36.0),
    min_crown_to_chin_mm: float = Form(31.0),
    target_crown_to_chin_mm: float = Form(34.0),
    max_extra_padding_px: int = Form(600),
) -> StreamingResponse:
    """Run the portrait-framing pipeline on an uploaded image."""
    print("ARGS:")
    print(f"  save_debug: {save_debug}")
    print(f"  target_w_over_h: {target_w_over_h}")
    print(f"  top_margin_ratio: {top_margin_ratio}")
    print(f"  bottom_upper_ratio: {bottom_upper_ratio}")
    print(f"  target_height_mm: {target_height_mm}")
    print(f"  min_height_px: {min_height_px}")
    print(f"  min_width_px: {min_width_px}")
    print(f"  resize_scaling: {resize_scaling}")
    print(f"  max_crown_to_chin_mm: {max_crown_to_chin_mm}")
    print(f"  min_crown_to_chin_mm: {min_crown_to_chin_mm}")
    print(f"  target_crown_to_chin_mm: {target_crown_to_chin_mm}")
    print(f"  max_extra_padding_px: {max_extra_padding_px}")

    # Use normalized resize scaling (the value actually used by the pipeline)
   

    # Build a descriptive filename suffix with key parameters
    def _fmt(v: float, ndigits: int) -> str:
        # fixed decimals, replace dot with 'p' to avoid confusion with file extensions
        return f"{v:.{ndigits}f}".replace(".", "p")
    image = _read_image_from_upload(file)
    fname_suffix = (
        f"tmr-{_fmt(top_margin_ratio, 3)}_"
        f"bur-{_fmt(bottom_upper_ratio, 3)}_"
        f"tcc-{_fmt(target_crown_to_chin_mm, 1)}_"
        f"rs-{_fmt(resize_scaling, 2)}"
    )

    params = RunParameters(
        target_w_over_h=target_w_over_h,
        top_margin_ratio=top_margin_ratio,
        bottom_upper_ratio=bottom_upper_ratio,
        target_height_mm=target_height_mm,
        min_height_px=min_height_px,
        min_width_px=min_width_px,
        resize_scaling=resize_scaling,
        max_crown_to_chin_mm=max_crown_to_chin_mm,
        min_crown_to_chin_mm=min_crown_to_chin_mm,
        target_crown_to_chin_mm=target_crown_to_chin_mm,
        max_extra_padding_px=max_extra_padding_px,
        lock_ratio_after_resize=True
    )

    results = framer.process_array(
        image,
        logdir="./api_logs",
        save_debug=save_debug,
        params=params,
    )

    if not results:
        raise HTTPException(status_code=422, detail="No faces detected in the uploaded image")

    archive = io.BytesIO()
    metadata: List[dict] = []

    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, face in enumerate(results, start=1):
            face_id = f"face{idx:02d}"

            # Compute physical target sizes from params (35×45 mm by default)
            height_mm = float(target_height_mm)
            width_mm = float(target_w_over_h * target_height_mm)

            final_bytes = _encode_jpeg_with_dpi(face.final_image, width_mm, height_mm)
            annotated_bytes = _encode_jpeg_with_dpi(face.annotated_image, width_mm, height_mm)

            zf.writestr(f"{face_id}_{fname_suffix}.jpg", final_bytes)
            zf.writestr(f"{face_id}_{fname_suffix}.jpg", annotated_bytes)

            metadata.append({"face_id": face_id, **face.to_dict()})

        zf.writestr(
            "metadata.json",
            json.dumps({"faces": metadata}, indent=2).encode("utf-8"),
        )

    archive.seek(0)
    headers = {
        "Content-Disposition": 'attachment; filename="portrait_results.zip"'
    }
    return StreamingResponse(archive, media_type="application/zip", headers=headers)


# --- SPA fallback: return index.html for all other GET paths so React Router can handle routing ---
@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str) -> FileResponse:
    # Let asset requests fall through to mounted StaticFiles; for anything else serve index.html
    # If the request contains a dot it likely targets a file; return 404 to let StaticFiles handle
    if "." in full_path:
        raise HTTPException(status_code=404, detail="File not found")
    if not Path(CLIENT_INDEX).exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(CLIENT_INDEX)
