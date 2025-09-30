"""FastAPI application exposing the portrait framing service."""

from __future__ import annotations

import io
import math
import json
import zipfile
from typing import Iterable, List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from portrait_framer import FaceProcessingResult, PortraitFramer, RunParameters

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


def _normalise_pipeline_name(raw: Optional[str]) -> str:
    if not raw:
        return "closed_form"
    value = raw.strip().lower()
    if value in {"legacy", "original", "classic"}:
        return "legacy"
    if value in {"closed", "closed-form", "closed_form", "mm", "mm-budget", "mm_budget"}:
        return "closed_form"
    return "closed_form"


def _format_suffix_value(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", "p")


def _build_filename_suffix(
    *,
    pipeline_mode: str,
    target_w_over_h: float,
    resize_scaling: float,
    target_crown_to_chin_mm: float,
    top_margin_ratio: float,
    bottom_upper_ratio: float,
    min_top_mm: float,
    min_bottom_mm: float,
    shoulder_clearance_mm: float,
) -> str:
    parts = [
        f"mode-{pipeline_mode}",
        f"ratio-{_format_suffix_value(target_w_over_h, 3)}",
        f"tcc-{_format_suffix_value(target_crown_to_chin_mm, 1)}",
        f"rs-{_format_suffix_value(resize_scaling, 2)}",
    ]
    if pipeline_mode == "legacy":
        parts.extend(
            [
                f"tmr-{_format_suffix_value(top_margin_ratio, 3)}",
                f"bur-{_format_suffix_value(bottom_upper_ratio, 3)}",
            ]
        )
    else:
        parts.extend(
            [
                f"tmin-{_format_suffix_value(min_top_mm, 1)}",
                f"bmin-{_format_suffix_value(min_bottom_mm, 1)}",
                f"sc-{_format_suffix_value(shoulder_clearance_mm, 1)}",
            ]
        )
    return "_".join(parts)


def _log_arguments(pairs: Iterable[Tuple[str, object]]) -> None:
    print("ARGS:")
    for key, value in pairs:
        print(f"  {key}: {value}")


def _build_response(
    *,
    results: List[FaceProcessingResult],
    target_height_mm: float,
    target_w_over_h: float,
    fname_suffix: str,
) -> StreamingResponse:
    archive = io.BytesIO()
    metadata: List[dict] = []
    suffix = f"_{fname_suffix}" if fname_suffix else ""

    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, face in enumerate(results, start=1):
            face_id = f"face{idx:02d}"
            height_mm = float(target_height_mm)
            width_mm = float(target_w_over_h * target_height_mm)

            final_bytes = _encode_jpeg_with_dpi(face.final_image, width_mm, height_mm)
            annotated_bytes = _encode_jpeg_with_dpi(face.annotated_image, width_mm, height_mm)

            zf.writestr(f"final_{face_id}{suffix}.jpg", final_bytes)
            zf.writestr(f"annotated_{face_id}{suffix}.jpg", annotated_bytes)

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


def _process_request(
    *,
    upload: UploadFile,
    save_debug: bool,
    pipeline_mode: str,
    target_w_over_h: float,
    top_margin_ratio: float,
    bottom_upper_ratio: float,
    target_height_mm: float,
    min_height_px: int,
    min_width_px: int,
    resize_scaling: float,
    max_crown_to_chin_mm: float,
    min_crown_to_chin_mm: float,
    target_crown_to_chin_mm: float,
    max_extra_padding_px: int,
    min_top_mm: float,
    min_bottom_mm: float,
    shoulder_clearance_mm: float,
) -> StreamingResponse:
    normalized_mode = _normalise_pipeline_name(pipeline_mode)
    scaling = _clamp_resize_scaling(resize_scaling)
    params = RunParameters(
        target_w_over_h=float(target_w_over_h),
        top_margin_ratio=float(top_margin_ratio),
        bottom_upper_ratio=float(bottom_upper_ratio),
        target_height_mm=float(target_height_mm),
        min_height_px=int(min_height_px),
        min_width_px=int(min_width_px),
        resize_scaling=scaling,
        max_crown_to_chin_mm=float(max_crown_to_chin_mm),
        min_crown_to_chin_mm=float(min_crown_to_chin_mm),
        target_crown_to_chin_mm=float(target_crown_to_chin_mm),
        max_extra_padding_px=int(max_extra_padding_px),
        lock_ratio_after_resize=True,
        min_top_mm=float(min_top_mm),
        min_bottom_mm=float(min_bottom_mm),
        shoulder_clearance_mm=float(shoulder_clearance_mm),
        use_closed_form=normalized_mode != "legacy",
    )

    _log_arguments(
        [
            ("pipeline_mode", normalized_mode),
            ("use_closed_form", params.use_closed_form),
            ("save_debug", save_debug),
            ("target_w_over_h", params.target_w_over_h),
            ("top_margin_ratio", params.top_margin_ratio),
            ("bottom_upper_ratio", params.bottom_upper_ratio),
            ("target_height_mm", params.target_height_mm),
            ("min_height_px", params.min_height_px),
            ("min_width_px", params.min_width_px),
            ("resize_scaling", params.resize_scaling),
            ("max_crown_to_chin_mm", params.max_crown_to_chin_mm),
            ("min_crown_to_chin_mm", params.min_crown_to_chin_mm),
            ("target_crown_to_chin_mm", params.target_crown_to_chin_mm),
            ("max_extra_padding_px", params.max_extra_padding_px),
            ("min_top_mm", params.min_top_mm),
            ("min_bottom_mm", params.min_bottom_mm),
            ("shoulder_clearance_mm", params.shoulder_clearance_mm),
        ]
    )

    upload.file.seek(0)
    image = _read_image_from_upload(upload)

    results = framer.process_array(
        image,
        logdir="./api_logs",
        save_debug=save_debug,
        params=params,
    )

    if not results:
        raise HTTPException(status_code=422, detail="No faces detected in the uploaded image")

    fname_suffix = _build_filename_suffix(
        pipeline_mode=normalized_mode,
        target_w_over_h=params.target_w_over_h,
        resize_scaling=params.resize_scaling,
        target_crown_to_chin_mm=params.target_crown_to_chin_mm,
        top_margin_ratio=params.top_margin_ratio,
        bottom_upper_ratio=params.bottom_upper_ratio,
        min_top_mm=params.min_top_mm,
        min_bottom_mm=params.min_bottom_mm,
        shoulder_clearance_mm=params.shoulder_clearance_mm,
    )

    return _build_response(
        results=results,
        target_height_mm=params.target_height_mm,
        target_w_over_h=params.target_w_over_h,
        fname_suffix=fname_suffix,
    )


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
    """Run the legacy portrait-framing pipeline on an uploaded image."""
    defaults = RunParameters()
    return _process_request(
        upload=file,
        save_debug=save_debug,
        pipeline_mode="legacy",
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
        min_top_mm=defaults.min_top_mm,
        min_bottom_mm=defaults.min_bottom_mm,
        shoulder_clearance_mm=defaults.shoulder_clearance_mm,
    )


@app.post(
    "/api/v2/process",
    summary="Process a portrait image (v2)",
    response_description="ZIP archive with crops and metadata",
)
async def process_image_v2(
    file: UploadFile = File(...),
    save_debug: bool = Form(False),
    pipeline: str = Form("closed_form"),
    use_closed_form_flag: Optional[bool] = Form(None),
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
    min_top_mm: float = Form(4.0),
    min_bottom_mm: float = Form(8.0),
    shoulder_clearance_mm: float = Form(3.0),
) -> StreamingResponse:
    """Run the configurable portrait-framing pipeline with closed-form support."""
    pipeline_mode = _normalise_pipeline_name(pipeline)
    if use_closed_form_flag is not None:
        pipeline_mode = "closed_form" if use_closed_form_flag else "legacy"

    return _process_request(
        upload=file,
        save_debug=save_debug,
        pipeline_mode=pipeline_mode,
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
        min_top_mm=min_top_mm,
        min_bottom_mm=min_bottom_mm,
        shoulder_clearance_mm=shoulder_clearance_mm,
    )


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
