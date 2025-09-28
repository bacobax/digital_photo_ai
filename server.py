"""FastAPI application exposing the portrait framing service."""

from __future__ import annotations

import io
import json
import zipfile
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from portrait_framer import PortraitFramer, RunParameters


app = FastAPI(title="Portrait Framing Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
framer = PortraitFramer()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def root_page() -> FileResponse:
    return FileResponse("static/index.html")


def _read_image_from_upload(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")
    return image


@app.post("/process", summary="Process a portrait image", response_description="ZIP archive with crops and metadata")
async def process_image(
    file: UploadFile = File(...),
    save_debug: bool = False,
    target_w_over_h: float = 7.0 / 9.0,
    top_margin_ratio: float = 0.10,
    bottom_upper_ratio: float = 0.80,
    target_height_mm: float = 45.0,
    min_height_px: int = 540,
    min_width_px: int = 420,
    max_crown_to_chin_mm: float = 36.0,
    min_crown_to_chin_mm: float = 31.0,
    target_crown_to_chin_mm: float = 34.0,
    max_extra_padding_px: int = 600,
) -> StreamingResponse:
    """Run the portrait-framing pipeline on an uploaded image."""

    image = _read_image_from_upload(file)
    params = RunParameters(
        target_w_over_h=target_w_over_h,
        top_margin_ratio=top_margin_ratio,
        bottom_upper_ratio=bottom_upper_ratio,
        target_height_mm=target_height_mm,
        min_height_px=min_height_px,
        min_width_px=min_width_px,
        max_crown_to_chin_mm=max_crown_to_chin_mm,
        min_crown_to_chin_mm=min_crown_to_chin_mm,
        target_crown_to_chin_mm=target_crown_to_chin_mm,
        max_extra_padding_px=max_extra_padding_px,
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
            ok_final, final_buf = cv2.imencode(".jpg", face.final_image)
            ok_annotated, annotated_buf = cv2.imencode(".jpg", face.annotated_image)
            if not (ok_final and ok_annotated):
                raise HTTPException(status_code=500, detail="Failed to encode processed images")

            zf.writestr(f"{face_id}_stage6_final_balanced.jpg", final_buf.tobytes())
            zf.writestr(f"{face_id}_stage6_final_annotated.jpg", annotated_buf.tobytes())

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
