"""
Vectorize API Router: Image and Video Flipbook vectorization endpoints.
CPU-heavy work is dispatched to a worker thread so the event loop stays responsive.
"""

import logging
import os
import tempfile
from functools import partial
from typing import Any, Dict, Optional

import anyio
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException, Request

from engine import (
    vectorize_image_file,
    vectorize_video_flipbook,
    MAX_UPLOAD_IMAGE_BYTES,
    MAX_UPLOAD_VIDEO_BYTES,
)
from engine.config import STYLE_PRESETS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vectorize", tags=["Vectorize"])

# Concurrency guards: VTracer/KMeans/mean-shift are CPU- and RAM-heavy native
# loops. Without a cap, N simultaneous uploads OOM the worker or starve it.
# Tune to the host: ~1 video job per 2GB free RAM, images are far cheaper.
_IMAGE_SEMAPHORE = anyio.Semaphore(4)
_VIDEO_SEMAPHORE = anyio.Semaphore(2)

_INT_KEYS = (
    "filter_speckle", "color_precision", "layer_difference", "corner_threshold",
    "bilateral_d", "max_raster_side", "meanshift_sp", "meanshift_sr",
    "max_iterations", "splice_threshold", "path_precision", "alpha_threshold",
)
_FLOAT_KEYS = (
    "length_threshold", "color_merge_threshold", "unsharp_strength",
    "min_path_area", "bilateral_sigma", "bezier_simplify_tolerance",
)

# (min, max) clamps — VTracer/OpenCV abortam ou alocam demais fora dessas faixas.
_TUNE_RANGES = {
    "filter_speckle": (0, 100),
    "color_precision": (1, 10),
    "layer_difference": (0, 50),
    "corner_threshold": (0, 180),
    "bilateral_d": (0, 15),
    "max_raster_side": (0, 2048),
    "meanshift_sp": (0, 30),
    "meanshift_sr": (0, 80),
    "max_iterations": (1, 20),
    "splice_threshold": (1, 100),
    "path_precision": (0, 5),
    "length_threshold": (0.0, 20.0),
    "color_merge_threshold": (0.0, 100.0),
    "unsharp_strength": (0.0, 2.0),
    "min_path_area": (0.0, 500.0),
    "bilateral_sigma": (0.0, 150.0),
    "bezier_simplify_tolerance": (0.0, 5.0),
    "alpha_threshold": (0, 255),
}


def _clamp(key: str, value: Any) -> Any:
    bounds = _TUNE_RANGES.get(key)
    if bounds is None:
        return value
    lo, hi = bounds
    try:
        return max(lo, min(hi, value))
    except TypeError:
        return value


async def _read_upload_capped(upload: UploadFile, limit_bytes: int) -> bytes:
    """
    Reads an upload in 1MB chunks, aborting with 413 as soon as the limit
    is exceeded — avoids buffering a 2GB body when the cap is 200MB.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo excede o limite de {limit_bytes // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _reject_if_content_length_exceeded(request: Request, limit_bytes: int) -> None:
    """Early 413 from the Content-Length header (multipart overhead tolerado)."""
    try:
        clen = request.headers.get("content-length")
    except Exception:
        return
    if not clen:
        return
    try:
        if int(clen) > limit_bytes + (5 * 1024 * 1024):
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo excede o limite de {limit_bytes // (1024 * 1024)} MB.",
            )
    except ValueError:
        return


async def vectorize_params(
    preset: str = Form("duolingo_flat"),
    filter_speckle: Optional[int] = Form(None),
    color_precision: Optional[int] = Form(None),
    layer_difference: Optional[int] = Form(None),
    corner_threshold: Optional[int] = Form(None),
    length_threshold: Optional[float] = Form(None),
    bilateral_d: Optional[int] = Form(None),
    bilateral_sigma: Optional[float] = Form(None),
    meanshift_sp: Optional[int] = Form(None),
    meanshift_sr: Optional[int] = Form(None),
    color_merge_threshold: Optional[float] = Form(None),
    unsharp_strength: Optional[float] = Form(None),
    min_path_area: Optional[float] = Form(None),
    max_raster_side: Optional[int] = Form(None),
    bezier_simplify_tolerance: Optional[float] = Form(None),
    max_iterations: Optional[int] = Form(None),
    splice_threshold: Optional[int] = Form(None),
    path_precision: Optional[int] = Form(None),
    alpha_threshold: Optional[int] = Form(None),
    color_metric: Optional[str] = Form(None),
    include_original: bool = Form(True),
) -> Dict[str, Any]:
    """
    Shared dependency: maps optional tuning Form fields to engine custom options,
    avoiding parameter duplication between the image and video endpoints.
    Unknown presets are rejected (400) so `preset_used` in the response never
    lies about which preset actually ran. Numeric tunings are clamped to safe
    ranges instead of crashing VTracer/OpenCV.
    """
    if not isinstance(preset, str):
        # Chamada direta fora do Depends (defaults são marcadores Form) — usa padrão.
        preset = "duolingo_flat"
    if preset not in STYLE_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Preset desconhecido: {preset!r}. Disponíveis: {sorted(STYLE_PRESETS)}.",
        )
    custom: Dict[str, Any] = {}
    raw = {
        "filter_speckle": filter_speckle,
        "color_precision": color_precision,
        "layer_difference": layer_difference,
        "corner_threshold": corner_threshold,
        "bilateral_d": bilateral_d,
        "bilateral_sigma": bilateral_sigma,
        "meanshift_sp": meanshift_sp,
        "meanshift_sr": meanshift_sr,
        "length_threshold": length_threshold,
        "color_merge_threshold": color_merge_threshold,
        "unsharp_strength": unsharp_strength,
        "min_path_area": min_path_area,
        "max_raster_side": max_raster_side,
        "bezier_simplify_tolerance": bezier_simplify_tolerance,
        "max_iterations": max_iterations,
        "splice_threshold": splice_threshold,
        "path_precision": path_precision,
        "alpha_threshold": alpha_threshold,
    }
    for key, value in raw.items():
        if value is None:
            continue
        try:
            coerced = int(value) if key in _INT_KEYS else float(value)
        except (TypeError, ValueError):
            # Marcador Form(...) quando chamado fora do Depends — ignora.
            continue
        custom[key] = _clamp(key, coerced)
    if color_metric is not None and isinstance(color_metric, str):
        metric = color_metric.lower()
        if metric not in ("redmean", "lab"):
            raise HTTPException(
                status_code=400,
                detail=f"Métrica desconhecida: {color_metric!r}. Use 'redmean' ou 'lab'.",
            )
        custom["color_metric"] = metric
    return {"preset": preset, "custom_options": custom or None, "include_original": include_original}


@router.post("/image")
async def api_vectorize_image(
    request: Request,
    image: UploadFile = File(...),
    params: Dict[str, Any] = Depends(vectorize_params),
):
    """
    Vectorizes a single raster image (PNG, JPG, WEBP) into clean SVG Bézier splines.
    Applies cel-shading hybrid smoothing, perceptual palette consolidation, and bounding box computation.
    """
    _reject_if_content_length_exceeded(request, MAX_UPLOAD_IMAGE_BYTES)
    contents = await _read_upload_capped(image, MAX_UPLOAD_IMAGE_BYTES)

    try:
        async with _IMAGE_SEMAPHORE:
            result = await anyio.to_thread.run_sync(
                partial(
                    vectorize_image_file,
                    file_bytes=contents,
                    preset_key=params["preset"],
                    custom_options=params["custom_options"],
                    include_original=params["include_original"],
                )
            )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Erro na vetorização de imagem (%s)", image.filename)
        raise HTTPException(status_code=500, detail="Erro interno na vetorização da imagem.")


@router.post("/video")
async def api_vectorize_video(
    request: Request,
    video: UploadFile = File(...),
    target_fps: int = Form(15),
    max_frames: int = Form(45),
    params: Dict[str, Any] = Depends(vectorize_params),
):
    """
    Extracts frames uniformly across an MP4 video, establishes a global master color palette
    (pre-quantization for temporal color stability), and converts each frame into a cel SVG
    with Bézier curves for an Anim8-style flipbook.
    """
    _reject_if_content_length_exceeded(request, MAX_UPLOAD_VIDEO_BYTES)
    contents = await _read_upload_capped(video, MAX_UPLOAD_VIDEO_BYTES)

    suffix = os.path.splitext(video.filename or ".mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_vid:
        tmp_vid_path = tmp_vid.name

    try:
        with open(tmp_vid_path, "wb") as f:
            f.write(contents)

        async with _VIDEO_SEMAPHORE:
            flipbook_res = await anyio.to_thread.run_sync(
                partial(
                    vectorize_video_flipbook,
                    video_path=tmp_vid_path,
                    target_fps=max(1, min(30, int(target_fps))),
                    max_frames=max(1, min(120, int(max_frames))),
                    preset_key=params["preset"],
                    custom_options=params["custom_options"],
                    include_original=params["include_original"],
                )
            )
        return flipbook_res

    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("Erro no processamento de vídeo (%s)", video.filename)
        raise HTTPException(status_code=500, detail="Erro interno no processamento do vídeo.")
    finally:
        if os.path.exists(tmp_vid_path):
            try:
                os.remove(tmp_vid_path)
            except Exception:
                pass
