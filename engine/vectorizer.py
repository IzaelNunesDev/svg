"""
Engine Vectorizer: VTracer Execution & Core Vector Pipeline
Orchestrates preprocessing, native curve-fitting, color consolidation, and SVG structure enrichment.
"""

import os
import time
import re
import base64
import logging
import tempfile
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import vtracer

from .config import STYLE_PRESETS, DEFAULT_MIN_PATH_AREA, VECTORIZE_MAX_SIDE, DEFAULT_BEZIER_SIMPLIFY_TOLERANCE, DEFAULT_SEMANTIC_PROXIMITY, resolve_color_metric
from .preprocessor import clean_frame_hybrid
from .palette import consolidate_palette, quantize_image_to_palette
from .svg_parser import parse_and_structure_svg, rescale_parsed_svg

logger = logging.getLogger(__name__)


def _prepare_input_image(img: np.ndarray) -> np.ndarray:
    """
    Normalizes any decoded image (grayscale, BGR, BGRA, gray+alpha, 16-bit)
    into an 8-bit BGR or BGRA array suitable for the preprocessing pipeline.
    - 16-bit inputs are scaled linearly (//257), NOT via NORM_MINMAX, so a
      fully-opaque alpha channel (all 65535) stays opaque instead of
      collapsing to 0 when min==max.
    - Gray+alpha (h,w,2) is expanded to BGRA, preserving transparency.
    """
    if img is None:
        return img
    # 1. Depth -> uint8 (linear for uint16; color-only normalize otherwise
    #    so the alpha channel is never stretched by MINMAX).
    if img.dtype == np.uint16:
        img = (img // np.uint16(257)).astype(np.uint8)
    elif img.dtype != np.uint8:
        if img.ndim == 3 and img.shape[2] == 4:
            bgr_part = img[:, :, :3]
            alpha_part = img[:, :, 3]
            try:
                bgr8 = cv2.normalize(bgr_part, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            except Exception:
                bgr8 = np.clip(bgr_part, 0, 255).astype(np.uint8)
            if np.issubdtype(np.asarray(alpha_part).dtype, np.floating):
                alpha8 = np.clip(np.asarray(alpha_part) * 255.0, 0, 255).astype(np.uint8)
            else:
                alpha8 = np.clip(np.asarray(alpha_part), 0, 255).astype(np.uint8)
            img = np.dstack([bgr8, alpha8])
        else:
            try:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            except Exception:
                img = np.clip(np.asarray(img), 0, 255).astype(np.uint8)
    # 2. Channel handling (now uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 2:
        # Gray+alpha -> BGRA (preserve transparency)
        gray = img[:, :, 0]
        alpha = img[:, :, 1]
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return np.dstack([bgr, alpha])
    return img


def _bleed_alpha_over_white(img: np.ndarray) -> np.ndarray:
    """
    Composites a BGRA image's RGB over white, PRESERVING the alpha channel.
    Transparent pixels usually store meaningless RGB (often black); without
    this, bilateral/mean-shift smear that black into edges and VTracer emits
    it as an opaque ghost path behind feathered shapes. Fully-transparent
    pixels keep alpha=0 so the tracer still drops them (transparency by
    omission); fringes blend toward white, matching the JPEG preview backdrop.
    Opaque images are returned unchanged (no copy when possible).
    """
    if img is None or getattr(img, "size", 0) == 0:
        return img
    if img.ndim != 3 or img.shape[2] != 4:
        return img
    if img[:, :, 3].min() == 255:
        return img
    flat = _composite_over_white(img)
    return np.dstack([flat, img[:, :, 3]])


def _binarize_alpha(img: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Snaps a BGRA alpha channel to fully opaque/transparent at `threshold`.
    Flat cel art has no half-opacity: soft alpha only produces stacked fringe
    rings the tracer cannot represent (no fill-opacity output). Thresholding
    keeps crisp stickers pixel-identical (binary alpha is a no-op) and cuts
    feathered edges at the 50% contour. `threshold <= 0` keeps soft alpha.
    Always copies when thresholding so the caller's array is never mutated.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return img
    if img.ndim != 3 or img.shape[2] != 4:
        return img
    try:
        thr = int(threshold)
    except (TypeError, ValueError):
        return img
    if thr <= 0 or thr > 255:
        return img
    out = img.copy()
    out[:, :, 3] = np.where(out[:, :, 3] >= thr, 255, 0).astype(np.uint8)
    return out


def _composite_over_white(img: np.ndarray) -> np.ndarray:
    """
    Flattens BGR/BGRA (or gray) onto a white background for JPEG preview.
    Transparent pixels become white instead of black fringes. Opaque images
    are returned unchanged (no copy when possible).
    """
    if img is None or getattr(img, "size", 0) == 0:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        comp = bgr * alpha + 255.0 * (1.0 - alpha)
        return np.clip(comp, 0, 255).astype(np.uint8)
    return img


def vectorize_image_array(
    img_bgr: np.ndarray,
    preset_key: str = "duolingo_flat",
    custom_options: Optional[Dict[str, Any]] = None,
    cel_index: int = 1,
    master_palette: Optional[List[str]] = None,
    include_original: bool = True
) -> Dict[str, Any]:
    """
    Complete vectorization pipeline for a single image array:
    0. Downscales huge inputs (longest side > VECTORIZE_MAX_SIDE) with INTER_AREA:
       kills pixel stair-steps before curve fitting and speeds up VTracer ~3x.
       Geometry is mapped back to the original coordinate space afterwards, and
       the`original_image` preview still uses the full-res source.
    1. Preprocesses image with hybrid bilateral + mean-shift cel-shading filter.
    2. Optionally pre-quantizes pixels to a fixed master palette (Lab distance) for
       temporal color consistency across video cels.
    3. Invokes VTracer native converter for Bézier curve fitting.
    4. Performs perceptual color consolidation (snapping near-duplicate edge artifacts).
    5. Enriches SVG with bounding boxes, background detection, and rounded coordinates.
    """
    start_time = time.time()
    resolved_preset = preset_key if preset_key in STYLE_PRESETS else "duolingo_flat"
    options = dict(STYLE_PRESETS[resolved_preset])
    if custom_options:
        options.update(custom_options)

    # Direct 2D grayscale arrays never reach _prepare_input_image (file path
    # only) and would crash mean-shift (3-channel required) — expand here.
    if img_bgr is not None and getattr(img_bgr, "ndim", 3) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    # 0. Downscale huge rasters (e.g. 1792px video frames) — vector output scales
    # cleanly anyway; tracing at full res only records pixel jaggies as nodes.
    # Geometry is mapped BACK to the original coordinate space after parsing
    # (rescale_parsed_svg) so bboxes stay consistent with the full-res
    # `original_image` preview, `video_info` dims and the tracking gates.
    work_img = img_bgr
    downscale = 1.0
    try:
        max_side = int(options.get("max_raster_side", VECTORIZE_MAX_SIDE))
    except (TypeError, ValueError):
        max_side = VECTORIZE_MAX_SIDE
    if max_side > 0 and img_bgr is not None and getattr(img_bgr, "size", 0):
        h0, w0 = img_bgr.shape[:2]
        longest = max(h0, w0)
        if longest > max_side:
            downscale = max_side / float(longest)
            work_img = cv2.resize(img_bgr, (int(round(w0 * downscale)), int(round(h0 * downscale))),
                                  interpolation=cv2.INTER_AREA)

    # 0b. Alpha bleed: composite translucent RGB over white (alpha kept) so
    # stored-black transparent pixels don't ghost into edges/tracing.
    work_img = _bleed_alpha_over_white(work_img)
    # 0c. Alpha snap for flat styles (no-op on opaque/binary-alpha inputs).
    try:
        alpha_thr = int(options.get("alpha_threshold", 127))
    except (TypeError, ValueError):
        alpha_thr = 127
    work_img = _binarize_alpha(work_img, alpha_thr)

    # 1. Edge-preserving Cel-Shading Preprocessing with Motion-Deblur
    d = int(options.get("bilateral_d", 5))
    sigma = float(options.get("bilateral_sigma", 35.0))
    sp = int(options.get("meanshift_sp", 6))
    sr = int(options.get("meanshift_sr", 16))
    unsharp = float(options.get("unsharp_strength", 0.0))
    cleaned_img = clean_frame_hybrid(work_img, d=d, sigma=sigma, sp=sp, sr=sr, unsharp_strength=unsharp)

    # 2. Optional master-palette pre-quantization (temporal color stability)
    if master_palette:
        cleaned_img = quantize_image_to_palette(cleaned_img, master_palette)

    # 3. VTracer Execution via Temp Files
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        tmp_in_path = tmp_in.name
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        if not cv2.imwrite(tmp_in_path, cleaned_img):
            raise IOError(f"Falha ao gravar imagem temporária: {tmp_in_path}")

        vtracer.convert_image_to_svg_py(
            tmp_in_path,
            tmp_out_path,
            colormode=options.get("colormode", "color"),
            hierarchical=options.get("hierarchical", "stacked"),
            mode=options.get("mode", "spline"),
            filter_speckle=int(options.get("filter_speckle", 12)),
            color_precision=int(options.get("color_precision", 6)),
            layer_difference=int(options.get("layer_difference", 18)),
            corner_threshold=int(options.get("corner_threshold", 65)),
            length_threshold=float(options.get("length_threshold", 4.5)),
            max_iterations=int(options.get("max_iterations", 10)),
            splice_threshold=int(options.get("splice_threshold", 45)),
            path_precision=int(options.get("path_precision", 2)),
        )

        with open(tmp_out_path, "r", encoding="utf-8") as f:
            raw_svg = f.read()

        # 4. Perceptual Color Consolidation (metric unified per run)
        raw_fills = re.findall(r'fill="([^"]+)"', raw_svg)
        merge_threshold = float(options.get("color_merge_threshold", 15.0))
        metric = resolve_color_metric(custom_options, resolved_preset)
        color_map, canonical_palette = consolidate_palette(raw_fills, threshold=merge_threshold, metric=metric)

        # 5. Parse & Structure SVG with Bounding Boxes & Clean Coordinates
        simplify_tol = float(options.get("bezier_simplify_tolerance", DEFAULT_BEZIER_SIMPLIFY_TOLERANCE))
        # min_path_area is documented in original-image px²: scale the cutoff
        # down to traced space so filtering behaves identically with/without
        # the pre-downscale above.
        min_area_traced = float(options.get("min_path_area", DEFAULT_MIN_PATH_AREA)) * downscale * downscale
        svg_content, layers, meta = parse_and_structure_svg(
            raw_svg,
            cel_index=cel_index,
            color_map=color_map,
            min_path_area=min_area_traced,
            bezier_simplify_tolerance=simplify_tol,
            # Margin documented in original px: scale to traced space like the area cutoff.
            semantic_proximity=float(DEFAULT_SEMANTIC_PROXIMITY) * downscale,
        )
        if downscale != 1.0:
            h_orig, w_orig = img_bgr.shape[:2]
            svg_content, layers, meta = rescale_parsed_svg(
                svg_content, layers, meta, 1.0 / downscale, w_orig, h_orig
            )
        else:
            meta["coordinate_scale"] = 1.0
        duration_ms = round((time.time() - start_time) * 1000, 1)

        # 6. Encode original image for frame-by-frame comparison view
        # (flatten transparency over white so JPEG has no black fringes)
        orig_data_url = ""
        if include_original:
            encode_src = _composite_over_white(img_bgr)
            _, buf = cv2.imencode('.jpg', encode_src, [cv2.IMWRITE_JPEG_QUALITY, 85])
            orig_data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode('ascii')

        return {
            "success": True,
            "cel_index": cel_index,
            "svg": svg_content,
            "original_image": orig_data_url,
            "layers": layers,
            "metadata": meta,
            "duration_ms": duration_ms,
            "preset_used": resolved_preset,
        }

    finally:
        for p in (tmp_in_path, tmp_out_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def vectorize_image_file(
    file_bytes: bytes,
    preset_key: str = "duolingo_flat",
    custom_options: Optional[Dict[str, Any]] = None,
    include_original: bool = True
) -> Dict[str, Any]:
    """Vectorizes an uploaded image file buffer (PNG, JPG, WEBP). Preserves alpha when present."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Formato de imagem inválido ou arquivo corrompido.")
    img = _prepare_input_image(img)
    return vectorize_image_array(
        img, preset_key, custom_options,
        cel_index=1, include_original=include_original
    )
