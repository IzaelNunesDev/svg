"""
Export API Router: SVG and Flipbook ZIP export endpoints.
"""

import io
import time
import json
import logging
import zipfile
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import Response

from engine import build_animated_svg
from engine.svg_parser import apply_recolors_to_svg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/export", tags=["Export"])


def _apply_recolors(svg_content: str, recolors: Dict[str, str]) -> str:
    """Applies live color replacements in a single pass (chained maps don't cascade)."""
    if not svg_content or not recolors or not isinstance(recolors, dict):
        return svg_content
    return apply_recolors_to_svg(svg_content, recolors)


@router.post("/svg")
async def api_export_svg(payload: Dict[str, Any] = Body(...)):
    """Exports a single SVG file with optional updated layer colors."""
    svg_content = payload.get("svg", "")
    filename = payload.get("filename", "vector_artwork.svg")
    if not svg_content:
        raise HTTPException(status_code=400, detail="Conteúdo SVG vazio.")

    # Apply any live color replacements if provided
    svg_content = _apply_recolors(svg_content, payload.get("recolors", {}))

    return Response(
        content=svg_content,
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.post("/zip")
async def api_export_zip(payload: Dict[str, Any] = Body(...)):
    """
    Exports all flipbook cels as individual .svg files + standalone animated SVG loop + JSON manifest.
    """
    cels = payload.get("cels", [])
    if not cels:
        raise HTTPException(status_code=400, detail="Nenhum cel para exportar.")

    try:
        fps = int(payload.get("fps", 15))
    except (TypeError, ValueError):
        fps = 15
    fps = max(1, min(60, fps))
    recolors = payload.get("recolors", {})

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Flipbook Manifest
        manifest = {
            "generator": "Anim8 Vector Engine",
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_cels": len(cels),
            "fps": fps,
            "cels": [
                {
                    "cel_index": c.get("cel_index"),
                    "timestamp": c.get("timestamp"),
                    "total_paths": c.get("total_paths"),
                    "colors": c.get("colors"),
                }
                for c in cels
            ]
        }
        zf.writestr("flipbook_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

        # 2. Individual Frame SVGs
        for cel in cels:
            c_idx = cel.get("cel_index", 1)
            svg_content = _apply_recolors(cel.get("svg", ""), recolors)
            zf.writestr(f"cels/cel_{c_idx:03d}.svg", svg_content)

        # 3. Single Standalone Animated SVG Loop. Cels were already recolored
        # above, so pass recolors=None to avoid applying the mapping twice.
        try:
            recolored_cels = [
                {**c, "svg": _apply_recolors(c.get("svg", ""), recolors)}
                for c in cels
            ]
            anim_svg = build_animated_svg(recolored_cels, fps=fps, recolors=None)
            if anim_svg:
                zf.writestr("animated_flipbook_loop.svg", anim_svg)
        except Exception:
            logger.exception("Não foi possível gerar animated_flipbook_loop.svg")

    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=anim8_vetor_flipbook.zip"}
    )
