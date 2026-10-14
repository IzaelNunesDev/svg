"""
System API Router: Health, diagnostics, RAM metrics, and presets catalog.
"""

from fastapi import APIRouter
from engine.config import STYLE_PRESETS

router = APIRouter(prefix="/api", tags=["System"])


@router.get("/status")
async def api_status():
    """Returns engine health and hardware diagnostics."""
    stats = {
        "status": "online",
        "engine": "vtracer + bilateral-meanshift hybrid cel-shading + perceptual palette consolidation",
        "available_presets": list(STYLE_PRESETS.keys()),
    }
    try:
        import psutil
        ram = psutil.virtual_memory()
        stats["ram_used_gb"] = round(ram.used / (1024 ** 3), 2)
        stats["ram_total_gb"] = round(ram.total / (1024 ** 3), 2)
        stats["ram_percent"] = ram.percent
    except Exception:
        pass
    return stats


@router.get("/presets")
async def api_presets():
    """Returns vectorization presets catalog."""
    return {"presets": STYLE_PRESETS}
