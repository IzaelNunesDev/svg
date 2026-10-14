"""
Engine Configuration: Presets and Vectorization Constants
Defines standard parameters for different illustration & animation styles.
"""

from typing import Dict, Any

# ---- Upload / Safety Limits ----
MAX_UPLOAD_IMAGE_BYTES: int = 25 * 1024 * 1024    # 25 MB
MAX_UPLOAD_VIDEO_BYTES: int = 200 * 1024 * 1024   # 200 MB

# Fallback unified palette threshold (weighted redmean RGB distance, see
# palette.color_distance). Per-preset `color_merge_threshold` takes precedence;
# the flipbook global consolidation reuses the active preset's value.
GLOBAL_PALETTE_THRESHOLD: float = 14.0

# Vector-side speckle floor: non-background paths with bbox area below this
# (SVG units^2) are dropped in svg_parser. Tunable per preset below.
DEFAULT_MIN_PATH_AREA: float = 4.0

# Bézier curve simplification tolerance: cubic curves with control point
# deviation below this threshold (SVG units/px) are simplified to straight lines,
# and consecutive collinear segments are merged (inspired by MARVEL/Mang2Vec).
DEFAULT_BEZIER_SIMPLIFY_TOLERANCE: float = 0.5

# Spatial proximity margin in SVG units: bounding boxes separated by at most this
# are considered connected components of the same semantic cluster (Vector Prism).
DEFAULT_SEMANTIC_PROXIMITY: float = 12.0

# Tracing resolution cap: inputs larger than this (longest side, px) are
# downscaled with INTER_AREA before VTracer. Flat geometric art gains nothing
# from 1792px tracing — it only records pixel stair-steps as Bézier nodes and
# explodes path counts. Tunable per preset via `max_raster_side` (0 disables).
VECTORIZE_MAX_SIDE: int = 960

# Consolidation distance metrics: 'redmean' (weighted RGB, fast, default) or
# 'lab' (CIE Delta E, perceptual). One metric is used end-to-end per run
# (master palette, per-cel and global consolidation) so near-duplicate shades
# are merged identically at every stage. Tracking gates stay in redmean space.
COLOR_METRICS = ("redmean", "lab")


def resolve_color_metric(
    custom_options: Dict[str, Any] | None = None,
    preset_key: str = "duolingo_flat",
) -> str:
    """Resolves the palette metric: custom override wins, else the preset's, else redmean."""
    if custom_options:
        try:
            custom_m = str(custom_options.get("color_metric", "")).lower()
        except (TypeError, ValueError, AttributeError):
            custom_m = ""
        if custom_m in COLOR_METRICS:
            return custom_m
    try:
        preset_m = str(STYLE_PRESETS.get(preset_key, {}).get("color_metric", "redmean")).lower()
    except (TypeError, ValueError, AttributeError):
        preset_m = "redmean"
    return preset_m if preset_m in COLOR_METRICS else "redmean"


STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "duolingo_flat": {
        "name": "Duolingo / Flat Cel",
        "description": "Cores sólidas limpas, curvas suaves e sem dentes de serra (estilo Anim8).",
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "spline",
        "filter_speckle": 12,
        "color_precision": 6,
        "layer_difference": 22,
        "corner_threshold": 72,
        "length_threshold": 6.0,
        "max_iterations": 10,
        "splice_threshold": 55,
        "path_precision": 2,
        "bilateral_d": 5,
        "bilateral_sigma": 35.0,
        "meanshift_sp": 10,
        "meanshift_sr": 28,
        "color_merge_threshold": 18.0,  # Distância redmean RGB ponderada p/ fusão de tons
        "color_metric": "redmean",      # Espaço da consolidação: 'redmean' (rápido) ou 'lab' (perceptual)
        "unsharp_strength": 0.0,        # Flat chapado não tem motion blur; unsharp cria halos/franjas
        "alpha_threshold": 127,         # Snap do alpha p/ opaco/transparente (0 preserva suave)
        "min_path_area": 4.0,           # Área mínima de bbox p/ manter path (SVG units²)
        "bezier_simplify_tolerance": 0.6,
    },
    "action_motion": {
        "name": "⚡ Ação Rápida / Anti-Smear",
        "description": "Máxima fidelidade para movimentos bruscos e rotações rápidas, preservando silhuetas.",
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "spline",
        "filter_speckle": 6,
        "color_precision": 8,
        "layer_difference": 14,
        "corner_threshold": 55,
        "length_threshold": 3.5,
        "max_iterations": 12,
        "splice_threshold": 45,
        "path_precision": 2,
        "bilateral_d": 5,
        "bilateral_sigma": 30.0,
        "meanshift_sp": 5,
        "meanshift_sr": 14,
        "color_merge_threshold": 12.0,
        "color_metric": "redmean",
        "unsharp_strength": 0.8,
        "alpha_threshold": 0,  # Ação preserva alpha suave (motion blur / glow)
        "min_path_area": 2.0,
        "bezier_simplify_tolerance": 0.4,
    },
    "detailed_vector": {
        "name": "Vetor Detalhado",
        "description": "Preserva detalhes menores e mais nuances de cores.",
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "spline",
        "filter_speckle": 6,
        "color_precision": 8,
        "layer_difference": 14,
        "corner_threshold": 50,
        "length_threshold": 3.0,
        "max_iterations": 12,
        "splice_threshold": 45,
        "path_precision": 2,
        "bilateral_d": 5,
        "bilateral_sigma": 35.0,
        "meanshift_sp": 4,
        "meanshift_sr": 12,
        "color_merge_threshold": 8.0,
        "color_metric": "redmean",
        "unsharp_strength": 0.4,
        "alpha_threshold": 0,  # Detalhado preserva alpha suave
        "min_path_area": 2.0,
        "bezier_simplify_tolerance": 0.3,
    },
    "minimal_logo": {
        "name": "Logotipo & Formas Puras",
        "description": "Geometria simplificada com número mínimo de pontos de controle.",
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "spline",
        "filter_speckle": 14,
        "color_precision": 7,
        "layer_difference": 20,
        "corner_threshold": 75,
        "length_threshold": 6.0,
        "max_iterations": 8,
        "splice_threshold": 50,
        "path_precision": 2,
        "bilateral_d": 7,
        "bilateral_sigma": 60.0,
        "meanshift_sp": 8,
        "meanshift_sr": 22,
        "color_merge_threshold": 18.0,
        "color_metric": "redmean",
        "unsharp_strength": 0.0,
        "alpha_threshold": 127,
        "min_path_area": 6.0,
        "bezier_simplify_tolerance": 0.7,
    },
    "sharp_polygon": {
        "name": "Polígono Preciso (Bordas Retas)",
        "description": "Arestas poligonais perfeitas sem curvas, ideal para gráficos geométricos.",
        "colormode": "color",
        "hierarchical": "stacked",
        "mode": "polygon",
        "filter_speckle": 8,
        "color_precision": 8,
        "layer_difference": 16,
        "corner_threshold": 60,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 2,
        "bilateral_d": 5,
        "bilateral_sigma": 40.0,
        "meanshift_sp": 6,
        "meanshift_sr": 16,
        "color_merge_threshold": 14.0,
        "color_metric": "redmean",
        "unsharp_strength": 0.5,
        "alpha_threshold": 127,
        "min_path_area": 4.0,
        "bezier_simplify_tolerance": 0.5,
    },
}
