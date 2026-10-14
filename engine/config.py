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
        "filter_speckle": 4,            # Reduzido de 12 p/ 4 (não apaga pupilas/detalhes dos olhos)
        "color_precision": 7,           # Aumentado de 6 p/ 7 (preserva sub-tons do bico/sombra)
        "layer_difference": 12,         # Reduzido de 22 p/ 12 (separa melhor sombra da cor base)
        "corner_threshold": 60,         # Equilíbrio entre cantos vivos do bico e curvas do corpo
        "length_threshold": 3.5,        # Reduzido de 6.0 p/ 3.5 (evita deformar formas pequenas)
        "max_iterations": 12,
        "splice_threshold": 45,
        "path_precision": 3,            # Aumentado de 2 p/ 3 (curvas Bézier muito mais fiéis)
        "bilateral_d": 5,
        "bilateral_sigma": 25.0,        # Limpa compressão MP4 sem borrar bordas
        "meanshift_sp": 0,              # DESLIGADO por padrão p/ flat art (evita derreter cantos)
        "meanshift_sr": 0,
        "color_merge_threshold": 10.0,  # Reduzido de 18.0 p/ 10.0 (evita fundir cores próximas intencionais)
        "color_metric": "lab",          # CIE Lab é muito mais preciso que redmean p/ flat art
        "unsharp_strength": 0.0,        # Mantém 0.0 (sem halos!)
        "alpha_threshold": 127,
        "min_path_area": 2.0,           # Reduzido de 4.0 p/ 2.0 (mantém brilhos dos olhos)
        "bezier_simplify_tolerance": 0.15, # Reduzido de 0.6 p/ 0.15 (não achata curvas em linhas retas)
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
