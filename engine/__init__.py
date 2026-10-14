"""
Engine Package: High-Fidelity Vectorization & Animation Engine
Specialized in flat-art, cel-shaded, and AI-generated video (Anim8 architecture).
"""

from .config import (
    STYLE_PRESETS,
    MAX_UPLOAD_IMAGE_BYTES,
    MAX_UPLOAD_VIDEO_BYTES,
    GLOBAL_PALETTE_THRESHOLD,
    DEFAULT_MIN_PATH_AREA,
    VECTORIZE_MAX_SIDE,
    DEFAULT_BEZIER_SIMPLIFY_TOLERANCE,
    DEFAULT_SEMANTIC_PROXIMITY,
    COLOR_METRICS,
    resolve_color_metric,
)
from .preprocessor import clean_frame_hybrid
from .palette import (
    consolidate_palette,
    extract_video_master_palette,
    quantize_image_to_palette,
    hex_to_rgb,
    rgb_to_hex,
    hex_to_lab,
    lab_distance,
    filter_transient_palette_colors,
)
from .svg_parser import (
    parse_and_structure_svg,
    compute_path_bbox,
    round_svg_coordinates,
    simplify_bezier_path,
    cluster_paths_semantically,
    bboxes_connect,
    scale_path_d,
    scale_translate,
    rescale_parsed_svg,
    apply_recolors_to_svg,
    PATH_TAG_RE,
    DEFAULT_MIN_PATH_AREA,
    DEFAULT_SEMANTIC_PROXIMITY,
)
from .vectorizer import vectorize_image_array, vectorize_image_file
from .flipbook import (
    vectorize_video_flipbook,
    build_animated_svg,
    bbox_iou,
    assign_stable_track_ids,
    stabilize_flipbook_z_order,
    stabilize_flipbook_group_ids,
)

__all__ = [
    "STYLE_PRESETS",
    "MAX_UPLOAD_IMAGE_BYTES",
    "MAX_UPLOAD_VIDEO_BYTES",
    "GLOBAL_PALETTE_THRESHOLD",
    "DEFAULT_MIN_PATH_AREA",
    "DEFAULT_BEZIER_SIMPLIFY_TOLERANCE",
    "DEFAULT_SEMANTIC_PROXIMITY",
    "VECTORIZE_MAX_SIDE",
    "COLOR_METRICS",
    "resolve_color_metric",
    "clean_frame_hybrid",
    "consolidate_palette",
    "extract_video_master_palette",
    "quantize_image_to_palette",
    "hex_to_rgb",
    "rgb_to_hex",
    "hex_to_lab",
    "lab_distance",
    "filter_transient_palette_colors",
    "parse_and_structure_svg",
    "compute_path_bbox",
    "round_svg_coordinates",
    "simplify_bezier_path",
    "cluster_paths_semantically",
    "bboxes_connect",
    "scale_path_d",
    "scale_translate",
    "rescale_parsed_svg",
    "apply_recolors_to_svg",
    "PATH_TAG_RE",
    "DEFAULT_MIN_PATH_AREA",
    "vectorize_image_array",
    "vectorize_image_file",
    "vectorize_video_flipbook",
    "build_animated_svg",
    "bbox_iou",
    "assign_stable_track_ids",
    "stabilize_flipbook_z_order",
    "stabilize_flipbook_group_ids",
]
