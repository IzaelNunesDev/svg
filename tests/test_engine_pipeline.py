"""
Validation tests for the improved vectorization pipeline.
Run: python -m pytest tests/test_engine_pipeline.py -v
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from engine import (
    consolidate_palette,
    quantize_image_to_palette,
    vectorize_image_array,
    vectorize_video_flipbook,
    simplify_bezier_path,
    DEFAULT_BEZIER_SIMPLIFY_TOLERANCE,
    cluster_paths_semantically,
    assign_stable_track_ids,
    stabilize_flipbook_z_order,
    hex_to_lab,
    lab_distance,
    filter_transient_palette_colors,
    extract_video_master_palette,
    parse_and_structure_svg,
    scale_path_d,
    rescale_parsed_svg,
    apply_recolors_to_svg,
    resolve_color_metric,
    stabilize_flipbook_group_ids,
)
from engine.flipbook import _detect_shared_background
from engine.palette import color_distance, hex_to_rgb


# ---------- Palette consolidation ----------

def test_consolidate_merges_nearby_colors():
    color_map, palette = consolidate_palette(
        ["#FF0000", "#FE0101", "#FF0202", "#00FF00"], threshold=10.0
    )
    # All reds snap to the same canonical color; green stays separate
    canon_reds = {color_map["#FF0000"], color_map["#FE0101"], color_map["#FF0202"]}
    assert len(canon_reds) == 1
    assert len(palette) == 2


def test_consolidate_centroid_is_between_members():
    color_map, palette = consolidate_palette(["#000000", "#202020"], threshold=120.0)
    assert len(palette) == 1
    # Weighted centroid of black & #202020 -> ~#101010 (not the raw seed)
    canon = palette[0]
    rgb = int(canon[1:3], 16)
    assert 8 <= rgb <= 20


def test_consolidate_respects_distance_cap():
    # Colors far apart in RGB must NOT merge even in a long chain
    color_map, palette = consolidate_palette(
        ["#000000", "#101010", "#202020", "#FFFFFF"], threshold=20.0
    )
    assert color_map["#FFFFFF"] != color_map["#000000"]


def test_consolidate_empty_input():
    assert consolidate_palette([], threshold=18.0) == ({}, [])


# ---------- Pre-quantization ----------

def test_quantize_snaps_to_palette():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :8] = (30, 30, 30)    # near-black BGR
    img[:, 8:] = (240, 40, 40)   # near-blue BGR
    out = quantize_image_to_palette(img, ["#000000", "#0000FF"])
    assert set(map(tuple, out.reshape(-1, 3))) == {(0, 0, 0), (255, 0, 0)}


def test_quantize_noop_without_palette():
    img = np.full((4, 4, 3), 128, dtype=np.uint8)
    assert quantize_image_to_palette(img, []) is img


# ---------- Single image pipeline ----------

@pytest.fixture(scope="module")
def flat_image():
    """Synthetic flat cel-style image: bright shapes on white background."""
    img = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.circle(img, (100, 120), 60, (40, 140, 250), -1)   # orange-ish circle
    cv2.rectangle(img, (180, 60), (300, 200), (60, 60, 220), -1)  # green rect
    return img


def test_vectorize_image_basic(flat_image):
    res = vectorize_image_array(flat_image)
    assert res["success"] is True
    assert res["svg"].startswith("<svg")
    assert res["metadata"]["total_paths"] > 0
    assert res["original_image"].startswith("data:image/jpeg;base64,")


def test_vectorize_image_no_original(flat_image):
    res = vectorize_image_array(flat_image, include_original=False)
    assert res["original_image"] == ""


def test_vectorize_image_with_master_palette(flat_image):
    palette = ["#FFFFFF", "#FF8C28", "#DC3C3C"]
    res = vectorize_image_array(flat_image, master_palette=palette)
    assert res["success"] is True
    # Every produced color must belong to (a merge of) the master palette
    for c in res["metadata"]["colors"]:
        assert c.upper().startswith("#")


def test_vectorize_rgba_input():
    img = np.full((64, 64, 4), 255, dtype=np.uint8)
    img[:, :, 3] = 200
    cv2.circle(img, (32, 32), 20, (0, 0, 255, 255), -1)
    res = vectorize_image_array(img)
    assert res["success"] is True


# ---------- Video flipbook ----------

@pytest.fixture(scope="module")
def synthetic_video():
    """12-frame synthetic video: a moving square over a static background."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    writer = cv2.VideoWriter(
        tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), 12, (160, 120)
    )
    for i in range(12):
        frame = np.full((120, 160, 3), 255, dtype=np.uint8)
        x = 20 + i * 10
        cv2.rectangle(frame, (x, 40), (x + 30, 80), (40, 140, 250), -1)
        writer.write(frame)
    writer.release()
    yield tmp.name
    os.unlink(tmp.name)


def test_flipbook_uniform_sampling(synthetic_video):
    res = vectorize_video_flipbook(
        synthetic_video, target_fps=4, max_frames=3,
        include_original=False, preset_key="duolingo_flat"
    )
    assert res["success"] is True
    assert 2 <= res["total_cels"] <= 3
    # Uniform sampling: cels must span the whole clip, not just the beginning
    frames = [c["source_frame"] for c in res["cels"]]
    assert min(frames) <= 2 and max(frames) >= 8


def test_flipbook_master_palette_applied(synthetic_video):
    res = vectorize_video_flipbook(
        synthetic_video, target_fps=6, max_frames=4, include_original=False
    )
    assert len(res["master_palette"]) > 0
    # With pre-quantization, per-cel colors should stay NEAR the master/global
    # palette. Exact set membership is too strict: per-cel canonicals are
    # frequency-weighted centroids (VTracer rounding ±1-2 units), so compare
    # by perceptual redmean distance against the active global threshold.
    threshold = float(res.get("global_palette_threshold", 14.0))
    merged = set(res["unique_colors"]) | set(res["master_palette"])
    assert len(merged) > 0
    for cel in res["cels"]:
        for c in cel["colors"]:
            if c in merged:
                continue
            rgb = hex_to_rgb(c)
            nearest = min(color_distance(rgb, hex_to_rgb(m)) for m in merged)
            assert nearest <= threshold, f"{c} drifts {nearest:.1f} > {threshold}"


def test_flipbook_skips_original_payload(synthetic_video):
    res = vectorize_video_flipbook(
        synthetic_video, target_fps=6, max_frames=3, include_original=False
    )
    assert all(c["original_frame"] == "" for c in res["cels"])


# ---------- Bézier Pruning & Collinear Line Merging (MARVEL / Mang2Vec) ----------

def test_simplify_collinear_cubic_to_line():
    # Cubic Bézier with control points lying right on chord (0,0) -> (100,0)
    raw_d = "M0 0 C33 0 66 0 100 0 Z"
    simplified = simplify_bezier_path(raw_d, tolerance=0.5)
    assert "C" not in simplified
    assert "L100 0" in simplified
    assert simplified.startswith("M0 0")
    assert simplified.endswith("Z")


def test_simplify_rectangle_path():
    # A rectangle path with 4 cubic curves generated by VTracer
    raw_d = "M0 0 C33 0 66 0 100 0 C100 33 100 66 100 100 C67 100 34 100 0 100 C0 67 0 34 0 0 Z"
    simplified = simplify_bezier_path(raw_d, tolerance=0.5)
    # All 4 cubic curves should become straight lines, reducing size significantly
    assert "C" not in simplified
    assert len(simplified) < len(raw_d) * 0.6  # over 40% shorter
    assert "L100 0" in simplified
    assert "L100 100" in simplified
    assert "L0 100" in simplified


def test_simplify_preserves_true_curves():
    # Quarter circle arc from (0,0) to (100,100) with control points (0,55) and (45,100)
    raw_d = "M0 0 C0 55.2 44.8 100 100 100 Z"
    simplified = simplify_bezier_path(raw_d, tolerance=0.5)
    # True curves MUST be preserved as 'C' commands
    assert "C" in simplified


def test_merge_consecutive_collinear_lines():
    # Three collinear line segments: (0,0) -> (33,0) -> (66,0) -> (100,0)
    raw_d = "M0 0 L33 0 L66 0 L100 0 Z"
    simplified = simplify_bezier_path(raw_d, tolerance=0.5)
    assert simplified == "M0 0 L100 0 Z"


def test_simplify_noop_on_zero_tolerance():
    raw_d = "M0 0 C33 0 66 0 100 0 Z"
    simplified = simplify_bezier_path(raw_d, tolerance=0.0)
    assert simplified == raw_d


def test_vectorize_image_pruning_metadata():
    # Create image with a crisp rectangle where collinear edges occur
    img = np.full((120, 120, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (100, 100), (0, 0, 200), -1)
    res = vectorize_image_array(img, preset_key="duolingo_flat")
    assert res["success"] is True
    meta = res["metadata"]
    assert "nodes_before_simplification" in meta
    assert "nodes_after_simplification" in meta
    assert "pruned_nodes_count" in meta
    # Rectangles definitely contain collinear Béziers that get pruned
    assert meta["pruned_nodes_count"] > 0
    assert meta["nodes_after_simplification"] < meta["nodes_before_simplification"]


# ---------- Semantic Stratification & Z-Order Stabilization (Vector Prism) ----------

def test_semantic_clustering_unifies_connected_paths():
    # Path 1: Body at (10, 10, 50, 50)
    # Path 2: Head at (20, 5, 30, 25) overlapping with Body
    # Path 3: Far detached ball at (200, 200, 20, 20)
    # Path 4: Canvas background
    items = [
        {"bbox": {"x": 0, "y": 0, "w": 300, "h": 300}, "is_bg": True},
        {"bbox": {"x": 10, "y": 10, "w": 50, "h": 50}, "is_bg": False},
        {"bbox": {"x": 20, "y": 5, "w": 30, "h": 25}, "is_bg": False},
        {"bbox": {"x": 200, "y": 200, "w": 20, "h": 20}, "is_bg": False},
    ]
    group_ids = cluster_paths_semantically(items, proximity_margin=10.0)
    assert group_ids[0] == "g-background"
    # Body and Head connect and share the same cluster
    assert group_ids[1] == group_ids[2]
    assert group_ids[1].startswith("g-cluster-")
    # Detached ball is in a distinct cluster
    assert group_ids[3] != group_ids[1]
    assert group_ids[3].startswith("g-cluster-")


def test_tracking_tolerates_minor_color_shift():
    # Cel 1: Shape at (30, 30) with #FAFAFA
    # Cel 2: Same shape at (32, 31) with #FEFFFF (subtle lighting shift)
    cels = [
        {
            "cel_index": 1,
            "svg": '<svg><path id="cel001_lyr001" data-layer-id="cel001_lyr001" fill="#FAFAFA" /></svg>',
            "layers": [{
                "id": "cel001_lyr001",
                "color_hex": "#FAFAFA",
                "bbox": {"x": 30, "y": 30, "w": 40, "h": 40, "cx": 50, "cy": 50},
            }],
        },
        {
            "cel_index": 2,
            "svg": '<svg><path id="cel002_lyr001" data-layer-id="cel002_lyr001" fill="#FEFFFF" /></svg>',
            "layers": [{
                "id": "cel002_lyr001",
                "color_hex": "#FEFFFF",
                "bbox": {"x": 32, "y": 31, "w": 40, "h": 40, "cx": 52, "cy": 51},
            }],
        },
    ]
    distinct_tracks = assign_stable_track_ids(cels, color_threshold=24.0)
    assert distinct_tracks == 1
    assert cels[0]["layers"][0]["track_id"] == cels[1]["layers"][0]["track_id"]
    assert 'data-track-id="trk_001"' in cels[1]["svg"]


def test_z_order_stabilization():
    # Cel 1 has Layer A (trk_001, z=1) underneath Layer B (trk_002, z=2)
    # Cel 2 has them emitted in inverted order by VTracer: Layer B first, then Layer A
    svg1 = (
        '<svg viewBox="0 0 100 100"><defs></defs>'
        '<g class="anim8-group" data-group="g-cluster-01">'
        '<path id="c1_l1" data-layer-id="c1_l1" data-track-id="trk_001" />'
        '<path id="c1_l2" data-layer-id="c1_l2" data-track-id="trk_002" />'
        '</g></svg>'
    )
    svg2 = (
        '<svg viewBox="0 0 100 100"><defs></defs>'
        '<g class="anim8-group" data-group="g-cluster-01">'
        '<path id="c2_l2" data-layer-id="c2_l2" data-track-id="trk_002" />'
        '<path id="c2_l1" data-layer-id="c2_l1" data-track-id="trk_001" />'
        '</g></svg>'
    )
    cels = [
        {
            "cel_index": 1,
            "svg": svg1,
            "groups": ["g-cluster-01"],
            "layers": [
                {"id": "c1_l1", "track_id": "trk_001", "z_order": 1, "group": "g-cluster-01", "bbox": {"x": 10, "y": 10, "w": 50, "h": 50}},
                {"id": "c1_l2", "track_id": "trk_002", "z_order": 2, "group": "g-cluster-01", "bbox": {"x": 20, "y": 20, "w": 40, "h": 40}},
            ],
        },
        {
            "cel_index": 2,
            "svg": svg2,
            "groups": ["g-cluster-01"],
            "layers": [
                {"id": "c2_l2", "track_id": "trk_002", "z_order": 1, "group": "g-cluster-01", "bbox": {"x": 22, "y": 21, "w": 40, "h": 40}},
                {"id": "c2_l1", "track_id": "trk_001", "z_order": 2, "group": "g-cluster-01", "bbox": {"x": 12, "y": 11, "w": 50, "h": 50}},
            ],
        },
    ]
    stabilize_flipbook_z_order(cels)
    # In Cel 2, layers must be reordered so that trk_001 is before trk_002
    assert cels[1]["layers"][0]["id"] == "c2_l1"
    assert cels[1]["layers"][1]["id"] == "c2_l2"
    # And in cel 2 SVG string, c2_l1 must appear before c2_l2
    pos_l1 = cels[1]["svg"].find("c2_l1")
    pos_l2 = cels[1]["svg"].find("c2_l2")
    assert pos_l1 < pos_l2


# ---------- CIE Lab & Temporal Persistence (Im2Vec / SVGBuilder) ----------

def test_hex_to_lab_and_distance():
    lab_white = hex_to_lab("#FFFFFF")
    lab_near_white = hex_to_lab("#FAFAFA")
    lab_black = hex_to_lab("#000000")

    assert isinstance(lab_white, np.ndarray)
    assert lab_white.shape == (3,)

    # White and near-white should have small perceptual distance (< 10)
    d_white_near = lab_distance(lab_white, lab_near_white)
    assert d_white_near < 10.0

    # White and black should have large perceptual distance (> 200 in OpenCV Lab)
    d_contrast = lab_distance(lab_white, lab_black)
    assert d_contrast > 200.0


def test_consolidate_palette_lab_metric():
    # Subtle variations of blue should merge into 1 canonical blue under Lab
    raw_blues = ["#0055AA", "#0256AB", "#0154A9", "#0055AA"]
    color_map, canonical = consolidate_palette(raw_blues, threshold=15.0, metric="lab")
    assert len(canonical) == 1
    for b in raw_blues:
        assert color_map[b] == canonical[0]


def test_master_palette_temporal_persistence_filters_transient():
    # 5 frames: 4 frames are pure white (with a red square),
    # 1 frame has a tiny transient green speck (compression noise)
    frames = []
    for i in range(5):
        img = np.full((90, 160, 3), 255, dtype=np.uint8) # BGR white
        cv2.rectangle(img, (20, 20), (60, 60), (0, 0, 255), -1) # BGR red square
        if i == 2:
            # Single transient compression artifact in frame 2 only (3 pixels)
            cv2.rectangle(img, (140, 70), (142, 71), (0, 255, 0), -1) # BGR green speck
        frames.append(img)

    palette = extract_video_master_palette(frames, max_colors=8, temporal_filter=True, metric="lab")
    # Palette should contain White (#FFFFFF) and Red (#FF0000), but NOT transient Green (#00FF00)
    assert len(palette) >= 2
    # Ensure white and red are present
    has_white = any(lab_distance(hex_to_lab(c), hex_to_lab("#FFFFFF")) < 15.0 for c in palette)
    has_red = any(lab_distance(hex_to_lab(c), hex_to_lab("#FF0000")) < 15.0 for c in palette)
    assert has_white is True
    assert has_red is True
    # Ensure green was filtered out
    has_green = any(lab_distance(hex_to_lab(c), hex_to_lab("#00FF00")) < 20.0 for c in palette)
    assert has_green is False


# ---------- Regression: path commands, tag forms, rescale, recolor ----------

def test_simplify_preserves_smooth_and_arc_commands():
    # S/T/A commands must keep their coordinates (previously emitted bare).
    assert "T50 50" in simplify_bezier_path("M0 0 T50 50 T100 0 Z", tolerance=0.5)
    assert "T100 0" in simplify_bezier_path("M0 0 T50 50 T100 0 Z", tolerance=0.5)
    assert "S50 50 100 0" in simplify_bezier_path("M0 0 S50 50 100 0 Z", tolerance=0.5)
    assert "A30 30 0 0 1 100 100" in simplify_bezier_path(
        "M0 0 A30 30 0 0 1 100 100 Z", tolerance=0.5
    )


def test_simplify_converts_relative_smooth_commands_to_absolute():
    out = simplify_bezier_path("m10 10 t40 40 z", tolerance=0.5)
    assert "T50 50" in out


def test_parse_handles_explicit_close_path_tags():
    svg = (
        '<svg width="100" height="100" viewBox="0 0 100 100">'
        '<path d="M0 0 L50 0 L50 50 Z" fill="#FF0000"></path></svg>'
    )
    _, layers, meta = parse_and_structure_svg(svg)
    assert meta["total_paths"] == 1
    assert layers[0]["color_hex"] == "#FF0000"


def test_stabilize_preserves_explicit_close_tags():
    svg1 = (
        '<svg viewBox="0 0 100 100"><defs></defs>'
        '<g class="anim8-group" data-group="g-cluster-01">'
        '<path id="c1_l1" data-layer-id="c1_l1" data-track-id="trk_001" d="M0 0 L20 0 Z" />'
        '<path id="c1_l2" data-layer-id="c1_l2" data-track-id="trk_002" d="M0 0 L10 0 Z" />'
        '</g></svg>'
    )
    svg2 = (
        '<svg viewBox="0 0 100 100"><defs></defs>'
        '<g class="anim8-group" data-group="g-cluster-01">'
        '<path id="c2_l2" data-layer-id="c2_l2" data-track-id="trk_002" d="M0 0 L10 0 Z"></path>'
        '<path id="c2_l1" data-layer-id="c2_l1" data-track-id="trk_001" d="M0 0 L20 0 Z"></path>'
        '</g></svg>'
    )
    cels = [
        {
            "cel_index": 1, "svg": svg1, "groups": ["g-cluster-01"],
            "layers": [
                {"id": "c1_l1", "track_id": "trk_001", "z_order": 1, "group": "g-cluster-01"},
                {"id": "c1_l2", "track_id": "trk_002", "z_order": 2, "group": "g-cluster-01"},
            ],
        },
        {
            "cel_index": 2, "svg": svg2, "groups": ["g-cluster-01"],
            "layers": [
                {"id": "c2_l2", "track_id": "trk_002", "z_order": 1, "group": "g-cluster-01"},
                {"id": "c2_l1", "track_id": "trk_001", "z_order": 2, "group": "g-cluster-01"},
            ],
        },
    ]
    stabilize_flipbook_z_order(cels)
    # Reordered without dropping the explicit-close paths
    assert [l["id"] for l in cels[1]["layers"]] == ["c2_l1", "c2_l2"]
    assert "c2_l1" in cels[1]["svg"] and "c2_l2" in cels[1]["svg"]
    assert cels[1]["svg"].find("c2_l1") < cels[1]["svg"].find("c2_l2")


def test_scale_path_d_scales_coords_not_arc_flags():
    assert scale_path_d("M10 10 A5 5 0 0 1 20 20 Z", 2.0) == "M20 20 A10 10 0 0 1 40 40 Z"
    assert scale_path_d("M0 0 L10 0 Z", 1.0) == "M0 0 L10 0 Z"


def test_rescale_parsed_svg_restores_original_geometry():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 70 60" width="70.0" height="60.0">'
        '<defs></defs><g class="anim8-group" data-group="g-cluster-01">'
        '<path id="cel001_lyr001" class="vector-path-element" data-layer-id="cel001_lyr001" '
        'data-color="#FF0000" data-cel="1" data-points="4" data-bbox="5,5,20,10" '
        'data-center="15,10" data-is-bg="false" data-group="g-cluster-01" '
        'd="M5 5 L25 5 L25 15 L5 15 Z" fill="#FF0000" /></g></svg>'
    )
    layers = [{
        "id": "cel001_lyr001",
        "bbox": {"x": 5.0, "y": 5.0, "w": 20.0, "h": 10.0, "cx": 15.0, "cy": 10.0},
        "area": 200.0, "transform": "",
    }]
    meta = {"width": 70.0, "height": 60.0, "viewbox": "0 0 70 60", "svg_size_bytes": 0}
    out_svg, out_layers, out_meta = rescale_parsed_svg(svg, layers, meta, 2.0, 140, 120)
    assert out_meta["width"] == 140.0
    assert out_meta["height"] == 120.0
    assert out_meta["viewbox"] == "0 0 140 120"
    assert out_layers[0]["bbox"] == {"x": 10.0, "y": 10.0, "w": 40.0, "h": 20.0, "cx": 30.0, "cy": 20.0}
    assert out_layers[0]["area"] == 800.0
    assert 'd="M10 10 L50 10 L50 30 L10 30 Z"' in out_svg
    assert 'data-bbox="10.0,10.0,40.0,20.0"' in out_svg


def test_vectorize_large_image_reports_original_coordinates():
    img = np.full((1200, 1400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (1300, 1100), (0, 0, 200), -1)
    res = vectorize_image_array(img, include_original=False)
    assert res["metadata"]["width"] == 1400.0
    assert res["metadata"]["height"] == 1200.0
    assert res["metadata"]["viewbox"] == "0 0 1400 1200"
    assert res["metadata"]["coordinate_scale"] > 1.0
    assert res["layers"], "expected at least the background path"
    for layer in res["layers"]:
        bb = layer["bbox"]
        assert 0.0 <= bb["x"] <= 1400.0 and 0.0 <= bb["y"] <= 1200.0
        assert bb["x"] + bb["w"] <= 1400.5 and bb["y"] + bb["h"] <= 1200.5


def test_recolor_mappings_do_not_chain():
    svg = ('<path fill="#FF0000" data-color="#FF0000"/>'
           '<path fill="#0000FF" data-color="#0000FF"/>')
    out = apply_recolors_to_svg(svg, {"#FF0000": "#0000FF", "#0000FF": "#00FF00"})
    # ex-red became blue (2 attrs), original blue became green (2 attrs)
    assert out.count("#0000FF") == 2
    assert out.count("#00FF00") == 2
    assert "#FF0000" not in out


# ---------- Round 2: shared bg, alpha, metric, stable groups ----------

def _bg_cel(color, cel_index=1):
    return {
        "cel_index": cel_index,
        "svg": "<svg></svg>",
        "layers": [{
            "id": f"cel{cel_index:03d}_lyr001",
            "color_hex": color,
            "is_background": True,
            "bbox": {"x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0, "cx": 50.0, "cy": 50.0},
        }],
    }


def test_shared_background_tolerates_centroid_drift():
    cels = [_bg_cel("#FFFFFF", 1), _bg_cel("#FEFEFE", 2)]
    assert _detect_shared_background(cels) is not None


def test_shared_background_rejects_different_color():
    cels = [_bg_cel("#FFFFFF", 1), _bg_cel("#FF0000", 2)]
    assert _detect_shared_background(cels) is None


def test_crisp_transparent_sticker_has_no_background_path():
    img = np.zeros((120, 120, 4), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (90, 90), (0, 0, 255, 255), -1)
    res = vectorize_image_array(img, include_original=False)
    assert res["success"] is True
    assert res["metadata"]["total_paths"] == 1
    assert "#FF0000" in res["metadata"]["colors"]


def test_feathered_alpha_has_no_black_ghost():
    img = np.zeros((120, 120, 4), dtype=np.uint8)
    img[30:91, 30:91, 0:3] = (0, 0, 255)
    img[30:91, 30:91, 3] = 255
    img[:, :, 3] = cv2.GaussianBlur(img[:, :, 3], (9, 9), 3.0)
    res = vectorize_image_array(img, include_original=False)
    assert "#000000" not in res["metadata"]["colors"]
    assert "#FF0000" in res["metadata"]["colors"]
    assert res["metadata"]["total_paths"] <= 4


def test_resolve_color_metric_prefers_custom_then_preset():
    assert resolve_color_metric({"color_metric": "LAB"}) == "lab"
    assert resolve_color_metric({"color_metric": "bogus"}) == "redmean"
    assert resolve_color_metric(None, "duolingo_flat") == "redmean"
    assert resolve_color_metric(None, "nope") == "redmean"


def test_vectorize_with_lab_metric(flat_image):
    res = vectorize_image_array(
        flat_image, custom_options={"color_metric": "lab"}, include_original=False
    )
    assert res["success"] is True
    assert res["metadata"]["total_paths"] > 0


def test_vectorize_with_invalid_metric_falls_back(flat_image):
    res = vectorize_image_array(
        flat_image, custom_options={"color_metric": "bogus"}, include_original=False
    )
    assert res["success"] is True


def test_group_ids_stable_when_size_rank_flips():
    # Cel 1: red (trk_001, area 100) bigger than blue (trk_002, area 10).
    # Cel 2: blue (area 200) outgrows red (area 100) — size-rank naming
    # would flip ids, dominant-track naming must not.
    def _cel(idx, red_area, blue_area):
        red = {
            "id": f"c{idx}_red", "track_id": "trk_001", "group": "g-cluster-01",
            "bbox": {"x": 0.0, "y": 0.0, "w": float(red_area), "h": 1.0,
                     "cx": 0.0, "cy": 0.0},
        }
        blue = {
            "id": f"c{idx}_blue", "track_id": "trk_002", "group": "g-cluster-02",
            "bbox": {"x": 0.0, "y": 0.0, "w": float(blue_area), "h": 1.0,
                     "cx": 0.0, "cy": 0.0},
        }
        svg = (
            '<svg viewBox="0 0 300 10"><defs></defs>'
            f'<g class="anim8-group" data-group="g-cluster-01">'
            f'<path id="{red["id"]}" data-layer-id="{red["id"]}" data-group="g-cluster-01" '
            f'd="M0 0 L{red_area} 0 Z" /></g>'
            f'<g class="anim8-group" data-group="g-cluster-02">'
            f'<path id="{blue["id"]}" data-layer-id="{blue["id"]}" data-group="g-cluster-02" '
            f'd="M0 0 L{blue_area} 0 Z" /></g></svg>'
        )
        return {
            "cel_index": idx, "svg": svg, "groups": ["g-cluster-01", "g-cluster-02"],
            "layers": [red, blue],
        }

    cels = [_cel(1, 100, 10), _cel(2, 100, 200)]
    stabilize_flipbook_group_ids(cels)
    red_groups = [c["layers"][0]["group"] for c in cels]
    blue_groups = [c["layers"][1]["group"] for c in cels]
    assert red_groups[0] == red_groups[1]
    assert blue_groups[0] == blue_groups[1]
    assert red_groups[0] != blue_groups[0]
    for cel in cels:
        assert f'data-group="{cel["layers"][0]["group"]}"' in cel["svg"]
        assert f'data-group="{cel["layers"][1]["group"]}"' in cel["svg"]
        assert sorted(cel["groups"]) == sorted(
            [cel["layers"][0]["group"], cel["layers"][1]["group"]]
        )


def test_vectorize_grayscale_array():
    gray = np.full((80, 80), 200, dtype=np.uint8)
    cv2.circle(gray, (40, 40), 25, 60, -1)
    res = vectorize_image_array(gray, include_original=False)
    assert res["success"] is True
    assert res["metadata"]["total_paths"] > 0


def test_transient_filter_handles_rgba_frames():
    frames = []
    for _ in range(3):
        f = np.full((90, 160, 4), 255, dtype=np.uint8)
        cv2.rectangle(f, (20, 20), (60, 60), (0, 0, 255, 255), -1)
        frames.append(f)
    out = filter_transient_palette_colors(frames, ["#FFFFFF", "#FF0000"])
    assert "#FFFFFF" in out and "#FF0000" in out


def test_group_rename_survives_id_permutation():
    # Layers ordered blue-first force a swap mapping ({02->01, 01->02}):
    # sequential string replacement would collapse both groups into one.
    blue = {
        "id": "c1_blue", "track_id": "trk_002", "group": "g-cluster-02",
        "bbox": {"x": 0.0, "y": 0.0, "w": 10.0, "h": 1.0, "cx": 0.0, "cy": 0.0},
    }
    red = {
        "id": "c1_red", "track_id": "trk_001", "group": "g-cluster-01",
        "bbox": {"x": 0.0, "y": 0.0, "w": 100.0, "h": 1.0, "cx": 0.0, "cy": 0.0},
    }
    svg = (
        '<svg viewBox="0 0 300 10"><defs></defs>'
        '<g class="anim8-group" data-group="g-cluster-02">'
        '<path id="c1_blue" data-layer-id="c1_blue" data-group="g-cluster-02" '
        'd="M0 0 L10 0 Z" /></g>'
        '<g class="anim8-group" data-group="g-cluster-01">'
        '<path id="c1_red" data-layer-id="c1_red" data-group="g-cluster-01" '
        'd="M0 0 L100 0 Z" /></g></svg>'
    )
    cels = [{
        "cel_index": 1, "svg": svg, "groups": ["g-cluster-02", "g-cluster-01"],
        "layers": [blue, red],
    }]
    stabilize_flipbook_group_ids(cels)
    assert blue["group"] == "g-cluster-01"
    assert red["group"] == "g-cluster-02"
    assert 'data-group="g-cluster-01"' in cels[0]["svg"]
    assert 'data-group="g-cluster-02"' in cels[0]["svg"]



