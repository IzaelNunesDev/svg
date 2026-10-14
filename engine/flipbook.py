"""
Engine Flipbook: Video Frame Extraction, Multi-Cel Vectorization & Animated SVG Loop
Coordinates video frame sampling, extracts unified global color palette,
and builds standalone CSS-animated SVGs for export.
"""

import time
import re
import math
import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional, Callable

import cv2
import numpy as np

from .config import STYLE_PRESETS, GLOBAL_PALETTE_THRESHOLD, resolve_color_metric
from .svg_parser import apply_recolors_to_svg
from .vectorizer import vectorize_image_array
from .palette import extract_video_master_palette, consolidate_palette, color_distance, hex_to_rgb

logger = logging.getLogger(__name__)

# Number of sample frames used to estimate the master palette
PALETTE_SAMPLE_COUNT = 10

# Flat cartoon needs FEW canonical colors. 24 yields near-duplicate whites that
# compete frame-to-frame (eye white merges into background white in some cels
# and vanishes). 14 keeps one white, one black, plus flat mid-tones.
VIDEO_MASTER_COLORS = 14

# Minimum bbox IoU (same color) to consider two paths the same object
# across consecutive cels. Below this, a new track id is issued.
TRACK_IOU_THRESHOLD = 0.15

# Background is shareable across cels only if bboxes overlap almost exactly.
BG_SHARE_IOU_THRESHOLD = 0.9


def bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Intersection-over-Union of two AABB dicts with x/y/w/h keys."""
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    iy = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _infer_canvas_diagonal(cels: List[Dict[str, Any]]) -> float:
    """
    Infere a diagonal do canvas (px) a partir do viewBox do 1º cel ou da
    extensão máxima dos bboxes. Fallback 400px (clipe sintético 160x120*2).
    """
    try:
        first_svg = cels[0].get("svg", "") if cels else ""
        vb = re.search(r'viewBox="([\d\.\-eE\s]+)"', first_svg)
        if vb:
            parts = [float(x) for x in vb.group(1).split()]
            if len(parts) == 4 and parts[2] > 0 and parts[3] > 0:
                return math.hypot(parts[2], parts[3])
        max_x = max_y = 0.0
        for cel in cels:
            for layer in cel.get("layers", []):
                bb = layer.get("bbox", {})
                max_x = max(max_x, float(bb.get("x", 0)) + float(bb.get("w", 0)))
                max_y = max(max_y, float(bb.get("y", 0)) + float(bb.get("h", 0)))
        if max_x > 0 and max_y > 0:
            return math.hypot(max_x, max_y)
    except Exception:
        pass
    return 400.0


def assign_stable_track_ids(
    cels: List[Dict[str, Any]],
    color_threshold: float = 24.0,
    iou_threshold: float = TRACK_IOU_THRESHOLD,
    canvas_diagonal: Optional[float] = None,
) -> int:
    """
    Robust multi-attribute object tracking across consecutive cels (inspired by Vector Prism).
    Matches candidates based on:
    - Perceptual color proximity (weighted redmean RGB distance <= color_threshold)
    - Spatial bounding box IoU
    - Centroid distance NORMALIZADA pela diagonal do canvas (invariante à escala:
      160px e 960px usam o mesmo gate relativo em vez de 35px/150px fixos).
    Mutates each layer in place (`track_id`) and injects `data-track-id`
    into the cel SVG. Per-cel `id`/`data-layer-id` stay unique and untouched
    so the frontend selection contract is preserved.
    Returns the number of distinct tracks.
    """
    next_track = 1
    prev_layers: List[Dict[str, Any]] = []

    try:
        _diag_val = float(canvas_diagonal) if canvas_diagonal else 0.0
    except (TypeError, ValueError):
        _diag_val = 0.0
    if not _diag_val or not math.isfinite(_diag_val) or _diag_val <= 0:
        _diag_val = _infer_canvas_diagonal(cels)
    diagonal = max(1.0, _diag_val)
    # Gates relativos (equiv. aos antigos 35px/150px num canvas de ~400px de diagonal):
    # gate rígido ~9% da diagonal, raio de proximidade ~37.5%.
    center_gate = 0.09 * diagonal
    prox_radius = 0.375 * diagonal

    for cel in cels:
        layers = cel.get("layers", [])
        used_cand_tracks = set()

        for layer in layers:
            c_rgb = hex_to_rgb(layer.get("color_hex", "#000000"))
            bbox = layer.get("bbox", {"x": 0, "y": 0, "w": 0, "h": 0, "cx": 0, "cy": 0})
            cx, cy = bbox.get("cx", 0.0), bbox.get("cy", 0.0)

            best_cand = None
            best_score = -1e9

            for cand in prev_layers:
                cand_tid = cand.get("track_id")
                if not cand_tid or cand_tid in used_cand_tracks:
                    continue

                # Check perceptual color proximity (allows matching across subtle lighting/quantization shifts)
                cand_rgb = hex_to_rgb(cand.get("color_hex", "#000000"))
                cdist = color_distance(c_rgb, cand_rgb)
                if cdist > color_threshold:
                    continue

                iou = bbox_iou(bbox, cand.get("bbox", {"x": 0, "y": 0, "w": 0, "h": 0}))
                cand_bbox = cand.get("bbox", {"x": 0, "y": 0, "w": 0, "h": 0, "cx": 0, "cy": 0})
                cdx = cx - cand_bbox.get("cx", 0.0)
                cdy = cy - cand_bbox.get("cy", 0.0)
                center_dist = math.hypot(cdx, cdy)

                # Combined matching score: high IoU + centroid proximity - color difference penalty
                # (proximidade normalizada: center_dist/prox_radius em vez de /150px fixo)
                pos_score = iou + max(0.0, 0.4 * (1.0 - (center_dist / prox_radius)))
                score = pos_score - (0.005 * cdist)

                if (iou >= iou_threshold or center_dist <= center_gate) and score > best_score:
                    best_score = score
                    best_cand = cand

            if best_cand is not None:
                layer["track_id"] = best_cand["track_id"]
                used_cand_tracks.add(best_cand["track_id"])
            else:
                layer["track_id"] = f"trk_{next_track:03d}"
                next_track += 1

        # Inject into SVG
        svg = cel.get("svg", "")
        for layer in layers:
            lid = layer["id"]
            tid = layer["track_id"]
            svg = svg.replace(
                f'data-layer-id="{lid}"',
                f'data-layer-id="{lid}" data-track-id="{tid}"',
                1,
            )
        cel["svg"] = svg
        prev_layers = list(layers)

    return next_track - 1


def stabilize_flipbook_z_order(cels: List[Dict[str, Any]]) -> None:
    """
    Stabilizes the layer stacking order (Z-order) across cels to eliminate
    flickering/z-fighting of overlapping shapes (inspired by Vector Prism).
    - Learns the reference depth of each track by MEDIANA das posições em
      todos os cels (voto majoritário: um cel invertido pelo VTracer não
      contamina a referência, ao contrário do first-seen).
    - Ordena grupos semânticos pela profundidade mínima dos membros, de modo
      que inversões ENTRE grupos diferentes também são corrigidas (antes só
      a ordem intra-grupo mudava e o z-fight cross-group persistia).
    - Reorders `cel["layers"]` and updates `cel["svg"]` to reflect the
      stabilized order while preserving unique per-path IDs and attributes.
    """
    if len(cels) <= 1:
        return

    # 1. Reference depth = median position per track (majority vote),
    # with first-seen order as tiebreak (2 cels em desacordo empatam na
    # mediana — o cel mais antigo decide, preservando o comportamento anterior).
    positions: Dict[str, List[int]] = defaultdict(list)
    first_seen: Dict[str, float] = {}
    for cel in cels:
        for z_idx, layer in enumerate(cel.get("layers", [])):
            tid = layer.get("track_id")
            if tid and not layer.get("is_background"):
                positions[tid].append(z_idx)
                if tid not in first_seen:
                    first_seen[tid] = float(z_idx)
    track_ref_depth: Dict[str, float] = {}
    for tid, lst in positions.items():
        ordered = sorted(lst)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            track_ref_depth[tid] = float(ordered[mid])
        else:
            track_ref_depth[tid] = (ordered[mid - 1] + ordered[mid]) / 2.0

    # 2. Reorder layers in each cel if an inversion of overlapping tracks is detected
    for cel in cels:
        layers = cel.get("layers", [])
        if len(layers) <= 1:
            continue

        svg_text = cel.get("svg", "")
        # Accepts both self-closing (<path ... />) and explicit-close
        # (<path ...></path>) tags so no path is silently dropped/reordered-out.
        path_pattern = re.compile(
            r'<path(?=[\s/>])[^>]*?data-layer-id="([^"]+)"[^>]*?(?:/>|>\s*</path\s*>)',
            re.DOTALL,
        )
        tag_by_id = {}
        for match in path_pattern.finditer(svg_text):
            lid = match.group(1)
            tag_by_id[lid] = match.group(0)

        def layer_sort_key(lyr: Dict[str, Any]) -> tuple:
            if lyr.get("is_background"):
                return (-100000.0, -100000.0)
            tid = lyr.get("track_id", "")
            return (
                track_ref_depth.get(tid, float(lyr.get("z_order", 0))),
                first_seen.get(tid, float(lyr.get("z_order", 0))),
            )

        sorted_layers = sorted(layers, key=layer_sort_key)
        if [l["id"] for l in sorted_layers] == [l["id"] for l in layers]:
            continue

        # Update z_order attribute on layers
        for new_z, lyr in enumerate(sorted_layers, start=1):
            lyr["z_order"] = new_z
        cel["layers"] = sorted_layers

        # Re-order path tags inside groups in SVG markup.
        # Grupos ordenados pela profundidade mínima dos membros (cross-group fix);
        # g-background sempre primeiro; dentro do grupo, ordem por track_ref_depth.
        group_order = cel.get("groups", [])
        paths_by_group: Dict[str, List[str]] = defaultdict(list)
        for lyr in sorted_layers:
            lid = lyr["id"]
            gid = lyr.get("group", "g-default")
            if lid in tag_by_id:
                paths_by_group[gid].append(tag_by_id[lid])

        if paths_by_group:
            def _group_depth(gid: str) -> tuple:
                if gid == "g-background":
                    return (float("-inf"), float("-inf"))
                depths = [
                    layer_sort_key(lyr)
                    for lyr in sorted_layers
                    if lyr.get("group", "g-default") == gid
                ]
                return min(depths) if depths else (float("inf"), float("inf"))

            ordered_gids = sorted(paths_by_group.keys(), key=_group_depth)
            # Preserva grupos originais desconhecidos? sorted acima já cobre;
            # garante estabilidade para empates mantendo a ordem de group_order.
            if group_order:
                rank = {g: i for i, g in enumerate(group_order)}
                ordered_gids = sorted(
                    ordered_gids, key=lambda g: (_group_depth(g), rank.get(g, 10**9))
                )

            new_grouped_body = "\n".join(
                f'<g class="anim8-group" data-group="{gid}">\n' + "\n".join(paths_by_group[gid]) + "\n</g>"
                for gid in ordered_gids
            )

            defs_end = svg_text.find("</defs>")
            svg_close = svg_text.rfind("</svg>")
            if defs_end != -1 and svg_close != -1:
                header = svg_text[:defs_end + 7]
                footer = svg_text[svg_close:]
                cel["svg"] = f"{header}\n{new_grouped_body}\n{footer}"
            # Keep the metadata group order in sync with the rebuilt markup.
            cel["groups"] = list(ordered_gids)


def stabilize_flipbook_group_ids(cels: List[Dict[str, Any]]) -> None:
    """
    Renames per-cel semantic groups to temporally STABLE ids (Vector Prism).
    `cluster_paths_semantically` numbers clusters per cel by size rank, so the
    same object flips between g-cluster-01/02 whenever relative sizes change.
    Here each cluster is keyed by its DOMINANT track (largest member area,
    track_id asc as tiebreak; background stays 'g-background') and numbered
    globally by first appearance across cels — the 'g-cluster-NN' format is
    preserved, but NN is now a stable identity instead of a per-cel size rank.
    Rewrites `layer["group"]`, `cel["groups"]` and every `data-group="..."`
    (path tags and <g> wrappers) in the cel SVG.
    """
    if not cels:
        return

    # Global numbering by first appearance of each dominant track.
    number_by_key: Dict[str, int] = {}
    for cel in cels:
        seen_here = set()
        for layer in cel.get("layers", []):
            gid = layer.get("group", "g-default")
            if gid == "g-background" or gid in seen_here:
                continue
            seen_here.add(gid)
            members = [l for l in cel.get("layers", []) if l.get("group") == gid]
            key = _dominant_track_key(members)
            if key not in number_by_key:
                number_by_key[key] = len(number_by_key) + 1

    for cel in cels:
        rename: Dict[str, str] = {}
        for layer in cel.get("layers", []):
            gid = layer.get("group", "g-default")
            if gid == "g-background" or gid in rename:
                continue
            members = [l for l in cel.get("layers", []) if l.get("group") == gid]
            num = number_by_key.get(_dominant_track_key(members))
            if num is not None:
                rename[gid] = f"g-cluster-{num:02d}"
        if not rename:
            continue
        for layer in cel.get("layers", []):
            if layer.get("group") in rename:
                layer["group"] = rename[layer["group"]]
        cel["groups"] = [rename.get(g, g) for g in cel.get("groups", [])]
        # Single-pass substitution: sequential replaces would cascade when
        # the mapping permutes ids (e.g. {01->02, 02->01} collapsing all
        # groups into one).
        def _sub(m: re.Match) -> str:
            return f'data-group="{rename.get(m.group(1), m.group(1))}"'

        svg = cel.get("svg", "")
        cel["svg"] = re.sub(r'data-group="([^"]+)"', _sub, svg)


def _dominant_track_key(members: List[Dict[str, Any]]) -> str:
    """Largest-area member's track (track_id asc tiebreak, layer id fallback)."""
    best_key = ""
    best_area = -1.0
    seen_best = False
    for layer in members:
        bb = layer.get("bbox", {}) or {}
        try:
            area = float(bb.get("w", 0)) * float(bb.get("h", 0))
        except (TypeError, ValueError):
            area = 0.0
        key = str(layer.get("track_id") or layer.get("id") or "")
        if not seen_best or area > best_area or (area == best_area and key < best_key):
            best_key, best_area, seen_best = key, area, True
    return best_key


def vectorize_video_flipbook(
    video_path: str,
    target_fps: int = 15,
    max_frames: int = 45,
    preset_key: str = "duolingo_flat",
    custom_options: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    include_original: bool = True
) -> Dict[str, Any]:
    """
    Extracts video frames at target_fps (uniformly distributed across the entire clip),
    calculates a unified master color palette across the clip, pre-quantizes each frame
    to that palette, and vectorizes each cel with consistent color mappings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o arquivo de vídeo: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        if not orig_fps or not math.isfinite(float(orig_fps)) or float(orig_fps) <= 0:
            orig_fps = 30.0
        else:
            orig_fps = float(orig_fps)
    except (TypeError, ValueError):
        orig_fps = 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if total_video_frames <= 0:
        # Alguns MP4/VFR reportam FRAME_COUNT=0 mas decodificam normalmente.
        # Conta via grab (rápido, sem decode) em vez de rejeitar o vídeo.
        _count = 0
        try:
            while cap.grab():
                _count += 1
                if _count > 50000:  # trava de segurança p/ streams patológicos
                    break
        except Exception:
            pass
        cap.release()
        if _count <= 0:
            raise ValueError("Vídeo não contém frames legíveis.")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Não foi possível reabrir o vídeo: {video_path}")
        total_video_frames = _count
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or video_w)
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or video_h)
    duration_sec = total_video_frames / orig_fps if orig_fps > 0 else 0

    # Uniform sampling across the WHOLE clip (previously: only the beginning)
    step = max(1, int(round(orig_fps / max(1, target_fps))))
    desired = int(np.ceil(total_video_frames / step))
    desired = max(1, min(max_frames, desired))
    frame_indices = sorted(set(
        int(i) for i in np.linspace(0, total_video_frames - 1, desired)
    ))
    last_index = frame_indices[-1]
    global_colors: set = set()
    total_t0 = time.time()

    # ---- Pass A (fast, sequential grab): sample frames for master palette ----
    # Thumbnails 160x90 (~100x menos RAM que full-res) — o bastante para KMeans.
    sample_step = max(1, len(frame_indices) // PALETTE_SAMPLE_COUNT)
    sample_set = set(frame_indices[::sample_step])
    sample_frames: List[np.ndarray] = []
    frame_num = 0
    while frame_num <= last_index:
        if not cap.grab():
            logger.warning("Pass A: stream terminou antes do esperado no frame %d", frame_num)
            break
        if frame_num in sample_set:
            ret, frame = cap.retrieve()
            if ret:
                try:
                    if frame.shape[0] > 90 or frame.shape[1] > 160:
                        frame = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass
                sample_frames.append(frame)
            else:
                logger.warning("Pass A: falha ao recuperar frame de amostra %d", frame_num)
        frame_num += 1

    master_palette = extract_video_master_palette(
        sample_frames, max_colors=VIDEO_MASTER_COLORS,
        metric=resolve_color_metric(custom_options, preset_key),
    )
    logger.info(
        "Master palette estimada com %d cores (%d amostras)",
        len(master_palette), len(sample_frames)
    )

    # Re-open the capture: Pass A consumed the stream up to the last target frame
    cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível reabrir o vídeo: {video_path}")

    # ---- Pass B (sequential grab): retrieve + vectorize target frames ----
    order = {f: i for i, f in enumerate(frame_indices)}
    cels: List[Dict[str, Any]] = []

    frame_num = 0
    while frame_num <= last_index:
        if not cap.grab():
            logger.warning("Pass B: stream terminou antes do esperado no frame %d", frame_num)
            break
        if frame_num in order:
            ret, frame = cap.retrieve()
            if not ret:
                logger.warning("Cel pulado: falha ao decodificar frame %d", frame_num)
            else:
                cel_idx = order[frame_num] + 1
                time_sec = round(frame_num / orig_fps, 3)

                cel_res = vectorize_image_array(
                    frame,
                    preset_key=preset_key,
                    custom_options=custom_options,
                    cel_index=cel_idx,
                    master_palette=master_palette,
                    include_original=include_original
                )

                for col in cel_res["metadata"]["colors"]:
                    global_colors.add(col)

                cels.append({
                    "cel_index": cel_idx,
                    "source_frame": frame_num,
                    "timestamp": time_sec,
                    "svg": cel_res["svg"],
                    "original_frame": cel_res.get("original_image", ""),
                    "layers": cel_res["layers"],
                    "total_paths": cel_res["metadata"]["total_paths"],
                    "colors": cel_res["metadata"]["colors"],
                    "duration_ms": cel_res["duration_ms"],
                    "svg_size_bytes": cel_res["metadata"].get("svg_size_bytes", 0),
                    "dropped_tiny_paths": cel_res["metadata"].get("dropped_tiny_paths", 0),
                    "dropped_covered_bg": cel_res["metadata"].get("dropped_covered_bg", 0),
                    "groups": cel_res["metadata"].get("groups", []),
                })

                if progress_callback:
                    # done = cels decodificados de fato; total = alvos (nunca pula número em falha)
                    try:
                        progress_callback(len(cels), len(frame_indices))
                    except Exception:
                        logger.warning("progress_callback falhou no cel %d", cel_idx)
        frame_num += 1

    cap.release()
    total_elapsed = round((time.time() - total_t0) * 1000, 1)

    if not cels:
        raise ValueError("Nenhum frame pôde ser extraído do vídeo.")

    # Cross-consolidate threshold do preset ATIVO primeiro: tracking e merge
    # global usam o mesmo valor para não ligar cores que a consolidação separa.
    preset_opts = STYLE_PRESETS.get(preset_key, {})
    if custom_options and "color_merge_threshold" in custom_options:
        try:
            global_threshold = float(custom_options["color_merge_threshold"])
        except (TypeError, ValueError):
            global_threshold = float(preset_opts.get("color_merge_threshold", GLOBAL_PALETTE_THRESHOLD))
    else:
        global_threshold = float(preset_opts.get("color_merge_threshold", GLOBAL_PALETTE_THRESHOLD))

    # Diagonal do canvas p/ tracking invariante à escala (fallback: 1º frame).
    try:
        _diag = math.hypot(float(video_w), float(video_h))
    except (TypeError, ValueError):
        _diag = 0.0
    canvas_diagonal: Optional[float] = _diag if _diag and _diag > 0 else None

    # Stable cross-cel object identity (color + bbox IoU). Keeps per-cel ids
    # unique for the frontend while enabling flicker measurement / smoothing.
    total_tracks = assign_stable_track_ids(
        cels,
        color_threshold=global_threshold,
        canvas_diagonal=canvas_diagonal,
    )

    # Stabilize layer stacking order across cels to eliminate Z-fighting (Vector Prism)
    stabilize_flipbook_z_order(cels)

    # Stable cross-cel group identity (dominant track, not per-cel size rank)
    stabilize_flipbook_group_ids(cels)

    _, unified_global_palette = consolidate_palette(
        list(global_colors), threshold=global_threshold,
        metric=resolve_color_metric(custom_options, preset_key),
    )

    return {
        "success": True,
        "video_info": {
            "width": video_w,
            "height": video_h,
            "original_fps": round(orig_fps, 2),
            "playback_fps": target_fps,
            "duration_sec": round(duration_sec, 2),
            "total_video_frames": total_video_frames,
            "cels_count": len(cels),
        },
        "total_cels": len(cels),
        "unique_colors": unified_global_palette,
        "master_palette": master_palette,
        "total_processing_ms": total_elapsed,
        "avg_ms_per_cel": round(total_elapsed / max(1, len(cels)), 1),
        "avg_paths_per_cel": round(sum(c["total_paths"] for c in cels) / max(1, len(cels)), 1),
        "total_dropped_tiny_paths": sum(c.get("dropped_tiny_paths", 0) for c in cels),
        "total_tracks": total_tracks,
        "global_palette_threshold": global_threshold,
        "cels": cels,
    }


def _detect_shared_background(
    cels: List[Dict[str, Any]],
    color_threshold: float = GLOBAL_PALETTE_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """
    Returns the VISIBLE (topmost) background layer dict from cel 1 if EVERY cel
    shares a topmost background with a perceptually matching color
    (redmean distance <= `color_threshold`, NOT exact equality: per-cel
    centroid drift of 1-2 units would otherwise defeat dedup on every clip)
    and near-identical bbox.
    Topmost matters: stacked VTracer output can carry a covered full-frame base
    coat underneath (e.g. black under white) — deduplicating the bottom one
    would paint the whole animation black.
    Per-cel SVGs stay self-contained; only the animated export deduplicates
    (using cel 1's shade for the shared copy).
    """
    first_bgs = [l for l in cels[0].get("layers", []) if l.get("is_background")]
    if not first_bgs:
        return None
    ref = first_bgs[-1]
    ref_rgb = hex_to_rgb(ref.get("color_hex", "#000000"))
    for cel in cels[1:]:
        bgs = [l for l in cel.get("layers", []) if l.get("is_background")]
        if not bgs:
            return None
        top = bgs[-1]
        if color_distance(ref_rgb, hex_to_rgb(top.get("color_hex", "#000000"))) > color_threshold:
            return None
        if bbox_iou(ref["bbox"], top["bbox"]) < BG_SHARE_IOU_THRESHOLD:
            return None
    return ref


def _apply_recolors_to_svg(svg_content: str, recolors: Optional[Dict[str, str]]) -> str:
    """Applies fill/data-color replacements in a single pass (no chained mapping)."""
    if not svg_content or not recolors or not isinstance(recolors, dict):
        return svg_content
    return apply_recolors_to_svg(svg_content, recolors)


def build_animated_svg(
    cels: List[Dict[str, Any]],
    fps: int = 15,
    recolors: Optional[Dict[str, str]] = None,
    bg_color_threshold: float = GLOBAL_PALETTE_THRESHOLD,
) -> str:
    """
    Builds a single standalone animated SVG file with pure CSS keyframes.
    Compatible with browsers, Figma, and vector software.
    Static backgrounds shared by all cels are emitted once as
    <g id="shared-background"> instead of being duplicated per frame.
    `bg_color_threshold` is the perceptual (redmean) tolerance for treating
    per-cel background shades as the same shared backdrop.
    Frame element ids (`cel-frame-1..N`) are dense sequence numbers so a
    skipped/failed source frame can never desync CSS keyframes from groups.
    The original `cel_index` is preserved as `data-cel-index` for traceability.
    Optional `recolors` ({old: new}) are applied case-insensitively.
    """
    if not cels or not isinstance(cels, list):
        return ""

    fps = max(1, int(fps)) if fps else 15
    first_cel = cels[0] if isinstance(cels[0], dict) else {}
    first_svg = first_cel.get("svg", "")
    if not first_svg:
        return ""

    viewbox_match = re.search(r'viewBox="([^"]+)"', first_svg)
    w_match = re.search(r'width="([^"]+)"', first_svg)
    h_match = re.search(r'height="([^"]+)"', first_svg)

    viewbox = viewbox_match.group(1) if viewbox_match else "0 0 1200 800"
    w = w_match.group(1) if w_match else "1200"
    h = h_match.group(1) if h_match else "800"

    num_frames = len(cels)
    total_duration = max(0.1, num_frames / fps)

    css_keyframes = []
    for i in range(num_frames):
        start_pct = (i / num_frames) * 100
        end_pct = ((i + 1) / num_frames) * 100
        css_keyframes.append(
            f"#cel-frame-{i+1} {{ animation: anim-cel-{i+1} {total_duration:.2f}s infinite steps(1); }}\n"
            f"@keyframes anim-cel-{i+1} {{\n"
            f"  0%, {start_pct:.2f}% {{ display: none; opacity: 0; }}\n"
            f"  {start_pct:.2f}%, {end_pct:.2f}% {{ display: inline; opacity: 1; }}\n"
            f"  {end_pct:.2f}%, 100% {{ display: none; opacity: 0; }}\n"
            f"}}"
        )

    css_block = "\n".join(css_keyframes)

    shared_bg = _detect_shared_background(cels, color_threshold=bg_color_threshold) if len(cels) > 1 else None
    shared_block = ""
    if shared_bg is not None:
        # Emit only the TOPMOST (visible) background path. Stacked VTracer
        # output can carry a covered full-frame base coat underneath (e.g.
        # black under white) — sharing all bg paths would resurrect dead weight.
        bg_paths = re.findall(
            r'<path(?=[\s/>])[^>]*?data-is-bg="true"[^>]*?(?:/>|>\s*</path\s*>)',
            first_svg,
        )
        if bg_paths:
            top_bg = _apply_recolors_to_svg(bg_paths[-1], recolors)
            shared_block = f'<g id="shared-background">\n{top_bg}\n</g>\n'

    groups = []
    for seq, cel in enumerate(cels, start=1):
        orig_idx = cel.get("cel_index", seq)
        svg_str = _apply_recolors_to_svg(cel.get("svg", ""), recolors)
        paths_only = re.sub(r'<svg[^>]*>', '', svg_str)
        paths_only = re.sub(r'</svg>', '', paths_only)
        paths_only = re.sub(r'<defs>.*?</defs>', '', paths_only, flags=re.DOTALL)
        if shared_block:
            # Background already rendered once below; strip per-frame copies.
            paths_only = re.sub(
                r'<path(?=[\s/>])[^>]*?data-is-bg="true"[^>]*?(?:/>|>\s*</path\s*>)',
                '',
                paths_only,
            )
            # Drop now-empty semantic groups to avoid empty <g> clutter.
            paths_only = re.sub(r'<g class="anim8-group"[^>]*>\s*</g>', '', paths_only)
        groups.append(
            f'<g id="cel-frame-{seq}" class="cel-frame-group" data-cel-index="{orig_idx}">\n{paths_only}\n</g>'
        )

    animated_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewbox}" width="{w}" height="{h}">\n'
        f'<style>\n'
        f'.cel-frame-group {{ display: none; }}\n'
        f'{css_block}\n'
        f'</style>\n'
        f'{shared_block}{"".join(groups)}\n'
        f'</svg>'
    )
    return animated_svg
