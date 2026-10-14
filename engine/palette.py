"""
Engine Palette: Perceptual Color Consolidation & Master Palette Indexing
Merges near-duplicate color tones resulting from edge-aliasing or compression ringing
into canonical discrete flat colors (Anim8 style), enabling multi-cel recoloring.
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import cv2


def hex_to_rgb(hex_str: str) -> np.ndarray:
    """Converts a #RRGGBB hex string into a float32 RGB array."""
    h = hex_str.lstrip('#')
    if len(h) == 3:
        h = ''.join([c * 2 for c in h])
    if len(h) != 6:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float32)


def rgb_to_hex(rgb: np.ndarray) -> str:
    """Converts an RGB array/tuple into an uppercase #RRGGBB hex string."""
    r = int(np.clip(round(rgb[0]), 0, 255))
    g = int(np.clip(round(rgb[1]), 0, 255))
    b = int(np.clip(round(rgb[2]), 0, 255))
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_lab(hex_str: str) -> np.ndarray:
    """
    Converts a #RRGGBB hex string into a float32 CIE Lab array,
    using OpenCV's standard uint8 BGR-to-Lab scaling [0..255] for exact
    consistency with quantize_image_to_palette.
    """
    rgb = hex_to_rgb(hex_str)
    bgr_u8 = np.ascontiguousarray(np.clip(rgb[::-1], 0, 255).astype(np.uint8)).reshape(1, 1, 3)
    return cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)


def lab_distance(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Computes Euclidean distance in CIE Lab space (Delta E*ab)."""
    d = lab1 - lab2
    return float(np.sqrt(np.dot(d, d)))


def color_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """
    Computes weighted Euclidean distance in RGB space approximating human perception
    (Compupic / redmean metric).
    """
    rmean = (c1[0] + c2[0]) / 2.0
    r = c1[0] - c2[0]
    g = c1[1] - c2[1]
    b = c1[2] - c2[2]
    weight_r = 2.0 + (rmean / 256.0)
    weight_g = 4.0
    weight_b = 2.0 + ((255.0 - rmean) / 256.0)
    return float(np.sqrt(weight_r * r * r + weight_g * g * g + weight_b * b * b))


def _rgb_centroid_to_lab(centroid_rgb: np.ndarray) -> np.ndarray:
    """Converts an RGB centroid (float) to CIE Lab via the same uint8 BGR path as hex_to_lab."""
    bgr_u8 = np.ascontiguousarray(np.clip(centroid_rgb[::-1], 0, 255).astype(np.uint8)).reshape(1, 1, 3)
    return cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)


def consolidate_palette(
    raw_colors: List[str],
    threshold: float = 18.0,
    metric: str = "redmean",
) -> Tuple[Dict[str, str], List[str]]:
    """
    Consolidates a list of raw hex colors from vector paths into a clean canonical palette.
    - Supported distance metrics:
      * 'redmean': weighted RGB Euclidean distance (default for backward compatibility).
      * 'lab': perceptually uniform CIE Lab Euclidean distance (Delta E*ab), inspired by Im2Vec/SVGBuilder.
    - Greedy centroid-radius clustering (anti-chain): colors are visited in
      frequency order and join the nearest existing cluster only if their
      distance to the cluster's frequency-weighted centroid is < `threshold`.
      Unlike single-linkage union-find, a long gradient ramp cannot collapse
      into one cluster through transitive A~B~C links when the endpoints are
      far apart.
    - The canonical color of each cluster is the frequency-weighted centroid of its members.
    - Returns:
      color_map: {original_hex: canonical_hex}
      canonical_palette: unique canonical hex colors ordered by cluster
        frequency (dominant first, hex tiebreak for determinism).
    """
    if not raw_colors:
        return {}, []

    # Count frequency of each raw hex (deterministic: freq desc, hex asc)
    freq: Dict[str, int] = Counter(c.upper() for c in raw_colors)
    unique_hexes = sorted(freq.keys(), key=lambda x: (-freq[x], x))
    rgbs = [hex_to_rgb(h) for h in unique_hexes]
    n = len(unique_hexes)

    use_lab = (metric.lower() == "lab")
    if use_lab:
        labs = [hex_to_lab(h) for h in unique_hexes]

    # Greedy clusters: each holds member indices, total freq, rgb centroid
    cluster_members: List[List[int]] = []
    cluster_freq: List[int] = []
    cluster_centroid: List[np.ndarray] = []  # float64 RGB
    if use_lab:
        cluster_centroid_lab: List[np.ndarray] = []

    for i in range(n):
        best_c = -1
        best_dist = float("inf")
        for c in range(len(cluster_members)):
            if use_lab:
                dist = lab_distance(labs[i], cluster_centroid_lab[c])
            else:
                dist = color_distance(rgbs[i], cluster_centroid[c])
            if dist < best_dist:
                best_dist = dist
                best_c = c
        if best_c >= 0 and best_dist < threshold:
            # Join nearest cluster within radius; update weighted centroid
            f = freq[unique_hexes[i]]
            tot = cluster_freq[best_c] + f
            w_old = cluster_freq[best_c] / tot
            w_new = f / tot
            new_centroid = cluster_centroid[best_c] * w_old + rgbs[i].astype(np.float64) * w_new
            cluster_centroid[best_c] = new_centroid
            if use_lab:
                cluster_centroid_lab[best_c] = _rgb_centroid_to_lab(new_centroid)
            cluster_members[best_c].append(i)
            cluster_freq[best_c] = tot
        else:
            cluster_members.append([i])
            cluster_freq.append(freq[unique_hexes[i]])
            centroid = rgbs[i].astype(np.float64)
            cluster_centroid.append(centroid)
            if use_lab:
                cluster_centroid_lab.append(_rgb_centroid_to_lab(centroid))

    # Order clusters by total frequency desc (dominant first)
    order = sorted(range(len(cluster_members)), key=lambda c: (-cluster_freq[c],))
    color_map: Dict[str, str] = {}
    canonical_list: List[str] = []

    for c in order:
        canon_hex = rgb_to_hex(cluster_centroid[c])
        canonical_list.append(canon_hex)
        for i in cluster_members[c]:
            color_map[unique_hexes[i]] = canon_hex

    return color_map, canonical_list


def quantize_image_to_palette(
    img_bgr: np.ndarray,
    palette_hexes: List[str]
) -> np.ndarray:
    """
    Snaps every pixel of a BGR(A) image to its nearest color in a fixed palette,
    using perceptually uniform CIE Lab distance. This pre-quantization before
    VTracer guarantees temporal color consistency across video cels and reduces
    the number of color clusters the tracer has to fit.
    The alpha channel, when present (BGRA), is preserved untouched: only the
    BGR channels are quantized and the original alpha is reattached.
    """
    if img_bgr is None or img_bgr.size == 0 or not palette_hexes:
        return img_bgr

    has_alpha = (img_bgr.ndim == 3 and img_bgr.shape[2] == 4)
    if has_alpha:
        alpha = img_bgr[:, :, 3].copy()
        img_bgr = img_bgr[:, :, :3]
    elif img_bgr.ndim == 3 and img_bgr.shape[2] == 2:
        # Gray+alpha -> quantize gray as BGR, preserve alpha
        alpha = img_bgr[:, :, 1].copy()
        img_bgr = cv2.cvtColor(img_bgr[:, :, 0], cv2.COLOR_GRAY2BGR)
        has_alpha = True
    elif img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    # NOTE: palette centers must go through the SAME uint8 Lab path as the
    # image above. cv2.cvtColor on float32 expects 0-1 ranges (L in 0-100),
    # while uint8 uses L 0-255 — mixing them collapses every pixel onto a
    # single wrong entry.
    centers_rgb = np.array([hex_to_rgb(c) for c in palette_hexes], dtype=np.float32)
    centers_bgr_u8 = np.ascontiguousarray(np.clip(centers_rgb[:, ::-1], 0, 255).astype(np.uint8))
    centers_lab = cv2.cvtColor(
        centers_bgr_u8.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)

    # Chunked nearest-center search to bound peak memory on large frames
    indices = np.empty(lab.shape[0], dtype=np.int32)
    chunk = 400_000
    for s in range(0, lab.shape[0], chunk):
        block = lab[s:s + chunk]
        d = ((block[:, None, :] - centers_lab[None, :, :]) ** 2).sum(axis=2)
        indices[s:s + chunk] = d.argmin(axis=1)

    quantized = centers_bgr_u8[indices].reshape(h, w, 3)
    if has_alpha:
        return np.dstack([quantized, alpha])
    return quantized


def filter_transient_palette_colors(
    sample_frames: List[np.ndarray],
    candidate_palette: List[str],
    min_frame_persistence: float = 0.20,
    min_pixel_ratio: float = 0.001
) -> List[str]:
    """
    Evaluates candidate palette colors across sample video frames and filters out
    transient compression artifacts/noise colors (inspired by Im2Vec / video restoration).
    - Checks in what fraction of sample frames each candidate color covers at least
      `min_pixel_ratio` of frame pixels.
    - If a color appears in fewer than `min_frame_persistence` of frames and has negligible
      overall volume, it is considered a transient artifact and dropped from the
      master palette. Dropped pixels implicitly remap to the nearest persistent
      color during the subsequent Lab pre-quantization step.
    """
    if not sample_frames or not candidate_palette or len(candidate_palette) <= 2:
        return candidate_palette

    color_presence: Dict[str, int] = {c: 0 for c in candidate_palette}
    color_volume: Dict[str, int] = {c: 0 for c in candidate_palette}
    total_pixels = 0

    for f in sample_frames:
        small = cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA) if (f.shape[0] > 90 or f.shape[1] > 160) else f
        q = quantize_image_to_palette(small, candidate_palette)
        h, w = small.shape[:2]
        frame_pixels = h * w
        total_pixels += frame_pixels

        min_px_threshold = max(5, int(round(frame_pixels * min_pixel_ratio)))
        # Drop alpha before statistics: a 4-channel array would silently
        # re-shape into garbage triplets spanning pixel boundaries.
        q_rgb = q[:, :, :3] if q.ndim == 3 and q.shape[2] == 4 else q
        unique_colors, counts = np.unique(q_rgb.reshape(-1, 3), axis=0, return_counts=True)
        for bgr_color, count in zip(unique_colors, counts):
            hex_c = rgb_to_hex(bgr_color[::-1])
            if hex_c in color_volume:
                color_volume[hex_c] += int(count)
                if count >= min_px_threshold:
                    color_presence[hex_c] += 1

    num_frames = len(sample_frames)
    min_frames = max(1, int(round(num_frames * min_frame_persistence)))

    persistent: List[str] = []
    transient: List[str] = []

    for c in candidate_palette:
        vol_ratio = (color_volume[c] / total_pixels) if total_pixels > 0 else 0.0
        if color_presence[c] >= min_frames or vol_ratio >= 0.005:
            persistent.append(c)
        else:
            transient.append(c)

    if not persistent:
        return candidate_palette

    return persistent


def extract_video_master_palette(
    frame_list: List[np.ndarray],
    max_colors: int = 24,
    temporal_filter: bool = True,
    metric: str = "lab",
) -> List[str]:
    """
    Analyzes multiple sample frames across a video to establish a unified master color palette.
    1. Uses K-Means clustering in Lab color space to find prominent flat colors.
    2. Applies temporal persistence filtering to eliminate transient compression artifacts.
    3. Consolidates palette centers in CIE Lab space.
    """
    if not frame_list:
        return []

    # Downsample frames and stack pixels
    pixels = []
    small_frames = []
    for f in frame_list:
        small = cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA)
        if small.ndim == 3 and small.shape[2] >= 3:
            pixels.append(small[:, :, :3].reshape(-1, 3))
            small_frames.append(small)

    if not pixels:
        return []

    all_pixels = np.vstack(pixels).astype(np.float32)

    # Convert to Lab for perceptually uniform clustering
    all_pixels_bgr = all_pixels.reshape(-1, 1, 3).astype(np.uint8)
    all_pixels_lab = cv2.cvtColor(all_pixels_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    k = min(max_colors, len(all_pixels_lab))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    flags = cv2.KMEANS_PP_CENTERS

    _, _, centers_lab = cv2.kmeans(all_pixels_lab, k, None, criteria, 5, flags)

    centers_lab_u8 = np.clip(centers_lab, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    centers_bgr = cv2.cvtColor(centers_lab_u8, cv2.COLOR_LAB2BGR).reshape(-1, 3)

    master_hexes = [rgb_to_hex(bgr[::-1]) for bgr in centers_bgr]

    # Filter transient noise colors if enabled
    if temporal_filter and len(small_frames) > 1:
        master_hexes = filter_transient_palette_colors(small_frames, master_hexes)

    _, consolidated = consolidate_palette(master_hexes, threshold=16.0, metric=metric)
    return consolidated
