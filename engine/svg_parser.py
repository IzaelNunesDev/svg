"""
Engine SVG Parser: Path Geometry, Bounding Box Extraction & Topology Structuring
Enriches raw VTracer output with interactive attributes (data-bbox, node count, canonical colors),
performs Bézier curve pruning and collinear line merging (inspired by MARVEL/Mang2Vec),
rounds coordinates to 1-2 decimal places to minimize payload size, and detects background elements.
"""

import re
import math
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

from .config import (
    DEFAULT_MIN_PATH_AREA,
    DEFAULT_BEZIER_SIMPLIFY_TOLERANCE,
    DEFAULT_SEMANTIC_PROXIMITY,
)

TOKEN_RE = re.compile(r'([a-zA-Z])|([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)')

# Matches both self-closing (<path ... />) and explicit-close
# (<path ...></path>) path elements.
PATH_TAG_RE = re.compile(r'<path(?=[\s/>])([^>]*?)\s*(?:/>|>\s*</path\s*>)', re.DOTALL)


def is_cubic_flat(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    tolerance: float
) -> bool:
    """
    Determines if a cubic Bézier segment deviates by at most `tolerance`
    from the straight chord connecting p0 -> p3.
    """
    dx = p3[0] - p0[0]
    dy = p3[1] - p0[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-8:
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1]) <= tolerance and math.hypot(p2[0] - p0[0], p2[1] - p0[1]) <= tolerance
    L = math.sqrt(L2)
    d1 = abs(dx * (p0[1] - p1[1]) - (p0[0] - p1[0]) * dy) / L
    d2 = abs(dx * (p0[1] - p2[1]) - (p0[0] - p2[0]) * dy) / L
    if d1 > tolerance or d2 > tolerance:
        return False
    t1 = ((p1[0] - p0[0]) * dx + (p1[1] - p0[1]) * dy) / L2
    t2 = ((p2[0] - p0[0]) * dx + (p2[1] - p0[1]) * dy) / L2
    return -0.05 <= t1 <= 1.05 and -0.05 <= t2 <= 1.05


def is_point_collinear(
    pa: Tuple[float, float],
    pb: Tuple[float, float],
    pc: Tuple[float, float],
    tolerance: float
) -> bool:
    """
    Determines if point pb lies on the straight segment between pa and pc
    within `tolerance`.
    """
    dx = pc[0] - pa[0]
    dy = pc[1] - pa[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-8:
        return math.hypot(pb[0] - pa[0], pb[1] - pa[1]) <= tolerance
    L = math.sqrt(L2)
    d = abs(dx * (pa[1] - pb[1]) - (pa[0] - pb[0]) * dy) / L
    if d > tolerance:
        return False
    t = ((pb[0] - pa[0]) * dx + (pb[1] - pa[1]) * dy) / L2
    return -0.01 <= t <= 1.01


def parse_path_commands(d_str: str) -> List[Tuple[str, Tuple[float, ...]]]:
    """Tokenizes and parses an SVG path 'd' string into a list of (command, args) tuples."""
    tokens = []
    for match in TOKEN_RE.finditer(d_str):
        cmd, num = match.groups()
        if cmd:
            tokens.append(('CMD', cmd))
        elif num:
            tokens.append(('NUM', float(num)))

    commands = []
    i = 0
    n = len(tokens)
    current_cmd = ''
    while i < n:
        tok_type, val = tokens[i]
        if tok_type == 'CMD':
            current_cmd = val
            i += 1

        cmd_upper = current_cmd.upper()
        if cmd_upper == 'Z':
            commands.append(('Z', ()))
        elif cmd_upper == 'M':
            if i + 1 < n and tokens[i][0] == 'NUM' and tokens[i+1][0] == 'NUM':
                commands.append((current_cmd, (tokens[i][1], tokens[i+1][1])))
                i += 2
                current_cmd = 'L' if current_cmd == 'M' else 'l'
            else:
                break
        elif cmd_upper in ('L', 'T'):
            if i + 1 < n and tokens[i][0] == 'NUM' and tokens[i+1][0] == 'NUM':
                commands.append((current_cmd, (tokens[i][1], tokens[i+1][1])))
                i += 2
            else:
                break
        elif cmd_upper == 'H':
            if i < n and tokens[i][0] == 'NUM':
                commands.append((current_cmd, (tokens[i][1],)))
                i += 1
            else:
                break
        elif cmd_upper == 'V':
            if i < n and tokens[i][0] == 'NUM':
                commands.append((current_cmd, (tokens[i][1],)))
                i += 1
            else:
                break
        elif cmd_upper == 'C':
            if i + 5 < n and all(tokens[i+k][0] == 'NUM' for k in range(6)):
                commands.append((current_cmd, tuple(tokens[i+k][1] for k in range(6))))
                i += 6
            else:
                break
        elif cmd_upper in ('S', 'Q'):
            if i + 3 < n and all(tokens[i+k][0] == 'NUM' for k in range(4)):
                commands.append((current_cmd, tuple(tokens[i+k][1] for k in range(4))))
                i += 4
            else:
                break
        elif cmd_upper == 'A':
            if i + 6 < n and all(tokens[i+k][0] == 'NUM' for k in range(7)):
                commands.append((current_cmd, tuple(tokens[i+k][1] for k in range(7))))
                i += 7
            else:
                break
        else:
            i += 1
    return commands


def simplify_bezier_path(
    d_str: str,
    tolerance: float = DEFAULT_BEZIER_SIMPLIFY_TOLERANCE,
    merge_collinear_lines: bool = True
) -> str:
    """
    Simplifies an SVG path 'd' string (inspired by MARVEL/Mang2Vec):
    1. Converts near-flat cubic Bézier curves (C) into straight lines (L)
       when control points deviate by less than `tolerance` from the chord line.
    2. Converts near-flat quadratic curves (Q) into straight lines (L).
    3. Merges consecutive collinear line segments (L ... L) where intermediate
       points lie on the segment within `tolerance`.
    4. Eliminates redundant closing lines immediately before 'Z'.
    """
    if tolerance <= 0 or not d_str:
        return d_str

    raw_cmds = parse_path_commands(d_str)
    if not raw_cmds:
        return d_str

    # Convert commands to absolute segments with (type, start_pt, end_pt / data)
    segments = []
    curr = (0.0, 0.0)
    subpath_start = (0.0, 0.0)

    for cmd, args in raw_cmds:
        c_up = cmd.upper()
        is_rel = (cmd != c_up)

        if c_up == 'M':
            pt = (curr[0] + args[0], curr[1] + args[1]) if is_rel else (args[0], args[1])
            curr = pt
            subpath_start = pt
            segments.append(('M', pt, pt))
        elif c_up == 'Z':
            segments.append(('Z', curr, subpath_start))
            curr = subpath_start
        elif c_up == 'L':
            pt = (curr[0] + args[0], curr[1] + args[1]) if is_rel else (args[0], args[1])
            segments.append(('L', curr, pt))
            curr = pt
        elif c_up == 'H':
            x = curr[0] + args[0] if is_rel else args[0]
            pt = (x, curr[1])
            segments.append(('L', curr, pt))
            curr = pt
        elif c_up == 'V':
            y = curr[1] + args[0] if is_rel else args[0]
            pt = (curr[0], y)
            segments.append(('L', curr, pt))
            curr = pt
        elif c_up == 'C':
            if is_rel:
                p1 = (curr[0] + args[0], curr[1] + args[1])
                p2 = (curr[0] + args[2], curr[1] + args[3])
                p3 = (curr[0] + args[4], curr[1] + args[5])
            else:
                p1 = (args[0], args[1])
                p2 = (args[2], args[3])
                p3 = (args[4], args[5])

            if is_cubic_flat(curr, p1, p2, p3, tolerance):
                segments.append(('L', curr, p3))
            else:
                segments.append(('C', curr, (p1, p2, p3)))
            curr = p3
        elif c_up == 'Q':
            if is_rel:
                p1 = (curr[0] + args[0], curr[1] + args[1])
                p2 = (curr[0] + args[2], curr[1] + args[3])
            else:
                p1 = (args[0], args[1])
                p2 = (args[2], args[3])

            dx = p2[0] - curr[0]
            dy = p2[1] - curr[1]
            L2 = dx * dx + dy * dy
            if L2 >= 1e-8 and (abs(dx * (curr[1] - p1[1]) - (curr[0] - p1[0]) * dy) / math.sqrt(L2)) <= tolerance:
                segments.append(('L', curr, p2))
            else:
                segments.append(('Q', curr, (p1, p2)))
            curr = p2
        elif c_up == 'S':
            # Smooth cubic Bézier: (x2, y2, x, y). The implicit reflection of
            # the previous control point is resolved by the renderer, so
            # converting relative -> absolute preserves the shape exactly.
            if is_rel:
                p2 = (curr[0] + args[0], curr[1] + args[1])
                p = (curr[0] + args[2], curr[1] + args[3])
            else:
                p2 = (args[0], args[1])
                p = (args[2], args[3])
            segments.append(('S', curr, (p2, p)))
            curr = p
        elif c_up == 'T':
            # Smooth quadratic: single endpoint (reflection resolved by renderer).
            pt = (curr[0] + args[0], curr[1] + args[1]) if is_rel else (args[0], args[1])
            segments.append(('T', curr, pt))
            curr = pt
        elif c_up == 'A':
            # Elliptical arc: (rx, ry, x-axis-rotation, large-arc, sweep, x, y).
            # Only rx/ry (lengths) and the endpoint scale/translate; rotation
            # and flags are passed through untouched.
            rx, ry, rot, large_arc, sweep, ex, ey = args
            pt = (curr[0] + ex, curr[1] + ey) if is_rel else (ex, ey)
            segments.append(('A', curr, (rx, ry, rot, large_arc, sweep, pt[0], pt[1])))
            curr = pt
        else:
            # Unknown command: preserve verbatim (command + args) so path
            # coordinates are never silently dropped.
            segments.append(('RAW', curr, (cmd, args)))

    # Pass 2: Merge consecutive collinear lines
    if merge_collinear_lines:
        changed = True
        while changed:
            changed = False
            merged = []
            for seg in segments:
                if not merged:
                    merged.append(seg)
                    continue

                prev_seg = merged[-1]
                if prev_seg[0] == 'L' and seg[0] == 'L':
                    pa = prev_seg[1]
                    pb = prev_seg[2]
                    pc = seg[2]
                    if is_point_collinear(pa, pb, pc, tolerance):
                        merged[-1] = ('L', pa, pc)
                        changed = True
                        continue
                merged.append(seg)
            segments = merged

    # Pass 3: Drop redundant closing line right before 'Z' if it ends at subpath_start
    filtered = []
    for i, seg in enumerate(segments):
        if seg[0] == 'L' and i + 1 < len(segments) and segments[i + 1][0] == 'Z':
            sub_start = segments[i + 1][2]
            if math.hypot(seg[2][0] - sub_start[0], seg[2][1] - sub_start[1]) <= tolerance:
                continue
        filtered.append(seg)
    segments = filtered

    # Format back to compact SVG path string
    def fmt(n: float) -> str:
        if abs(n - round(n)) < 1e-4:
            return str(int(round(n)))
        return f"{n:.2f}".rstrip('0').rstrip('.')

    parts = []
    for seg in segments:
        stype = seg[0]
        if stype == 'M':
            parts.append(f"M{fmt(seg[2][0])} {fmt(seg[2][1])}")
        elif stype == 'Z':
            parts.append("Z")
        elif stype == 'L':
            parts.append(f"L{fmt(seg[2][0])} {fmt(seg[2][1])}")
        elif stype == 'C':
            p1, p2, p3 = seg[2]
            parts.append(f"C{fmt(p1[0])} {fmt(p1[1])} {fmt(p2[0])} {fmt(p2[1])} {fmt(p3[0])} {fmt(p3[1])}")
        elif stype == 'Q':
            p1, p2 = seg[2]
            parts.append(f"Q{fmt(p1[0])} {fmt(p1[1])} {fmt(p2[0])} {fmt(p2[1])}")
        elif stype == 'S':
            p2, p = seg[2]
            parts.append(f"S{fmt(p2[0])} {fmt(p2[1])} {fmt(p[0])} {fmt(p[1])}")
        elif stype == 'T':
            parts.append(f"T{fmt(seg[2][0])} {fmt(seg[2][1])}")
        elif stype == 'A':
            parts.append("A" + " ".join(fmt(a) for a in seg[2]))
        elif stype == 'RAW':
            cmd, args = seg[2]
            parts.append(cmd + (" " + " ".join(fmt(a) for a in args) if args else ""))
        else:
            # Defensive fallback (unreachable with current producers):
            # emit any numeric payload instead of a bare command letter so
            # path data is never silently dropped.
            payload = seg[2] if len(seg) > 2 else ()
            nums: List[float] = []

            def _collect(v: Any) -> None:
                if isinstance(v, (int, float)):
                    nums.append(float(v))
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        _collect(x)

            _collect(payload)
            parts.append(stype + (" " + " ".join(fmt(a) for a in nums) if nums else ""))

    return " ".join(parts)


def round_svg_coordinates(svg_text: str, precision: int = 1) -> str:
    """
    Rounds long floating-point coordinates in SVG path 'd' strings to `precision` decimals.
    Removes bulky 16-decimal floats (e.g. 410.52000000000004 -> 410.5) reducing SVG size by up to 60%.
    Only numbers inside `d="..."` are touched so `<style>` values
    (e.g. `transition: fill 0.12s`) are preserved byte-for-byte.
    """
    def repl_float(m):
        val = float(m.group(0))
        if precision <= 0:
            return str(int(round(val)))
        return f"{val:.{precision}f}".rstrip('0').rstrip('.')

    def repl_d(match: re.Match) -> str:
        inner = re.sub(r'-?\d+\.\d{2,}', repl_float, match.group(1))
        return f'd="{inner}"'

    # Matches decimal numbers with 2 or more decimals (lookbehind avoids id="...").
    return re.sub(r'(?<![\w-])d="([^"]*)"', repl_d, svg_text)


def _fmt_scaled(n: float) -> str:
    """Formats a rescaled coordinate at 1-decimal precision (matches pipeline output)."""
    r = round(float(n), 1)
    if abs(r - round(r)) < 1e-9:
        return str(int(round(r)))
    return f"{r:.1f}"


def scale_path_d(d_str: str, s: float) -> str:
    """
    Scales every coordinate in an SVG path 'd' string by factor `s`.
    Arc rotation/flags are passed through untouched (only rx/ry and the
    endpoint scale); scaling is linear so it applies equally to relative
    and absolute commands. Unknown commands are preserved verbatim to
    avoid corrupting flags whose semantics are not known.
    """
    if s == 1.0 or not d_str:
        return d_str
    try:
        commands = parse_path_commands(d_str)
    except Exception:
        return d_str
    if not commands:
        return d_str

    parts: List[str] = []
    for cmd, args in commands:
        c_up = cmd.upper()
        try:
            if c_up in ("M", "L", "T"):
                parts.append(f"{cmd}{_fmt_scaled(args[0] * s)} {_fmt_scaled(args[1] * s)}")
            elif c_up in ("H", "V"):
                parts.append(f"{cmd}{_fmt_scaled(args[0] * s)}")
            elif c_up in ("C", "S", "Q"):
                parts.append(cmd + " ".join(_fmt_scaled(a * s) for a in args))
            elif c_up == "A":
                rx, ry, rot, large_arc, sweep, ex, ey = args
                parts.append(
                    f"{cmd}{_fmt_scaled(rx * s)} {_fmt_scaled(ry * s)} "
                    f"{_fmt_scaled(rot)} {int(large_arc)} {int(sweep)} "
                    f"{_fmt_scaled(ex * s)} {_fmt_scaled(ey * s)}"
                )
            elif c_up == "Z":
                parts.append(cmd)
            else:
                # Unknown command: keep verbatim (see simplify_bezier_path).
                parts.append(cmd + (" " + " ".join(_fmt_scaled(a) for a in args) if args else ""))
        except (IndexError, TypeError, ValueError):
            parts.append(cmd)
    return " ".join(parts)


def scale_translate(transform_str: str, s: float) -> str:
    """Scales the (tx, ty) offsets of an SVG translate() transform by `s`."""
    if s == 1.0 or not transform_str:
        return transform_str

    def _repl(m: re.Match) -> str:
        tx = float(m.group(1)) * s
        if m.group(2) is None:
            return f"translate({_fmt_scaled(tx)})"
        ty = float(m.group(2)) * s
        return f"translate({_fmt_scaled(tx)} {_fmt_scaled(ty)})"

    num = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    return re.sub(rf'translate\s*\(\s*({num})(?:\s*[,\s]\s*({num}))?\s*\)', _repl, transform_str)


def rescale_parsed_svg(
    svg_text: str,
    layers: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    scale: float,
    orig_w: float,
    orig_h: float,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Maps a parsed SVG traced at downscaled resolution back to the original
    raster coordinate space (inverse of the INTER_AREA downscale in
    vectorizer.vectorize_image_array).
    - Scales every path `d`, `translate()`, bbox, center and area by `scale`
      (= original_side / traced_side).
    - Restores the `<svg>` viewBox/width/height and metadata to the original
      dimensions so bboxes stay consistent with `original_image` previews,
      `video_info` dimensions and the scale-invariant tracking gates.
    Mutates `layers`/`metadata` in place and returns (svg, layers, metadata).
    """
    if scale == 1.0 or not svg_text or not layers:
        return svg_text, layers, metadata

    by_id = {l.get("id"): l for l in layers}
    for layer in layers:
        bbox = layer.get("bbox", {})
        for key in ("x", "y", "cx", "cy"):
            if key in bbox:
                bbox[key] = round(float(bbox[key]) * scale, 1)
        for key in ("w", "h"):
            if key in bbox:
                bbox[key] = round(float(bbox[key]) * scale, 1)
        if "area" in layer:
            try:
                layer["area"] = round(float(layer["area"]) * scale * scale, 1)
            except (TypeError, ValueError):
                pass
        tr = layer.get("transform", "")
        if tr:
            layer["transform"] = scale_translate(tr, scale)

    def _patch_tag(m: re.Match) -> str:
        tag = m.group(0)
        lid = m.group(1)
        layer = by_id.get(lid)
        if layer is None:
            return tag

        def _repl_d(dm: re.Match) -> str:
            return f'd="{scale_path_d(dm.group(1), scale)}"'

        # Note the lookbehind: bare d="..." would also match the tail of
        # id="..." / data-layer-id="...".
        tag = re.sub(r'(?<![\w-])d="([^"]*)"', _repl_d, tag, count=1)
        bbox = layer.get("bbox", {})
        try:
            tag = re.sub(
                r'data-bbox="[^"]*"',
                f'data-bbox="{bbox["x"]},{bbox["y"]},{bbox["w"]},{bbox["h"]}"',
                tag,
                count=1,
            )
            tag = re.sub(
                r'data-center="[^"]*"',
                f'data-center="{bbox["cx"]},{bbox["cy"]}"',
                tag,
                count=1,
            )
        except KeyError:
            pass
        tr = layer.get("transform", "")
        if tr:
            tag = re.sub(r'transform="[^"]*"', f'transform="{tr}"', tag, count=1)
        return tag

    svg_text = re.sub(
        r'<path(?=[\s/>])[^>]*?data-layer-id="([^"]+)"[^>]*?(?:/>|>\s*</path\s*>)',
        _patch_tag,
        svg_text,
        flags=re.DOTALL,
    )

    def _patch_root(m: re.Match) -> str:
        tag = m.group(0)
        tag = re.sub(r'viewBox="[^"]*"', f'viewBox="0 0 {orig_w} {orig_h}"', tag, count=1)
        tag = re.sub(r'width="[^"]*"', f'width="{orig_w}"', tag, count=1)
        tag = re.sub(r'height="[^"]*"', f'height="{orig_h}"', tag, count=1)
        return tag

    svg_text = re.sub(r'<svg\b[^>]*>', _patch_root, svg_text, count=1)

    metadata["width"] = float(orig_w)
    metadata["height"] = float(orig_h)
    metadata["viewbox"] = f"0 0 {orig_w} {orig_h}"
    metadata["svg_size_bytes"] = len(svg_text.encode("utf-8"))
    metadata["coordinate_scale"] = round(float(scale), 4)
    return svg_text, layers, metadata


# Matches fill=/data-color= hex colors for single-pass recoloring.
RECOLOR_ATTR_RE = re.compile(r'(fill|data-color)="(#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3}))"')


def apply_recolors_to_svg(svg_content: str, recolors: Optional[Dict[str, str]]) -> str:
    """
    Applies color replacements to `fill`/`data-color` attributes in a SINGLE
    pass, so chained mappings ({A->B, B->C}) resolve simultaneously: A becomes
    B and the original B becomes C, instead of everything cascading into C.
    Keys match case-insensitively; unmatched colors keep their original text.
    """
    if not svg_content or not recolors or not isinstance(recolors, dict):
        return svg_content
    norm: Dict[str, str] = {}
    for old_col, new_col in recolors.items():
        try:
            norm[str(old_col).upper()] = str(new_col).upper()
        except Exception:
            continue
    if not norm:
        return svg_content

    def _repl(m: re.Match) -> str:
        return f'{m.group(1)}="{norm.get(m.group(2).upper(), m.group(2))}"'

    return RECOLOR_ATTR_RE.sub(_repl, svg_content)


def count_path_points(d_attr: str) -> int:
    """Counts anchor and control command points in an SVG path."""
    commands = re.findall(r"[MLCQSTAZmlcqstaz]", d_attr)
    return max(len(commands), 1)


def _parse_translate(transform_str: str = "") -> Tuple[float, float]:
    """
    Parses an SVG `translate()` transform, accepting both comma and
    space separators plus the single-argument form:
    `translate(x)`, `translate(x,y)`, `translate(x y)`.
    Returns (tx, ty). Unknown/other transforms yield (0, 0).
    """
    if not transform_str:
        return 0.0, 0.0
    num = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    m = re.search(
        rf'translate\s*\(\s*({num})(?:\s*[,\s]\s*({num}))?\s*\)',
        transform_str,
    )
    if not m:
        return 0.0, 0.0
    try:
        tx = float(m.group(1))
    except (TypeError, ValueError):
        tx = 0.0
    try:
        ty = float(m.group(2)) if m.group(2) is not None else 0.0
    except (TypeError, ValueError):
        ty = 0.0
    return tx, ty


def compute_path_bbox(d_attr: str, transform_str: str = "") -> Dict[str, float]:
    """
    Computes the axis-aligned bounding box (AABB) and centroid of an SVG
    path 'd' attribute, applying any translate() offset.
    Geometry is derived from parsed path commands (absolute positions):
    - H/V contribute a single axis; the other axis keeps the current point.
    - C/Q/S control points are included (conservative, matches curve extent).
    - A (arc) contributes only its endpoint; radii/rotation/flags are ignored
      (they are not coordinates and must not pull the bbox toward 0/1).
    """
    # Extract translate offset (comma- or space-separated, 1- or 2-arg)
    tx, ty = _parse_translate(transform_str)

    if not d_attr:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0, "cx": 0.0, "cy": 0.0}

    try:
        commands = parse_path_commands(d_attr)
    except Exception:
        commands = []
    if not commands:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0, "cx": 0.0, "cy": 0.0}

    xs: List[float] = []
    ys: List[float] = []

    def _add(pt: Tuple[float, float]) -> None:
        xs.append(pt[0])
        ys.append(pt[1])

    curr: Tuple[float, float] = (0.0, 0.0)
    subpath_start: Tuple[float, float] = (0.0, 0.0)

    for cmd, args in commands:
        c_up = cmd.upper()
        is_rel = (cmd != c_up)
        try:
            if c_up == 'M':
                pt = (curr[0] + args[0], curr[1] + args[1]) if is_rel else (args[0], args[1])
                curr = pt
                subpath_start = pt
                _add(pt)
            elif c_up == 'Z':
                curr = subpath_start
            elif c_up == 'L' or c_up == 'T':
                pt = (curr[0] + args[0], curr[1] + args[1]) if is_rel else (args[0], args[1])
                curr = pt
                _add(pt)
            elif c_up == 'H':
                x = curr[0] + args[0] if is_rel else args[0]
                curr = (x, curr[1])
                _add(curr)
            elif c_up == 'V':
                y = curr[1] + args[0] if is_rel else args[0]
                curr = (curr[0], y)
                _add(curr)
            elif c_up == 'C':
                if is_rel:
                    p1 = (curr[0] + args[0], curr[1] + args[1])
                    p2 = (curr[0] + args[2], curr[1] + args[3])
                    p3 = (curr[0] + args[4], curr[1] + args[5])
                else:
                    p1 = (args[0], args[1])
                    p2 = (args[2], args[3])
                    p3 = (args[4], args[5])
                _add(p1)
                _add(p2)
                _add(p3)
                curr = p3
            elif c_up == 'Q' or c_up == 'S':
                if is_rel:
                    p1 = (curr[0] + args[0], curr[1] + args[1])
                    p2 = (curr[0] + args[2], curr[1] + args[3])
                else:
                    p1 = (args[0], args[1])
                    p2 = (args[2], args[3])
                _add(p1)
                _add(p2)
                curr = p2
            elif c_up == 'A':
                # rx, ry, x-axis-rotation, large-arc-flag, sweep-flag, x, y
                ex, ey = args[5], args[6]
                pt = (curr[0] + ex, curr[1] + ey) if is_rel else (ex, ey)
                curr = pt
                _add(pt)
            else:
                continue
        except (IndexError, TypeError, ValueError):
            continue

    if not xs or not ys:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0, "cx": 0.0, "cy": 0.0}

    min_x = min(xs) + tx
    max_x = max(xs) + tx
    min_y = min(ys) + ty
    max_y = max(ys) + ty

    w = max(0.0, max_x - min_x)
    h = max(0.0, max_y - min_y)
    cx = min_x + (w / 2.0)
    cy = min_y + (h / 2.0)

    return {
        "x": round(min_x, 1),
        "y": round(min_y, 1),
        "w": round(w, 1),
        "h": round(h, 1),
        "cx": round(cx, 1),
        "cy": round(cy, 1),
    }


def bboxes_connect(
    b1: Dict[str, float],
    b2: Dict[str, float],
    margin: float = DEFAULT_SEMANTIC_PROXIMITY
) -> bool:
    """
    Checks whether two bounding boxes intersect, touch, or are separated by at
    most `margin` pixels. Also covers containment (one box inside another).
    """
    return not (
        b1["x"] + b1["w"] + margin < b2["x"] or
        b2["x"] + b2["w"] + margin < b1["x"] or
        b1["y"] + b1["h"] + margin < b2["y"] or
        b2["y"] + b2["h"] + margin < b1["y"]
    )


def cluster_paths_semantically(
    items: List[Dict[str, Any]],
    proximity_margin: float = DEFAULT_SEMANTIC_PROXIMITY
) -> List[str]:
    """
    Assigns semantic cluster IDs to a list of parsed path items based on
    spatial connectivity (inspired by Vector Prism: Stratifying Semantic Structure).
    - Background paths are assigned 'g-background'.
    - Non-background paths that overlap, touch, or lie within `proximity_margin`
      of each other are clustered together via Union-Find.
    - Clusters are sorted by total bounding area in descending order and assigned
      stable group IDs ('g-cluster-01', 'g-cluster-02', ...).
    Returns a list of group_id strings corresponding 1-to-1 with `items`.
    """
    n = len(items)
    if n == 0:
        return []

    group_ids = ["" for _ in range(n)]
    non_bg_indices = [i for i, p in enumerate(items) if not p.get("is_bg")]

    for i, p in enumerate(items):
        if p.get("is_bg"):
            group_ids[i] = "g-background"

    if not non_bg_indices:
        return group_ids

    # Union-Find over non-background paths
    parent = {i: i for i in non_bg_indices}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    m = len(non_bg_indices)
    for idx_a in range(m):
        i = non_bg_indices[idx_a]
        bbox_i = items[i]["bbox"]
        for idx_b in range(idx_a + 1, m):
            j = non_bg_indices[idx_b]
            if find(i) != find(j) and bboxes_connect(bbox_i, items[j]["bbox"], proximity_margin):
                union(i, j)

    # Group member indices by root
    raw_clusters: Dict[int, List[int]] = defaultdict(list)
    for i in non_bg_indices:
        raw_clusters[find(i)].append(i)

    # Sort clusters by aggregate bounding area descending so main character/object is cluster-01
    def cluster_area(member_idxs: List[int]) -> float:
        min_x = min(items[i]["bbox"]["x"] for i in member_idxs)
        max_x = max(items[i]["bbox"]["x"] + items[i]["bbox"]["w"] for i in member_idxs)
        min_y = min(items[i]["bbox"]["y"] for i in member_idxs)
        max_y = max(items[i]["bbox"]["y"] + items[i]["bbox"]["h"] for i in member_idxs)
        return (max_x - min_x) * (max_y - min_y)

    sorted_clusters = sorted(raw_clusters.values(), key=cluster_area, reverse=True)

    for cluster_num, member_idxs in enumerate(sorted_clusters, start=1):
        cid = f"g-cluster-{cluster_num:02d}"
        for i in member_idxs:
            group_ids[i] = cid

    return group_ids


def parse_and_structure_svg(
    svg_raw: str,
    cel_index: int = 1,
    color_map: Optional[Dict[str, str]] = None,
    min_path_area: float = DEFAULT_MIN_PATH_AREA,
    bezier_simplify_tolerance: float = DEFAULT_BEZIER_SIMPLIFY_TOLERANCE,
    semantic_proximity: float = DEFAULT_SEMANTIC_PROXIMITY,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parses VTracer SVG output:
    - Simplifies Bézier curves (C -> L) and merges collinear lines (MARVEL / Mang2Vec).
    - Applies color consolidation mapping to snap colors to canonical palette.
    - Drops vector-side speckle: non-background paths with bbox area < min_path_area.
    - Computes bounding box (AABB) for direct selection & transform controls.
    - Identifies background rect if present.
    - Clusters surviving paths into semantic <g> groups by spatial
      connectivity within `semantic_proximity` (Vector Prism).
    - Optimizes coordinate strings with rounding.
    - Produces clean interactive SVG markup.
    Contract: per-path id/data-layer-id stay unique (`celXXX_lyrYYY`,
    reindexed after filtering); layers keep id/cel_index/color_hex/bbox/etc.
    """
    width_match = re.search(r'width="(\d+(?:\.\d+)?)"', svg_raw)
    height_match = re.search(r'height="(\d+(?:\.\d+)?)"', svg_raw)
    viewbox_match = re.search(r'viewBox="([^"]+)"', svg_raw)

    svg_w = float(width_match.group(1)) if width_match else 1200.0
    svg_h = float(height_match.group(1)) if height_match else 800.0
    viewbox = viewbox_match.group(1) if viewbox_match else f"0 0 {svg_w} {svg_h}"

    path_pattern = PATH_TAG_RE
    paths = path_pattern.findall(svg_raw)

    # First pass: parse geometry + color so background detection uses raw order,
    # then filter speckle and reindex surviving paths sequentially.
    parsed = []
    total_screen_area = svg_w * svg_h
    raw_nodes_total = 0
    simplified_nodes_total = 0

    for raw_idx, attrs in enumerate(paths, start=1):
        # Lookbehind: bare d="/fill=" would also match the tail of
        # id="..." or data-layer-id="...".
        fill_match = re.search(r'(?<![\w-])fill="([^"]+)"', attrs)
        d_match = re.search(r'(?<![\w-])d="([^"]+)"', attrs)
        transform_match = re.search(r'(?<![\w-])transform="([^"]+)"', attrs)

        raw_color = fill_match.group(1).upper() if fill_match else "#000000"
        d_str = d_match.group(1) if d_match else ""
        transform_str = transform_match.group(1) if transform_match else ""

        raw_pts = count_path_points(d_str)
        raw_nodes_total += raw_pts

        # Apply Bézier & collinearity pruning (MARVEL / Mang2Vec)
        if bezier_simplify_tolerance > 0 and d_str:
            d_opt = simplify_bezier_path(d_str, tolerance=bezier_simplify_tolerance)
        else:
            d_opt = d_str

        # Replace d in attrs with optimized d (lookbehind avoids id="...").
        attrs_opt = re.sub(r'(?<![\w-])d="[^"]*"', f'd="{d_opt}"', attrs)

        # Map to canonical color if color_map provided
        canon_color = color_map.get(raw_color, raw_color) if color_map else raw_color
        points_count = count_path_points(d_opt)
        simplified_nodes_total += points_count

        # Compute Bounding Box
        bbox = compute_path_bbox(d_opt, transform_str)
        path_area = bbox["w"] * bbox["h"]

        # Background detection: covers > 88% of canvas and is one of first layers
        is_bg = (raw_idx <= 2 and total_screen_area > 0 and (path_area / total_screen_area) >= 0.88)
        parsed.append({
            "attrs": attrs_opt, "canon_color": canon_color, "d_str": d_opt,
            "transform_str": transform_str, "bbox": bbox, "area": path_area,
            "points": points_count, "is_bg": is_bg, "raw_index": raw_idx,
        })

    dropped_tiny = sum(1 for p in parsed if not p["is_bg"] and p["area"] < min_path_area)
    surviving = [p for p in parsed if p["is_bg"] or p["area"] >= min_path_area]

    # VTracer stacked mode emits opaque full-frame "base coats" (e.g. a black
    # full-frame under a white full-frame). Any full-frame path below a LATER
    # full-frame path is pixel-invisible — drop it instead of shipping dead
    # weight (and misleading the animated-export background dedup).
    dropped_covered_bg = 0
    if total_screen_area > 0:
        full_idx = [i for i, p in enumerate(surviving)
                    if p["area"] / total_screen_area >= 0.995]
        if len(full_idx) > 1:
            drop = set(full_idx[:-1])
            dropped_covered_bg = len(drop)
            surviving = [p for i, p in enumerate(surviving) if i not in drop]

    assigned_group_ids = cluster_paths_semantically(surviving, proximity_margin=semantic_proximity)

    enriched_by_group: Dict[str, List[str]] = defaultdict(list)
    group_order: List[str] = []
    layers_meta: List[Dict[str, Any]] = []
    color_histogram: Dict[str, int] = {}

    for idx, p in enumerate(surviving, start=1):
        canon_color = p["canon_color"]
        bbox = p["bbox"]
        is_bg = p["is_bg"]
        color_histogram[canon_color] = color_histogram.get(canon_color, 0) + 1
        layer_id = f"cel{cel_index:03d}_lyr{idx:03d}"
        layer_name = f"Frame {cel_index:03d} - {canon_color}" + (" (Background)" if is_bg else "")
        group_id = assigned_group_ids[idx - 1]

        # Clean old attributes
        cleaned_attrs = re.sub(r'id="[^"]*"', '', p["attrs"])
        cleaned_attrs = re.sub(r'class="[^"]*"', '', cleaned_attrs)
        cleaned_attrs = re.sub(r'fill="[^"]*"', f'fill="{canon_color}"', cleaned_attrs)

        new_path_tag = (
            f'<path id="{layer_id}" '
            f'class="vector-path-element{" is-background" if is_bg else ""}" '
            f'data-layer-id="{layer_id}" '
            f'data-color="{canon_color}" '
            f'data-cel="{cel_index}" '
            f'data-points="{p["points"]}" '
            f'data-bbox="{bbox["x"]},{bbox["y"]},{bbox["w"]},{bbox["h"]}" '
            f'data-center="{bbox["cx"]},{bbox["cy"]}" '
            f'data-is-bg="{str(is_bg).lower()}" '
            f'data-group="{group_id}" '
            f'{cleaned_attrs.strip()} />'
        )
        if group_id not in enriched_by_group:
            group_order.append(group_id)
        enriched_by_group[group_id].append(new_path_tag)

        layers_meta.append({
            "id": layer_id,
            "cel_index": cel_index,
            "layer_index": idx,
            "name": layer_name,
            "color_hex": canon_color,
            "points_count": p["points"],
            "bbox": bbox,
            "area": round(p["area"], 1),
            "is_background": is_bg,
            "group": group_id,
            "transform": p["transform_str"],
            "z_order": idx,
            "visible": True,
        })

    svg_header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{viewbox}" width="{svg_w}" height="{svg_h}" '
        f'class="anim8-vector-svg" id="svg-cel-{cel_index}">'
        f'<defs>'
        f'<style>'
        f'.vector-path-element {{ transition: fill 0.12s ease, opacity 0.12s ease; }}'
        f'.vector-path-element:hover {{ filter: drop-shadow(0 0 4px rgba(0,255,136,0.6)); cursor: pointer; }}'
        f'.vector-path-element.selected {{ stroke: #00ff88; stroke-width: 2px; stroke-dasharray: 4,4; }}'
        f'.vector-path-element.is-background {{ pointer-events: all; }}'
        f'</style>'
        f'</defs>'
    )
    grouped_body = "\n".join(
        f'<g class="anim8-group" data-group="{gid}">\n' + "\n".join(enriched_by_group[gid]) + "\n</g>"
        for gid in group_order
    )
    svg_footer = "</svg>"

    raw_assembled = f"{svg_header}\n{grouped_body}\n{svg_footer}"
    final_svg = round_svg_coordinates(raw_assembled, precision=1)

    metadata = {
        "width": svg_w,
        "height": svg_h,
        "viewbox": viewbox,
        "total_paths": len(layers_meta),
        "raw_total_paths": len(parsed),
        "dropped_tiny_paths": dropped_tiny,
        "dropped_covered_bg": dropped_covered_bg,
        "total_colors": len(color_histogram),
        "colors": sorted(list(color_histogram.keys())),
        "color_counts": color_histogram,
        "groups": sorted(group_order),
        "semantic_clusters_count": len(set(gid for gid in assigned_group_ids if gid != "g-background")),
        "svg_size_bytes": len(final_svg.encode("utf-8")),
        "nodes_before_simplification": raw_nodes_total,
        "nodes_after_simplification": simplified_nodes_total,
        "pruned_nodes_count": max(0, raw_nodes_total - simplified_nodes_total),
    }

    return final_svg, layers_meta, metadata
