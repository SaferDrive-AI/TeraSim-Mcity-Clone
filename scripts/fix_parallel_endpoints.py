#!/usr/bin/env python3
"""
Fix Parallel Endpoints for Center-Lane Edges

After adding center lanes and running netconvert, the edge endpoints at
junctions may not be parallel to their reverse edge. This creates a flared
appearance at junctions.

This script post-processes the compiled .net.xml to fix this by:
1. Finding all edge pairs with centerLaneAdded=true
2. At each end of each pair, checking if the endpoint segments are parallel
3. If only one endpoint is misaligned with its interior, moving only that one
4. If both are misaligned, sliding both the same distance toward each other
5. Applying the same adjustment to all lane shapes

Geometry at each end of a forward/reverse edge pair:

    Forward edge:  P1 ────────→ P2 ────→ P3 ──── ... ────
    Reverse edge:  Q1 ←──────── Q2 ←──── Q3 ──── ... ────

    P1, Q1 are the endpoints at the junction (adjusted if needed).
    P2, Q2 are the next interior points (fixed).
    P3, Q3 are the second interior points (used to check interior alignment).

    Case A — One endpoint already aligned with its interior:
        Only move the other endpoint to match the good one's direction.

    Case B — Both endpoints misaligned (or no interior reference):
        Slide both P1 and Q1 along the connection line P1↔Q1 toward each
        other by the same distance t:

            t = cross(d_f, d_r) / cross(d_f - d_r, ĉ)

        where d_f = P2 - P1, d_r = Q1 - Q2, ĉ = normalize(Q1 - P1).

Usage:
    python fix_parallel_endpoints.py network.net.xml
    python fix_parallel_endpoints.py network.net.xml -o fixed.net.xml --tolerance 0.5

Author: TeraSim Team
License: MIT
"""

import xml.etree.ElementTree as ET
import math
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Set

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ── Vector helpers ───────────────────────────────────────────────────────

def parse_shape(shape_str: str) -> List[Tuple[float, float]]:
    """Parse SUMO shape string into coordinate list."""
    coords = []
    for pair in shape_str.strip().split():
        parts = pair.split(',')
        coords.append((float(parts[0]), float(parts[1])))
    return coords


def shape_to_str(coords: List[Tuple[float, float]]) -> str:
    """Convert coordinate list to SUMO shape string."""
    return ' '.join(f"{x:.2f},{y:.2f}" for x, y in coords)


def cross2d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """2D cross product: a × b."""
    return a[0] * b[1] - a[1] * b[0]


def vec_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def vec_add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def vec_scale(v: Tuple[float, float], s: float) -> Tuple[float, float]:
    return (v[0] * s, v[1] * s)


def vec_len(v: Tuple[float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


def vec_normalize(v: Tuple[float, float]) -> Tuple[float, float]:
    length = vec_len(v)
    if length > 1e-10:
        return (v[0] / length, v[1] / length)
    return (0.0, 0.0)


def parallel_deviation_deg(
    d1: Tuple[float, float], d2: Tuple[float, float]
) -> float:
    """
    Deviation from parallel/anti-parallel in degrees.

    Returns 0 for exactly parallel or anti-parallel, 90 for perpendicular.
    Works regardless of whether edges point in the same or opposite direction.
    """
    len1 = vec_len(d1)
    len2 = vec_len(d2)
    if len1 < 1e-10 or len2 < 1e-10:
        return 0.0
    sin_angle = min(abs(cross2d(d1, d2)) / (len1 * len2), 1.0)
    return math.degrees(math.asin(sin_angle))


# ── Core logic ───────────────────────────────────────────────────────────

def compute_endpoint_fix(
    P1: Tuple[float, float],
    P2: Tuple[float, float],
    Q1: Tuple[float, float],
    Q2: Tuple[float, float],
    P3: Optional[Tuple[float, float]],
    Q3: Optional[Tuple[float, float]],
    tolerance_deg: float = 1.0,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], float, str]:
    """
    Compute adjustments to make edge endpoint segments parallel to each other.

    P1, Q1 are the endpoints at the junction (to be adjusted).
    P2, Q2 are the next interior points (fixed).
    P3, Q3 are the second interior points (used to check interior alignment).

    Logic:
      - If P1→P2 is already aligned with its interior (P2→P3) but Q2→Q1 is
        not aligned with its interior (Q3→Q2): only move Q1.
      - Vice versa: only move P1.
      - Otherwise: move both P1 and Q1 the same distance toward each other.

    Returns:
        (delta_P1, delta_Q1, deviation_before, mode)
        delta_P1 is None if already parallel within tolerance.
        mode is 'fwd_only', 'rev_only', or 'both'.
    """
    d_f = vec_sub(P2, P1)   # forward endpoint segment (away from junction)
    d_r = vec_sub(Q1, Q2)   # reverse endpoint segment (toward junction)

    # ── Check if endpoints are already parallel to each other ────────
    dev = parallel_deviation_deg(d_f, d_r)

    if dev < tolerance_deg:
        return None, None, dev, ''

    # ── Check each endpoint's alignment with its own interior ────────
    fwd_aligned = False
    rev_aligned = False
    if P3 is not None:
        ref_fwd = vec_sub(P3, P2)  # forward interior segment
        fwd_aligned = parallel_deviation_deg(d_f, ref_fwd) < tolerance_deg
    if Q3 is not None:
        ref_rev = vec_sub(Q3, Q2)  # reverse interior segment (away from junction)
        # d_r points toward junction, ref_rev away — they should be anti-parallel
        rev_aligned = parallel_deviation_deg(d_r, ref_rev) < tolerance_deg

    # ── Connection vector from P1 to Q1 ─────────────────────────────
    conn = vec_sub(Q1, P1)
    conn_len = vec_len(conn)
    if conn_len < 1e-6:
        logger.warning("      Endpoints coincide — cannot fix")
        return None, None, dev, ''
    c = vec_normalize(conn)

    # ── Decide which endpoints to move ───────────────────────────────
    if fwd_aligned and not rev_aligned:
        # Forward is good — only move Q1 to match forward's direction
        d_target = vec_normalize(d_f)
        denom = cross2d(c, d_target)
        if abs(denom) < 1e-10:
            logger.warning("      Degenerate geometry — cannot solve")
            return None, None, dev, ''
        r = cross2d(d_r, d_target) / denom
        delta_P1 = (0.0, 0.0)
        delta_Q1 = vec_scale(c, -r)
        mode = 'rev_only'

    elif rev_aligned and not fwd_aligned:
        # Reverse is good — only move P1 to match reverse's direction
        d_target = vec_normalize(d_r)
        denom = cross2d(c, d_target)
        if abs(denom) < 1e-10:
            logger.warning("      Degenerate geometry — cannot solve")
            return None, None, dev, ''
        s = cross2d(d_f, d_target) / denom
        delta_P1 = vec_scale(c, s)
        delta_Q1 = (0.0, 0.0)
        mode = 'fwd_only'

    else:
        # Both need fixing — move both the same distance toward each other
        d_diff = vec_sub(d_f, d_r)
        denom = cross2d(d_diff, c)
        if abs(denom) < 1e-10:
            logger.warning("      Degenerate geometry — cannot solve")
            return None, None, dev, ''
        t = cross2d(d_f, d_r) / denom
        delta_P1 = vec_scale(c, t)       # P1 slides toward Q1
        delta_Q1 = vec_scale(c, -t)      # Q1 slides toward P1
        mode = 'both'

    # ── Verify the fix ───────────────────────────────────────────────
    new_d_f = vec_sub(P2, vec_add(P1, delta_P1))
    new_d_r = vec_sub(vec_add(Q1, delta_Q1), Q2)
    new_dev = parallel_deviation_deg(new_d_f, new_d_r)

    if new_dev > dev:
        logger.warning(
            f"      Fix would worsen: {dev:.2f}° → {new_dev:.2f}°, skipping"
        )
        return None, None, dev, ''

    # Warn on large adjustments
    max_move = max(vec_len(delta_P1), vec_len(delta_Q1))
    if max_move > 10.0:
        logger.warning(f"      Large adjustment: {max_move:.2f}m — verify geometry")

    return delta_P1, delta_Q1, dev, mode


def fix_edge_pair_end(
    fwd_elem: ET.Element,
    rev_elem: ET.Element,
    fwd_shape: List[Tuple[float, float]],
    rev_shape: List[Tuple[float, float]],
    end: str,
    tolerance_deg: float,
    dry_run: bool,
) -> Tuple[bool, Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Check and fix parallelism at one end of an edge pair.

    Args:
        end: 'start' (start of fwd = end of rev) or 'end' (end of fwd = start of rev)

    Returns (fixed, delta_fwd, delta_rev).
    """
    if end == 'start':
        # At the start-of-forward / end-of-reverse junction
        P1, P2 = fwd_shape[0], fwd_shape[1]
        Q1, Q2 = rev_shape[-1], rev_shape[-2]
        P3 = fwd_shape[2] if len(fwd_shape) >= 3 else None
        Q3 = rev_shape[-3] if len(rev_shape) >= 3 else None
    else:
        # At the end-of-forward / start-of-reverse junction
        P1, P2 = fwd_shape[-1], fwd_shape[-2]
        Q1, Q2 = rev_shape[0], rev_shape[1]
        P3 = fwd_shape[-3] if len(fwd_shape) >= 3 else None
        Q3 = rev_shape[2] if len(rev_shape) >= 3 else None

    delta_P, delta_Q, dev, mode = compute_endpoint_fix(P1, P2, Q1, Q2, P3, Q3, tolerance_deg)

    if delta_P is None:
        logger.info(f"    {end:5s} end: OK (dev={dev:.2f}°)")
        return False, None, None

    move_p = vec_len(delta_P)
    move_q = vec_len(delta_Q)
    if mode == 'rev_only':
        detail = f"moving rev only by {move_q:.4f}m"
    elif mode == 'fwd_only':
        detail = f"moving fwd only by {move_p:.4f}m"
    else:
        detail = f"moving both by {move_p:.4f}m"
    logger.info(
        f"    {end:5s} end: NOT parallel (dev={dev:.2f}°), {detail}"
    )

    if dry_run:
        return False, None, None

    # ── Apply delta to edge shapes ──────────────────────────────────────
    if end == 'start':
        fwd_shape[0] = vec_add(fwd_shape[0], delta_P)
        rev_shape[-1] = vec_add(rev_shape[-1], delta_Q)
    else:
        fwd_shape[-1] = vec_add(fwd_shape[-1], delta_P)
        rev_shape[0] = vec_add(rev_shape[0], delta_Q)

    # ── Apply delta to lane shapes ──────────────────────────────────────
    # Forward edge lanes: adjust the same end as the edge
    fwd_lane_idx = 0 if end == 'start' else -1
    for lane_elem in fwd_elem.findall('lane'):
        lane_shape_str = lane_elem.get('shape')
        if not lane_shape_str:
            continue
        lane_shape = parse_shape(lane_shape_str)
        lane_shape[fwd_lane_idx] = vec_add(lane_shape[fwd_lane_idx], delta_P)
        lane_elem.set('shape', shape_to_str(lane_shape))

    # Reverse edge lanes: adjust the opposite end
    rev_lane_idx = -1 if end == 'start' else 0
    for lane_elem in rev_elem.findall('lane'):
        lane_shape_str = lane_elem.get('shape')
        if not lane_shape_str:
            continue
        lane_shape = parse_shape(lane_shape_str)
        lane_shape[rev_lane_idx] = vec_add(lane_shape[rev_lane_idx], delta_Q)
        lane_elem.set('shape', shape_to_str(lane_shape))

    return True, delta_P, delta_Q


# ── Junction shape update ─────────────────────────────────────────────────

def update_junction_shapes(
    root: ET.Element,
    junction_edge_deltas: Dict[str, Dict[str, Tuple[float, float]]],
) -> int:
    """
    Update junction shapes after edge endpoints have been modified.

    For each affected junction, finds the nearest edge endpoint for each
    junction shape vertex and applies the corresponding delta. Uses inverse
    distance weighting when a vertex is between multiple edges.

    Based on SUMO's NBNodeShapeComputer: junction shape vertices are derived
    from edge boundary lines (outermost lane ± half lane width). When an edge
    endpoint moves by delta, the nearby boundary also moves by delta, so the
    junction shape vertex should move by the same amount.

    Args:
        junction_edge_deltas: {junction_id: {edge_id: (dx, dy)}}

    Returns number of junctions updated.
    """
    if not junction_edge_deltas:
        return 0

    # Build junction-to-edges map (only for affected junctions)
    junc_edges: Dict[str, List[ET.Element]] = {jid: [] for jid in junction_edge_deltas}
    for edge_elem in root.findall('edge'):
        eid = edge_elem.get('id')
        if not eid or eid.startswith(':'):
            continue
        fn = edge_elem.get('from')
        tn = edge_elem.get('to')
        if fn in junc_edges:
            junc_edges[fn].append(edge_elem)
        if tn in junc_edges:
            junc_edges[tn].append(edge_elem)

    # Build junction element map
    junc_map: Dict[str, ET.Element] = {}
    for junc_elem in root.findall('junction'):
        jid = junc_elem.get('id')
        if jid in junction_edge_deltas:
            junc_map[jid] = junc_elem

    updated = 0

    for junc_id, edge_delta_map in junction_edge_deltas.items():
        junc_elem = junc_map.get(junc_id)
        if junc_elem is None:
            continue

        shape_str = junc_elem.get('shape')
        if not shape_str:
            continue
        shape = parse_shape(shape_str)
        if not shape:
            continue

        # Collect ALL edge lane endpoints at this junction, with their deltas.
        # Unmodified edges get delta=(0,0) so they anchor nearby shape points.
        endpoints: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        for edge_elem in junc_edges.get(junc_id, []):
            eid = edge_elem.get('id')
            delta = edge_delta_map.get(eid, (0.0, 0.0))
            fn = edge_elem.get('from')

            for lane_elem in edge_elem.findall('lane'):
                ls_str = lane_elem.get('shape')
                if not ls_str:
                    continue
                ls = parse_shape(ls_str)
                if not ls:
                    continue
                # Endpoint at this junction (already moved if edge was modified)
                pt = ls[0] if fn == junc_id else ls[-1]
                endpoints.append((pt, delta))

        if not endpoints:
            continue

        # Update each shape vertex using inverse-distance-weighted deltas.
        # Nearby modified edge endpoints pull the vertex; nearby unmodified
        # endpoints (delta=0) anchor it in place.
        new_shape = []
        any_moved = False

        for sp in shape:
            total_weight = 0.0
            weighted_dx = 0.0
            weighted_dy = 0.0

            for pt, delta in endpoints:
                d = vec_len(vec_sub(sp, pt))
                if d < 1e-6:
                    # Exact match — use this delta directly
                    weighted_dx = delta[0]
                    weighted_dy = delta[1]
                    total_weight = 1.0
                    break
                w = 1.0 / (d * d)  # inverse distance squared
                weighted_dx += delta[0] * w
                weighted_dy += delta[1] * w
                total_weight += w

            if total_weight > 0:
                avg_dx = weighted_dx / total_weight
                avg_dy = weighted_dy / total_weight
                if abs(avg_dx) > 1e-4 or abs(avg_dy) > 1e-4:
                    any_moved = True
                new_shape.append((sp[0] + avg_dx, sp[1] + avg_dy))
            else:
                new_shape.append(sp)

        if any_moved:
            junc_elem.set('shape', shape_to_str(new_shape))
            updated += 1

    return updated




# ── Network Validation (inspired by SUMO netconvert) ─────────────────────

def _segment_angle(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Absolute angle of segment a→b in radians."""
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _angle_diff(a: float, b: float) -> float:
    """Signed shortest angle difference in radians, range [-pi, pi]."""
    d = b - a
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def _segments_intersect(
    a1: Tuple[float, float], a2: Tuple[float, float],
    b1: Tuple[float, float], b2: Tuple[float, float],
) -> bool:
    """Check if segments a1-a2 and b1-b2 properly intersect (not just touch)."""
    d1 = cross2d(vec_sub(b2, b1), vec_sub(a1, b1))
    d2 = cross2d(vec_sub(b2, b1), vec_sub(a2, b1))
    d3 = cross2d(vec_sub(a2, a1), vec_sub(b1, a1))
    d4 = cross2d(vec_sub(a2, a1), vec_sub(b2, a1))
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return False


def _polygon_area_signed(pts: List[Tuple[float, float]]) -> float:
    """Signed area of polygon using shoelace formula."""
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return area / 2.0


def validate_edge_geometry(
    edge_elem: ET.Element,
    max_angle_deg: float = 99.0,
    min_segment_len: float = 0.1,
) -> List[str]:
    """
    Validate edge geometry (inspired by NBEdge::checkGeometry).

    Checks:
      - Sharp angles between consecutive segments (> max_angle_deg)
      - Zero-length or very short segments (< min_segment_len)
      - Self-intersecting edge shapes

    Returns list of warning strings.
    """
    warnings = []
    edge_id = edge_elem.get('id', '?')
    shape_str = edge_elem.get('shape')
    if not shape_str:
        return warnings

    shape = parse_shape(shape_str)
    if len(shape) < 2:
        warnings.append(f"Edge '{edge_id}': shape has fewer than 2 points")
        return warnings

    # -- Zero-length / very short segments --
    for i in range(len(shape) - 1):
        seg_len = vec_len(vec_sub(shape[i + 1], shape[i]))
        if seg_len < 1e-6:
            warnings.append(
                f"Edge '{edge_id}': zero-length segment at index {i}"
            )
        elif seg_len < min_segment_len:
            warnings.append(
                f"Edge '{edge_id}': very short segment ({seg_len:.4f}m) at index {i}"
            )

    # -- Sharp angles between consecutive segments --
    if len(shape) >= 3:
        angles = []
        for i in range(len(shape) - 1):
            angles.append(_segment_angle(shape[i], shape[i + 1]))
        for i in range(len(angles) - 1):
            rel_angle = abs(_angle_diff(angles[i], angles[i + 1]))
            if rel_angle > math.radians(max_angle_deg):
                warnings.append(
                    f"Edge '{edge_id}': sharp angle ({math.degrees(rel_angle):.1f}°) "
                    f"at segment {i}"
                )

    # -- Self-intersection of edge shape --
    if len(shape) >= 4:
        for i in range(len(shape) - 1):
            for j in range(i + 2, len(shape) - 1):
                if i == 0 and j == len(shape) - 2:
                    continue  # skip first/last (they share no endpoint)
                if _segments_intersect(shape[i], shape[i + 1], shape[j], shape[j + 1]):
                    warnings.append(
                        f"Edge '{edge_id}': self-intersection between "
                        f"segments {i} and {j}"
                    )

    return warnings


def validate_lane_consistency(
    edge_elem: ET.Element,
    max_endpoint_drift: float = 2.0,
) -> List[str]:
    """
    Validate lane shapes are consistent with their edge shape.

    Checks that lane endpoints haven't drifted too far from the edge
    endpoints (beyond normal lane offset). Large drift indicates a
    lane that wasn't properly updated when the edge was modified.

    Returns list of warning strings.
    """
    warnings = []
    edge_id = edge_elem.get('id', '?')
    shape_str = edge_elem.get('shape')
    if not shape_str:
        return warnings

    edge_shape = parse_shape(shape_str)
    if len(edge_shape) < 2:
        return warnings

    lanes = edge_elem.findall('lane')
    num_lanes = len(lanes)
    if num_lanes == 0:
        return warnings

    # Estimate max reasonable offset: (num_lanes) * lane_width
    # Typical lane width is ~3.2m; allow generous margin
    max_lane_offset = num_lanes * 5.0

    for lane_elem in lanes:
        lane_id = lane_elem.get('id', '?')
        ls_str = lane_elem.get('shape')
        if not ls_str:
            continue
        lane_shape = parse_shape(ls_str)
        if len(lane_shape) < 2:
            continue

        # Check start endpoint drift
        d_start = vec_len(vec_sub(lane_shape[0], edge_shape[0]))
        if d_start > max_lane_offset + max_endpoint_drift:
            warnings.append(
                f"Lane '{lane_id}': start point {d_start:.2f}m from edge start "
                f"(expected < {max_lane_offset + max_endpoint_drift:.1f}m)"
            )

        # Check end endpoint drift
        d_end = vec_len(vec_sub(lane_shape[-1], edge_shape[-1]))
        if d_end > max_lane_offset + max_endpoint_drift:
            warnings.append(
                f"Lane '{lane_id}': end point {d_end:.2f}m from edge end "
                f"(expected < {max_lane_offset + max_endpoint_drift:.1f}m)"
            )

    return warnings


def validate_junction_shapes(
    root: ET.Element,
    junction_ids: Optional[Set[str]] = None,
) -> List[str]:
    """
    Validate junction shapes (inspired by NBNodeShapeComputer).

    Checks:
      - NaN values in shape coordinates
      - Degenerate shapes (zero area, too few points)
      - Self-intersecting junction polygons

    Args:
        junction_ids: if given, only validate these junctions;
                      otherwise validate all.

    Returns list of warning strings.
    """
    warnings = []

    for junc_elem in root.findall('junction'):
        jid = junc_elem.get('id', '?')
        if junction_ids is not None and jid not in junction_ids:
            continue

        shape_str = junc_elem.get('shape')
        if not shape_str:
            continue

        shape = parse_shape(shape_str)

        # -- NaN check --
        for i, (x, y) in enumerate(shape):
            if math.isnan(x) or math.isnan(y):
                warnings.append(
                    f"Junction '{jid}': NaN coordinate at shape point {i}"
                )

        # -- Degenerate shape --
        if len(shape) < 3:
            # Only warn for non-internal junctions
            jtype = junc_elem.get('type', '')
            if jtype != 'internal':
                warnings.append(
                    f"Junction '{jid}': degenerate shape ({len(shape)} points)"
                )
            continue

        area = abs(_polygon_area_signed(shape))
        if area < 0.01:
            warnings.append(
                f"Junction '{jid}': near-zero area ({area:.4f} m²)"
            )

        # -- Self-intersection --
        n = len(shape)
        for i in range(n):
            ni = (i + 1) % n
            for j in range(i + 2, n):
                nj = (j + 1) % n
                if ni == j or nj == i:
                    continue  # adjacent segments share a vertex
                if _segments_intersect(shape[i], shape[ni], shape[j], shape[nj]):
                    warnings.append(
                        f"Junction '{jid}': self-intersecting shape "
                        f"(segments {i}-{ni} and {j}-{nj})"
                    )

    return warnings


def validate_endpoint_node_distance(
    root: ET.Element,
    center_edges: Dict[str, ET.Element],
    max_distance: float = 50.0,
) -> List[str]:
    """
    Validate edge endpoints are within reasonable distance of their junction node.

    After modifying endpoints, they should still be near the junction position.
    Inspired by NBNodeShapeComputer's distance check.

    Returns list of warning strings.
    """
    warnings = []

    # Build node position map
    node_pos: Dict[str, Tuple[float, float]] = {}
    for junc_elem in root.findall('junction'):
        jid = junc_elem.get('id', '')
        x_str = junc_elem.get('x')
        y_str = junc_elem.get('y')
        if jid and x_str and y_str:
            try:
                node_pos[jid] = (float(x_str), float(y_str))
            except ValueError:
                pass

    for eid, edge_elem in center_edges.items():
        shape_str = edge_elem.get('shape')
        if not shape_str:
            continue
        shape = parse_shape(shape_str)
        if len(shape) < 2:
            continue

        fn = edge_elem.get('from', '')
        tn = edge_elem.get('to', '')

        if fn in node_pos:
            d = vec_len(vec_sub(shape[0], node_pos[fn]))
            if d > max_distance:
                warnings.append(
                    f"Edge '{eid}': start point {d:.2f}m from node '{fn}' "
                    f"(max {max_distance}m)"
                )

        if tn in node_pos:
            d = vec_len(vec_sub(shape[-1], node_pos[tn]))
            if d > max_distance:
                warnings.append(
                    f"Edge '{eid}': end point {d:.2f}m from node '{tn}' "
                    f"(max {max_distance}m)"
                )

    return warnings


def validate_pair_alignment(
    pairs: List[Tuple[str, str]],
    center_edges: Dict[str, ET.Element],
    tolerance_deg: float = 1.0,
) -> List[str]:
    """
    Re-verify that all edge pair endpoints are now parallel after fixing.

    This is the verification pass: any remaining non-parallel endpoints
    indicate the fix was incomplete or introduced new issues.

    Returns list of warning strings.
    """
    warnings = []

    for fwd_id, rev_id in pairs:
        fwd_elem = center_edges.get(fwd_id)
        rev_elem = center_edges.get(rev_id)
        if fwd_elem is None or rev_elem is None:
            continue

        fwd_shape_str = fwd_elem.get('shape')
        rev_shape_str = rev_elem.get('shape')
        if not fwd_shape_str or not rev_shape_str:
            continue

        fwd_shape = parse_shape(fwd_shape_str)
        rev_shape = parse_shape(rev_shape_str)
        if len(fwd_shape) < 2 or len(rev_shape) < 2:
            continue

        # Check start end
        d_f = vec_sub(fwd_shape[1], fwd_shape[0])
        d_r = vec_sub(rev_shape[-1], rev_shape[-2])
        dev = parallel_deviation_deg(d_f, d_r)
        if dev >= tolerance_deg:
            warnings.append(
                f"Pair '{fwd_id}' <-> '{rev_id}': start end still not parallel "
                f"({dev:.2f}°)"
            )

        # Check end end
        d_f = vec_sub(fwd_shape[-2], fwd_shape[-1])
        d_r = vec_sub(rev_shape[0], rev_shape[1])
        dev = parallel_deviation_deg(d_f, d_r)
        if dev >= tolerance_deg:
            warnings.append(
                f"Pair '{fwd_id}' <-> '{rev_id}': end end still not parallel "
                f"({dev:.2f}°)"
            )

    return warnings


def validate_network(
    root: ET.Element,
    center_edges: Dict[str, ET.Element],
    pairs: List[Tuple[str, str]],
    affected_junctions: Optional[Set[str]] = None,
    tolerance_deg: float = 1.0,
    max_angle_deg: float = 99.0,
) -> Dict[str, List[str]]:
    """
    Run all post-fix network validation checks.

    Inspired by SUMO netconvert validation (NBEdge::checkGeometry,
    NBEdgeCont::checkOverlap, NBNodeShapeComputer, NBEdge::recheckLanes).

    Returns dict of {check_name: [warning_strings]}.
    """
    results: Dict[str, List[str]] = {}

    # 1. Edge geometry: sharp angles, zero-length segments, self-intersection
    geom_warnings = []
    for eid, elem in center_edges.items():
        geom_warnings.extend(validate_edge_geometry(elem, max_angle_deg))
    results['edge_geometry'] = geom_warnings

    # 2. Lane-edge consistency
    lane_warnings = []
    for eid, elem in center_edges.items():
        lane_warnings.extend(validate_lane_consistency(elem))
    results['lane_consistency'] = lane_warnings

    # 3. Junction shape validity
    results['junction_shapes'] = validate_junction_shapes(root, affected_junctions)

    # 4. Edge endpoint-to-node distance
    results['endpoint_node_distance'] = validate_endpoint_node_distance(
        root, center_edges
    )

    # 5. Pair alignment verification (re-check parallelism)
    results['pair_alignment'] = validate_pair_alignment(
        pairs, center_edges, tolerance_deg
    )

    return results


def print_validation_report(results: Dict[str, List[str]]) -> int:
    """Print validation results and return total warning count."""
    total = 0
    check_labels = {
        'edge_geometry': 'Edge Geometry (sharp angles, short segments, self-intersection)',
        'lane_consistency': 'Lane-Edge Consistency',
        'junction_shapes': 'Junction Shape Validity',
        'endpoint_node_distance': 'Endpoint-to-Node Distance',
        'pair_alignment': 'Pair Alignment Verification',
    }

    logger.info("\n" + "=" * 60)
    logger.info("  NETWORK VALIDATION REPORT")
    logger.info("=" * 60)

    for check_name, warnings in results.items():
        label = check_labels.get(check_name, check_name)
        count = len(warnings)
        total += count
        status = "PASS" if count == 0 else f"WARN ({count})"
        logger.info(f"\n  [{status}] {label}")
        for w in warnings[:20]:  # limit output per check
            logger.warning(f"    {w}")
        if count > 20:
            logger.warning(f"    ... and {count - 20} more")

    logger.info("\n" + "-" * 60)
    if total == 0:
        logger.info("  All checks passed — network is valid")
    else:
        logger.info(f"  Total warnings: {total}")
    logger.info("=" * 60)

    return total


def main():
    parser = argparse.ArgumentParser(
        description='Fix parallel endpoints for center-lane edges in SUMO net.xml',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
After adding center lanes and running netconvert, edge endpoints at junctions
may not be parallel. This script slides the endpoints along the connection
line between the forward and reverse edge until the segments become parallel.

Example:
    python fix_parallel_endpoints.py Ann_Arbor/aa_modified.net.xml
    python fix_parallel_endpoints.py aa.net.xml -o aa_fixed.net.xml --tolerance 0.5
        """,
    )

    parser.add_argument('input_file', help='Input SUMO net.xml file')
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output file (default: overwrites input)',
    )
    parser.add_argument(
        '--tolerance', type=float, default=1.0,
        help='Parallelism tolerance in degrees (default: 1.0)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Report issues without modifying the file',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable verbose logging',
    )
    parser.add_argument(
        '--no-validate', action='store_true',
        help='Skip post-fix network validation',
    )
    parser.add_argument(
        '--max-angle', type=float, default=99.0,
        help='Max angle (degrees) before warning in geometry check (default: 99.0)',
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Reading: {args.input_file}")
    tree = ET.parse(args.input_file)
    root = tree.getroot()

    # ── Find center-lane edges ──────────────────────────────────────────
    center_edges: Dict[str, ET.Element] = {}
    for edge_elem in root.findall('edge'):
        edge_id = edge_elem.get('id')
        if not edge_id or edge_id.startswith(':'):
            continue
        for param in edge_elem.findall('param'):
            if param.get('key') == 'centerLaneAdded' and param.get('value') == 'true':
                center_edges[edge_id] = edge_elem
                break

    logger.info(f"Found {len(center_edges)} edges with centerLaneAdded=true")

    if not center_edges:
        logger.info("Nothing to fix")
        return

    # ── Pair forward/reverse edges by matching from/to nodes ────────────
    edge_nodes: Dict[str, Tuple[str, str]] = {}
    for eid, elem in center_edges.items():
        fn = elem.get('from')
        tn = elem.get('to')
        if fn and tn:
            edge_nodes[eid] = (fn, tn)

    pairs: List[Tuple[str, str]] = []
    paired: Set[str] = set()
    for eid, (fn, tn) in edge_nodes.items():
        if eid in paired:
            continue
        for rid, (rfn, rtn) in edge_nodes.items():
            if rid == eid or rid in paired:
                continue
            if rfn == tn and rtn == fn:
                pairs.append((eid, rid))
                paired.add(eid)
                paired.add(rid)
                break

    unpaired = set(center_edges.keys()) - paired
    if unpaired:
        logger.warning(
            f"  {len(unpaired)} center-lane edge(s) without a reverse pair"
        )

    logger.info(f"Found {len(pairs)} edge pairs to check")

    # ── Check and fix each pair ─────────────────────────────────────────
    total_fixed = 0
    # Track junction modifications: {junc_id: {edge_id: (dx, dy)}}
    junction_edge_deltas: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for fwd_id, rev_id in pairs:
        fwd_elem = center_edges[fwd_id]
        rev_elem = center_edges[rev_id]

        fwd_shape_str = fwd_elem.get('shape')
        rev_shape_str = rev_elem.get('shape')

        if not fwd_shape_str or not rev_shape_str:
            logger.warning(f"  Pair {fwd_id} <-> {rev_id}: missing shape — skipping")
            continue

        fwd_shape = parse_shape(fwd_shape_str)
        rev_shape = parse_shape(rev_shape_str)

        if len(fwd_shape) < 2 or len(rev_shape) < 2:
            logger.warning(f"  Pair {fwd_id} <-> {rev_id}: shape too short — skipping")
            continue

        pair_fixed = 0

        logger.info(f"\n  Pair: {fwd_id} <-> {rev_id}")

        for end in ['start', 'end']:
            fixed, delta_fwd, delta_rev = fix_edge_pair_end(
                fwd_elem, rev_elem,
                fwd_shape, rev_shape,
                end, args.tolerance, args.dry_run,
            )
            if fixed:
                pair_fixed += 1
                # Record which junction was affected
                if end == 'start':
                    junc_id = fwd_elem.get('from')
                else:
                    junc_id = fwd_elem.get('to')
                if junc_id:
                    if junc_id not in junction_edge_deltas:
                        junction_edge_deltas[junc_id] = {}
                    junction_edge_deltas[junc_id][fwd_id] = delta_fwd
                    junction_edge_deltas[junc_id][rev_id] = delta_rev

        # Update edge shapes if any end was fixed
        if pair_fixed > 0 and not args.dry_run:
            fwd_elem.set('shape', shape_to_str(fwd_shape))
            rev_elem.set('shape', shape_to_str(rev_shape))

        total_fixed += pair_fixed

    # ── Update junction shapes ───────────────────────────────────────
    junctions_updated = 0
    if not args.dry_run and junction_edge_deltas:
        logger.info(f"\nUpdating junction shapes for {len(junction_edge_deltas)} affected junctions...")
        junctions_updated = update_junction_shapes(root, junction_edge_deltas)
        logger.info(f"Updated {junctions_updated} junction shape(s)")

    # ── Network validation ────────────────────────────────────────────
    if not args.no_validate and not args.dry_run and total_fixed > 0:
        affected_junctions = set(junction_edge_deltas.keys()) if junction_edge_deltas else None
        validation_results = validate_network(
            root, center_edges, pairs,
            affected_junctions=affected_junctions,
            tolerance_deg=args.tolerance,
            max_angle_deg=args.max_angle,
        )
        print_validation_report(validation_results)

    # ── Write output ────────────────────────────────────────────────────
    output_file = args.output or args.input_file

    uturns_removed = 0
    any_changes = total_fixed > 0 or uturns_removed > 0
    if not args.dry_run and any_changes:
        logger.info(f"\nWriting: {output_file}")
        ET.indent(tree, space="    ")
        tree.write(output_file, encoding='UTF-8', xml_declaration=True)

    logger.info(
        f"\n{'[DRY RUN] ' if args.dry_run else ''}"
        f"Fixed {total_fixed} endpoint(s) across {len(pairs)} edge pairs"
        f"{f', updated {junctions_updated} junction(s)' if junctions_updated else ''}"
        f"{f', removed {uturns_removed} U-turn connection(s)' if uturns_removed else ''}"
    )


if __name__ == '__main__':
    main()
