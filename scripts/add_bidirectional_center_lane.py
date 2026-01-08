#!/usr/bin/env python3
"""
SUMO Bidirectional Center Lane Creator

This script adds a bidirectional center lane to roads that should have:
- 2 lanes in each direction (on the sides) - EXISTING, kept as-is
- 1 shared bidirectional lane in the middle - NEW, added by this script

For roads like West Huron Street where the original data shows a center lane
that can be used in either direction, but SUMO conversion only created
the single-directional lanes on each side.

The script:
1. Takes a pair of edges (forward and reverse)
2. Adds one lane to each edge (inserting it as the new center lane)
3. The center lanes overlap exactly (spreadType="center") for bidirectional use
4. Shifts existing lane indices and updates all connections

Lane layout change:
  Before: Edge has lanes [0, 1] (2 lanes)
  After:  Edge has lanes [0, 1, 2] (3 lanes)
          - Lane 0: rightmost lane (original lane 0)
          - Lane 1: NEW center bidirectional lane (overlaps with reverse edge)
          - Lane 2: leftmost lane (original lane 1, shifted)

Author: TeraSim Team
License: MIT
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from copy import deepcopy
import argparse
import logging
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EdgeInfo:
    """Information about an edge"""
    id: str
    from_node: str
    to_node: str
    shape: str
    num_lanes: int
    element: ET.Element
    name: Optional[str] = None
    speed: Optional[str] = None
    priority: Optional[str] = None
    edge_type: Optional[str] = None

    @property
    def reverse_node_pair(self) -> Tuple[str, str]:
        """Return reverse node pair"""
        return (self.to_node, self.from_node)


def parse_shape_coordinates(shape_str: str) -> List[Tuple[float, float]]:
    """Parse shape string into list of coordinate tuples."""
    if not shape_str:
        return []
    coords = []
    for coord_pair in shape_str.strip().split():
        x, y = coord_pair.split(',')
        coords.append((float(x), float(y)))
    return coords


def reverse_shape(shape_str: str) -> str:
    """Reverse a shape string."""
    if not shape_str:
        return ""
    coords = parse_shape_coordinates(shape_str)
    reversed_coords = coords[::-1]
    return ' '.join(f"{x:.2f},{y:.2f}" for x, y in reversed_coords)


def coords_to_shape(coords: List[Tuple[float, float]]) -> str:
    """Convert coordinate list to shape string."""
    return ' '.join(f"{x:.2f},{y:.2f}" for x, y in coords)


def offset_shape(shape_str: str, offset_distance: float, direction: str = 'right') -> str:
    """
    Offset a shape perpendicular to its overall direction.

    Uses a single consistent offset vector based on the overall edge direction
    (from start to end point) to preserve the original shape geometry.

    Args:
        shape_str: Original shape string
        offset_distance: Distance to offset (in meters, typically lane width ~3.2m)
        direction: 'right' or 'left' relative to the edge direction

    Returns:
        New shape string with offset coordinates
    """
    import math

    coords = parse_shape_coordinates(shape_str)
    if len(coords) < 2:
        return shape_str

    # Calculate overall direction from start to end point
    start_x, start_y = coords[0]
    end_x, end_y = coords[-1]
    dx = end_x - start_x
    dy = end_y - start_y

    # Normalize direction vector
    length = math.sqrt(dx * dx + dy * dy)
    if length > 0:
        dx /= length
        dy /= length
    else:
        dx, dy = 1, 0

    # Calculate perpendicular vector (rotate 90 degrees)
    # For 'right': rotate clockwise (-90 degrees): (dx, dy) -> (dy, -dx)
    # For 'left': rotate counter-clockwise (+90 degrees): (dx, dy) -> (-dy, dx)
    if direction == 'right':
        perp_x = dy
        perp_y = -dx
    else:  # left
        perp_x = -dy
        perp_y = dx

    # Apply the same offset to all points to preserve shape
    offset_coords = []
    for x, y in coords:
        new_x = x + perp_x * offset_distance
        new_y = y + perp_y * offset_distance
        offset_coords.append((new_x, new_y))

    return coords_to_shape(offset_coords)


def find_reverse_edge(edge_id: str, edge_map: Dict[str, EdgeInfo]) -> Optional[EdgeInfo]:
    """Find the reverse edge for a given edge ID."""
    if edge_id not in edge_map:
        return None

    edge = edge_map[edge_id]
    reverse_node_pair = edge.reverse_node_pair

    for other_id, other_edge in edge_map.items():
        if other_id == edge_id:
            continue
        if (other_edge.from_node, other_edge.to_node) == reverse_node_pair:
            return other_edge

    return None


def shift_specified_edges(
    edge_map: Dict[str, EdgeInfo],
    edges_up: List[str],
    edges_down: List[str],
    shift_distance: float,
    dry_run: bool = False
) -> int:
    """
    Shift specified edges up or down by half lane width.

    Both the edge and its reverse are shifted in the same absolute direction
    (perpendicular to the road), keeping them together as a pair.

    Args:
        edge_map: Dictionary of edge ID to EdgeInfo
        edges_up: List of edge IDs to shift UP
        edges_down: List of edge IDs to shift DOWN
        shift_distance: Distance to shift (typically lane_width / 2)
        dry_run: If True, only report changes

    Returns:
        Number of edges shifted
    """
    shifted_count = 0

    # Helper to shift an edge pair in the same absolute direction
    def shift_edge_pair(edge_id: str, shift_up: bool) -> int:
        if edge_id not in edge_map:
            logger.warning(f"Edge not found: {edge_id}")
            return 0

        edge = edge_map[edge_id]
        reverse_edge = find_reverse_edge(edge_id, edge_map)

        dir_name = "UP" if shift_up else "DOWN"
        logger.info(f"    Shifting {dir_name}: {edge.id}" + (f" & {reverse_edge.id}" if reverse_edge else ""))

        count = 0
        if not dry_run:
            # For the primary edge: shift right if UP, left if DOWN
            primary_direction = 'right' if shift_up else 'left'
            edge.element.set('shape', offset_shape(edge.shape, shift_distance, direction=primary_direction))
            count += 1

            if reverse_edge:
                # For the reverse edge: shift in OPPOSITE direction relative to its own direction
                # This keeps both edges moving together in the same absolute direction
                reverse_direction = 'left' if shift_up else 'right'
                reverse_edge.element.set('shape', offset_shape(reverse_edge.shape, shift_distance, direction=reverse_direction))
                count += 1

        return count if not dry_run else (2 if reverse_edge else 1)

    if edges_up or edges_down:
        logger.info(f"\nShifting specified edges:")

    # Shift edges UP
    for edge_id in edges_up:
        shifted_count += shift_edge_pair(edge_id, shift_up=True)

    # Shift edges DOWN
    for edge_id in edges_down:
        shifted_count += shift_edge_pair(edge_id, shift_up=False)

    return shifted_count


def add_center_lane_to_edges(
    input_edge_file: str,
    output_edge_file: str,
    edge_ids: List[str],
    dry_run: bool = False,
    lane_width: float = 3.2,
    edges_shift_up: Optional[List[str]] = None,
    edges_shift_down: Optional[List[str]] = None
) -> int:
    """
    Add a bidirectional center lane to specified edge pairs.

    For each specified edge:
    1. Find its reverse edge
    2. Offset both edges outward to create a gap in the middle
    3. Add one lane to each edge (making them 3 lanes if they were 2)
    4. The new center lane (index 1) occupies the gap between the two edges

    The geometry modification:
    - Edge 1 is shifted to the RIGHT by (lane_width / 2)
    - Edge 2 is shifted to the RIGHT by (lane_width / 2) - which is LEFT relative to Edge 1
    - This creates a gap of lane_width in the middle for the bidirectional center lane

    Args:
        input_edge_file: Path to input edge XML file
        output_edge_file: Path to output edge XML file
        edge_ids: List of edge IDs to modify
        dry_run: If True, only report what would be changed
        lane_width: Width of the center lane in meters (default 3.2m)

    Returns:
        Number of edge pairs modified
    """
    logger.info(f"Reading edge file: {input_edge_file}")

    tree = ET.parse(input_edge_file)
    root = tree.getroot()

    # Build edge map
    edge_map: Dict[str, EdgeInfo] = {}

    for edge_elem in root.findall('edge'):
        edge_id = edge_elem.get('id')
        from_node = edge_elem.get('from')
        to_node = edge_elem.get('to')

        if edge_id is None or from_node is None or to_node is None:
            continue

        edge_map[edge_id] = EdgeInfo(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            shape=edge_elem.get('shape', ''),
            num_lanes=int(edge_elem.get('numLanes', '1')),
            element=edge_elem,
            name=edge_elem.get('name'),
            speed=edge_elem.get('speed'),
            priority=edge_elem.get('priority'),
            edge_type=edge_elem.get('type')
        )

    logger.info(f"Found {len(edge_map)} total edges")

    # Process specified edges
    processed: Set[str] = set()
    modified_count = 0

    for edge_id in edge_ids:
        if edge_id in processed:
            continue

        if edge_id not in edge_map:
            logger.warning(f"Edge not found: {edge_id}")
            continue

        edge = edge_map[edge_id]
        reverse_edge = find_reverse_edge(edge_id, edge_map)

        if not reverse_edge:
            logger.warning(f"No reverse edge found for {edge_id}")
            continue

        logger.info(f"\nProcessing edge pair: {edge.id} <-> {reverse_edge.id}")
        logger.info(f"  Name: {edge.name}")
        logger.info(f"  Current lanes: {edge.num_lanes} / {reverse_edge.num_lanes}")

        # Calculate new lane count (add 1 for center lane)
        new_lane_count = edge.num_lanes + 1

        logger.info(f"  New lanes: {new_lane_count} (adding center bidirectional lane)")

        # Calculate offset to make only the center (leftmost) lane overlap
        # With spreadType="center", lanes spread equally on both sides of the shape
        #
        # For N lanes with width W, the leftmost lane (index N-1) center is at:
        #   position = (N - 1) / 2 * W  (to the left of shape centerline)
        #
        # After adding 1 lane (N+1 total), the new leftmost lane center is at:
        #   position = N / 2 * W
        #
        # To make the leftmost lanes of both edges overlap at the road center,
        # we offset each edge to the RIGHT by: ((N - 1) / 2) * W
        # Each edge shifts, bringing them one lane closer together.
        #
        # Example: 3 lanes -> 4 lanes, W = 3.2m
        #   offset = ((3 - 1) / 2) * 3.2 = 3.2m
        #   Each edge shifts right 3.2m, their leftmost lanes now overlap at center
        offset_distance = ((edge.num_lanes - 2) / 2) * lane_width

        logger.info(f"  Offset distance: {offset_distance:.2f}m (to overlap only center lane)")

        if not dry_run:
            # Get original shapes
            original_shape = edge.shape
            original_reverse_shape = reverse_edge.shape

            # If reverse edge doesn't have a shape, derive it from the forward edge
            if not original_reverse_shape and original_shape:
                original_reverse_shape = reverse_shape(original_shape)

            # Offset edge 1 to the RIGHT (shifts lanes toward right side of road)
            offset_shape_1 = offset_shape(original_shape, offset_distance, direction='right')

            # Offset edge 2 to the RIGHT (relative to its direction)
            # IMPORTANT: Use the actual reverse edge's shape, not a reversed copy of forward edge
            # This preserves the correct endpoint positions for each edge
            offset_shape_2 = offset_shape(original_reverse_shape, offset_distance, direction='right')

            logger.info(f"  Original shape (edge 1): {original_shape[:50]}...")
            logger.info(f"  Offset shape (edge 1):   {offset_shape_1[:50]}...")
            logger.info(f"  Offset shape (edge 2):   {offset_shape_2[:50]}...")

            # Update edge 1
            edge.element.set('numLanes', str(new_lane_count))
            edge.element.set('shape', offset_shape_1)
            edge.element.set('spreadType', 'center')

            # Update edge 2
            reverse_edge.element.set('numLanes', str(new_lane_count))
            reverse_edge.element.set('shape', offset_shape_2)
            reverse_edge.element.set('spreadType', 'center')

            # Update lane elements if they exist
            _update_lane_elements(edge.element, new_lane_count)
            _update_lane_elements(reverse_edge.element, new_lane_count)

            logger.info(f"  ✓ Updated edge {edge.id} - shifted right by {offset_distance:.2f}m")
            logger.info(f"  ✓ Updated edge {reverse_edge.id} - shifted right by {offset_distance:.2f}m")

        processed.add(edge_id)
        processed.add(reverse_edge.id)
        modified_count += 1

    # Shift connected edges if manually specified
    if (edges_shift_up or edges_shift_down) and modified_count > 0:
        shift_distance = lane_width / 2  # Half lane width
        shift_specified_edges(
            edge_map,
            edges_shift_up or [],
            edges_shift_down or [],
            shift_distance,
            dry_run
        )

    if not dry_run and modified_count > 0:
        logger.info(f"\nWriting modified edge file: {output_edge_file}")
        ET.indent(tree, space="    ")
        tree.write(output_edge_file, encoding='UTF-8', xml_declaration=True)

    logger.info(f"\n{'[DRY RUN] ' if dry_run else ''}Modified {modified_count} edge pairs")

    return modified_count


def _update_lane_elements(edge_elem: ET.Element, new_lane_count: int) -> None:
    """Update lane child elements to match new lane count."""
    existing_lanes = edge_elem.findall('lane')

    if not existing_lanes:
        return

    # Get the last lane as template
    template_lane = existing_lanes[-1]
    orig_id = None
    for param in template_lane.findall('param'):
        if param.get('key') == 'origId':
            orig_id = param.get('value')
            break

    # Remove existing lanes
    for lane in existing_lanes:
        edge_elem.remove(lane)

    # Add new lanes
    for i in range(new_lane_count):
        new_lane = ET.SubElement(edge_elem, 'lane')
        new_lane.set('index', str(i))
        if orig_id:
            param = ET.SubElement(new_lane, 'param')
            param.set('key', 'origId')
            param.set('value', orig_id)


def calculate_edge_angle(shape_str: str, at_end: bool = True) -> float:
    """
    Calculate the heading angle of an edge based on its shape.

    Args:
        shape_str: Shape string
        at_end: If True, calculate angle at the end of the edge; otherwise at the start

    Returns:
        Angle in degrees (0-360, where 0 is East, 90 is North)
    """
    import math

    if not shape_str:
        return 0.0

    coords = parse_shape_coordinates(shape_str)
    if len(coords) < 2:
        return 0.0

    if at_end:
        # Use last two points to determine heading at the end
        x1, y1 = coords[-2]
        x2, y2 = coords[-1]
    else:
        # Use first two points to determine heading at the start
        x1, y1 = coords[0]
        x2, y2 = coords[1]

    angle = math.atan2(y2 - y1, x2 - x1)
    return math.degrees(angle) % 360


def determine_turn_direction(from_angle: float, to_angle: float) -> str:
    """
    Determine the turn direction based on incoming and outgoing angles.

    Returns: 'r' (right), 's' (straight), 'l' (left), or 't' (u-turn)
    """
    # Calculate the turn angle
    turn = (to_angle - from_angle) % 360

    # Normalize to -180 to 180
    if turn > 180:
        turn -= 360

    # Classify the turn
    if -45 <= turn <= 45:
        return 's'  # straight
    elif 45 < turn <= 135:
        return 'l'  # left
    elif -135 <= turn < -45:
        return 'r'  # right
    else:
        return 't'  # u-turn (turn around)


def update_connections_for_center_lane(
    input_edge_file: str,
    input_con_file: str,
    output_con_file: str,
    edge_ids: List[str],
    dry_run: bool = False,
    skip_updates: bool = False
) -> int:
    """
    Update connections after adding center lane.

    When we add a center lane, we need to:
    1. Shift existing lane indices to make room for the new lane
    2. Assign left turns and U-turns to the new center lane (leftmost lane)
    3. Remove left turns and U-turns from other lanes

    Args:
        input_edge_file: Path to edge XML file
        input_con_file: Path to connection XML file
        output_con_file: Path to output connection XML file
        edge_ids: List of edge IDs that were modified
        dry_run: If True, only report changes
        skip_updates: If True, don't modify connections at all

    Returns:
        Number of connections modified
    """
    if skip_updates:
        logger.info("\nSkipping connection updates (--skip-connection-updates specified)")
        return 0

    logger.info(f"\nReading edge file: {input_edge_file}")
    logger.info(f"Reading connection file: {input_con_file}")

    # Parse edge file to get edge info
    edge_tree = ET.parse(input_edge_file)
    edge_root = edge_tree.getroot()

    edge_map: Dict[str, EdgeInfo] = {}
    for edge_elem in edge_root.findall('edge'):
        edge_id = edge_elem.get('id')
        from_node = edge_elem.get('from')
        to_node = edge_elem.get('to')

        if edge_id is None or from_node is None or to_node is None:
            continue

        edge_map[edge_id] = EdgeInfo(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            shape=edge_elem.get('shape', ''),
            num_lanes=int(edge_elem.get('numLanes', '1')),
            element=edge_elem,
            name=edge_elem.get('name')
        )

    # Find reverse edges for all specified edges
    edge_pairs: Dict[str, str] = {}
    for edge_id in edge_ids:
        reverse = find_reverse_edge(edge_id, edge_map)
        if reverse:
            edge_pairs[edge_id] = reverse.id
            edge_pairs[reverse.id] = edge_id

    modified_edges = set(edge_pairs.keys())

    # Parse connection file
    con_tree = ET.parse(input_con_file)
    con_root = con_tree.getroot()

    modified_count = 0

    # First pass: analyze all connections and determine turn directions
    # Collect connection info for each edge
    edge_conn_info: Dict[str, List[Tuple[ET.Element, str, int, Optional[str], int]]] = {}

    for conn_elem in con_root.findall('connection'):
        from_edge = conn_elem.get('from')
        to_edge = conn_elem.get('to')

        if from_edge is None or to_edge is None:
            continue

        # Only process connections FROM modified edges
        if from_edge not in modified_edges:
            continue

        from_lane = int(conn_elem.get('fromLane', '0'))
        to_lane = int(conn_elem.get('toLane', '0'))

        # Get the edge info
        if from_edge not in edge_map:
            continue

        from_edge_info = edge_map[from_edge]
        reverse_edge_id = edge_pairs.get(from_edge)

        # Determine the turn direction
        turn_direction = None

        # Check if this is a U-turn to the reverse edge
        if reverse_edge_id and to_edge == reverse_edge_id:
            turn_direction = 't'
        elif to_edge in edge_map:
            to_edge_info = edge_map[to_edge]
            from_angle = calculate_edge_angle(from_edge_info.shape, at_end=True)
            to_angle = calculate_edge_angle(to_edge_info.shape, at_end=False)
            turn_direction = determine_turn_direction(from_angle, to_angle)

        if from_edge not in edge_conn_info:
            edge_conn_info[from_edge] = []
        edge_conn_info[from_edge].append((conn_elem, to_edge, from_lane, turn_direction, to_lane))

    # Second pass: fix connections for each modified edge
    for from_edge, conn_list in edge_conn_info.items():
        from_edge_info = edge_map[from_edge]
        num_lanes = from_edge_info.num_lanes  # Current number of lanes (before adding)
        left_lane = num_lanes - 1  # Leftmost lane index (this is the center lane)

        # For proper lane assignment:
        # - Leftmost lane (index num_lanes-1): left turns + U-turns ONLY
        # - Other lanes: straight + right (no left/U-turn)

        logger.info(f"\n  Processing {from_edge} ({num_lanes} lanes, leftmost={left_lane}):")

        for conn_elem, to_edge, from_lane, turn_direction, to_lane in conn_list:
            dir_names: Dict[str, str] = {'r': 'right', 's': 'straight', 'l': 'left', 't': 'U-turn'}
            dir_name = dir_names.get(turn_direction or '', 'unknown')

            if turn_direction in ('l', 't'):
                # Left turn or U-turn: should be on leftmost lane ONLY
                if from_lane != left_lane:
                    # Move to leftmost lane
                    logger.info(f"    Moving {dir_name}: lane {from_lane} -> lane {left_lane} (to {to_edge})")
                    if not dry_run:
                        conn_elem.set('fromLane', str(left_lane))
                    modified_count += 1
                else:
                    logger.info(f"    Keeping {dir_name}: lane {from_lane} (to {to_edge})")
            elif turn_direction in ('r', 's'):
                # Right or straight: should NOT be on leftmost lane
                if from_lane == left_lane:
                    # Move straight/right from leftmost lane to the next lane (left_lane - 1)
                    new_lane = left_lane - 1 if left_lane > 0 else 0
                    logger.info(f"    Moving {dir_name}: lane {from_lane} -> lane {new_lane} (to {to_edge})")
                    if not dry_run:
                        conn_elem.set('fromLane', str(new_lane))
                    modified_count += 1
                else:
                    logger.info(f"    Keeping {dir_name}: lane {from_lane} (to {to_edge})")

    # Add U-turn connections for the leftmost lane if not already present
    for edge_id in edge_ids:
        if edge_id not in edge_pairs:
            continue

        reverse_id = edge_pairs[edge_id]
        edge_info = edge_map.get(edge_id)
        if not edge_info:
            continue

        left_lane = edge_info.num_lanes - 1

        # Check if U-turn connection already exists from left lane
        uturn_exists = False
        for conn_elem in con_root.findall('connection'):
            if (conn_elem.get('from') == edge_id and
                conn_elem.get('to') == reverse_id and
                int(conn_elem.get('fromLane', '0')) == left_lane):
                uturn_exists = True
                break

        if not uturn_exists:
            logger.info(f"  Adding U-turn connection: {edge_id}:{left_lane} -> {reverse_id}:{left_lane}")

            if not dry_run:
                new_conn = ET.SubElement(con_root, 'connection')
                new_conn.set('from', edge_id)
                new_conn.set('to', reverse_id)
                new_conn.set('fromLane', str(left_lane))
                new_conn.set('toLane', str(left_lane))

            modified_count += 1

    if not dry_run and modified_count > 0:
        logger.info(f"\nWriting modified connection file: {output_con_file}")
        ET.indent(con_tree, space="    ")
        con_tree.write(output_con_file, encoding='UTF-8', xml_declaration=True)

    logger.info(f"\n{'[DRY RUN] ' if dry_run else ''}Modified/added {modified_count} connections")

    return modified_count


def main():
    parser = argparse.ArgumentParser(
        description='Add bidirectional center lane to SUMO road edges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add center lane to specific edges (dry run)
  python add_bidirectional_center_lane.py Ann_Arbor/aa_plain.edg.xml \\
      --edges "23481137010#0" --dry-run

  # Add center lane with manual edge shifting
  # For edge 8974919201#0: shift 8974919180#0/8974919181#0 pair UP
  #                        shift 4126464930#0/4126464931#0 pair DOWN
  python add_bidirectional_center_lane.py Ann_Arbor/aa_plain.edg.xml \\
      --edges "8974919201#0" \\
      --shift-up "8974919180#0" \\
      --shift-down "4126464930#0"

  # Add center lane and update connections
  python add_bidirectional_center_lane.py Ann_Arbor/aa_plain.edg.xml \\
      --edges "23481137010#0" --update-connections --modify-connections

  # Specify output files
  python add_bidirectional_center_lane.py Ann_Arbor/aa_plain.edg.xml \\
      --edges "23481137010#0" -o Ann_Arbor/aa_plain_modified.edg.xml

Road Configuration:
  Before: 2 lanes each direction (4 total lanes, 2 per edge)
    Edge A: [Lane 0] [Lane 1] ->
    Edge B: <- [Lane 0] [Lane 1]

  After: 2 lanes each direction + 1 shared center lane (6 total physical, 3 per edge)
    Edge A: [Lane 0 (right)] [Lane 1 (center, shared)] [Lane 2 (left)] ->
    Edge B: <- [Lane 0 (right)] [Lane 1 (center, shared)] [Lane 2 (left)]

    - Lane 0: Original lane 0 (rightmost, single-direction)
    - Lane 1: NEW center lane (overlaps between edges, bidirectional)
    - Lane 2: Original lane 1 shifted (leftmost, single-direction)
        """
    )

    parser.add_argument(
        'input_edge_file',
        help='Input SUMO plain edge XML file'
    )

    parser.add_argument(
        '--edges',
        nargs='+',
        required=True,
        help='Edge IDs to add center lane to'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output edge XML file (default: overwrites input)',
        default=None
    )

    parser.add_argument(
        '--con-file',
        help='Connection XML file (default: inferred from edge file)',
        default=None
    )

    parser.add_argument(
        '--con-output',
        help='Output connection XML file (default: overwrites input)',
        default=None
    )

    parser.add_argument(
        '--update-connections',
        action='store_true',
        help='Also update connection file for new lane configuration'
    )

    parser.add_argument(
        '--skip-connection-updates',
        action='store_true',
        default=True,
        help='Skip updating connection lane assignments (default: True, only update edge file)'
    )

    parser.add_argument(
        '--modify-connections',
        action='store_false',
        dest='skip_connection_updates',
        help='Enable connection lane assignment modifications (overrides --skip-connection-updates)'
    )

    parser.add_argument(
        '--lane-width',
        type=float,
        default=3.2,
        help='Width of the center lane in meters (default: 3.2m)'
    )

    parser.add_argument(
        '--shift-up',
        nargs='+',
        default=None,
        help='Edge IDs to shift UP (e.g., incoming edges at junction)'
    )

    parser.add_argument(
        '--shift-down',
        nargs='+',
        default=None,
        help='Edge IDs to shift DOWN (e.g., outgoing edges at junction)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not os.path.exists(args.input_edge_file):
        logger.error(f"Edge file not found: {args.input_edge_file}")
        sys.exit(1)

    output_edge_file = args.output if args.output else args.input_edge_file

    try:
        # Add center lane to edges
        modified = add_center_lane_to_edges(
            input_edge_file=args.input_edge_file,
            output_edge_file=output_edge_file,
            edge_ids=args.edges,
            dry_run=args.dry_run,
            lane_width=args.lane_width,
            edges_shift_up=args.shift_up,
            edges_shift_down=args.shift_down
        )

        # Update connections if requested
        if args.update_connections:
            con_file = args.con_file
            if not con_file:
                con_file = args.input_edge_file.replace('.edg.xml', '.con.xml')

            if not os.path.exists(con_file):
                logger.error(f"Connection file not found: {con_file}")
                sys.exit(1)

            con_output = args.con_output if args.con_output else con_file

            update_connections_for_center_lane(
                input_edge_file=output_edge_file if not args.dry_run else args.input_edge_file,
                input_con_file=con_file,
                output_con_file=con_output,
                edge_ids=args.edges,
                dry_run=args.dry_run,
                skip_updates=args.skip_connection_updates
            )

        if modified > 0 and not args.dry_run:
            logger.info("\nNext steps:")
            logger.info("1. Review the changes in the output files")
            logger.info("2. Rebuild the SUMO network using netconvert:")
            logger.info(f"   netconvert --node-files=aa_plain.nod.xml \\")
            logger.info(f"              --edge-files={os.path.basename(output_edge_file)} \\")
            logger.info(f"              --connection-files=aa_plain.con.xml \\")
            logger.info(f"              --output-file=aa_modified.net.xml")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
