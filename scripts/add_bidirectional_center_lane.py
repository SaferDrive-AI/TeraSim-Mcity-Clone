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
    Offset a shape perpendicular to its direction.

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

    offset_coords = []

    for i in range(len(coords)):
        x, y = coords[i]

        # Calculate the direction vector at this point
        if i == 0:
            # First point: use direction to next point
            dx = coords[i + 1][0] - x
            dy = coords[i + 1][1] - y
        elif i == len(coords) - 1:
            # Last point: use direction from previous point
            dx = x - coords[i - 1][0]
            dy = y - coords[i - 1][1]
        else:
            # Middle points: average of incoming and outgoing directions
            dx1 = x - coords[i - 1][0]
            dy1 = y - coords[i - 1][1]
            dx2 = coords[i + 1][0] - x
            dy2 = coords[i + 1][1] - y
            # Normalize and average
            len1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
            len2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
            if len1 > 0 and len2 > 0:
                dx = (dx1 / len1 + dx2 / len2) / 2
                dy = (dy1 / len1 + dy2 / len2) / 2
            elif len1 > 0:
                dx, dy = dx1 / len1, dy1 / len1
            elif len2 > 0:
                dx, dy = dx2 / len2, dy2 / len2
            else:
                dx, dy = 1, 0

        # Normalize direction vector
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx /= length
            dy /= length

        # Calculate perpendicular vector (rotate 90 degrees)
        # For 'right': rotate clockwise (-90 degrees): (dx, dy) -> (dy, -dx)
        # For 'left': rotate counter-clockwise (+90 degrees): (dx, dy) -> (-dy, dx)
        if direction == 'right':
            perp_x = dy
            perp_y = -dx
        else:  # left
            perp_x = -dy
            perp_y = dx

        # Apply offset
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


def add_center_lane_to_edges(
    input_edge_file: str,
    output_edge_file: str,
    edge_ids: List[str],
    dry_run: bool = False,
    lane_width: float = 3.2
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
        # For n lanes: lanes 0 to (n/2-1) are on right, lanes (n/2) to (n-1) are on left
        # The leftmost lane (index n-1) should overlap between the two edges
        #
        # Original: 3 lanes each, shapes overlap -> all 3 lanes overlap (wrong)
        # We want: 4 lanes each, only lane 3 (leftmost) overlaps
        #
        # With 4 lanes and spreadType="center":
        #   - Total width = 4 * lane_width = 12.8m
        #   - Lanes spread 6.4m on each side of shape
        #   - Lane 3 (leftmost) center is at: 1.5 * lane_width from shape = 4.8m left
        #
        # To make lane 3 from both edges overlap at the same position:
        #   - We need to offset each edge to the RIGHT by half a lane width
        #   - This shifts all lanes right, moving the leftmost lane toward center
        offset_distance = lane_width

        logger.info(f"  Offset distance: {offset_distance:.2f}m (to overlap only center lane)")

        if not dry_run:
            # Get original shapes
            original_shape = edge.shape
            original_reverse_shape = reverse_edge.shape

            # If reverse edge doesn't have a shape, derive it from the forward edge
            if not original_reverse_shape and original_shape:
                original_reverse_shape = reverse_shape(original_shape)

            # Make the reverse edge shape exactly reversed from the forward edge
            reversed_shape_str = reverse_shape(original_shape)

            # Offset edge 1 to the RIGHT (shifts lanes toward right side of road)
            offset_shape_1 = offset_shape(original_shape, offset_distance, direction='right')

            # Offset edge 2 to the RIGHT (relative to its direction)
            # Since it goes opposite direction, this also shifts its lanes toward the outside
            offset_shape_2 = offset_shape(reversed_shape_str, offset_distance, direction='right')

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


def update_connections_for_center_lane(
    input_edge_file: str,
    input_con_file: str,
    output_con_file: str,
    edge_ids: List[str],
    dry_run: bool = False
) -> int:
    """
    Update connections after adding center lane.

    When we add a center lane (new lane index 1), we need to:
    1. Shift existing lane 1 connections to lane 2
    2. Add connections for the new center lane (lane 1)

    The center lane should connect to:
    - The reverse edge's center lane (for U-turns / through traffic)
    - Optionally to other edges

    Args:
        input_edge_file: Path to edge XML file
        input_con_file: Path to connection XML file
        output_con_file: Path to output connection XML file
        edge_ids: List of edge IDs that were modified
        dry_run: If True, only report changes

    Returns:
        Number of connections modified
    """
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
    new_connections = []

    for conn_elem in con_root.findall('connection'):
        from_edge = conn_elem.get('from')
        to_edge = conn_elem.get('to')

        if from_edge is None or to_edge is None:
            continue

        from_lane = int(conn_elem.get('fromLane', '0'))
        to_lane = int(conn_elem.get('toLane', '0'))

        # Check if this connection involves a modified edge
        from_modified = from_edge in modified_edges
        to_modified = to_edge in modified_edges

        if from_modified or to_modified:
            # Shift lane indices for lanes >= 1 (to make room for center lane)
            new_from_lane = from_lane
            new_to_lane = to_lane

            if from_modified and from_lane >= 1:
                new_from_lane = from_lane + 1

            if to_modified and to_lane >= 1:
                new_to_lane = to_lane + 1

            if new_from_lane != from_lane or new_to_lane != to_lane:
                logger.info(f"  Shifting connection: {from_edge}:{from_lane}->{to_edge}:{to_lane}")
                logger.info(f"    -> {from_edge}:{new_from_lane}->{to_edge}:{new_to_lane}")

                if not dry_run:
                    conn_elem.set('fromLane', str(new_from_lane))
                    conn_elem.set('toLane', str(new_to_lane))

                modified_count += 1

    # Add connections for the new center lane (lane 1)
    for edge_id in edge_ids:
        if edge_id not in edge_pairs:
            continue

        reverse_id = edge_pairs[edge_id]

        # Add connection from edge's center lane to reverse edge's center lane
        # This allows bidirectional use
        logger.info(f"  Adding center lane connection: {edge_id}:1 -> {reverse_id}:1")

        if not dry_run:
            new_conn = ET.SubElement(con_root, 'connection')
            new_conn.set('from', edge_id)
            new_conn.set('to', reverse_id)
            new_conn.set('fromLane', '1')
            new_conn.set('toLane', '1')

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

  # Add center lane and update connections
  python add_bidirectional_center_lane.py Ann_Arbor/aa_plain.edg.xml \\
      --edges "23481137010#0" --update-connections

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
        '--lane-width',
        type=float,
        default=3.2,
        help='Width of the center lane in meters (default: 3.2m)'
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
            lane_width=args.lane_width
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
                dry_run=args.dry_run
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
