#!/usr/bin/env python3
"""
SUMO Connection Lane Assignment Fixer

This script identifies and fixes incorrect lane assignments in SUMO connection files.
A common issue is when multi-lane roads have all turn directions assigned to one lane
(e.g., right, straight, and left on lane 0) while another lane only has U-turns.

The correct pattern for a 2-lane road should be:
- Lane 0 (right lane): right turn + straight
- Lane 1 (left lane): left turn + U-turn

Author: TeraSim Team
License: MIT
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import argparse
import logging
import os
import sys
import math

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
    name: Optional[str] = None
    num_lanes: int = 1

    @property
    def node_pair(self) -> Tuple[str, str]:
        """Return node pair"""
        return (self.from_node, self.to_node)

    @property
    def reverse_node_pair(self) -> Tuple[str, str]:
        """Return reverse node pair"""
        return (self.to_node, self.from_node)


@dataclass
class ConnectionInfo:
    """Information about a connection"""
    from_edge: str
    to_edge: str
    from_lane: int
    to_lane: int
    element: ET.Element
    direction: Optional[str] = None  # 'r' (right), 's' (straight), 'l' (left), 't' (turn/u-turn)


def parse_shape_coordinates(shape_str: str) -> List[Tuple[float, float]]:
    """Parse shape string into list of coordinate tuples."""
    if not shape_str:
        return []

    coords = []
    for coord_pair in shape_str.strip().split():
        x, y = coord_pair.split(',')
        coords.append((float(x), float(y)))
    return coords


def calculate_angle(shape_str: str) -> float:
    """
    Calculate the heading angle of an edge based on its shape.

    Returns angle in degrees (0-360, where 0 is East, 90 is North)
    """
    if not shape_str:
        return 0.0

    coords = parse_shape_coordinates(shape_str)
    if len(coords) < 2:
        return 0.0

    # Use last two points to determine heading at the end
    x1, y1 = coords[-2]
    x2, y2 = coords[-1]

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


def find_bidirectional_pairs(edges: List[EdgeInfo]) -> Dict[str, str]:
    """
    Find pairs of edges that form bidirectional lanes.

    Returns a dictionary mapping edge ID to its reverse edge ID.
    """
    reverse_map: Dict[Tuple[str, str], List[EdgeInfo]] = {}

    for edge in edges:
        reverse_pair = edge.reverse_node_pair
        if reverse_pair not in reverse_map:
            reverse_map[reverse_pair] = []
        reverse_map[reverse_pair].append(edge)

    # Build edge -> reverse edge mapping
    result: Dict[str, str] = {}
    processed: Set[str] = set()

    for edge in edges:
        if edge.id in processed:
            continue

        node_pair = edge.node_pair
        if node_pair in reverse_map:
            for reverse_edge in reverse_map[node_pair]:
                if reverse_edge.id not in processed:
                    result[edge.id] = reverse_edge.id
                    result[reverse_edge.id] = edge.id
                    processed.add(edge.id)
                    processed.add(reverse_edge.id)
                    break

    return result


def analyze_edge_connections(
    edge_id: str,
    connections: List[ConnectionInfo],
    edge_map: Dict[str, EdgeInfo],
    reverse_edge_id: Optional[str] = None
) -> None:
    """
    Analyze connections from an edge and set their direction.
    """
    edge_connections = [c for c in connections if c.from_edge == edge_id]

    if edge_id not in edge_map:
        return

    from_edge = edge_map[edge_id]
    from_angle = calculate_angle(from_edge.shape)

    for conn in edge_connections:
        to_edge_id = conn.to_edge

        # Check if this is a U-turn to the reverse edge
        if reverse_edge_id and to_edge_id == reverse_edge_id:
            conn.direction = 't'
            continue

        if to_edge_id not in edge_map:
            continue

        to_edge = edge_map[to_edge_id]

        # For outgoing edge, we need to look at the start angle
        to_coords = parse_shape_coordinates(to_edge.shape)
        if len(to_coords) >= 2:
            x1, y1 = to_coords[0]
            x2, y2 = to_coords[1]
            to_angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 360
            conn.direction = determine_turn_direction(from_angle, to_angle)


def has_incorrect_lane_assignment(
    edge_id: str,
    connections: List[ConnectionInfo],
    edge_map: Dict[str, EdgeInfo]
) -> bool:
    """
    Check if an edge has incorrect lane assignments.

    The typical incorrect pattern is:
    - Lane 0 (right): right, straight, AND left turns
    - Lane 1 (left): only U-turn
    """
    edge_connections = [c for c in connections if c.from_edge == edge_id]

    if edge_id not in edge_map:
        return False

    edge = edge_map[edge_id]
    if edge.num_lanes < 2:
        return False

    # Get connections by lane
    lane_connections: Dict[int, List[ConnectionInfo]] = defaultdict(list)
    for conn in edge_connections:
        lane_connections[conn.from_lane].append(conn)

    right_lane = 0
    left_lane = edge.num_lanes - 1

    right_lane_conns = lane_connections.get(right_lane, [])
    left_lane_conns = lane_connections.get(left_lane, [])

    # Check for the problematic pattern:
    # Right lane has left turns AND left lane only has U-turns
    right_lane_has_left = any(c.direction == 'l' for c in right_lane_conns if c.direction)
    left_lane_only_uturn = (
        len(left_lane_conns) > 0 and
        all(c.direction == 't' for c in left_lane_conns if c.direction)
    )

    if right_lane_has_left and left_lane_only_uturn:
        return True

    return False


def fix_connections_for_edge(
    edge_id: str,
    connections: List[ConnectionInfo],
    edge_map: Dict[str, EdgeInfo],
    dry_run: bool = False
) -> int:
    """
    Fix connection lane assignments for an edge.

    Correct pattern:
    - Lane 0 (right): right turn + straight
    - Lane 1 (left): left turn + U-turn
    """
    if edge_id not in edge_map:
        return 0

    edge = edge_map[edge_id]
    if edge.num_lanes < 2:
        return 0

    edge_connections = [c for c in connections if c.from_edge == edge_id]

    fixed_count = 0
    right_lane = 0
    left_lane = edge.num_lanes - 1

    # Fix each connection based on its turn direction
    for conn in edge_connections:
        if conn.direction is None:
            continue

        new_lane = conn.from_lane

        if conn.direction == 'r':
            # Right turn should be on right lane
            new_lane = right_lane
        elif conn.direction == 's':
            # Straight should be on right lane for 2-lane roads
            new_lane = right_lane
        elif conn.direction == 'l':
            # Left turn should be on left lane
            new_lane = left_lane
        elif conn.direction == 't':
            # U-turn should be on left lane
            new_lane = left_lane

        if new_lane != conn.from_lane:
            logger.info(f"    Fixing: {edge_id}:{conn.from_lane} -> {conn.to_edge}:{conn.to_lane}")
            logger.info(f"      Direction: {conn.direction}, changing fromLane {conn.from_lane} -> {new_lane}")

            if not dry_run:
                conn.element.set('fromLane', str(new_lane))
                conn.from_lane = new_lane

            fixed_count += 1

    return fixed_count


def fix_connections(
    input_edge_file: str,
    input_con_file: str,
    output_con_file: str,
    specific_edges: Optional[List[str]] = None,
    dry_run: bool = False
) -> int:
    """
    Fix connection lane assignments for edges with incorrect patterns.
    """
    logger.info(f"Reading edge file: {input_edge_file}")
    logger.info(f"Reading connection file: {input_con_file}")

    # Parse edge XML
    edge_tree = ET.parse(input_edge_file)
    edge_root = edge_tree.getroot()

    # Extract edges
    edge_map: Dict[str, EdgeInfo] = {}
    edges: List[EdgeInfo] = []

    for edge_elem in edge_root.findall('edge'):
        edge_id = edge_elem.get('id')
        from_node = edge_elem.get('from')
        to_node = edge_elem.get('to')

        if edge_id is None or from_node is None or to_node is None:
            continue

        shape = edge_elem.get('shape', '')
        name = edge_elem.get('name')
        num_lanes = int(edge_elem.get('numLanes', '1'))

        edge_info = EdgeInfo(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            shape=shape,
            name=name,
            num_lanes=num_lanes
        )
        edges.append(edge_info)
        edge_map[edge_id] = edge_info

    logger.info(f"Found {len(edges)} total edges")

    # Find bidirectional pairs for U-turn detection
    reverse_edge_map = find_bidirectional_pairs(edges)

    # Parse connection XML
    con_tree = ET.parse(input_con_file)
    con_root = con_tree.getroot()

    # Extract connections
    connections: List[ConnectionInfo] = []

    for conn_elem in con_root.findall('connection'):
        from_edge = conn_elem.get('from')
        to_edge = conn_elem.get('to')

        if from_edge is None or to_edge is None:
            continue

        from_lane = int(conn_elem.get('fromLane', '0'))
        to_lane = int(conn_elem.get('toLane', '0'))

        connections.append(ConnectionInfo(
            from_edge=from_edge,
            to_edge=to_edge,
            from_lane=from_lane,
            to_lane=to_lane,
            element=conn_elem
        ))

    logger.info(f"Found {len(connections)} total connections")

    # Determine which edges to check
    if specific_edges:
        edges_to_check = specific_edges
    else:
        edges_to_check = [e.id for e in edges if e.num_lanes >= 2]

    # Analyze and fix connections
    total_fixed = 0

    for edge_id in edges_to_check:
        if edge_id not in edge_map:
            continue

        edge = edge_map[edge_id]
        if edge.num_lanes < 2:
            continue

        reverse_edge_id = reverse_edge_map.get(edge_id)

        # Analyze connections to determine turn directions
        analyze_edge_connections(edge_id, connections, edge_map, reverse_edge_id)

        # Check if this edge has incorrect lane assignments
        if has_incorrect_lane_assignment(edge_id, connections, edge_map):
            logger.info(f"\nEdge {edge_id} ({edge.name}) has incorrect lane assignments")

            # Log current state
            edge_conns = [c for c in connections if c.from_edge == edge_id]
            logger.info(f"  Current connections:")
            for conn in edge_conns:
                dir_map = {'r': 'right', 's': 'straight', 'l': 'left', 't': 'u-turn'}
                dir_name = dir_map.get(conn.direction, '?') if conn.direction else '?'
                logger.info(f"    Lane {conn.from_lane} -> {conn.to_edge}:{conn.to_lane} ({dir_name})")

            # Fix the connections
            fixed = fix_connections_for_edge(edge_id, connections, edge_map, dry_run)
            total_fixed += fixed

    if not dry_run and total_fixed > 0:
        logger.info(f"\nWriting fixed connection file: {output_con_file}")
        ET.indent(con_tree, space="    ")
        con_tree.write(output_con_file, encoding='UTF-8', xml_declaration=True)

    logger.info(f"\n{'[DRY RUN] ' if dry_run else ''}Fixed {total_fixed} connections")

    return total_fixed


def find_edges_by_name_pattern(input_edge_file: str, name_patterns: List[str]) -> List[str]:
    """Find edge IDs that match given name patterns."""
    tree = ET.parse(input_edge_file)
    root = tree.getroot()

    matching_edges = []

    for edge_elem in root.findall('edge'):
        edge_id = edge_elem.get('id')
        edge_name = edge_elem.get('name', '').lower()

        if edge_id is None:
            continue

        for pattern in name_patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in edge_name or pattern_lower.replace('_', ' ') in edge_name:
                matching_edges.append(edge_id)
                logger.info(f"Found matching edge: {edge_id} (name: {edge_name})")
                break

    return matching_edges


def main():
    parser = argparse.ArgumentParser(
        description='Fix lane assignments in SUMO connection files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix specific edges by ID
  python fix_connections.py Ann_Arbor/aa_plain.edg.xml --edges "87220590#0" "2227868411#0"

  # Dry run to see what would be changed
  python fix_connections.py Ann_Arbor/aa_plain.edg.xml --edges "87220590#0" --dry-run

  # Fix edges by name pattern
  python fix_connections.py Ann_Arbor/aa_plain.edg.xml --names "hubbard" "maple"

  # Specify connection file explicitly
  python fix_connections.py Ann_Arbor/aa_plain.edg.xml --con-file Ann_Arbor/aa_plain.con.xml

  # Specify output file
  python fix_connections.py Ann_Arbor/aa_plain.edg.xml -o Ann_Arbor/aa_plain_fixed.con.xml
        """
    )

    parser.add_argument(
        'input_edge_file',
        help='Input SUMO plain edge XML file (e.g., aa_plain.edg.xml)'
    )

    parser.add_argument(
        '--con-file',
        help='Connection XML file (default: inferred from edge file)',
        default=None
    )

    parser.add_argument(
        '-o', '--output',
        help='Output connection XML file (default: overwrites input)',
        default=None
    )

    parser.add_argument(
        '--edges',
        nargs='+',
        help='Specific edge IDs to fix',
        default=None
    )

    parser.add_argument(
        '--names',
        nargs='+',
        help='Find edges by name patterns (e.g., "hubbard" "maple")',
        default=None
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

    # Validate input file
    if not os.path.exists(args.input_edge_file):
        logger.error(f"Edge file not found: {args.input_edge_file}")
        sys.exit(1)

    # Determine connection file
    if args.con_file:
        con_file = args.con_file
    else:
        con_file = args.input_edge_file.replace('.edg.xml', '.con.xml')

    if not os.path.exists(con_file):
        logger.error(f"Connection file not found: {con_file}")
        logger.error("Use --con-file to specify the connection file path")
        sys.exit(1)

    # Determine output file
    output_file = args.output if args.output else con_file

    # Find edges to fix
    specific_edges = args.edges
    if args.names:
        logger.info(f"Searching for edges matching names: {args.names}")
        found_edges = find_edges_by_name_pattern(args.input_edge_file, args.names)
        if specific_edges:
            specific_edges.extend(found_edges)
        else:
            specific_edges = found_edges

        if not specific_edges:
            logger.warning("No edges found matching the specified names")

    if not specific_edges:
        logger.error("No edges specified. Use --edges or --names to specify edges to fix.")
        sys.exit(1)

    try:
        fixed_count = fix_connections(
            input_edge_file=args.input_edge_file,
            input_con_file=con_file,
            output_con_file=output_file,
            specific_edges=specific_edges,
            dry_run=args.dry_run
        )

        if fixed_count > 0 and not args.dry_run:
            logger.info(f"\nOutput written to: {output_file}")
            logger.info("\nNext steps:")
            logger.info("1. Review the changes in the output file")
            logger.info("2. Rebuild the SUMO network using netconvert:")
            logger.info(f"   netconvert --node-files=aa_plain.nod.xml --edge-files=aa_plain.edg.xml \\")
            logger.info(f"              --connection-files={os.path.basename(output_file)} --output-file=aa_fixed.net.xml")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
