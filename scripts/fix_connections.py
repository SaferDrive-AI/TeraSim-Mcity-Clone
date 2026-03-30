#!/usr/bin/env python3
"""
SUMO Connection Lane Assignment Fixer

This script identifies and fixes incorrect lane assignments in SUMO connection files.
It can automatically detect all multi-lane edges with incorrect lane assignments
and fix them based on standard traffic rules.

Lane Assignment Rules for 4-way intersections:
- Rightmost lane (0): right turns only
- Middle lanes (1 to N-2): straight only
- Leftmost lane (N-1): left turns + U-turns only

For 2-lane roads:
- Lane 0 (right lane): right turn + straight
- Lane 1 (left lane): left turn + U-turn

Usage:
  # Auto-detect and fix all incorrect assignments
  python fix_connections.py network.edg.xml --auto-detect

  # Dry run to see what would be changed
  python fix_connections.py network.edg.xml --auto-detect --dry-run

  # If network has sidewalks on lane 0
  python fix_connections.py network.edg.xml --auto-detect --has-sidewalk

Author: TeraSim Team
License: MIT
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import argparse
import logging
import math
import os
import sys

from sumo_net_utils import (
    parse_osm_centre_turn_lanes,
    find_edges_by_osm_ids,
    parse_shape_coordinates,
    determine_turn_direction,
)

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


def calculate_angle(shape_str: str) -> float:
    """
    Calculate the heading angle of an edge at its end based on its shape.

    Returns angle in degrees (0-360, where 0 is East, 90 is North)
    """
    if not shape_str:
        return 0.0

    coords = parse_shape_coordinates(shape_str)
    if len(coords) < 2:
        return 0.0

    x1, y1 = coords[-2]
    x2, y2 = coords[-1]

    angle = math.atan2(y2 - y1, x2 - x1)
    return math.degrees(angle) % 360


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


def _get_vehicle_lanes(
    edge_id: str,
    edge: EdgeInfo,
    has_sidewalk: bool,
    center_turn_lane_edges: Optional[Set[str]] = None
) -> Tuple[int, int, int, List[int], Set[int]]:
    """
    Calculate the effective vehicle lane indices for an edge, excluding
    sidewalk and center turn lanes.

    Returns:
        Tuple of (num_vehicle_lanes, right_lane, left_lane, middle_lanes, excluded_lanes)
    """
    excluded_lanes: Set[int] = set()

    # Exclude sidewalk (lane 0)
    if has_sidewalk:
        excluded_lanes.add(0)

    # Exclude center turn lane (lane 1 after center lane insertion)
    # The center lane is inserted at index 1 by add_bidirectional_center_lane.py
    if center_turn_lane_edges and edge_id in center_turn_lane_edges:
        center_idx = 1 if not has_sidewalk else 2
        if center_idx < edge.num_lanes:
            excluded_lanes.add(center_idx)

    # Build list of vehicle lanes (all lanes minus excluded)
    vehicle_lanes = [i for i in range(edge.num_lanes) if i not in excluded_lanes]
    num_vehicle_lanes = len(vehicle_lanes)

    if num_vehicle_lanes < 2:
        return num_vehicle_lanes, -1, -1, [], excluded_lanes

    right_lane = vehicle_lanes[0]
    left_lane = vehicle_lanes[-1]
    middle_lanes = vehicle_lanes[1:-1]

    return num_vehicle_lanes, right_lane, left_lane, middle_lanes, excluded_lanes


def has_incorrect_lane_assignment(
    edge_id: str,
    connections: List[ConnectionInfo],
    edge_map: Dict[str, EdgeInfo],
    has_sidewalk: bool = False,
    center_turn_lane_edges: Optional[Set[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Check if an edge has incorrect lane assignments.

    Correct lane assignment rules for multi-lane roads:
    - Rightmost vehicle lane: right turns only
    - Middle lanes: straight only
    - Leftmost lane: left turns + U-turns only

    Excluded lanes (sidewalk, center turn lane) are skipped.

    Args:
        edge_id: Edge ID to check
        connections: List of all connections
        edge_map: Dictionary of edge ID to EdgeInfo
        has_sidewalk: If True, lane 0 is sidewalk
        center_turn_lane_edges: Set of edge IDs that have a center turn lane

    Returns:
        Tuple of (has_incorrect, list_of_issues)
    """
    edge_connections = [c for c in connections if c.from_edge == edge_id]

    if edge_id not in edge_map:
        return False, []

    edge = edge_map[edge_id]
    num_vehicle_lanes, right_lane, left_lane, middle_lanes, excluded_lanes = \
        _get_vehicle_lanes(edge_id, edge, has_sidewalk, center_turn_lane_edges)

    if num_vehicle_lanes < 2:
        return False, []

    # Get connections by lane (only for vehicle lanes)
    lane_connections: Dict[int, List[ConnectionInfo]] = defaultdict(list)
    for conn in edge_connections:
        if conn.from_lane in excluded_lanes:
            continue
        lane_connections[conn.from_lane].append(conn)

    issues = []

    # Check rightmost vehicle lane
    right_lane_conns = lane_connections.get(right_lane, [])
    for conn in right_lane_conns:
        if conn.direction == 'l':
            issues.append(f"Lane {right_lane} has left turn (should be on leftmost lane {left_lane})")
        elif conn.direction == 't':
            issues.append(f"Lane {right_lane} has U-turn (should be on leftmost lane {left_lane})")

    # Check middle lanes (should only have straight)
    for mid_lane in middle_lanes:
        mid_conns = lane_connections.get(mid_lane, [])
        for conn in mid_conns:
            if conn.direction == 'l':
                issues.append(f"Lane {mid_lane} has left turn (should be on leftmost lane {left_lane})")
            elif conn.direction == 't':
                issues.append(f"Lane {mid_lane} has U-turn (should be on leftmost lane {left_lane})")
            elif conn.direction == 'r':
                issues.append(f"Lane {mid_lane} has right turn (should be on rightmost lane {right_lane})")

    # Check leftmost lane (should only have left turns and U-turns)
    left_lane_conns = lane_connections.get(left_lane, [])
    for conn in left_lane_conns:
        if conn.direction == 'r':
            issues.append(f"Lane {left_lane} has right turn (should be on rightmost lane {right_lane})")

    return len(issues) > 0, issues


def fix_connections_for_edge(
    edge_id: str,
    connections: List[ConnectionInfo],
    edge_map: Dict[str, EdgeInfo],
    dry_run: bool = False,
    has_sidewalk: bool = False,
    center_turn_lane_edges: Optional[Set[str]] = None
) -> int:
    """
    Fix connection lane assignments for an edge.

    Correct pattern for multi-lane roads:
    - Rightmost vehicle lane: right turns
    - Middle vehicle lanes: straight
    - Leftmost vehicle lane: left turns + U-turns

    Excluded lanes (sidewalk, center turn lane) are not assigned regular traffic.

    For 2-vehicle-lane roads:
    - Rightmost vehicle lane: right + straight
    - Leftmost vehicle lane: left + U-turn

    Args:
        edge_id: Edge ID to fix
        connections: List of all connections
        edge_map: Dictionary of edge ID to EdgeInfo
        dry_run: If True, only report changes without modifying
        has_sidewalk: If True, lane 0 is sidewalk
        center_turn_lane_edges: Set of edge IDs that have a center turn lane
    """
    if edge_id not in edge_map:
        return 0

    edge = edge_map[edge_id]
    num_vehicle_lanes, right_lane, left_lane, middle_lanes, excluded_lanes = \
        _get_vehicle_lanes(edge_id, edge, has_sidewalk, center_turn_lane_edges)

    if num_vehicle_lanes < 2:
        return 0

    edge_connections = [c for c in connections if c.from_edge == edge_id]

    fixed_count = 0

    # For roads with more than 2 vehicle lanes, use middle lanes for straight
    # For 2-vehicle-lane roads, straight goes on the right vehicle lane
    if num_vehicle_lanes > 2 and middle_lanes:
        # Middle lane for straight (use the lane closest to center)
        straight_lane = middle_lanes[len(middle_lanes) // 2]
    else:
        straight_lane = right_lane

    # Fix each connection based on its turn direction
    for conn in edge_connections:
        if conn.direction is None:
            continue

        # Skip excluded lane connections (sidewalk, center turn lane)
        if conn.from_lane in excluded_lanes:
            continue

        new_lane = conn.from_lane

        if conn.direction == 'r':
            # Right turn should be on rightmost vehicle lane
            new_lane = right_lane
        elif conn.direction == 's':
            # Straight should be on middle lane(s) or right vehicle lane for 2-lane roads
            if conn.from_lane == left_lane and num_vehicle_lanes > 2:
                new_lane = straight_lane
            elif conn.from_lane == left_lane and num_vehicle_lanes == 2:
                # For 2-lane, straight can stay on right or left vehicle lane, prefer not moving
                pass
            elif num_vehicle_lanes > 2 and conn.from_lane == right_lane:
                # Move straight from rightmost to middle if there are middle lanes
                new_lane = straight_lane
        elif conn.direction == 'l':
            # Left turn should be on leftmost vehicle lane
            new_lane = left_lane
        elif conn.direction == 't':
            # U-turn should be on leftmost vehicle lane
            new_lane = left_lane

        if new_lane != conn.from_lane:
            dir_map = {'r': 'right', 's': 'straight', 'l': 'left', 't': 'u-turn'}
            dir_name = dir_map.get(conn.direction, '?')
            logger.info(f"    Fixing: {edge_id}:{conn.from_lane} -> {conn.to_edge}:{conn.to_lane}")
            logger.info(f"      Direction: {dir_name}, changing fromLane {conn.from_lane} -> {new_lane}")

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
    dry_run: bool = False,
    auto_detect: bool = False,
    has_sidewalk: bool = False,
    center_turn_lane_edges: Optional[Set[str]] = None
) -> int:
    """
    Fix connection lane assignments for edges with incorrect patterns.

    Args:
        input_edge_file: Path to edge XML file
        input_con_file: Path to connection XML file
        output_con_file: Path to output connection XML file
        specific_edges: List of specific edge IDs to fix (optional)
        dry_run: If True, only report changes without modifying files
        auto_detect: If True, automatically scan all multi-lane edges
        has_sidewalk: If True, lane 0 is sidewalk and vehicle lanes start from 1
        center_turn_lane_edges: Set of edge IDs that have a center turn lane
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
    elif auto_detect:
        # Auto-detect: check all multi-lane edges
        edges_to_check = [e.id for e in edges if e.num_lanes >= 2]
        logger.info(f"Auto-detecting: checking {len(edges_to_check)} multi-lane edges")
    else:
        edges_to_check = [e.id for e in edges if e.num_lanes >= 2]

    # First pass: analyze all connections to determine turn directions
    logger.info("Analyzing turn directions for all connections...")
    for edge_id in edges_to_check:
        if edge_id not in edge_map:
            continue
        reverse_edge_id = reverse_edge_map.get(edge_id)
        analyze_edge_connections(edge_id, connections, edge_map, reverse_edge_id)

    # Second pass: detect and fix incorrect assignments
    total_fixed = 0
    edges_with_issues = 0

    for edge_id in edges_to_check:
        if edge_id not in edge_map:
            continue

        edge = edge_map[edge_id]
        if edge.num_lanes < 2:
            continue

        # Check if this edge has incorrect lane assignments
        has_issues, issues = has_incorrect_lane_assignment(edge_id, connections, edge_map, has_sidewalk, center_turn_lane_edges)

        if has_issues:
            edges_with_issues += 1
            sidewalk_note = " (with sidewalk on lane 0)" if has_sidewalk else ""
            logger.info(f"\nEdge {edge_id} ({edge.name or 'unnamed'}) - {edge.num_lanes} lanes{sidewalk_note}")
            logger.info(f"  Issues detected:")
            for issue in issues:
                logger.info(f"    - {issue}")

            # Log current state
            edge_conns = [c for c in connections if c.from_edge == edge_id]
            logger.info(f"  Current connections:")
            for conn in edge_conns:
                dir_map = {'r': 'right', 's': 'straight', 'l': 'left', 't': 'u-turn'}
                dir_name = dir_map.get(conn.direction, '?') if conn.direction else '?'
                lane_note = " (sidewalk)" if has_sidewalk and conn.from_lane == 0 else ""
                logger.info(f"    Lane {conn.from_lane}{lane_note} -> {conn.to_edge}:{conn.to_lane} ({dir_name})")

            # Fix the connections
            fixed = fix_connections_for_edge(edge_id, connections, edge_map, dry_run, has_sidewalk, center_turn_lane_edges)
            total_fixed += fixed

    if not dry_run and total_fixed > 0:
        logger.info(f"\nWriting fixed connection file: {output_con_file}")
        ET.indent(con_tree, space="    ")
        con_tree.write(output_con_file, encoding='UTF-8', xml_declaration=True)

    logger.info(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
    logger.info(f"  Edges with issues: {edges_with_issues}")
    logger.info(f"  Connections fixed: {total_fixed}")

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
  # Auto-detect edges from OSM file (recommended)
  python fix_connections.py Ann_Arbor/aa_plain.edg.xml --osm-file Ann_Arbor/aa_plain.osm --dry-run

  # Auto-detect from OSM with sidewalk support
  python fix_connections.py Ann_Arbor/aa_plain.edg.xml --osm-file Ann_Arbor/aa_plain.osm --has-sidewalk

  # Auto-detect and fix all incorrect lane assignments in multi-lane edges
  python fix_connections.py san_jose/osm.edg.xml --auto-detect --dry-run

  # If the network has sidewalks on lane 0, use --has-sidewalk
  python fix_connections.py san_jose/osm.edg.xml --auto-detect --has-sidewalk --dry-run

  # Fix specific edges by ID
  python fix_connections.py Ann_Arbor/aa_plain.edg.xml --edges "87220590#0" "2227868411#0"

  # Fix edges by name pattern
  python fix_connections.py Ann_Arbor/aa_plain.edg.xml --names "hubbard" "maple"

  # Specify connection file explicitly
  python fix_connections.py Ann_Arbor/aa_plain.edg.xml --con-file Ann_Arbor/aa_plain.con.xml

Lane Assignment Rules:
  For a 4-way intersection with multi-lane roads (without sidewalk):
  - Lane 0 (rightmost): right turns only
  - Middle lanes: straight only
  - Lane N-1 (leftmost): left turns + U-turns only

  For roads with sidewalk (--has-sidewalk):
  - Lane 0: sidewalk (ignored)
  - Lane 1 (rightmost vehicle): right turns only
  - Middle lanes: straight only
  - Lane N-1 (leftmost): left turns + U-turns only

  For 2-lane vehicle roads:
  - Rightmost vehicle lane: right + straight
  - Leftmost lane: left + U-turn
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
        '--auto-detect',
        action='store_true',
        help='Automatically detect and fix all incorrect lane assignments in multi-lane edges'
    )

    parser.add_argument(
        '--has-sidewalk',
        action='store_true',
        help='If set, lane 0 is treated as sidewalk and vehicle lanes start from lane 1'
    )

    parser.add_argument(
        '--osm-file',
        help='OSM XML file to auto-detect edges with centre_turn_lane=yes tag'
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

    # Auto-detect from OSM file if provided
    # Also build the set of center turn lane edges for lane exclusion
    center_turn_lane_edges: Optional[Set[str]] = None
    if args.osm_file:
        if not os.path.exists(args.osm_file):
            logger.error(f"OSM file not found: {args.osm_file}")
            sys.exit(1)

        osm_way_ids = parse_osm_centre_turn_lanes(args.osm_file)
        if osm_way_ids:
            osm_edges = find_edges_by_osm_ids(args.input_edge_file, osm_way_ids)
            if osm_edges:
                # Track ALL matched edges as center turn lane edges
                center_turn_lane_edges = set(osm_edges)
                logger.info(f"Identified {len(center_turn_lane_edges)} center turn lane edges")

                if specific_edges:
                    specific_edges.extend(osm_edges)
                else:
                    specific_edges = osm_edges
            else:
                logger.warning("No matching edges found in edge file for OSM way IDs")
        else:
            logger.warning("No ways with centre_turn_lane=yes found in OSM file")

    if args.names:
        logger.info(f"Searching for edges matching names: {args.names}")
        found_edges = find_edges_by_name_pattern(args.input_edge_file, args.names)
        if specific_edges:
            specific_edges.extend(found_edges)
        else:
            specific_edges = found_edges

        if not specific_edges:
            logger.warning("No edges found matching the specified names")

    # Check if we have edges to process or auto-detect is enabled
    if not specific_edges and not args.auto_detect:
        logger.error("No edges specified. Use --auto-detect, --osm-file, --edges, or --names to specify edges.")
        sys.exit(1)

    try:
        fixed_count = fix_connections(
            input_edge_file=args.input_edge_file,
            input_con_file=con_file,
            output_con_file=output_file,
            specific_edges=specific_edges,
            dry_run=args.dry_run,
            auto_detect=args.auto_detect,
            has_sidewalk=args.has_sidewalk,
            center_turn_lane_edges=center_turn_lane_edges
        )

        if fixed_count > 0 and not args.dry_run:
            logger.info(f"\nOutput written to: {output_file}")
            logger.info("\nNext steps:")
            logger.info("1. Review the changes in the output file")
            logger.info("2. Rebuild the SUMO network using netconvert:")
            logger.info(f"   netconvert --node-files=<node_file>.nod.xml --edge-files=<edge_file>.edg.xml \\")
            logger.info(f"              --connection-files={os.path.basename(output_file)} --output-file=<output>.net.xml")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
