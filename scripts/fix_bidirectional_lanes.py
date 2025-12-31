#!/usr/bin/env python3
"""
SUMO Bidirectional Lane Fixer

This script identifies and fixes bidirectional lanes in SUMO plain XML files.
Bidirectional lanes are common in rail networks and some road scenarios where
a single track/lane can be used in both directions.

According to SUMO documentation, bidirectional track usage must be enabled
explicitly using two edges with exactly reversed geometries and
spreadType="center" attribute.

Author: TeraSim Team
License: MIT
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
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
    spread_type: Optional[str]
    element: ET.Element
    name: Optional[str] = None

    @property
    def node_pair(self) -> Tuple[str, str]:
        """Return node pair for comparison"""
        return (self.from_node, self.to_node)

    @property
    def reverse_node_pair(self) -> Tuple[str, str]:
        """Return reverse node pair"""
        return (self.to_node, self.from_node)


def parse_shape_coordinates(shape_str: str) -> List[Tuple[float, float]]:
    """
    Parse shape string into list of coordinate tuples.

    Args:
        shape_str: Space-separated coordinate pairs "x1,y1 x2,y2 x3,y3"

    Returns:
        List of (x, y) tuples
    """
    if not shape_str:
        return []

    coords = []
    for coord_pair in shape_str.strip().split():
        x, y = coord_pair.split(',')
        coords.append((float(x), float(y)))
    return coords


def reverse_shape(shape_str: str) -> str:
    """
    Reverse a shape string.

    Args:
        shape_str: Space-separated coordinate pairs

    Returns:
        Reversed shape string
    """
    if not shape_str:
        return ""

    coords = parse_shape_coordinates(shape_str)
    reversed_coords = coords[::-1]
    return ' '.join(f"{x:.2f},{y:.2f}" for x, y in reversed_coords)


def shapes_are_reverse(shape1: str, shape2: str, tolerance: float = 0.5) -> bool:
    """
    Check if two shapes are reverses of each other (within tolerance).

    Args:
        shape1: First shape string
        shape2: Second shape string
        tolerance: Maximum distance tolerance in meters

    Returns:
        True if shapes are reverses of each other
    """
    if not shape1 or not shape2:
        return False

    coords1 = parse_shape_coordinates(shape1)
    coords2 = parse_shape_coordinates(shape2)

    if len(coords1) != len(coords2):
        return False

    # Check if coords2 is reverse of coords1
    for (x1, y1), (x2, y2) in zip(coords1, reversed(coords2)):
        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if dist > tolerance:
            return False

    return True


def find_bidirectional_pairs(edges: List[EdgeInfo]) -> List[Tuple[EdgeInfo, EdgeInfo]]:
    """
    Find pairs of edges that form bidirectional lanes.

    Two edges form a bidirectional pair if:
    1. They connect the same two nodes in opposite directions

    Args:
        edges: List of EdgeInfo objects

    Returns:
        List of (edge1, edge2) tuples representing bidirectional pairs
    """
    # Build a map of reverse connections
    reverse_map: Dict[Tuple[str, str], List[EdgeInfo]] = {}

    for edge in edges:
        reverse_pair = edge.reverse_node_pair
        if reverse_pair not in reverse_map:
            reverse_map[reverse_pair] = []
        reverse_map[reverse_pair].append(edge)

    # Find bidirectional pairs
    pairs = []
    processed_edges: Set[str] = set()

    for edge in edges:
        if edge.id in processed_edges:
            continue

        # Look for reverse edge
        node_pair = edge.node_pair
        if node_pair in reverse_map:
            for reverse_edge in reverse_map[node_pair]:
                if reverse_edge.id not in processed_edges:
                    # Found a bidirectional pair
                    pairs.append((edge, reverse_edge))
                    processed_edges.add(edge.id)
                    processed_edges.add(reverse_edge.id)
                    break

    return pairs


def find_reverse_edge(edge_id: str, edge_map: Dict[str, EdgeInfo]) -> Optional[EdgeInfo]:
    """
    Find the reverse edge for a given edge ID.

    Args:
        edge_id: The edge ID to find the reverse for
        edge_map: Dictionary mapping edge IDs to EdgeInfo

    Returns:
        The reverse EdgeInfo if found, None otherwise
    """
    if edge_id not in edge_map:
        return None

    edge = edge_map[edge_id]
    reverse_node_pair = edge.reverse_node_pair

    # Search for an edge with reversed from/to nodes
    for other_id, other_edge in edge_map.items():
        if other_id == edge_id:
            continue
        if other_edge.node_pair == reverse_node_pair:
            return other_edge

    return None


def fix_bidirectional_lanes(
    input_edge_file: str,
    output_edge_file: str,
    specific_edges: Optional[List[str]] = None,
    dry_run: bool = False
) -> int:
    """
    Fix bidirectional lanes in SUMO plain edge XML file.

    Args:
        input_edge_file: Path to input aa_plain.edg.xml
        output_edge_file: Path to output fixed aa_plain.edg.xml
        specific_edges: List of specific edge IDs to fix (optional)
        dry_run: If True, only report what would be changed

    Returns:
        Number of edge pairs fixed
    """
    logger.info(f"Reading edge file: {input_edge_file}")

    # Parse XML
    tree = ET.parse(input_edge_file)
    root = tree.getroot()

    # Extract edges
    edges = []
    edge_map: Dict[str, EdgeInfo] = {}

    for edge_elem in root.findall('edge'):
        edge_id = edge_elem.get('id')
        from_node = edge_elem.get('from')
        to_node = edge_elem.get('to')

        if edge_id is None or from_node is None or to_node is None:
            continue

        shape = edge_elem.get('shape', '')
        spread_type = edge_elem.get('spreadType')
        name = edge_elem.get('name')

        edge_info = EdgeInfo(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            shape=shape,
            spread_type=spread_type,
            element=edge_elem,
            name=name
        )
        edges.append(edge_info)
        edge_map[edge_id] = edge_info

    logger.info(f"Found {len(edges)} total edges")

    # Find bidirectional pairs
    if specific_edges:
        # For specific edges, find their reverse edges directly
        pairs: List[Tuple[EdgeInfo, EdgeInfo]] = []
        processed_edges: Set[str] = set()

        for edge_id in specific_edges:
            if edge_id in processed_edges:
                continue

            if edge_id not in edge_map:
                logger.warning(f"Edge not found: {edge_id}")
                continue

            edge = edge_map[edge_id]
            reverse_edge = find_reverse_edge(edge_id, edge_map)

            if reverse_edge:
                pairs.append((edge, reverse_edge))
                processed_edges.add(edge_id)
                processed_edges.add(reverse_edge.id)
                logger.info(f"Found reverse edge for {edge_id}: {reverse_edge.id}")
            else:
                logger.warning(f"No reverse edge found for {edge_id}")

        logger.info(f"Found {len(pairs)} pairs for specified edges")
    else:
        # Find all bidirectional pairs
        pairs = find_bidirectional_pairs(edges)
        logger.info(f"Found {len(pairs)} potential bidirectional pairs")

    # Fix the pairs
    fixed_count = 0

    for edge1, edge2 in pairs:
        logger.info(f"\nProcessing pair: {edge1.id} <-> {edge2.id}")
        logger.info(f"  Names: {edge1.name} <-> {edge2.name}")
        logger.info(f"  Nodes: {edge1.from_node}->{edge1.to_node} <-> {edge2.from_node}->{edge2.to_node}")

        # Check if shapes are already reversed
        if shapes_are_reverse(edge1.shape, edge2.shape):
            logger.info(f"  ✓ Shapes are already reversed")
        else:
            logger.info(f"  ⚠ Shapes are NOT reversed - will fix")

        if not dry_run:
            # Use edge1's shape as the reference, reverse it for edge2
            reference_shape = edge1.shape if edge1.shape else edge2.shape
            reversed_shape = reverse_shape(reference_shape)

            # Update shapes
            edge1.element.set('shape', reference_shape)
            edge2.element.set('shape', reversed_shape)
            logger.info(f"  ✓ Updated shapes to be exact reverses")

            # Set spreadType="center" for both edges
            edge1.element.set('spreadType', 'center')
            edge2.element.set('spreadType', 'center')
            logger.info(f"  ✓ Set spreadType='center' for both edges")

        fixed_count += 1

    if dry_run:
        logger.info(f"\n[DRY RUN] Would fix {fixed_count} bidirectional pairs")
    else:
        # Write output
        logger.info(f"\nWriting fixed edge file: {output_edge_file}")

        # Ensure pretty printing
        ET.indent(tree, space="    ")
        tree.write(output_edge_file, encoding='UTF-8', xml_declaration=True)

        logger.info(f"✓ Fixed {fixed_count} bidirectional pairs")

    return fixed_count


def find_edges_by_name_pattern(input_edge_file: str, name_patterns: List[str]) -> List[str]:
    """
    Find edge IDs that match given name patterns.

    Args:
        input_edge_file: Path to edge XML file
        name_patterns: List of name patterns to search for (case-insensitive substring match)

    Returns:
        List of matching edge IDs
    """
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
        description='Fix bidirectional lanes in SUMO plain XML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix all bidirectional pairs in Ann_Arbor directory
  python fix_bidirectional_lanes.py Ann_Arbor/aa_plain.edg.xml

  # Dry run to see what would be changed
  python fix_bidirectional_lanes.py Ann_Arbor/aa_plain.edg.xml --dry-run

  # Fix only specific edges by name pattern
  python fix_bidirectional_lanes.py Ann_Arbor/aa_plain.edg.xml --names "hubbard" "maple" "fuller"

  # Fix specific edges by ID
  python fix_bidirectional_lanes.py Ann_Arbor/aa_plain.edg.xml --edges "edge_id_1" "edge_id_2"

  # Specify output file
  python fix_bidirectional_lanes.py Ann_Arbor/aa_plain.edg.xml -o Ann_Arbor/aa_plain_fixed.edg.xml

Note: For fixing lane connection assignments (e.g., left turn on wrong lane),
use the separate fix_connections.py script.
        """
    )

    parser.add_argument(
        'input_file',
        help='Input SUMO plain edge XML file (e.g., aa_plain.edg.xml)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Output edge XML file (default: overwrites input file)',
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
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    # Determine output file
    output_file = args.output if args.output else args.input_file

    # Find edges by name if requested
    specific_edges = args.edges
    if args.names:
        logger.info(f"Searching for edges matching names: {args.names}")
        found_edges = find_edges_by_name_pattern(args.input_file, args.names)
        if specific_edges:
            specific_edges.extend(found_edges)
        else:
            specific_edges = found_edges

        if not specific_edges:
            logger.warning("No edges found matching the specified names")

    try:
        fixed_count = fix_bidirectional_lanes(
            input_edge_file=args.input_file,
            output_edge_file=output_file,
            specific_edges=specific_edges,
            dry_run=args.dry_run
        )

        if fixed_count > 0:
            logger.info(f"\n{'[DRY RUN] ' if args.dry_run else ''}Successfully processed {fixed_count} bidirectional pairs")
            if not args.dry_run:
                logger.info(f"Output written to: {output_file}")
                logger.info("\nNext steps:")
                logger.info("1. Review the changes in the output file")
                logger.info("2. Rebuild the SUMO network using netconvert:")
                logger.info(f"   netconvert --node-files=aa_plain.nod.xml --edge-files={os.path.basename(output_file)} \\")
                logger.info("              --connection-files=aa_plain.con.xml --output-file=aa_fixed.net.xml")
        else:
            logger.info("No bidirectional pairs found to fix")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
