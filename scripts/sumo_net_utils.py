#!/usr/bin/env python3
"""
Shared utilities for SUMO network manipulation scripts.

Contains common functions used by add_bidirectional_center_lane.py
and fix_connections.py.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Set
import logging
import math

logger = logging.getLogger(__name__)


def parse_osm_centre_turn_lanes(osm_file: str) -> List[str]:
    """
    Parse an OSM XML file and find all way IDs with centre_turn_lane=yes.

    Args:
        osm_file: Path to the OSM XML file

    Returns:
        List of OSM way ID strings that have centre_turn_lane=yes
    """
    logger.info(f"Parsing OSM file for centre_turn_lane tags: {osm_file}")

    tree = ET.parse(osm_file)
    root = tree.getroot()

    way_ids = []
    for way in root.findall('way'):
        way_id = way.get('id')
        if way_id is None:
            continue

        for tag in way.findall('tag'):
            if tag.get('k') == 'centre_turn_lane' and tag.get('v') == 'yes':
                name_tag = way.find("tag[@k='name']")
                name = name_tag.get('v') if name_tag is not None else 'unnamed'
                logger.info(f"  Found centre_turn_lane way: {way_id} ({name})")
                way_ids.append(way_id)
                break

    logger.info(f"Found {len(way_ids)} ways with centre_turn_lane=yes")
    return way_ids


def find_edges_by_osm_ids(
    edge_file: str,
    osm_way_ids: List[str],
    deduplicate_segments: bool = False
) -> List[str]:
    """
    Find SUMO edge IDs that correspond to given OSM way IDs.

    The mapping works by checking if the edge ID's base (before '#') matches
    an OSM way ID (e.g., "87235091#0" matches "87235091").

    Args:
        edge_file: Path to the SUMO edge XML file
        osm_way_ids: List of OSM way ID strings to match
        deduplicate_segments: If True, only return one edge per OSM way ID
            (the first segment). Useful when processing edge pairs.

    Returns:
        List of matched SUMO edge IDs
    """
    logger.info(f"Matching {len(osm_way_ids)} OSM way IDs to edge IDs in: {edge_file}")

    tree = ET.parse(edge_file)
    root = tree.getroot()

    osm_id_set = set(osm_way_ids)
    matched_edges = []

    for edge_elem in root.findall('edge'):
        edge_id = edge_elem.get('id')
        if edge_id is None:
            continue

        base_id = edge_id.split('#')[0]

        if base_id in osm_id_set:
            matched_edges.append(edge_id)
            logger.debug(f"  Matched edge {edge_id} -> OSM way {base_id}")

    if deduplicate_segments:
        seen_base_ids: Set[str] = set()
        unique_edges = []
        for edge_id in matched_edges:
            base_id = edge_id.split('#')[0]
            if base_id not in seen_base_ids:
                seen_base_ids.add(base_id)
                unique_edges.append(edge_id)

        logger.info(f"Matched {len(unique_edges)} unique edges ({len(matched_edges)} total with segments)")
        for edge_id in unique_edges:
            logger.info(f"  {edge_id}")
        return unique_edges

    logger.info(f"Matched {len(matched_edges)} edges to OSM way IDs")
    return matched_edges


def parse_shape_coordinates(shape_str: str) -> List[Tuple[float, float]]:
    """Parse SUMO shape string into list of (x, y) coordinate tuples."""
    if not shape_str:
        return []
    coords = []
    for coord_pair in shape_str.strip().split():
        x, y = coord_pair.split(',')
        coords.append((float(x), float(y)))
    return coords


def determine_turn_direction(from_angle: float, to_angle: float) -> str:
    """
    Determine the turn direction based on incoming and outgoing angles.

    Args:
        from_angle: Heading angle of the incoming edge (degrees, 0=East, 90=North)
        to_angle: Heading angle of the outgoing edge (degrees)

    Returns:
        'r' (right), 's' (straight), 'l' (left), or 't' (u-turn)
    """
    turn = (to_angle - from_angle) % 360

    if turn > 180:
        turn -= 360

    if -45 <= turn <= 45:
        return 's'
    elif 45 < turn <= 135:
        return 'l'
    elif -135 <= turn < -45:
        return 'r'
    else:
        return 't'
