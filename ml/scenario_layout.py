"""
Shared helpers for keeping convoy scenario geometry and timing stable.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET


EGO_VEHICLE_ID = "V001"
DEFAULT_EGO_DEPART_POS_M = 0.0
DEFAULT_MIN_GAP_M = 25.0
DEFAULT_MAX_CONVOY_SPAN_M = 600.0
DEFAULT_SINGLE_PEER_GAP_M = 50.0
DEFAULT_MIN_GAP_AT_PEER_COUNT = 5
DEFAULT_GAP_GRANULARITY_M = 5.0
DEFAULT_SUMO_END_TIME_S = 65.0


def format_position(value: float) -> str:
    return f"{value:.3f}"


def _format_config_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def require_vehicle(root: ET.Element, vehicle_id: str = EGO_VEHICLE_ID) -> ET.Element:
    for vehicle in root.findall("vehicle"):
        if vehicle.get("id") == vehicle_id:
            return vehicle
    raise ValueError(f"vehicles.rou.xml missing {vehicle_id} vehicle.")


def vehicle_depart_pos(vehicle: ET.Element) -> float:
    raw = vehicle.get("departPos")
    if raw is None:
        return float("inf")
    try:
        return float(raw)
    except ValueError:
        return float("inf")


def determine_peer_spacing(
    peer_count: int,
    *,
    min_gap: float = DEFAULT_MIN_GAP_M,
    max_convoy_span: float = DEFAULT_MAX_CONVOY_SPAN_M,
    single_peer_gap: float = DEFAULT_SINGLE_PEER_GAP_M,
    min_gap_at_peer_count: int = DEFAULT_MIN_GAP_AT_PEER_COUNT,
    gap_granularity: float = DEFAULT_GAP_GRANULARITY_M,
) -> float:
    if peer_count < 0:
        raise ValueError(f"peer_count must be non-negative, got {peer_count}")
    if peer_count == 0:
        return min_gap
    if max_convoy_span <= 0.0:
        raise ValueError(f"max_convoy_span must be positive, got {max_convoy_span}")
    if min_gap <= 0.0:
        raise ValueError(f"min_gap must be positive, got {min_gap}")

    if peer_count == 1:
        raw_spacing = single_peer_gap
    elif peer_count >= min_gap_at_peer_count:
        raw_spacing = min_gap
    else:
        span = max(1, min_gap_at_peer_count - 1)
        progress = (peer_count - 1) / span
        raw_spacing = single_peer_gap - progress * (single_peer_gap - min_gap)

    if gap_granularity > 0.0:
        raw_spacing = math.floor(raw_spacing / gap_granularity) * gap_granularity

    spacing = max(min_gap, raw_spacing)
    if spacing * peer_count > max_convoy_span:
        max_supported_spacing = max_convoy_span / peer_count
        if max_supported_spacing < min_gap:
            raise ValueError(
                f"Cannot fit {peer_count} peers within span {max_convoy_span}m "
                f"while maintaining min_gap={min_gap}m"
            )
        spacing = max_supported_spacing
    return spacing


def redistribute_peer_depart_positions(
    root: ET.Element,
    *,
    ego_vehicle_id: str = EGO_VEHICLE_ID,
    ego_pos: float = DEFAULT_EGO_DEPART_POS_M,
    min_gap: float = DEFAULT_MIN_GAP_M,
    max_convoy_span: float = DEFAULT_MAX_CONVOY_SPAN_M,
) -> None:
    ego_vehicle = require_vehicle(root, ego_vehicle_id)
    peers = sorted(
        (
            vehicle
            for vehicle in root.findall("vehicle")
            if vehicle.get("id") != ego_vehicle_id
        ),
        key=lambda vehicle: (vehicle_depart_pos(vehicle), vehicle.get("id") or ""),
    )

    ego_vehicle.set("departPos", format_position(ego_pos))
    if not peers:
        return

    spacing = determine_peer_spacing(
        len(peers),
        min_gap=min_gap,
        max_convoy_span=max_convoy_span,
    )
    for index, peer in enumerate(peers, start=1):
        peer.set("departPos", format_position(ego_pos + spacing * index))


def set_sumocfg_end_time(
    sumocfg_tree: ET.ElementTree,
    *,
    end_time_s: float = DEFAULT_SUMO_END_TIME_S,
) -> None:
    root = sumocfg_tree.getroot()
    time_element = root.find("time")
    if time_element is None:
        time_element = ET.SubElement(root, "time")

    end_element = time_element.find("end")
    if end_element is None:
        end_element = ET.SubElement(time_element, "end")

    end_element.set("value", _format_config_value(end_time_s))
