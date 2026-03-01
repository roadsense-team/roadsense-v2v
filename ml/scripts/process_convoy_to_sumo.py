#!/usr/bin/env python3
"""Convert convoy trajectory outputs into a SUMO base_real scenario."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _load_trajectory_csv(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Trajectory not found: {path}")

    data = {
        "t_local_ms": [],
        "t_rel_s": [],
        "lat": [],
        "lon": [],
        "x_m": [],
        "y_m": [],
        "speed_ms": [],
        "heading_deg": [],
    }

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = set(data.keys())
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing trajectory columns: {sorted(missing)}")

        for row in reader:
            for k in data.keys():
                data[k].append(float(row[k]))

    if len(data["t_local_ms"]) < 2:
        raise ValueError(f"{path} has insufficient trajectory rows")

    return data


def _find_active_start_index(speed_ms: Sequence[float], threshold: float = 0.5) -> int:
    for i, v in enumerate(speed_ms):
        if v >= threshold:
            return i
    return 0


def _find_active_end_index(speed_ms: Sequence[float], threshold: float = 0.5) -> int:
    for i in range(len(speed_ms) - 1, -1, -1):
        if speed_ms[i] >= threshold:
            return i
    return max(0, len(speed_ms) - 1)


def _estimate_depart_speed(
    speed_ms: Sequence[float],
    start_idx: int,
    window_samples: int = 30,
) -> float:
    if not speed_ms:
        return 0.0
    i0 = max(0, min(start_idx, len(speed_ms) - 1))
    i1 = min(len(speed_ms), i0 + max(1, window_samples))
    window = [float(v) for v in speed_ms[i0:i1]]
    if not window:
        return float(speed_ms[i0])
    window.sort()
    mid = len(window) // 2
    if len(window) % 2 == 1:
        return window[mid]
    return 0.5 * (window[mid - 1] + window[mid])


def _dedupe_consecutive(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def _edge_is_usable(edge) -> bool:
    edge_id = edge.getID()
    if edge_id.startswith(":"):
        return False
    try:
        return bool(edge.allows("passenger"))
    except Exception:
        return True


def _nearest_edge(net, x: float, y: float, radii_m: Sequence[float]) -> Optional[object]:
    for radius in radii_m:
        candidates = net.getNeighboringEdges(x, y, radius)
        usable = []
        for edge, dist in candidates:
            if _edge_is_usable(edge):
                usable.append((edge, dist))
        if usable:
            usable.sort(key=lambda t: t[1])
            return usable[0][0]
    return None


def _neighbor_edge_candidates(
    net,
    x: float,
    y: float,
    radii_m: Sequence[float],
    max_candidates: int = 8,
) -> List[Tuple[object, float]]:
    best: List[Tuple[object, float]] = []
    seen: Dict[str, float] = {}
    for radius in radii_m:
        for edge, dist in net.getNeighboringEdges(x, y, radius):
            if not _edge_is_usable(edge):
                continue
            edge_id = edge.getID()
            prev = seen.get(edge_id)
            if prev is None or dist < prev:
                seen[edge_id] = float(dist)
        if seen:
            break

    for edge_id, dist in seen.items():
        best.append((net.getEdge(edge_id), dist))
    best.sort(key=lambda t: t[1])
    return best[:max_candidates]


def _shortest_path_edges(net, start_edge, end_edge) -> Optional[List[object]]:
    try:
        shortest = net.getShortestPath(start_edge, end_edge, vClass="passenger")
        if shortest is not None and shortest[0]:
            return shortest[0]
    except TypeError:
        shortest = net.getShortestPath(start_edge, end_edge)
        if shortest is not None and shortest[0]:
            return shortest[0]
    except Exception:
        return None
    return None


def _infer_route_edges_from_endpoints(
    net,
    traj: Dict[str, List[float]],
    start_idx: int,
    end_idx: int,
) -> Optional[List[str]]:
    lats = traj["lat"]
    lons = traj["lon"]

    start_idx = max(0, min(start_idx, len(lats) - 1))
    end_idx = max(0, min(end_idx, len(lats) - 1))
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx

    sx, sy = net.convertLonLat2XY(lons[start_idx], lats[start_idx])
    ex, ey = net.convertLonLat2XY(lons[end_idx], lats[end_idx])

    start_candidates = _neighbor_edge_candidates(net, sx, sy, radii_m=(20.0, 40.0, 80.0, 150.0))
    end_candidates = _neighbor_edge_candidates(net, ex, ey, radii_m=(20.0, 40.0, 80.0, 150.0))
    if not start_candidates or not end_candidates:
        return None

    best_ids: Optional[List[str]] = None
    best_cost = float("inf")

    for s_edge, s_dist in start_candidates:
        for e_edge, e_dist in end_candidates:
            if s_edge.getID() == e_edge.getID():
                candidate_ids = [s_edge.getID()]
                route_cost = float(s_edge.getLength())
            else:
                path = _shortest_path_edges(net, s_edge, e_edge)
                if not path:
                    continue
                candidate_ids = [edge.getID() for edge in path]
                route_cost = float(sum(edge.getLength() for edge in path))

            total_cost = route_cost + float(s_dist) + float(e_dist)
            if total_cost < best_cost and candidate_ids:
                best_cost = total_cost
                best_ids = candidate_ids

    return best_ids


def _bridge_edges(net, edge_ids: Sequence[str]) -> List[str]:
    if not edge_ids:
        return []
    if len(edge_ids) == 1:
        return [edge_ids[0]]

    bridged: List[str] = [edge_ids[0]]
    for next_id in edge_ids[1:]:
        prev_id = bridged[-1]
        if prev_id == next_id:
            continue

        prev_edge = net.getEdge(prev_id)
        next_edge = net.getEdge(next_id)

        path_edges = None
        try:
            shortest = net.getShortestPath(prev_edge, next_edge, vClass="passenger")
            if shortest is not None and shortest[0]:
                path_edges = shortest[0]
        except TypeError:
            shortest = net.getShortestPath(prev_edge, next_edge)
            if shortest is not None and shortest[0]:
                path_edges = shortest[0]
        except Exception:
            path_edges = None

        if path_edges:
            for edge in path_edges[1:]:
                eid = edge.getID()
                if not bridged or bridged[-1] != eid:
                    bridged.append(eid)
            continue

        # Fallback: keep continuity best-effort.
        bridged.append(next_id)

    return _dedupe_consecutive(bridged)


def _infer_route_edges(net, traj: Dict[str, List[float]], sample_stride: int = 5) -> List[str]:
    lats = traj["lat"]
    lons = traj["lon"]

    raw_edges: List[str] = []
    for i in range(0, len(lats), max(1, sample_stride)):
        x, y = net.convertLonLat2XY(lons[i], lats[i])
        edge = _nearest_edge(net, x, y, radii_m=(20.0, 40.0, 80.0, 150.0))
        if edge is None:
            continue
        raw_edges.append(edge.getID())

    raw_edges = _dedupe_consecutive(raw_edges)
    if len(raw_edges) < 2:
        raise RuntimeError("Could not infer a valid route from trajectory points and network.")

    return _bridge_edges(net, raw_edges)


def _route_length_m(net, edge_ids: Sequence[str]) -> float:
    total = 0.0
    for eid in edge_ids:
        total += float(net.getEdge(eid).getLength())
    return total


def _spacing_defaults(analysis_summary: Dict) -> Tuple[float, float]:
    stats = analysis_summary.get("formation", {}).get("stats", {})
    v001_v002 = stats.get("V001_V002", {}).get("mean_m", 11.0)
    v001_v003 = stats.get("V001_V003", {}).get("mean_m", 17.0)
    return float(v001_v002), float(v001_v003)


def _extract_hard_brake(events_json: Optional[Dict], fallback_peak_ms2: float = -8.63) -> float:
    fallback = float(fallback_peak_ms2)
    if not events_json:
        return fallback
    hard = events_json.get("detected", {}).get("hard_event")
    if hard and "min_accel_x_ms2" in hard:
        detected = float(hard["min_accel_x_ms2"])
        # Preserve known recording peak if detector underestimates the event.
        return min(detected, fallback)
    return fallback


def _build_vehicle_rows(
    route_length_m: float,
    spacing_v001_v002: float,
    spacing_v001_v003: float,
    depart_speed_v001: float,
    depart_speed_v002: float,
    depart_speed_v003: float,
    depart_lane: str = "0",
) -> List[Dict[str, str]]:
    # Position is measured from route start; larger departPos is further ahead.
    v001_pos = 0.0
    v002_pos = max(5.0, spacing_v001_v002)
    v003_pos = max(v002_pos + 5.0, spacing_v001_v003)

    synthetic_gap = 22.0
    v004_pos = v003_pos + synthetic_gap
    v005_pos = v004_pos + synthetic_gap
    v006_pos = v005_pos + synthetic_gap

    max_pos = max(0.0, route_length_m - 5.0)
    positions = {
        "V001": min(v001_pos, max_pos),
        "V002": min(v002_pos, max_pos),
        "V003": min(v003_pos, max_pos),
        "V004": min(v004_pos, max_pos),
        "V005": min(v005_pos, max_pos),
        "V006": min(v006_pos, max_pos),
    }

    synthetic_speed = max(depart_speed_v002, depart_speed_v003, depart_speed_v001)
    speeds = {
        "V001": depart_speed_v001,
        "V002": depart_speed_v002,
        "V003": depart_speed_v003,
        "V004": synthetic_speed,
        "V005": synthetic_speed,
        "V006": synthetic_speed,
    }

    colors = {
        "V001": "0,0,1",
        "V002": "0,1,0",
        "V003": "1,0,0",
        "V004": "1,0.5,0",
        "V005": "1,0.5,0",
        "V006": "1,0.5,0",
    }

    # Front of convoy first.
    order = ["V006", "V005", "V004", "V003", "V002", "V001"]
    rows: List[Dict[str, str]] = []
    for vid in order:
        rows.append(
            {
                "id": vid,
                "type": "car",
                "depart": "0",
                "route": "convoy_route",
                "departSpeed": _format_float(max(0.0, speeds[vid])),
                "departPos": _format_float(max(0.0, positions[vid])),
                "departLane": depart_lane,
                "color": colors[vid],
            }
        )
    return rows


def _write_routes_xml(
    out_path: Path,
    route_edges: Sequence[str],
    vehicles: Sequence[Dict[str, str]],
    vtype_decel: float,
    vtype_tau: float,
) -> None:
    root = ET.Element("routes")

    vtype = ET.SubElement(
        root,
        "vType",
        {
            "id": "car",
            "accel": "2.6",
            "decel": _format_float(vtype_decel),
            "sigma": "0.5",
            "length": "5",
            "maxSpeed": "50",
            "tau": _format_float(vtype_tau),
        },
    )
    ET.SubElement(vtype, "param", {"key": "has.ssm.device", "value": "true"})
    ET.SubElement(vtype, "param", {"key": "device.ssm.measures", "value": "TTC DRAC PET"})
    ET.SubElement(vtype, "param", {"key": "device.ssm.thresholds", "value": "4.0 3.4 2.0"})
    ET.SubElement(vtype, "param", {"key": "device.ssm.range", "value": "100.0"})
    ET.SubElement(vtype, "param", {"key": "device.ssm.file", "value": "ssm_output.xml"})

    ET.SubElement(root, "route", {"id": "convoy_route", "edges": " ".join(route_edges)})
    for vehicle in vehicles:
        ET.SubElement(root, "vehicle", vehicle)

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")  # Python 3.9+
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=False)


def _write_sumocfg(out_path: Path) -> None:
    root = ET.Element("configuration")
    input_node = ET.SubElement(root, "input")
    ET.SubElement(input_node, "net-file", {"value": "network.net.xml"})
    ET.SubElement(input_node, "route-files", {"value": "vehicles.rou.xml"})

    time_node = ET.SubElement(root, "time")
    ET.SubElement(time_node, "begin", {"value": "0"})
    ET.SubElement(time_node, "end", {"value": "120"})
    ET.SubElement(time_node, "step-length", {"value": "0.1"})

    output_node = ET.SubElement(root, "output")
    ET.SubElement(output_node, "fcd-output", {"value": "fcd_output.csv"})
    ET.SubElement(output_node, "fcd-output.geo", {"value": "false"})
    ET.SubElement(output_node, "fcd-output.acceleration", {"value": "true"})

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=False)


def _write_metadata(
    out_path: Path,
    analysis_summary: Dict,
    hard_brake_peak_ms2: float,
    source_tx: str,
    source_rx: str,
) -> None:
    spacing = float(analysis_summary.get("formation", {}).get("avg_spacing_m", 13.9))

    payload = {
        "recording_date": "2026-02-28",
        "location": "Rural road outside Deir Hanna, near Sonol station by Nahal Tzalmon, between Mghar and Eilaboun",
        "gps_start": {"lat": 32.8596, "lon": 35.4031},
        "board_placement": "Front windshield of each car (line-of-sight, production-realistic)",
        "duration_s": float(analysis_summary.get("trajectories", {}).get("V001", {}).get("duration_s", 195.5)),
        "vehicles_recorded": ["V001", "V002", "V003"],
        "vehicles_synthetic": ["V004", "V005", "V006"],
        "formation_avg_spacing_m": spacing,
        "hard_braking_peak_ms2": hard_brake_peak_ms2,
        "source_files": {"tx": source_tx, "rx": source_rx},
        "analysis_dir": "ml/data/convoy_analysis_site/",
        "notes": "Ego-only mesh recording. V002/V003 trajectories extracted from V001 RX log. V004-V006 are synthetic peers placed ahead of V003.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_sumolib_net(network_path: Path):
    try:
        import sumolib  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "sumolib is required. Run this script inside the ML Docker/container environment."
        ) from exc
    try:
        import pyproj  # type: ignore # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "pyproj is required for sumolib geo conversion (convertLonLat2XY). "
            "Install it in your active environment or run in Docker where it is available."
        ) from exc
    return sumolib.net.readNet(str(network_path), withInternal=False)


def _resolve_route_edges(
    net,
    traj_v001: Dict[str, List[float]],
    start_idx: int,
    end_idx: int,
    route_edge_id: Optional[str],
) -> List[str]:
    if route_edge_id:
        try:
            edge = net.getEdge(route_edge_id)
        except Exception as exc:
            raise RuntimeError(f"Requested route edge '{route_edge_id}' not found in network.") from exc
        if not _edge_is_usable(edge):
            raise RuntimeError(f"Requested route edge '{route_edge_id}' is not usable for passenger vehicles.")
        return [route_edge_id]

    route_edges = _infer_route_edges_from_endpoints(net, traj_v001, start_idx=start_idx, end_idx=end_idx)
    if not route_edges:
        route_edges = _infer_route_edges(net, traj_v001, sample_stride=5)
    return _dedupe_consecutive(route_edges)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate base_real SUMO files from convoy trajectories.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("data/convoy_analysis_site"),
        help="Directory containing analysis_summary.json and trajectories/*.csv",
    )
    parser.add_argument(
        "--network",
        type=Path,
        default=Path("scenarios/base_real/network.net.xml"),
        help="SUMO network file for the recording area.",
    )
    parser.add_argument(
        "--output-routes",
        type=Path,
        default=Path("scenarios/base_real/vehicles.rou.xml"),
        help="Output vehicles.rou.xml path.",
    )
    parser.add_argument(
        "--output-sumocfg",
        type=Path,
        default=Path("scenarios/base_real/scenario.sumocfg"),
        help="Output scenario.sumocfg path.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=Path("scenarios/base_real/recording_metadata.json"),
        help="Output recording_metadata.json path.",
    )
    parser.add_argument(
        "--vtype-decel",
        type=float,
        default=5.0,
        help="vType decel in m/s^2 (default: 5.0).",
    )
    parser.add_argument(
        "--vtype-tau",
        type=float,
        default=1.0,
        help="vType tau in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--route-edge-id",
        type=str,
        default="-635191227",
        help="Force a single-edge route by ID (default: -635191227 for base_real).",
    )
    parser.add_argument(
        "--depart-lane",
        type=str,
        default="0",
        help="Vehicle departLane value (default: 0).",
    )
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    analysis_summary = _load_json(analysis_dir / "analysis_summary.json")
    events_path = analysis_dir / "convoy_events.json"
    events_json = _load_json(events_path) if events_path.exists() else None

    traj_v001 = _load_trajectory_csv(analysis_dir / "trajectories/V001_trajectory.csv")
    traj_v002 = _load_trajectory_csv(analysis_dir / "trajectories/V002_trajectory.csv")
    traj_v003 = _load_trajectory_csv(analysis_dir / "trajectories/V003_trajectory.csv")

    i1 = _find_active_start_index(traj_v001["speed_ms"])
    i1_end = _find_active_end_index(traj_v001["speed_ms"])
    i2 = _find_active_start_index(traj_v002["speed_ms"])
    i3 = _find_active_start_index(traj_v003["speed_ms"])

    try:
        net = _load_sumolib_net(args.network)
        route_edges = _resolve_route_edges(
            net=net,
            traj_v001=traj_v001,
            start_idx=i1,
            end_idx=i1_end,
            route_edge_id=args.route_edge_id,
        )
        route_len = _route_length_m(net, route_edges)
    except Exception as exc:
        print(f"Failed route inference from network/trajectory: {exc}")
        return 1

    spacing_v001_v002, spacing_v001_v003 = _spacing_defaults(analysis_summary)
    vehicles = _build_vehicle_rows(
        route_length_m=route_len,
        spacing_v001_v002=spacing_v001_v002,
        spacing_v001_v003=spacing_v001_v003,
        depart_speed_v001=_estimate_depart_speed(traj_v001["speed_ms"], i1),
        depart_speed_v002=_estimate_depart_speed(traj_v002["speed_ms"], i2),
        depart_speed_v003=_estimate_depart_speed(traj_v003["speed_ms"], i3),
        depart_lane=args.depart_lane,
    )

    _write_routes_xml(
        out_path=args.output_routes,
        route_edges=route_edges,
        vehicles=vehicles,
        vtype_decel=max(3.5, min(6.0, args.vtype_decel)),
        vtype_tau=max(0.5, min(2.0, args.vtype_tau)),
    )
    _write_sumocfg(args.output_sumocfg)
    _write_metadata(
        out_path=args.output_metadata,
        analysis_summary=analysis_summary,
        hard_brake_peak_ms2=_extract_hard_brake(events_json),
        source_tx="Convoy_recording_02282026/V001_tx_004.csv",
        source_rx="Convoy_recording_02282026/V001_rx_004.csv",
    )

    print("Generated base_real scenario files:")
    print(f"- route edges: {len(route_edges)}")
    print(f"- route length: {_format_float(route_len)} m")
    print(f"- vehicles: {len(vehicles)}")
    print(f"- {args.output_routes}")
    print(f"- {args.output_sumocfg}")
    print(f"- {args.output_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
