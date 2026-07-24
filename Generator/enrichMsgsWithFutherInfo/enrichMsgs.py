import argparse
import json
import traci
import sys
import math
import os
from typing import List, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_nearest_edge_neighbors(x: float, y: float, radius: float = 100):
    try:
        edges = traci.simulation.getNeighboringEdges(x, y, radius)
        if edges:
            edges_sorted = sorted(edges, key=lambda e: e[1])
            return edges_sorted[0][0]
    except:
        return None


def get_distance_to_nearest_road(x: float, y: float) -> float:
    try:
        edge_id = None
        lane_pos = None
        lane_index = None

        try:
            edge_id, lane_pos, lane_index = traci.simulation.convertRoad(x, y)
        except:
            pass

        if edge_id is None:
            try:
                edges = traci.simulation.getNeighboringEdges(x, y, 500)
                if edges:
                    edge_id = edges[0][0]
                    edge_shape = traci.edge.getShape(edge_id)
                    min_dist = float('inf')
                    for i, point in enumerate(edge_shape):
                        dist = traci.simulation.getDistance2D(x, y, point[0], point[1])
                        if dist < min_dist:
                            min_dist = dist
                            if i == 0:
                                lane_pos = 0
                            elif i == len(edge_shape) - 1:
                                lane_pos = traci.edge.getLength(edge_id)
                            else:
                                lane_pos = (i / (len(edge_shape) - 1)) * traci.edge.getLength(edge_id)
                    lane_index = 0
            except Exception as e:
                print(f"Fallback failed for position ({x}, {y}): {e}")
                return 0

        if edge_id is None:
            return 0

        num_lanes = traci.edge.getLaneNumber(edge_id)
        lane_id = f"{edge_id}_{lane_index}"
        lane_length = traci.lane.getLength(lane_id)
        lane_pos = max(0, min(lane_pos, lane_length))
        heading = traci.edge.getAngle(edge_id, lane_pos)

        center_x, center_y = traci.simulation.convert2D(edge_id, lane_pos, lane_index)

        total_offset = 0
        for i in range(lane_index, num_lanes):
            lane_width = traci.lane.getWidth(f"{edge_id}_{i}")
            if i > lane_index:
                total_offset += lane_width
            else:
                total_offset += lane_width / 2

        new_heading = (heading - 90) % 360
        heading_rad = math.radians(new_heading)
        right_angle = heading_rad

        mittle_edge_x = center_x + math.sin(right_angle) * total_offset
        mittle_edge_y = center_y + math.cos(right_angle) * total_offset
        right_lat, right_lon = traci.simulation.convertGeo(mittle_edge_x, mittle_edge_y)

        distance_mittle = traci.simulation.getDistance2D(mittle_edge_x, mittle_edge_y, x, y) * -1
        total_width = sum(traci.lane.getWidth(f"{edge_id}_{i}") for i in range(num_lanes))
        distance_edge = distance_mittle + total_width

        return distance_edge

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def parse_position(pos_string) -> Tuple[float, float, float]:
    if isinstance(pos_string, str):
        parts = pos_string.split(',')
    else:
        parts = pos_string
    return float(parts[0]), float(parts[1]), float(parts[2])


def worker_process_batch(json_files: List[str], sumo_config: str, worker_id: int):
    """Worker processes multiple JSON files with a single SUMO instance."""
    results = []
    port = 8873 + worker_id

    try:
        # Start SUMO once
        sumo_binary = "sumo"
        traci.start([sumo_binary, "-c", sumo_config, "--no-step-log", "true"],
                    port=port, label=str(port))
        traci.switch(str(port))

        # Process all assigned files
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    messages = json.load(f)

                for message in messages:
                    sender = message.get('sender', {})
                    pos_string = sender.get('pos', '')
                    if pos_string:
                        x, y, z = parse_position(pos_string)
                        sender['distance_to_road_edge'] = get_distance_to_nearest_road(x, y)

                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(messages, f, indent=2)

                results.append({'status': 'ok', 'file': json_file})

            except Exception as e:
                results.append({'status': 'error', 'file': json_file, 'error': str(e)})

    finally:
        traci.close()

    return results


def discover_message_files(input_folder: Path) -> list[Path]:
    if input_folder.name in {"cam", "cpm"}:
        return sorted(input_folder.glob("*.json"))

    nested_files = sorted(
        path for path in input_folder.rglob("*.json")
        if path.parent.name in {"cam", "cpm"}
    )
    if nested_files:
        return nested_files
    return sorted(input_folder.glob("*.json"))


def infer_sumo_config(input_folder: Path) -> Path:
    search_roots = [input_folder, *input_folder.parents]
    config_descriptors = []
    for root in search_roots:
        scenarios_dir = root / "scenarios"
        if scenarios_dir.is_dir():
            config_descriptors.extend(scenarios_dir.glob("*/sumo/sumo_config.json"))
            break

    resolved = []
    for descriptor in config_descriptors:
        with descriptor.open("r", encoding="utf-8") as file:
            config_name = json.load(file).get("sumoConfigurationFile")
        if config_name:
            candidate = descriptor.parent / config_name
            if candidate.is_file():
                resolved.append(candidate)

    if len(resolved) == 1:
        return resolved[0]
    if not resolved:
        raise ValueError(
            "Could not infer the SUMO configuration. Pass it explicitly as the "
            "second positional argument."
        )
    raise ValueError(
        "Multiple SUMO configurations were found. Pass the intended .sumocfg "
        "file explicitly as the second positional argument."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add sender road-edge distance to CAM and CPM JSON messages."
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="A flat message folder, a json_* folder, or an urban/highway test root",
    )
    parser.add_argument(
        "sumo_config",
        nargs="?",
        type=Path,
        help="SUMO .sumocfg file; inferred from the test structure when omitted",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel SUMO workers",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_folder = args.input_folder.resolve()
    if not input_folder.is_dir():
        print(f"Error: input folder does not exist: {input_folder}", file=sys.stderr)
        return 1

    try:
        sumo_config = (
            args.sumo_config.resolve()
            if args.sumo_config is not None
            else infer_sumo_config(input_folder)
        )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if not sumo_config.is_file():
        print(f"Error: SUMO configuration does not exist: {sumo_config}", file=sys.stderr)
        return 1

    json_files = discover_message_files(input_folder)
    total_files = len(json_files)

    if total_files == 0:
        print(
            f"Error: no CAM or CPM JSON files found under {input_folder}",
            file=sys.stderr,
        )
        return 1

    max_workers = max(1, args.workers)
    cam_count = sum(path.parent.name == "cam" for path in json_files)
    cpm_count = sum(path.parent.name == "cpm" for path in json_files)
    print(
        f"Found {total_files} message files ({cam_count} CAM, {cpm_count} CPM).",
        file=sys.stderr,
    )
    print(f"Using SUMO configuration: {sumo_config}", file=sys.stderr)
    print(f"Starting processing with {max_workers} workers...", file=sys.stderr)
    count = 0
    errors = []

    # Split files across workers
    chunk_size = max(1, total_files // max_workers)
    chunks = [json_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

    with ProcessPoolExecutor(max_workers=min(max_workers, len(chunks))) as executor:
        futures = []

        for worker_id, chunk in enumerate(chunks):
            f = executor.submit(worker_process_batch,
                                [str(f) for f in chunk],
                                str(sumo_config),
                                worker_id)
            futures.append(f)

        # Collect results
        for future in as_completed(futures):
            try:
                for result in future.result():
                    count += 1
                    if result['status'] == 'ok':
                        print(f"Processed file {count}/{total_files}: {Path(result['file']).name}")
                    else:
                        print(f"[ERROR] {Path(result['file']).name}: {result.get('error')}", file=sys.stderr)
                        errors.append((Path(result['file']).name, result.get('error')))
            except Exception as e:
                print(f"[ERROR] Worker failed: {e}", file=sys.stderr)
                errors.append(("<worker>", str(e)))

    # Summary
    if errors:
        print(f"\n{len(errors)} files with errors:", file=sys.stderr)
        for file, error in errors:
            print(f"  - {file}: {error}", file=sys.stderr)
        return 1
    else:
        print(f"\nAll {total_files} files processed successfully!")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
