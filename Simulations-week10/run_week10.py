#!/usr/bin/env python3
"""Prepare, run, and validate the week-10 conference dataset matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MANIFEST_PATH = HERE / "experiment.json"
RUNTIME = HERE / ".runtime"
WORK = HERE / ".work"
CLEAN_ROOT = HERE / "seeded-simulations-300sec"
ATTACK_ROOT = HERE / "attacks"
LOG_ROOT = HERE / "run-logs"
ATTACK_GENERATOR = REPO / "Generator/attackGenerator/attackGenerator.py"
ATTACK_PYTHON = Path(sys.executable)


class PipelineError(RuntimeError):
    pass


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() or "unknown"


def run_checked(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd or REPO, check=True)


def build_application(manifest: dict[str, Any]) -> Path:
    project = REPO / manifest["application_project"]
    run_checked(["mvn", "-q", "test", "package"], cwd=project)
    jar = project / "target/CamApp-0.0.1.jar"
    if not jar.is_file():
        raise PipelineError(f"Application build did not produce {jar}")
    return jar


def copy_tree_reflink(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".copying")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    run_checked(["cp", "-a", "--reflink=auto", f"{source}/.", str(temporary)])
    temporary.replace(destination)


def set_sumo_seed_and_end(path: Path, seed: int, stop_time: int) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    time_node = root.find("time")
    if time_node is None:
        time_node = ET.SubElement(root, "time")
    end = time_node.find("end")
    if end is None:
        end = ET.SubElement(time_node, "end")
    end.set("value", str(stop_time))
    random_node = root.find("random_number")
    if random_node is None:
        random_node = ET.SubElement(root, "random_number")
    seed_node = random_node.find("seed")
    if seed_node is None:
        seed_node = ET.SubElement(random_node, "seed")
    seed_node.set("value", str(seed))
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def set_omnet_seed(path: Path, seed_set: int) -> None:
    text = path.read_text(encoding="utf-8")
    line = f"seed-set = {seed_set}"
    if re.search(r"(?m)^\s*seed-set\s*=", text):
        text = re.sub(r"(?m)^\s*seed-set\s*=.*$", line, text)
    else:
        text = text.replace("[General]", f"[General]\n{line}", 1)
    path.write_text(text, encoding="utf-8")


def scenario_runtime_name(setting: str, seed_id: str) -> str:
    return f"seed-{seed_id}__{setting}"


def prepare_scenario(
    manifest: dict[str, Any], setting: str, seed_id: str, jar: Path,
    *, smoke: bool = False,
) -> tuple[str, Path, dict[str, Any]]:
    definition = dict(manifest["scenarios"][setting])
    seed = manifest["simulation_seeds"][seed_id]
    runtime_name = scenario_runtime_name(setting, seed_id)
    scenario_root = RUNTIME / "scenarios" / runtime_name
    source = REPO / manifest["source_scenario"]
    copy_tree_reflink(source, scenario_root)

    start = int(definition["communication_start"])
    stop = int(definition["communication_end"])
    if smoke:
        runtime_name = "smoke__InTAS_urban_300sec_pipeline"
        scenario_root = RUNTIME / "smoke-scenarios" / runtime_name
        copy_tree_reflink(source, scenario_root)
        # The InTAS urban communication area has no equipped vehicles in the
        # opening seconds. A 300 s warm-up keeps this smoke test short while
        # exercising actual CAM, CPM, and ego output.
        start, stop = 300, 330

    scenario_config_path = scenario_root / "scenario_config.json"
    scenario_config = json.loads(scenario_config_path.read_text(encoding="utf-8-sig"))
    scenario_config["simulation"]["duration"] = f"{stop}s"
    scenario_config["simulation"]["randomSeed"] = int(seed["mosaic"])
    scenario_config["federates"]["output"] = False
    scenario_config.pop("duration", None)
    write_json(scenario_config_path, scenario_config)

    bounds = definition["bounds"]
    etsi = {
        "minimalPayloadLength": 200,
        "maxStartOffset": "1s",
        "minInterval": "500ms",
        "maxInterval": "1s",
        "positionChange": 4,
        "headingChange": 4,
        "velocityChange": 0.5,
        "pseudonymInterval": "50s",
        "jsonPath": "/opt/mosaic/output/",
        "pseudonymDebugPath": "/opt/mosaic/logs/pseudonym_debug.json",
        "simulationTime": {"start": f"{start}s", "end": f"{stop}s"},
        "simulationArea": bounds,
        "enableDriverProfiles": True,
    }
    write_json(scenario_root / "application/EtsiApplication.json", etsi)
    application_config = {
        "perceptionConfiguration": {
            "vehicleIndex": {"enabled": True, "type": "grid", "cellWidth": "5m", "cellHeight": "5m"},
            "trafficLightIndex": {"enabled": False},
            "wallIndex": {"enabled": True},
            "perceptionArea": {
                "a": {"latitude": bounds["maxX"], "longitude": bounds["minY"]},
                "b": {"latitude": bounds["minX"], "longitude": bounds["maxY"]},
            },
        }
    }
    write_json(scenario_root / "application/application_config.json", application_config)
    shutil.copy2(jar, scenario_root / "application/CamApp-0.0.1.jar")
    shutil.copy2(REPO / manifest["application_database"], scenario_root / "application/InTAS.db")

    set_sumo_seed_and_end(
        scenario_root / "sumo/InTAS_full_poly.sumocfg", int(seed["sumo"]), stop
    )
    sumo_ambassador_path = scenario_root / "sumo/sumo_config.json"
    sumo_ambassador = json.loads(sumo_ambassador_path.read_text(encoding="utf-8-sig"))
    sumo_ambassador["additionalSumoParameters"] = (
        f"--time-to-teleport 0 --seed {int(seed['sumo'])}"
    )
    write_json(sumo_ambassador_path, sumo_ambassador)
    set_omnet_seed(scenario_root / "omnetpp/omnetpp.ini", int(seed["omnet"]))
    definition.update({"communication_start": start, "communication_end": stop})
    return runtime_name, scenario_root, definition


def inspect_image(manifest: dict[str, Any]) -> None:
    image = manifest["container"]["image"]
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode:
        raise PipelineError(f"Pinned container image is unavailable: {image}\n{result.stderr.strip()}")
    if manifest["container"]["digest"] not in result.stdout:
        raise PipelineError("Installed container does not match the pinned experiment digest")


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise PipelineError("Unsupported manifest schema")
    if set(manifest["active_simulation_seeds"]) - set(manifest["simulation_seeds"]):
        raise PipelineError("An active simulation seed is not defined")
    if int(manifest["container"].get("watchdog_seconds", -1)) != 0:
        raise PipelineError("Conference runs require the watchdog to be disabled for dense InTAS traffic")
    for name, scenario in manifest["scenarios"].items():
        start, end = scenario["communication_start"], scenario["communication_end"]
        if end - start != 300:
            raise PipelineError(f"{name} does not have a 300-second window")
        if start not in (7200, 25200) or end not in (7500, 25500):
            raise PipelineError(f"{name} has an unexpected conference window")
    if manifest["attacks"] != ["constantPositionOffset", "randomPositionOffset"]:
        raise PipelineError("Conference attack list changed unexpectedly")
    if float(manifest["attack_ratio"]) != 0.2:
        raise PipelineError("Conference attack ratio must remain 0.2")
    dependency_check = subprocess.run(
        [str(ATTACK_PYTHON), "-c", "import numpy, pandas, traci"], check=False
    )
    if dependency_check.returncode:
        raise PipelineError(
            "The active Python environment lacks numpy, pandas, or traci; "
            "install Simulations-week10/requirements.txt"
        )


def validate_prepared_scenario(
    manifest: dict[str, Any], scenario_root: Path, definition: dict[str, Any], seed_id: str, jar: Path
) -> None:
    seed = manifest["simulation_seeds"][seed_id]
    config = json.loads((scenario_root / "scenario_config.json").read_text())
    expected_stop = definition["communication_end"]
    if config["simulation"]["duration"] != f"{expected_stop}s":
        raise PipelineError(f"Bad MOSAIC duration in {scenario_root}")
    if config["simulation"]["randomSeed"] != seed["mosaic"] or config["federates"]["output"]:
        raise PipelineError(f"Bad MOSAIC seed/federates in {scenario_root}")
    app = json.loads((scenario_root / "application/EtsiApplication.json").read_text())
    if app["simulationTime"] != {
        "start": f"{definition['communication_start']}s", "end": f"{expected_stop}s"
    }:
        raise PipelineError(f"Bad application window in {scenario_root}")
    if app["jsonPath"] != "/opt/mosaic/output/" or app["pseudonymInterval"] != "50s":
        raise PipelineError(f"Bad container output or pseudonym configuration in {scenario_root}")
    if sha256(jar) != sha256(scenario_root / "application/CamApp-0.0.1.jar"):
        raise PipelineError(f"Application JAR differs in {scenario_root}")
    tree = ET.parse(scenario_root / "sumo/InTAS_full_poly.sumocfg")
    end = tree.getroot().find("./time/end")
    sumo_seed = tree.getroot().find("./random_number/seed")
    if end is None or end.get("value") != str(expected_stop):
        raise PipelineError(f"Bad SUMO end time in {scenario_root}")
    if sumo_seed is None or sumo_seed.get("value") != str(seed["sumo"]):
        raise PipelineError(f"Bad SUMO seed in {scenario_root}")
    sumo_ambassador = json.loads((scenario_root / "sumo/sumo_config.json").read_text())
    expected_parameters = f"--time-to-teleport 0 --seed {int(seed['sumo'])}"
    if sumo_ambassador.get("additionalSumoParameters") != expected_parameters:
        raise PipelineError(f"Bad SUMO ambassador seed in {scenario_root}")
    ini = (scenario_root / "omnetpp/omnetpp.ini").read_text()
    if not re.search(rf"(?m)^seed-set\s*=\s*{seed['omnet']}\s*$", ini):
        raise PipelineError(f"Bad OMNeT++ seed in {scenario_root}")


def finalize_streamed_file(path: Path) -> int:
    with path.open("r", encoding="utf-8") as source:
        first = source.read(1)
    if first == "[":
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
        if not isinstance(value, list):
            raise PipelineError(f"Expected a JSON array in {path}")
        return len(value)

    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with path.open("r", encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as output:
        output.write("[\n")
        for line_number, line in enumerate(source, 1):
            record = line.strip()
            if not record:
                continue
            try:
                json.loads(record)
            except json.JSONDecodeError as error:
                raise PipelineError(f"Invalid streamed JSON in {path}:{line_number}: {error}") from error
            if count:
                output.write(",\n")
            output.write(record)
            count += 1
        output.write("\n]\n")
    temporary.replace(path)
    return count


def finalize_dataset(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kind in ("cam", "cpm", "ego"):
        directory = root / kind
        if not directory.is_dir():
            raise PipelineError(f"Missing {kind} directory in {root}")
        files = sorted(directory.glob("*.json"))
        if not files:
            raise PipelineError(f"No {kind} JSON files in {root}")
        counts[kind] = sum(finalize_streamed_file(path) for path in files)
        if counts[kind] == 0:
            raise PipelineError(f"No {kind} messages in {root}")
    return counts


def iter_records(directory: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as source:
            records = json.load(source)
        if not isinstance(records, list):
            raise PipelineError(f"Expected JSON array in {path}")
        yield from records


def normalized_sender_state(sender: dict[str, Any]) -> dict[str, Any]:
    """Normalize CAM string fields and CPM numeric fields for semantic comparison."""
    normalized: dict[str, Any] = {}
    for field in ("pos", "pos_noise"):
        value = sender.get(field)
        normalized[field] = tuple(float(part) for part in str(value).split(","))
    for field in ("spd", "spd_noise", "acl", "acl_noise", "hed", "hed_noise"):
        normalized[field] = float(sender.get(field))
    normalized["driversProfile"] = str(sender.get("driversProfile"))
    return normalized


def validate_dataset(root: Path, start: int, end: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"files": {}, "messages": {}, "min_send_time_ns": {}, "max_send_time_ns": {}}
    lower, upper = start * 1_000_000_000, end * 1_000_000_000
    for kind in ("cam", "cpm", "ego"):
        directory = root / kind
        files = list(directory.glob("*.json"))
        times: list[int] = []
        count = 0
        for record in iter_records(directory):
            count += 1
            if "sendTime" in record:
                times.append(int(record["sendTime"]))
        if not files or count == 0 or not times:
            raise PipelineError(f"Empty or invalid {kind} dataset in {root}")
        if min(times) < lower or max(times) > upper:
            raise PipelineError(f"{kind} send time falls outside {start}-{end}s in {root}")
        summary["files"][kind] = len(files)
        summary["messages"][kind] = count
        summary["min_send_time_ns"][kind] = min(times)
        summary["max_send_time_ns"][kind] = max(times)

    # Compare a bounded sample of synchronized CAM/CPM sender observations.
    cam_states: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in iter_records(root / "cam"):
        key = (str(record.get("sender_id")), int(record.get("sender_alias", 0)), int(record["sendTime"]))
        cam_states.setdefault(key, normalized_sender_state(record.get("sender", {})))
        if len(cam_states) >= 10000:
            break
    compared = 0
    for record in iter_records(root / "cpm"):
        key = (str(record.get("sender_id")), int(record.get("sender_alias", 0)), int(record["sendTime"]))
        if key in cam_states:
            if normalized_sender_state(record.get("sender", {})) != cam_states[key]:
                raise PipelineError(f"CAM/CPM sender state mismatch for {key} in {root}")
            compared += 1
            if compared >= 200:
                break
    if compared == 0:
        raise PipelineError(f"Could not find synchronized CAM/CPM observations in {root}")
    summary["synchronized_sender_samples"] = compared

    # Check a bounded set of unique transmissions for implausibly fast alias
    # changes. Receptions duplicate transmissions across receiver files.
    observations: dict[str, dict[int, int]] = {}
    unique_observations = 0
    for record in iter_records(root / "cam"):
        sender = str(record.get("sender_id"))
        if sender not in observations and len(observations) >= 500:
            continue
        send_time = int(record["sendTime"])
        sender_observations = observations.setdefault(sender, {})
        if send_time not in sender_observations:
            sender_observations[send_time] = int(record.get("sender_alias", 0))
            unique_observations += 1
        if unique_observations >= 250000:
            break
    transitions = 0
    for sender, timeline in observations.items():
        previous_alias: int | None = None
        sender_transitions = 0
        for send_time, alias in sorted(timeline.items()):
            if previous_alias is not None and alias != previous_alias:
                sender_transitions += 1
                transitions += 1
            previous_alias = alias
        if timeline:
            observed_span = max(timeline) - min(timeline)
            # Receiver files reveal changes only on successfully received CAMs.
            # Bound the transition rate, allowing one partially observed change at
            # each edge of the observation span.
            maximum_transitions = observed_span // 50_000_000_000 + 2
            if sender_transitions > maximum_transitions:
                raise PipelineError(f"Pseudonym transition rate is too high for {sender} in {root}")
    summary["pseudonym_transitions_checked"] = transitions
    return summary


def latest_mosaic_log(log_root: Path) -> Path:
    candidates = sorted(log_root.glob("log-*/MOSAIC.log"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise PipelineError(f"No MOSAIC log produced in {log_root}")
    return candidates[-1]


def run_container(
    image: str, watchdog_seconds: int, scenario_mount: Path, runtime_name: str,
    output: Path, logs: Path, transcript: Path,
) -> None:
    command = [
        "docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{scenario_mount}:/opt/mosaic/scenarios",
        "-v", f"{output}:/opt/mosaic/output",
        "-v", f"{logs}:/opt/mosaic/logs",
        "--entrypoint", "/opt/mosaic/mosaic.sh",
        image, "-s", runtime_name, "-w", str(watchdog_seconds),
    ]
    transcript.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    with transcript.open("w", encoding="utf-8", buffering=1) as capture:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            capture.write(line)
        return_code = process.wait()
    if return_code:
        raise PipelineError(f"Docker simulation exited with status {return_code}; see {transcript}")


def assert_simulation_complete(log_path: Path, stop: int) -> None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    marker = f"Simulation ended after {stop}s of {stop}s (100%)"
    if marker not in text or "Simulation finished: 101" not in text:
        raise PipelineError(f"MOSAIC did not reach the configured stop time; see {log_path}")
    critical = text.find("Stopping simulation due to a critical error")
    if critical >= 0:
        shutdown = text[critical:]
        known_shutdown = (
            critical > text.find(marker)
            and "InternalFederateException: Socket closed" in shutdown
            and "SocketException: Socket closed" in shutdown
        )
        if not known_shutdown:
            raise PipelineError(f"MOSAIC reported a critical error; see {log_path}")


def simulation_destination(seed_id: str, setting: str) -> Path:
    return CLEAN_ROOT / seed_id / setting


def promote_simulation_output(
    manifest: dict[str, Any], setting: str, seed_id: str, jar: Path,
    scenario_root: Path, definition: dict[str, Any], output: Path,
    logs: Path, mosaic_log: Path, destination: Path,
) -> Path:
    finalize_dataset(output)
    summary = validate_dataset(output, definition["communication_start"], definition["communication_end"])
    seed = manifest["simulation_seeds"][seed_id]
    final_logs = LOG_ROOT / "simulations" / seed_id / setting
    archived_mosaic_log = final_logs / mosaic_log.relative_to(logs)
    metadata = {
        "status": "complete",
        "kind": "clean-simulation",
        "setting": setting,
        "communication_window_s": [definition["communication_start"], definition["communication_end"]],
        "communication_activation": "window-and-area-only",
        "simulation_seed_id": seed_id,
        "seeds": seed,
        "git_revision": git_revision(),
        "container_image": manifest["container"]["image"],
        "container_watchdog_seconds": manifest["container"]["watchdog_seconds"],
        "application_jar_sha256": sha256(jar),
        "scenario_config_sha256": sha256(scenario_root / "scenario_config.json"),
        "sumo_config_sha256": sha256(scenario_root / "sumo/InTAS_full_poly.sumocfg"),
        "sumo_ambassador_config_sha256": sha256(scenario_root / "sumo/sumo_config.json"),
        "validation": summary,
        "mosaic_log": str(archived_mosaic_log.relative_to(HERE)),
    }
    write_json(output / "metadata.json", metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.replace(destination)
    if final_logs.exists():
        shutil.rmtree(final_logs)
    logs.replace(final_logs)
    print(f"[ok] {destination}")
    return destination


def run_simulation_job(manifest: dict[str, Any], setting: str, seed_id: str, jar: Path) -> Path:
    destination = simulation_destination(seed_id, setting)
    if destination.exists():
        print(f"[skip] completed destination already exists: {destination}")
        validate_dataset(
            destination,
            manifest["scenarios"][setting]["communication_start"],
            manifest["scenarios"][setting]["communication_end"],
        )
        return destination

    staging = WORK / "simulations" / seed_id / setting
    if staging.exists():
        try:
            runtime_name, scenario_root, definition = prepare_scenario(manifest, setting, seed_id, jar)
            validate_prepared_scenario(manifest, scenario_root, definition, seed_id, jar)
            staged_output, staged_logs = staging / "output", staging / "logs"
            staged_mosaic_log = latest_mosaic_log(staged_logs)
            assert_simulation_complete(staged_mosaic_log, definition["communication_end"])
            print(f"[recover] validating completed staging output: {staging}")
            return promote_simulation_output(
                manifest, setting, seed_id, jar, scenario_root, definition,
                staged_output, staged_logs, staged_mosaic_log, destination,
            )
        except (PipelineError, OSError, ValueError, json.JSONDecodeError) as error:
            print(f"[discard] unusable staging output {staging}: {error}")
            shutil.rmtree(staging)

    runtime_name, scenario_root, definition = prepare_scenario(manifest, setting, seed_id, jar)
    validate_prepared_scenario(manifest, scenario_root, definition, seed_id, jar)
    output, logs = staging / "output", staging / "logs"
    output.mkdir(parents=True)
    logs.mkdir(parents=True)
    transcript = LOG_ROOT / "simulations" / seed_id / f"{setting}.container.log"
    run_container(
        manifest["container"]["image"], manifest["container"]["watchdog_seconds"],
        RUNTIME / "scenarios", runtime_name, output, logs, transcript,
    )
    mosaic_log = latest_mosaic_log(logs)
    assert_simulation_complete(mosaic_log, definition["communication_end"])
    return promote_simulation_output(
        manifest, setting, seed_id, jar, scenario_root, definition,
        output, logs, mosaic_log, destination,
    )


def attack_destination(attack: str, attack_seed: int, simulation_seed: str, setting: str) -> Path:
    return ATTACK_ROOT / attack / str(attack_seed) / f"simulation-seed-{simulation_seed}" / setting


def run_attack_job(
    manifest: dict[str, Any], attack: str, attack_seed: int, simulation_seed: str, setting: str
) -> Path:
    source = simulation_destination(simulation_seed, setting)
    if not source.is_dir():
        raise PipelineError(f"Clean source dataset is missing: {source}")
    destination = attack_destination(attack, attack_seed, simulation_seed, setting)
    definition = manifest["scenarios"][setting]
    if destination.exists():
        print(f"[skip] completed destination already exists: {destination}")
        validate_dataset(destination, definition["communication_start"], definition["communication_end"])
        return destination

    staging = WORK / "attacks" / attack / str(attack_seed) / f"simulation-seed-{simulation_seed}" / setting
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    scenario = RUNTIME / "scenarios" / scenario_runtime_name(setting, simulation_seed)
    sumo_config = scenario / "sumo/InTAS_full_poly.sumocfg"
    transcript = LOG_ROOT / "attacks" / attack / str(attack_seed) / f"simulation-seed-{simulation_seed}" / f"{setting}.log"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ATTACK_PYTHON), str(ATTACK_GENERATOR), str(source), attack, str(sumo_config),
        "--seed", str(attack_seed), "--output-dir", str(staging),
    ]
    print("+", " ".join(command), flush=True)
    with transcript.open("w", encoding="utf-8") as output:
        subprocess.run(command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT, check=True)
    summary = validate_dataset(staging, definition["communication_start"], definition["communication_end"])
    source_metadata = json.loads((source / "metadata.json").read_text())
    metadata = {
        "status": "complete",
        "kind": "attacked-dataset",
        "setting": setting,
        "attack": attack,
        "attack_ratio": manifest["attack_ratio"],
        "attack_seed": attack_seed,
        "simulation_seed_id": simulation_seed,
        "simulation_seeds": source_metadata["seeds"],
        "source_dataset": str(source.relative_to(HERE)),
        "source_metadata_sha256": sha256(source / "metadata.json"),
        "git_revision": git_revision(),
        "validation": summary,
    }
    write_json(staging / "metadata.json", metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(destination)
    print(f"[ok] {destination}")
    return destination


def selected_values(requested: list[str] | None, available: Iterable[str], label: str) -> list[str]:
    values = list(available) if not requested else requested
    unknown = set(values) - set(available)
    if unknown:
        raise PipelineError(f"Unknown {label}: {', '.join(sorted(unknown))}")
    return values


def run_parallel(function: Any, jobs: list[tuple[Any, ...]], workers: int) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(function, *job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise


def validate_command(manifest: dict[str, Any], *, prepare: bool = True) -> Path:
    validate_manifest(manifest)
    inspect_image(manifest)
    jar = build_application(manifest)
    if prepare:
        for seed_id in manifest["active_simulation_seeds"]:
            for setting in manifest["scenarios"]:
                _, root, definition = prepare_scenario(manifest, setting, seed_id, jar)
                validate_prepared_scenario(manifest, root, definition, seed_id, jar)
    print(f"[ok] configuration, image, and application validated; JAR sha256={sha256(jar)}")
    return jar


def smoke_command(manifest: dict[str, Any], jar: Path) -> None:
    setting = "InTAS_urban_2AM_300sec"
    runtime_name, root, definition = prepare_scenario(manifest, setting, "1", jar, smoke=True)
    staging = WORK / "smoke"
    if staging.exists():
        shutil.rmtree(staging)
    output, logs = staging / "output", staging / "logs"
    output.mkdir(parents=True)
    logs.mkdir()
    run_container(
        manifest["container"]["image"], manifest["container"]["watchdog_seconds"],
        RUNTIME / "smoke-scenarios", runtime_name,
        output, logs, LOG_ROOT / "smoke/container.log",
    )
    assert_simulation_complete(latest_mosaic_log(logs), definition["communication_end"])
    finalize_dataset(output)
    summary = validate_dataset(output, definition["communication_start"], definition["communication_end"])
    print(f"[ok] Docker smoke test: {json.dumps(summary['messages'], sort_keys=True)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Build the app and statically validate all active scenarios")
    subparsers.add_parser("smoke", help="Run a short end-to-end Docker simulation")
    for command in ("simulate", "attack", "all"):
        item = subparsers.add_parser(command)
        item.add_argument("--settings", nargs="+")
        item.add_argument("--simulation-seeds", nargs="+")
        item.add_argument("--jobs", type=int)
        if command in ("attack", "all"):
            item.add_argument("--attacks", nargs="+")
            item.add_argument("--attack-seeds", nargs="+", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    try:
        if args.command == "validate":
            validate_command(manifest)
            return 0
        jar = validate_command(manifest, prepare=args.command != "smoke")
        if args.command == "smoke":
            smoke_command(manifest, jar)
            return 0

        settings = selected_values(args.settings, manifest["scenarios"].keys(), "setting")
        simulation_seeds = selected_values(
            args.simulation_seeds, manifest["active_simulation_seeds"], "active simulation seed"
        )
        if args.command in ("simulate", "all"):
            jobs = [(manifest, setting, seed_id, jar) for seed_id in simulation_seeds for setting in settings]
            run_parallel(
                run_simulation_job, jobs,
                args.jobs if args.jobs is not None else int(manifest["simulation_jobs"]),
            )
        if args.command in ("attack", "all"):
            attacks = selected_values(args.attacks, manifest["attacks"], "attack")
            attack_seeds = args.attack_seeds or manifest["attack_seeds"]
            unknown_attack_seeds = set(attack_seeds) - set(manifest["attack_seeds"])
            if unknown_attack_seeds:
                raise PipelineError(f"Unknown attack seeds: {sorted(unknown_attack_seeds)}")
            jobs = [
                (manifest, attack, attack_seed, simulation_seed, setting)
                for attack in attacks for attack_seed in attack_seeds
                for simulation_seed in simulation_seeds for setting in settings
            ]
            run_parallel(
                run_attack_job, jobs,
                args.jobs if args.jobs is not None else int(manifest["attack_jobs"]),
            )
        return 0
    except (PipelineError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
