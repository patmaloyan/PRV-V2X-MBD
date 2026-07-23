import json
from pathlib import Path

import numpy as np
import pandas as pd

from Generator.MBD_systems.complex_kalman_types.kalman_filter import KalmanTrack as NisKalmanTrack
from Generator.MBD_systems.complex_kalman_types.kalman_filter import measurement_gate as nis_measurement_gate
from Generator.MBD_systems.complex_kalman_types.kalman_filter import parse_position
from Generator.MBD_systems.complex_kalman_types.kalman_filter_fixed_gate import KalmanTrack as FixedGateKalmanTrack
from Generator.MBD_systems.complex_kalman_types.kalman_filter_fixed_gate import measurement_gate as fixed_measurement_gate
from Generator.MBD_systems.complex_kalman_types.kalman_filter_fixed_gate import POSITION_THRESHOLD_M, SPEED_THRESHOLD_MPS


WIRELESS_RANGE_M = 300.0
RANGE_MARGIN_M = 50.0
EGO_LOOKBACK_NS = 2_000_000_000
CPM_SENSOR_RANGE_M = 80.0


class CamOnlyKalmanDetector:
    def __init__(self, kalman_type="nis"):
        if kalman_type == "nis":
            self.track_class = NisKalmanTrack
            self.measurement_gate = nis_measurement_gate
            self.gate_rank = lambda gate: gate.nis
        elif kalman_type == "fixed":
            self.track_class = FixedGateKalmanTrack
            self.measurement_gate = fixed_measurement_gate
            self.gate_rank = lambda gate: (
                gate.position_error / POSITION_THRESHOLD_M + gate.speed_error / SPEED_THRESHOLD_MPS
            )
        else:
            raise ValueError(f"Unknown kalman_type: {kalman_type}")
        self.tracks = []
        self.tracks_by_station_id = {}

    def process_receiver(self, cam_path: Path, ego_path: Path | None):
        cam_messages = sorted(load_json_list(cam_path), key=lambda msg: int(msg["rcvTime"]))
        ego_snapshots = sorted(load_json_list(ego_path), key=lambda msg: int(msg["sendTime"]))

        rows = []
        for cam in cam_messages:
            decision = self.process_cam(cam, ego_snapshots)
            rows.append({
                "receiver_id": cam_path.stem,
                "messageID": cam.get("messageID"),
                "sender_id": cam.get("sender_id"),
                "sender_alias": cam.get("sender_alias", 0),
                "sender_just_entered_communication_zone": sender_just_entered(cam),
                "receiver_just_entered_communication_zone": receiver_just_entered(cam),
                "sendTime": int(cam.get("sendTime", 0)),
                "rcvTime": int(cam.get("rcvTime", 0)),
                "attacker": int(cam.get("attacker", 0)),
                "prediction": 0 if decision["accepted"] else 1,
                **decision,
            })

        return pd.DataFrame(rows)

    def process_cam(self, cam: dict, ego_snapshots: list[dict]):
        # Flowchart order: known ID, pseudonym change, then new vehicle.
        known_result = self.known_station_id_check(cam)
        if known_result is not None:
            return known_result

        pseudonym_result = self.pseudonym_change_check(cam)
        if pseudonym_result is not None:
            return pseudonym_result

        return self.new_vehicle_check(cam, ego_snapshots)

    def known_station_id_check(self, cam: dict):
        track = self.tracks_by_station_id.get(str(cam["sender_id"]))
        if track is None:
            return None

        gate = track.gate(cam)
        if gate.accepted:
            track.update_from_cam(cam)
            return decision(True, "known_id_accept", track.station_id, gate)

        return decision(False, "known_id_reject", track.station_id, gate)

    def pseudonym_change_check(self, cam: dict):
        best_track = None
        best_gate = None

        # Associate an unknown station ID with the statistically closest track.
        for track in self.tracks:
            gate = track.gate(cam)
            if best_gate is None or self.gate_rank(gate) < self.gate_rank(best_gate):
                best_track = track
                best_gate = gate

        if best_track is None or not best_gate.accepted:
            return None

        old_station_id = best_track.station_id
        self.tracks_by_station_id.pop(old_station_id, None)
        best_track.station_id = str(cam["sender_id"])
        self.tracks_by_station_id[best_track.station_id] = best_track
        best_track.update_from_cam(cam)
        return decision(True, "pseudonym_accept", old_station_id, best_gate)

    def new_vehicle_check(self, cam: dict, ego_snapshots: list[dict]):
        # A new ID is accepted if it just entered, appears near range edge, or is seen by the receiver.
        if sender_just_entered(cam) == 1:
            self.add_track(cam)
            return decision(True, "new_vehicle_sender_zone_entry_accept", None)

        if receiver_just_entered(cam) == 1:
            self.add_track(cam)
            return decision(True, "new_vehicle_receiver_zone_entry_accept", None)

        if self.within_wireless_margin(cam):
            self.add_track(cam)
            return decision(True, "new_vehicle_margin_accept", None)

        ego_match = self.find_ego_sensor_match(cam, ego_snapshots)
        if ego_match is not None:
            self.add_track(cam)
            return decision(True, "new_vehicle_ego_accept", ego_match["object_id"], ego_match["gate"])

        return decision(False, "new_vehicle_reject", None)

    def add_track(self, cam: dict):
        track = self.track_class.from_cam(cam)
        self.tracks.append(track)
        self.tracks_by_station_id[track.station_id] = track

    def within_wireless_margin(self, cam: dict):
        receiver_pos = parse_position(cam["receiver"]["pos"])
        sender_pos = parse_position(cam["sender"]["pos"])
        distance = float(np.linalg.norm(receiver_pos[0:2] - sender_pos[0:2]))
        return WIRELESS_RANGE_M - RANGE_MARGIN_M <= distance <= WIRELESS_RANGE_M + RANGE_MARGIN_M

    def find_ego_sensor_match(self, cam: dict, ego_snapshots: list[dict]):
        snapshot = latest_ego_snapshot(ego_snapshots, int(cam["rcvTime"]))
        if snapshot is None:
            return None

        best_match = None

        for obj in snapshot.get("perceivedObjects", []):
            measurement = perceived_object_as_cam(obj, int(cam["rcvTime"]))
            if measurement is None:
                continue
            gate = self.measurement_gate(measurement, cam)
            if best_match is None or self.gate_rank(gate) < self.gate_rank(best_match["gate"]):
                best_match = {"object_id": obj.get("object_id"), "gate": gate}

        if best_match and best_match["gate"].accepted:
            return best_match
        return None


class CamCpmKalmanDetector(CamOnlyKalmanDetector):
    """Tsukada detector using one time-ordered CAM/CPM stream per receiver."""

    def __init__(self, kalman_type="nis"):
        super().__init__(kalman_type)
        self.anonymous_track_counter = 0

    def process_receiver(self, cam_path: Path | None, cpm_path: Path | None, ego_path: Path | None):
        # CAMs and CPMs received by this vehicle share one chronological track history.
        messages = combined_message_frame(cam_path, cpm_path)
        ego_snapshots = sorted(load_json_list(ego_path), key=lambda msg: int(msg["sendTime"]))
        receiver_id = (cam_path or cpm_path).stem

        rows = []
        for message in messages.to_dict(orient="records"):
            message_type = str(message["message_type"]).upper()
            source_decision = self.process_cam(message, ego_snapshots)
            object_counts = empty_object_counts()
            if message_type == "CPM":
                perceived_objects = message.get("perceivedObjects", [])
                # The orange flowchart branch runs only after the CPM source is accepted.
                if source_decision["accepted"]:
                    object_counts = self.process_perceived_objects(message)
                elif isinstance(perceived_objects, list):
                    # Count skipped objects so debug totals still reconcile with raw CPM data.
                    object_counts["cpm_objects_observed"] = len(perceived_objects)
                    object_counts["cpm_objects_source_rejected"] = len(perceived_objects)
                else:
                    object_counts["cpm_objects_malformed"] = 1

            rows.append({
                "receiver_id": receiver_id,
                "message_type": message_type,
                "messageID": message.get("messageID"),
                "sender_id": message.get("sender_id"),
                "sender_alias": message.get("sender_alias", 0),
                "sender_just_entered_communication_zone": sender_just_entered(message),
                "receiver_just_entered_communication_zone": receiver_just_entered(message),
                "sendTime": int(message.get("sendTime", 0)),
                "rcvTime": int(message.get("rcvTime", 0)),
                "attacker": int(message.get("attacker", 0)),
                "prediction": 0 if source_decision["accepted"] else 1,
                **source_decision,
                **object_counts,
            })

        return pd.DataFrame(rows)

    def process_perceived_objects(self, cpm: dict):
        counts = empty_object_counts()
        objects = cpm.get("perceivedObjects", [])
        if not isinstance(objects, list):
            counts["cpm_objects_malformed"] = 1
            return counts

        counts["cpm_objects_observed"] = len(objects)
        for perceived_object in objects:
            object_id = perceived_object.get("object_id") if isinstance(perceived_object, dict) else None
            measurement = perceived_object_as_cam(perceived_object, int(cpm["rcvTime"]))
            if measurement is None:
                counts["cpm_objects_malformed"] += 1
                counts["cpm_object_events"].append({"object_id": object_id, "action": "malformed"})
                continue

            best_track, gate = self.closest_track(measurement)
            # object_id is simulation ground truth; association uses only Kalman deviation.
            if best_track is not None and gate.accepted:
                # CPM object data is indirect: it may confirm a track, but must not update it.
                counts["cpm_objects_matched"] += 1
                counts["cpm_object_events"].append({
                    "object_id": object_id, "action": "matched", **gate_debug(gate)
                })
                continue

            if not perceived_object_within_sensor_range(perceived_object):
                counts["cpm_objects_out_of_range"] += 1
                event = {"object_id": object_id, "action": "out_of_range"}
                if gate is not None:
                    event.update(gate_debug(gate))
                counts["cpm_object_events"].append(event)
                continue

            self.add_anonymous_object_track(measurement)
            counts["cpm_objects_initialized"] += 1
            event = {"object_id": object_id, "action": "initialized"}
            if gate is not None:
                event.update(gate_debug(gate))
            counts["cpm_object_events"].append(event)

        return counts

    def closest_track(self, measurement: dict):
        best_track = None
        best_gate = None
        # NIS normalizes each residual by that track's predicted uncertainty.
        for track in self.tracks:
            gate = track.gate(measurement)
            if best_gate is None or self.gate_rank(gate) < self.gate_rank(best_gate):
                best_track = track
                best_gate = gate
        return best_track, best_gate

    def add_anonymous_object_track(self, measurement: dict):
        self.anonymous_track_counter += 1
        # Do not expose the CPM ground-truth object_id as an observable station identity.
        measurement["sender_id"] = f"cpm_object_{self.anonymous_track_counter}"
        track = self.track_class.from_cam(measurement)
        self.tracks.append(track)

def process_kalman_folder(input_folder: Path, kalman_type="nis"):
    cam_dir = input_folder / "cam"
    ego_dir = input_folder / "ego"
    if not cam_dir.is_dir():
        raise ValueError(f"Kalman detector expects CAM folder at {cam_dir}")

    receiver_results = []
    for cam_path in sorted(cam_dir.glob("*.json")):
        ego_path = ego_dir / cam_path.name if ego_dir.is_dir() else None
        detector = CamOnlyKalmanDetector(kalman_type)
        receiver_results.append(detector.process_receiver(cam_path, ego_path))

    if not receiver_results:
        raise ValueError(f"No CAM JSON files found in {cam_dir}")

    results = pd.concat(receiver_results, ignore_index=True)
    metrics = calculate_metrics(results)
    metrics["wireless_range_m"] = WIRELESS_RANGE_M
    metrics["range_margin_m"] = RANGE_MARGIN_M
    metrics["total_messages"] = int(len(results))
    metrics["kalman_type"] = kalman_type
    return metrics, results


def process_cam_cpm_kalman_folder(input_folder: Path, kalman_type="nis"):
    cam_dir = input_folder / "cam"
    cpm_dir = input_folder / "cpm"
    ego_dir = input_folder / "ego"
    if not cam_dir.is_dir() or not cpm_dir.is_dir():
        raise ValueError(f"Type 3 expects CAM and CPM folders at {cam_dir} and {cpm_dir}")

    cam_paths = {path.stem: path for path in cam_dir.glob("*.json")}
    cpm_paths = {path.stem: path for path in cpm_dir.glob("*.json")}
    receiver_ids = sorted(set(cam_paths) | set(cpm_paths))
    if not receiver_ids:
        raise ValueError(f"No CAM or CPM JSON files found in {input_folder}")

    receiver_results = []
    for receiver_id in receiver_ids:
        ego_path = ego_dir / f"{receiver_id}.json" if ego_dir.is_dir() else None
        # Kalman state is local to a receiver and must never leak between vehicles.
        detector = CamCpmKalmanDetector(kalman_type)
        receiver_results.append(detector.process_receiver(
            cam_paths.get(receiver_id), cpm_paths.get(receiver_id), ego_path
        ))

    results = pd.concat(receiver_results, ignore_index=True)
    metrics = calculate_metrics(results)
    metrics["wireless_range_m"] = WIRELESS_RANGE_M
    metrics["range_margin_m"] = RANGE_MARGIN_M
    metrics["cpm_sensor_range_m"] = CPM_SENSOR_RANGE_M
    metrics["total_messages"] = int(len(results))
    metrics["cam_messages"] = int((results["message_type"] == "CAM").sum())
    metrics["cpm_messages"] = int((results["message_type"] == "CPM").sum())
    metrics["kalman_type"] = kalman_type
    return metrics, results


def combined_message_frame(cam_path: Path | None, cpm_path: Path | None):
    records = []
    for message_type, path in (("CAM", cam_path), ("CPM", cpm_path)):
        for message in load_json_list(path):
            row = dict(message)
            row["message_type"] = message_type
            records.append(row)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["rcvTime"] = frame["rcvTime"].astype("int64")
    frame["sendTime"] = frame["sendTime"].astype("int64")
    # Stable secondary keys make equal-reception-time processing reproducible.
    return frame.sort_values(
        ["rcvTime", "sendTime", "messageID"], kind="mergesort", ignore_index=True
    )


def perceived_object_as_cam(perceived_object: dict, rcv_time: int):
    if not isinstance(perceived_object, dict):
        return None
    required = ("global_pos", "spd", "hed")
    if any(key not in perceived_object for key in required):
        return None
    try:
        parse_position(perceived_object["global_pos"])
        float(perceived_object["spd"])
        float(perceived_object["hed"])
    except (TypeError, ValueError, IndexError):
        return None
    return {
        "sender_id": "",
        "sender_alias": 0,
        "rcvTime": int(rcv_time),
        "sender": {
            "pos": perceived_object["global_pos"],
            "spd": perceived_object["spd"],
            "hed": perceived_object["hed"],
        },
    }


def perceived_object_within_sensor_range(perceived_object: dict):
    try:
        relative_position = parse_position(perceived_object["rel_pos"])
    except (KeyError, TypeError, ValueError, IndexError):
        return False
    return float(np.linalg.norm(relative_position[0:2])) <= CPM_SENSOR_RANGE_M


def empty_object_counts():
    return {
        "cpm_objects_observed": 0,
        "cpm_objects_matched": 0,
        "cpm_objects_initialized": 0,
        "cpm_objects_out_of_range": 0,
        "cpm_objects_source_rejected": 0,
        "cpm_objects_malformed": 0,
        "cpm_object_events": [],
    }


def calculate_metrics(results: pd.DataFrame):
    tp = ((results["attacker"] == 1) & (results["prediction"] == 1)).sum()
    tn = ((results["attacker"] == 0) & (results["prediction"] == 0)).sum()
    fp = ((results["attacker"] == 0) & (results["prediction"] == 1)).sum()
    fn = ((results["attacker"] == 1) & (results["prediction"] == 0)).sum()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def load_json_list(path: Path | None):
    if path is None or not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def latest_ego_snapshot(ego_snapshots: list[dict], rcv_time: int):
    latest = None
    for snapshot in ego_snapshots:
        send_time = int(snapshot.get("sendTime", 0))
        if send_time > rcv_time:
            break
        if rcv_time - send_time <= EGO_LOOKBACK_NS:
            latest = snapshot
    return latest


def sender_just_entered(cam: dict):
    return int(cam.get(
        "just_entered_communication_zone",
        cam.get("just_entered_communication_zone_cpm", cam.get("just_entered_communication_zone_cam", 0)),
    ))


def receiver_just_entered(cam: dict):
    return int(cam.get("receiver", {}).get("just_entered_communication_zone", 0))


def gate_debug(gate):
    return {
        "nis": gate.nis,
        "nis_threshold": gate.nis_threshold,
        "dt": gate.dt,
        "innovation_x": float(gate.innovation[0]),
        "innovation_y": float(gate.innovation[1]),
        "innovation_vx": float(gate.innovation[2]),
        "innovation_vy": float(gate.innovation[3]),
        "pos_error": gate.position_error,
        "speed_error": gate.speed_error,
        "innovation_covariance_trace": gate.innovation_covariance_trace,
        "innovation_covariance_condition": gate.innovation_covariance_condition,
    }


def decision(accepted: bool, reason: str, matched_id, gate=None):
    result = {
        "accepted": accepted,
        "reason": reason,
        "matched_id": matched_id,
    }
    if gate is None:
        result.update({
            "nis": None, "nis_threshold": None, "dt": None,
            "innovation_x": None, "innovation_y": None,
            "innovation_vx": None, "innovation_vy": None,
            "pos_error": None, "speed_error": None,
            "innovation_covariance_trace": None,
            "innovation_covariance_condition": None,
        })
    else:
        result.update(gate_debug(gate))
    return result
