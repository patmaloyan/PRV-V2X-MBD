import json
from pathlib import Path

import numpy as np
import pandas as pd

from kalman_filter import KalmanTrack, parse_position


POSITION_THRESHOLD_M = 7.5
SPEED_THRESHOLD_MPS = 5.0
WIRELESS_RANGE_M = 300.0
RANGE_MARGIN_M = 50.0
EGO_LOOKBACK_NS = 2_000_000_000


class CamOnlyKalmanDetector:
    def __init__(self):
        self.tracks = []
        self.tracks_by_station_id = {}
        self.initial_covariance = np.diag([25.0, 25.0, 9.0, 9.0])  # Initial x/y/vx/vy uncertainty.
        self.measurement_noise = np.diag([9.0, 9.0, 4.0, 4.0])  # CAM x/y/vx/vy measurement noise.

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

        pos_error, speed_error = track.errors_against_cam(cam)
        if errors_within_threshold(pos_error, speed_error):
            track.update_from_cam(cam, self.measurement_noise)
            return decision(True, "known_id_accept", pos_error, speed_error, track.station_id)

        return decision(False, "known_id_reject", pos_error, speed_error, track.station_id)

    def pseudonym_change_check(self, cam: dict):
        best_track = None
        best_pos_error = None
        best_speed_error = None
        best_score = float("inf")

        for track in self.tracks:
            pos_error, speed_error = track.errors_against_cam(cam)
            score = (pos_error / POSITION_THRESHOLD_M) + (speed_error / SPEED_THRESHOLD_MPS)
            if score < best_score:
                best_track = track
                best_pos_error = pos_error
                best_speed_error = speed_error
                best_score = score

        if best_track is None or not errors_within_threshold(best_pos_error, best_speed_error):
            return None

        old_station_id = best_track.station_id
        self.tracks_by_station_id.pop(old_station_id, None)
        best_track.station_id = str(cam["sender_id"])
        self.tracks_by_station_id[best_track.station_id] = best_track
        best_track.update_from_cam(cam, self.measurement_noise)
        return decision(True, "pseudonym_accept", best_pos_error, best_speed_error, old_station_id)

    def new_vehicle_check(self, cam: dict, ego_snapshots: list[dict]):
        # A new ID is accepted if it just entered, appears near range edge, or is seen by the receiver.
        if sender_just_entered(cam) == 1:
            self.add_track(cam)
            return decision(True, "new_vehicle_sender_zone_entry_accept", None, None, None)

        if receiver_just_entered(cam) == 1:
            self.add_track(cam)
            return decision(True, "new_vehicle_receiver_zone_entry_accept", None, None, None)

        if self.within_wireless_margin(cam):
            self.add_track(cam)
            return decision(True, "new_vehicle_margin_accept", None, None, None)

        ego_match = self.find_ego_sensor_match(cam, ego_snapshots)
        if ego_match is not None:
            self.add_track(cam)
            return decision(
                True,
                "new_vehicle_ego_accept",
                ego_match["pos_error"],
                ego_match["speed_error"],
                ego_match["object_id"],
            )

        return decision(False, "new_vehicle_reject", None, None, None)

    def add_track(self, cam: dict):
        track = KalmanTrack.from_cam(cam, self.initial_covariance)
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

        cam_pos = parse_position(cam["sender"]["pos"])
        cam_speed = float(cam["sender"]["spd"])
        best_match = None
        best_score = float("inf")

        for obj in snapshot.get("perceivedObjects", []):
            if "global_pos" not in obj or "spd" not in obj:
                continue

            obj_pos = parse_position(obj["global_pos"])
            pos_error = float(np.linalg.norm(cam_pos[0:2] - obj_pos[0:2]))
            speed_error = abs(cam_speed - float(obj["spd"]))
            score = (pos_error / POSITION_THRESHOLD_M) + (speed_error / SPEED_THRESHOLD_MPS)
            if score < best_score:
                best_score = score
                best_match = {"object_id": obj.get("object_id"), "pos_error": pos_error, "speed_error": speed_error}

        if best_match and errors_within_threshold(best_match["pos_error"], best_match["speed_error"]):
            return best_match
        return None


def process_kalman_folder(input_folder: Path):
    cam_dir = input_folder / "cam"
    ego_dir = input_folder / "ego"
    if not cam_dir.is_dir():
        raise ValueError(f"Kalman detector expects CAM folder at {cam_dir}")

    receiver_results = []
    for cam_path in sorted(cam_dir.glob("*.json")):
        ego_path = ego_dir / cam_path.name if ego_dir.is_dir() else None
        detector = CamOnlyKalmanDetector()
        receiver_results.append(detector.process_receiver(cam_path, ego_path))

    if not receiver_results:
        raise ValueError(f"No CAM JSON files found in {cam_dir}")

    results = pd.concat(receiver_results, ignore_index=True)
    metrics = calculate_metrics(results)
    metrics["wireless_range_m"] = WIRELESS_RANGE_M
    metrics["range_margin_m"] = RANGE_MARGIN_M
    metrics["total_messages"] = int(len(results))
    return metrics, results


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


def errors_within_threshold(pos_error, speed_error):
    return pos_error <= POSITION_THRESHOLD_M and speed_error <= SPEED_THRESHOLD_MPS


def sender_just_entered(cam: dict):
    return int(cam.get("just_entered_communication_zone", 0))


def receiver_just_entered(cam: dict):
    return int(cam.get("receiver", {}).get("just_entered_communication_zone", 0))


def decision(accepted: bool, reason: str, pos_error, speed_error, matched_id):
    return {
        "accepted": accepted,
        "reason": reason,
        "pos_error": pos_error,
        "speed_error": speed_error,
        "matched_id": matched_id,
    }
