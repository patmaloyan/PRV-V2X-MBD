"""Type 6: weighted reciprocal CPM checks in one-second intervals."""

from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import pandas as pd

from kalman_detector import (
    CPM_SENSOR_RANGE_M,
    CamCpmKalmanDetector,
    combined_message_frame,
    decision,
    empty_object_counts,
    load_json_list,
    parse_position,
    process_cam_cpm_kalman_folder,
    receiver_just_entered,
    sender_just_entered,
)


INTERVAL_NS = 1_000_000_000
SCORE_HISTORY_LENGTH = 3
RECIPROCITY_BOTH_DIRECTIONS_COEFFICIENT = 2.0
RECIPROCITY_INBOUND_ONLY_COEFFICIENT = 1.0
RECIPROCITY_OUTBOUND_ONLY_COEFFICIENT = -2.0
TRUST_ALPHA = 1.0 / 3.0
MAX_PAIR_SCORE_MAGNITUDE = 2.0
RECIPROCITY_NIS_THRESHOLD = 18.47


@dataclass
class EdgeEvidence:
    nis: float
    distance: float
    confidence: float
    opportunity: float
    weight: float


@dataclass
class VehicleTrust:
    accepted: bool = True
    scores: deque = field(
        default_factory=lambda: deque(maxlen=SCORE_HISTORY_LENGTH)
    )


@dataclass
class MaintainedVehicleTrust:
    accepted: bool = True
    score: float = 0.0


def nis_confidence(nis):
    return 1.0 - float(nis) / RECIPROCITY_NIS_THRESHOLD


def distance_opportunity(distance):
    return max(0.0, min(1.0, 1.0 - float(distance) / CPM_SENSOR_RANGE_M))


def edge_evidence(nis, distance):
    confidence = nis_confidence(nis)
    opportunity = distance_opportunity(distance)
    return EdgeEvidence(
        nis=float(nis),
        distance=float(distance),
        confidence=confidence,
        opportunity=opportunity,
        weight=confidence * opportunity,
    )


def pair_score(inbound, outbound):
    """Score evidence about A; inbound is B->A and outbound is A->B."""
    if inbound is not None and outbound is not None:
        return RECIPROCITY_BOTH_DIRECTIONS_COEFFICIENT * math.sqrt(
            inbound.weight * outbound.weight
        )
    if inbound is not None:
        return RECIPROCITY_INBOUND_ONLY_COEFFICIENT * inbound.weight
    if outbound is not None:
        return (
            RECIPROCITY_OUTBOUND_ONLY_COEFFICIENT
            * outbound.opportunity
            * outbound.weight
        )
    return 0.0


def maintained_trust_pair_score(inbound, outbound):
    """Type 7/20 score; the outbound-only penalty uses w_AB directly."""
    if inbound is not None and outbound is not None:
        return RECIPROCITY_BOTH_DIRECTIONS_COEFFICIENT * math.sqrt(
            inbound.weight * outbound.weight
        )
    if inbound is not None:
        return RECIPROCITY_INBOUND_ONLY_COEFFICIENT * inbound.weight
    if outbound is not None:
        return RECIPROCITY_OUTBOUND_ONLY_COEFFICIENT * outbound.weight
    return 0.0


class WeightedReciprocityDetector(CamCpmKalmanDetector):
    def __init__(self, catch_params=None, catch_enabled=True):
        super().__init__(catch_params, catch_enabled)
        self.current_bucket = None
        self.bucket_edges = {}
        self.track_ids = {}
        self.next_track_id = 1
        self.trust = {}
        self.last_closed_intervals = []
        self.current_source_track_id = None

    def perceived_object_matches(self, measurement):
        best_track, deviation = self.closest_track(measurement)
        if (
            best_track is not None
            and deviation.nis <= RECIPROCITY_NIS_THRESHOLD
        ):
            return [(best_track, deviation)]
        return []

    def track_id(self, track, create_trust=False):
        key = id(track)
        if key not in self.track_ids:
            self.track_ids[key] = self.next_track_id
            self.next_track_id += 1
        track_id = self.track_ids[key]
        if create_trust:
            self.trust.setdefault(track_id, VehicleTrust())
        return track_id

    def advance_to(self, receive_time):
        bucket = int(receive_time) // INTERVAL_NS
        self.last_closed_intervals = []
        if self.current_bucket is None:
            self.current_bucket = bucket
            return
        while self.current_bucket < bucket:
            self.last_closed_intervals.append(self.close_current_bucket())
            self.current_bucket += 1
            self.bucket_edges = {}

    def close_current_bucket(self):
        # Freeze trust for the whole interval so iteration order cannot change scores.
        accepted_snapshot = {
            track_id: state.accepted for track_id, state in self.trust.items()
        }
        scores = {}
        for subject_id, state in self.trust.items():
            score = 0.0
            for counterpart_id, counterpart_accepted in accepted_snapshot.items():
                if counterpart_id == subject_id or not counterpart_accepted:
                    continue
                inbound = self.bucket_edges.get((counterpart_id, subject_id))
                outbound = self.bucket_edges.get((subject_id, counterpart_id))
                score += pair_score(inbound, outbound)
            scores[subject_id] = score

        transitions = []
        for track_id, score in scores.items():
            state = self.trust[track_id]
            was_accepted = state.accepted
            state.scores.append(score)
            state.accepted = self.history_is_accepted(state)
            if state.accepted != was_accepted:
                transitions.append({
                    "track_id": track_id,
                    "state": "accepted" if state.accepted else "quarantined",
                })

        return {
            "bucket_id": self.current_bucket,
            "scores": scores,
            "transitions": transitions,
        }

    def history_is_accepted(self, state):
        return self.rolling_score(state) >= 0.0

    def rolling_score(self, state):
        if not state.scores:
            return 0.0
        return sum(state.scores) / len(state.scores)

    def process_receiver(
        self, cam_path: Path | None, cpm_path: Path | None,
        ego_path: Path | None,
    ):
        messages = self.catch_messages(
            combined_message_frame(cam_path, cpm_path)
        )
        ego_snapshots = sorted(
            load_json_list(ego_path), key=lambda msg: int(msg["sendTime"])
        )
        receiver_id = (cam_path or cpm_path).stem
        rows = []

        for message in messages.to_dict(orient="records"):
            message_type = str(message["message_type"]).upper()
            catch_prediction = int(message["catch_prediction"])
            source_state = None
            self.current_source_track_id = None
            if catch_prediction:
                base_decision = None
                source_decision = self.catch_rejection()
            else:
                self.advance_to(message["rcvTime"])
                base_decision = self.process_cam(
                    message, ego_snapshots, commit=False
                )
                source_track = self.tracks_by_station_alias.get(
                    int(message.get("sender_alias", 0))
                )
                if (
                    source_track is None
                    and base_decision["reason"] == "pseudonym_accept"
                ):
                    source_track = self.tracks_by_station_alias.get(
                        int(base_decision["matched_id"])
                    )
                if source_track is not None:
                    self.current_source_track_id = self.track_ids.get(
                        id(source_track)
                    )
                    source_state = self.trust.get(self.current_source_track_id)

                externally_accepted = base_decision["accepted"] and (
                    source_state is None or source_state.accepted
                )
                source_decision = dict(base_decision)
                if externally_accepted:
                    source_decision = self.process_cam(
                        message, ego_snapshots, commit=True
                    )
                    source_track = self.tracks_by_station_alias.get(
                        int(message.get("sender_alias", 0))
                    )
                    if source_track is not None:
                        self.current_source_track_id = self.track_id(
                            source_track, create_trust=True
                        )
                        source_state = self.trust[self.current_source_track_id]
                elif base_decision["accepted"]:
                    source_decision.update(decision(
                        False, "weighted_reciprocity_quarantine",
                        base_decision.get("pos_error"),
                        base_decision.get("speed_error"),
                        base_decision.get("matched_id"), base_decision.get("nis"),
                    ))

            object_counts = empty_object_counts()
            if message_type == "CPM":
                if source_decision["accepted"]:
                    object_counts = self.process_perceived_objects(message)
                else:
                    objects = message.get("perceivedObjects", [])
                    if isinstance(objects, list):
                        object_counts["cpm_objects_observed"] = len(objects)
                        object_counts["cpm_objects_source_rejected"] = len(objects)
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
                **self.catch_debug(message, base_decision or source_decision),
                **object_counts,
                "reciprocity_bucket": self.current_bucket,
                "reciprocity_track_id": self.current_source_track_id,
                "reciprocity_state": (
                    "accepted" if source_state is None or source_state.accepted
                    else "quarantined"
                ),
                "reciprocity_score_history": (
                    self.score_history(source_state)
                    if source_state is not None else []
                ),
                "reciprocity_rolling_score": (
                    self.decision_score(source_state)
                    if source_state is not None else 0.0
                ),
                "closed_reciprocity_intervals": self.last_closed_intervals,
            })

        if self.current_bucket is not None:
            self.last_closed_intervals = [self.close_current_bucket()]
        return pd.DataFrame(rows)

    def score_history(self, state):
        return list(state.scores)

    def decision_score(self, state):
        return self.rolling_score(state)

    def on_perceived_object_match(
        self, cpm, matched_track, deviation=None, perceived_object=None
    ):
        if (
            self.current_source_track_id is None
            or perceived_object is None
            or deviation is None
        ):
            return {"edge_added": False}
        target_id = self.track_id(matched_track)
        if target_id == self.current_source_track_id:
            return {"edge_added": False}
        try:
            distance = float(np.linalg.norm(
                parse_position(perceived_object["rel_pos"])[0:2]
            ))
        except (KeyError, TypeError, ValueError, IndexError):
            return {"edge_added": False}
        evidence = edge_evidence(deviation.nis, distance)
        self.bucket_edges[(self.current_source_track_id, target_id)] = evidence
        return {
            "edge_added": True,
            "edge_weight": evidence.weight,
            "edge_confidence": evidence.confidence,
            "edge_opportunity": evidence.opportunity,
        }


def process_weighted_reciprocity_folder(
    input_folder: Path, catch_params, catch_enabled=True,
):
    metrics, results = process_cam_cpm_kalman_folder(
        input_folder, catch_params, WeightedReciprocityDetector, catch_enabled
    )
    metrics["reciprocity_nis_threshold"] = RECIPROCITY_NIS_THRESHOLD
    return metrics, results


class MaintainedTrustReciprocityDetector(WeightedReciprocityDetector):
    """Type 7: EWMA trust over normalized one-second reciprocity evidence."""

    def track_id(self, track, create_trust=False):
        key = id(track)
        if key not in self.track_ids:
            self.track_ids[key] = self.next_track_id
            self.next_track_id += 1
        track_id = self.track_ids[key]
        if create_trust:
            self.trust.setdefault(track_id, MaintainedVehicleTrust())
        return track_id

    def close_current_bucket(self):
        accepted_snapshot = {
            track_id: state.accepted for track_id, state in self.trust.items()
        }
        raw_scores = {}
        normalized_scores = {}
        evidence_counts = {}

        for subject_id in self.trust:
            raw_score = 0.0
            evidence_count = 0
            for counterpart_id, counterpart_accepted in accepted_snapshot.items():
                if counterpart_id == subject_id or not counterpart_accepted:
                    continue
                inbound = self.bucket_edges.get((counterpart_id, subject_id))
                outbound = self.bucket_edges.get((subject_id, counterpart_id))
                if inbound is None and outbound is None:
                    continue
                raw_score += self.pair_score(inbound, outbound)
                evidence_count += 1

            raw_scores[subject_id] = raw_score
            evidence_counts[subject_id] = evidence_count
            normalized_scores[subject_id] = (
                raw_score / (MAX_PAIR_SCORE_MAGNITUDE * evidence_count)
                if evidence_count else None
            )

        transitions = []
        trust_scores = {}
        for track_id, normalized_score in normalized_scores.items():
            state = self.trust[track_id]
            was_accepted = state.accepted
            if normalized_score is not None:
                state.score += TRUST_ALPHA * (normalized_score - state.score)
                state.accepted = state.score >= 0.0
            trust_scores[track_id] = state.score
            if state.accepted != was_accepted:
                transitions.append({
                    "track_id": track_id,
                    "state": "accepted" if state.accepted else "quarantined",
                })

        return {
            "bucket_id": self.current_bucket,
            "scores": raw_scores,
            "normalized_scores": normalized_scores,
            "evidence_counts": evidence_counts,
            "trust_scores": trust_scores,
            "transitions": transitions,
        }

    def score_history(self, state):
        return []

    def decision_score(self, state):
        return state.score

    def pair_score(self, inbound, outbound):
        return maintained_trust_pair_score(inbound, outbound)


def process_maintained_trust_reciprocity_folder(
    input_folder: Path, catch_params, catch_enabled=True,
):
    metrics, results = process_cam_cpm_kalman_folder(
        input_folder, catch_params, MaintainedTrustReciprocityDetector,
        catch_enabled,
    )
    metrics["reciprocity_nis_threshold"] = RECIPROCITY_NIS_THRESHOLD
    return metrics, results


class NoAnonymousMaintainedTrustDetector(MaintainedTrustReciprocityDetector):
    """Type 20: maintained trust without CPM-initialized anonymous tracks."""

    def add_anonymous_object_track(self, measurement):
        return False


def process_no_anonymous_maintained_trust_folder(
    input_folder: Path, catch_params, catch_enabled=True,
):
    metrics, results = process_cam_cpm_kalman_folder(
        input_folder, catch_params, NoAnonymousMaintainedTrustDetector,
        catch_enabled,
    )
    metrics["reciprocity_nis_threshold"] = RECIPROCITY_NIS_THRESHOLD
    return metrics, results
