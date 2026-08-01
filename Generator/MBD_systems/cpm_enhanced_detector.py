"""Type 4: type 3 plus reciprocal CPM perception checks."""

from pathlib import Path

from kalman_detector import (
    CamCpmKalmanDetector,
    decision,
    nis_prediction_is_fresh,
    nis_within_threshold,
    process_cam_cpm_kalman_folder,
)


EDGE_GRACE_NS = 2_000_000_000
EDGE_TTL_NS = 6_000_000_000


def valid_alias(value):
    try:
        alias = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return alias if alias != 0 else None


class CpmEnhancedDetector(CamCpmKalmanDetector):
    def __init__(self, required_unreciprocated_edges=1):
        super().__init__()
        self.required_unreciprocated_edges = required_unreciprocated_edges
        self.edges = {}
        self.last_unreciprocated_targets = []

    def perceived_object_matches(self, measurement):
        matches = []
        for track in self.tracks:
            if not nis_prediction_is_fresh(track, int(measurement["rcvTime"])):
                continue
            deviation = track.deviation_against_cam(
                measurement, self.measurement_noise
            )
            if nis_within_threshold(deviation.nis):
                matches.append((track, deviation))
        return matches

    def prune_edges(self, now):
        # Keep only observations inside the configured edge lifetime.
        for source in list(self.edges):
            for target in list(self.edges[source]):
                timestamps = [
                    timestamp for timestamp in self.edges[source][target]
                    if now - timestamp <= EDGE_TTL_NS
                ]
                if timestamps:
                    self.edges[source][target] = timestamps
                else:
                    del self.edges[source][target]
            if not self.edges[source]:
                del self.edges[source]

    def add_edge(self, source, target, timestamp):
        source = valid_alias(source)
        target = valid_alias(target)
        if source is None or target is None or source == target:
            return False
        self.edges.setdefault(source, {}).setdefault(target, []).append(int(timestamp))
        return True

    def unreciprocated_targets(self, source, now):
        source = valid_alias(source)
        if source is None:
            return []
        return sorted(
            target for target, timestamps in self.edges.get(source, {}).items()
            # Allow grace seconds for the reverse observation to arrive.
            if now - timestamps[0] >= EDGE_GRACE_NS
            and source not in self.edges.get(target, {})
        )

    def pre_source_decision(self, message):
        now = int(message["rcvTime"])
        self.prune_edges(now)
        self.last_unreciprocated_targets = self.unreciprocated_targets(
            message.get("sender_alias"), now
        )
        if len(self.last_unreciprocated_targets) >= self.required_unreciprocated_edges:
            # Reject before the normal flow so this message cannot update Kalman state.
            return decision(False, "reciprocity_reject", None, None, None)
        return None

    def on_perceived_object_match(
        self, cpm, matched_track, deviation=None, perceived_object=None
    ):
        if self.tracks_by_station_alias.get(matched_track.station_alias) is not matched_track:
            return {"edge_added": False}

        # Graph identities are observable aliases, never simulation-only object_id values.
        edge_added = self.add_edge(
            cpm.get("sender_alias"), matched_track.station_alias, cpm["rcvTime"]
        )
        return {"edge_added": edge_added}

    def edge_count(self):
        return sum(len(targets) for targets in self.edges.values())

    def message_debug(self):
        return {
            "edge_count": self.edge_count(),
            "unreciprocated_targets": self.last_unreciprocated_targets,
            "required_unreciprocated_edges": self.required_unreciprocated_edges,
        }


def process_cpm_enhanced_folder(input_folder: Path, required_unreciprocated_edges=1):
    return process_cam_cpm_kalman_folder(
        input_folder,
        lambda: CpmEnhancedDetector(required_unreciprocated_edges),
    )
