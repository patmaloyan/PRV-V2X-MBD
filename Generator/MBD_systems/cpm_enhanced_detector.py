"""Type 4: type 3 plus reciprocal CPM perception checks."""

from pathlib import Path

from kalman_detector import (
    CamCpmKalmanDetector,
    decision,
    process_cam_cpm_kalman_folder,
)


EDGE_GRACE_NS = 2_000_000_000
EDGE_TTL_NS = 4_000_000_000


def valid_alias(value):
    try:
        alias = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return alias if alias != 0 else None


class CpmEnhancedDetector(CamCpmKalmanDetector):
    def __init__(self):
        super().__init__()
        self.edges = {}
        self.last_unreciprocated_targets = []

    def prune_edges(self, now):
        # Keep only observations from the last four seconds.
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
            # Allow two seconds for the reverse observation to arrive.
            if now - timestamps[0] >= EDGE_GRACE_NS
            and source not in self.edges.get(target, {})
        )

    def pre_source_decision(self, message):
        now = int(message["rcvTime"])
        self.prune_edges(now)
        self.last_unreciprocated_targets = self.unreciprocated_targets(
            message.get("sender_alias"), now
        )
        if self.last_unreciprocated_targets:
            # Reject before the normal flow so this message cannot update Kalman state.
            return decision(False, "reciprocity_reject", None, None, None)
        return None

    def on_perceived_object_match(self, cpm, matched_track):
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
        }


def process_cpm_enhanced_folder(input_folder: Path):
    return process_cam_cpm_kalman_folder(input_folder, CpmEnhancedDetector)
