#!/usr/bin/env python3
"""Validate raw CAM/CPM/ego output before attack injection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STATE_FIELDS = (
    "pos",
    "pos_noise",
    "spd",
    "spd_noise",
    "acl",
    "acl_noise",
    "hed",
    "hed_noise",
    "driversProfile",
)
CPM_SOURCE_FIELDS = (
    "sendTime",
    "sender_id",
    "sender_alias",
    "messageID",
    "just_entered_communication_zone",
    "sender",
    "perceivedObjects",
)
MAX_ERROR_EXAMPLES = 25


class DatasetValidator:
    def __init__(
        self,
        root: Path,
        expect_sensor_errors: bool,
        expect_perceived_object_errors: bool,
    ) -> None:
        self.root = root
        self.expect_sensor_errors = expect_sensor_errors
        self.expect_perceived_object_errors = expect_perceived_object_errors
        self.errors: list[str] = []
        self.counts: Counter[str] = Counter()
        self.transmissions: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
            "cam": {},
            "cpm": {},
        }
        self.ego: dict[tuple[str, str], dict[str, Any]] = {}
        self.ego_times: defaultdict[str, list[int]] = defaultdict(list)
        self.nonzero_sensor_errors: Counter[str] = Counter()

    def error(self, location: str, message: str) -> None:
        self.counts["errors"] += 1
        if len(self.errors) < MAX_ERROR_EXAMPLES:
            self.errors.append(f"{location}: {message}")

    def validate(self) -> bool:
        for stream in ("cam", "cpm"):
            self._validate_received_stream(stream)
        self._validate_ego_stream()
        self._validate_cpm_ego_consistency()
        self._validate_cpm_timing()
        self._validate_expected_sensor_errors()

        for stream in ("cam", "cpm", "ego"):
            if self.counts[f"{stream}_files"] == 0:
                self.error(stream, "stream directory has no JSON files")
            if self.counts[f"{stream}_messages"] == 0:
                self.error(stream, "stream contains no messages")
        if self.counts["perceived_objects"] == 0:
            self.error("cpm", "no CPM contains a perceived object")

        return self.counts["errors"] == 0

    def _load_file(self, path: Path) -> list[Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.error(str(path), f"cannot read JSON array: {exc}")
            return None
        if not isinstance(data, list):
            self.error(str(path), "top-level value is not an array")
            return None
        if not data:
            self.error(str(path), "JSON array is empty")
        return data

    def _validate_received_stream(self, stream: str) -> None:
        directory = self.root / stream
        files = sorted(directory.glob("*.json")) if directory.is_dir() else []
        self.counts[f"{stream}_files"] = len(files)

        for path in files:
            receiver_id = path.stem
            data = self._load_file(path)
            if data is None:
                continue
            for index, message in enumerate(data):
                location = f"{path}:{index}"
                self.counts[f"{stream}_messages"] += 1
                if not isinstance(message, dict):
                    self.error(location, "message is not an object")
                    continue
                self._validate_message_common(message, stream, location, received=True)

                sender_id = str(message.get("sender_id", ""))
                if sender_id == receiver_id:
                    self.error(location, "self-reception was written to the dataset")

                key = (sender_id, str(message.get("messageID", "")))
                canonical = self._canonical_source(message, stream)
                prior = self.transmissions[stream].get(key)
                if prior is None:
                    self.transmissions[stream][key] = canonical
                elif prior != canonical:
                    self.error(location, "sender payload differs across receivers")

    def _validate_ego_stream(self) -> None:
        directory = self.root / "ego"
        files = sorted(directory.glob("*.json")) if directory.is_dir() else []
        self.counts["ego_files"] = len(files)

        for path in files:
            data = self._load_file(path)
            if data is None:
                continue
            for index, message in enumerate(data):
                location = f"{path}:{index}"
                self.counts["ego_messages"] += 1
                if not isinstance(message, dict):
                    self.error(location, "message is not an object")
                    continue
                self._validate_message_common(message, "cpm", location, received=False)
                key = (
                    str(message.get("sender_id", "")),
                    str(message.get("messageID", "")),
                )
                canonical = self._canonical_source(message, "cpm")
                if key in self.ego:
                    self.error(location, "duplicate ego CPM identity")
                else:
                    self.ego[key] = canonical
                try:
                    self.ego_times[key[0]].append(int(message["sendTime"]))
                except (KeyError, TypeError, ValueError):
                    pass

    def _validate_message_common(
        self,
        message: dict[str, Any],
        stream: str,
        location: str,
        received: bool,
    ) -> None:
        required = [
            "type",
            "sendTime",
            "sender_id",
            "sender_alias",
            "messageID",
            "just_entered_communication_zone",
            "sender",
        ]
        if received:
            required.extend(("rcvTime", "receiver"))
        if stream == "cpm":
            required.append("perceivedObjects")

        for field in required:
            if field not in message:
                self.error(location, f"missing field {field!r}")
        if any(field not in message for field in required):
            return

        if message["type"] != stream.upper():
            self.error(location, f"type is {message['type']!r}, expected {stream.upper()!r}")
        if "attacker" in message:
            self.error(location, "raw pre-attack message already contains an attacker field")

        send_time = self._integer(message["sendTime"], location, "sendTime")
        receive_time = (
            self._integer(message["rcvTime"], location, "rcvTime") if received else None
        )
        if send_time is not None and send_time < 0:
            self.error(location, "sendTime is negative")
        if (
            send_time is not None
            and receive_time is not None
            and receive_time < send_time
        ):
            self.error(location, "rcvTime precedes sendTime")

        alias = self._integer(message["sender_alias"], location, "sender_alias")
        if alias is not None and not 1_000_000_000 <= alias < 10_000_000_000:
            self.error(location, "sender_alias is outside the 10-digit pseudonym range")

        prefix = f"{stream}_"
        if not str(message["messageID"]).startswith(prefix):
            self.error(location, f"messageID does not start with {prefix!r}")

        entered = self._integer(
            message["just_entered_communication_zone"],
            location,
            "just_entered_communication_zone",
        )
        if entered not in (0, 1):
            self.error(location, "just_entered_communication_zone is not binary")

        self._validate_state(message["sender"], f"{location}.sender", "sender")
        if received:
            self._validate_state(message["receiver"], f"{location}.receiver", "receiver")
        if stream == "cpm":
            self._validate_perceived_objects(
                message["perceivedObjects"],
                message["sender"],
                f"{location}.perceivedObjects",
            )

    def _validate_state(self, state: Any, location: str, side: str) -> None:
        if not isinstance(state, dict):
            self.error(location, "vehicle state is not an object")
            return
        for field in STATE_FIELDS:
            if field not in state:
                self.error(location, f"missing state field {field!r}")
        if any(field not in state for field in STATE_FIELDS):
            return

        self._vector(state["pos"], location, "pos")
        position_noise = self._vector(state["pos_noise"], location, "pos_noise")
        for field in ("spd", "spd_noise", "acl", "acl_noise", "hed", "hed_noise"):
            self._number(state[field], location, field)
        heading = self._number(state["hed"], location, "hed")
        if heading is not None and not 0.0 <= heading < 360.0:
            self.error(location, "heading is outside [0, 360)")

        if position_noise and any(value != 0.0 for value in position_noise):
            self.nonzero_sensor_errors[f"{side}.pos"] += 1
        for field in ("spd_noise", "acl_noise", "hed_noise"):
            value = self._number(state[field], location, field)
            if value is not None and value != 0.0:
                self.nonzero_sensor_errors[f"{side}.{field}"] += 1

    def _validate_perceived_objects(
        self,
        objects: Any,
        sender: Any,
        location: str,
    ) -> None:
        if not isinstance(objects, list):
            self.error(location, "perceivedObjects is not an array")
            return
        if not isinstance(sender, dict) or "pos" not in sender:
            return
        sender_position = self._vector(sender["pos"], location, "sender.pos")

        for index, perceived in enumerate(objects):
            object_location = f"{location}:{index}"
            self.counts["perceived_objects"] += 1
            if not isinstance(perceived, dict):
                self.error(object_location, "perceived object is not an object")
                continue
            required = (
                "object_id",
                "global_pos",
                "rel_pos",
                "spd",
                "acl",
                "hed",
                "dimensions",
            )
            if self.expect_perceived_object_errors:
                required += ("pos_noise", "spd_noise", "hed_noise")
            for field in required:
                if field not in perceived:
                    self.error(object_location, f"missing field {field!r}")
            if any(field not in perceived for field in required):
                continue

            global_position = self._vector(
                perceived["global_pos"], object_location, "global_pos"
            )
            relative_position = self._vector(
                perceived["rel_pos"], object_location, "rel_pos"
            )
            self._vector(perceived["dimensions"], object_location, "dimensions")
            self._number(perceived["spd"], object_location, "spd")
            heading = self._number(perceived["hed"], object_location, "hed")
            if heading is not None and not 0.0 <= heading < 360.0:
                self.error(object_location, "heading is outside [0, 360)")
            if perceived["acl"] is not None:
                self._number(perceived["acl"], object_location, "acl")

            if sender_position and global_position and relative_position:
                expected = tuple(
                    global_position[i] - sender_position[i] for i in range(3)
                )
                if any(
                    not math.isclose(expected[i], relative_position[i], abs_tol=1e-6)
                    for i in range(3)
                ):
                    self.error(
                        object_location,
                        "rel_pos is inconsistent with sender and global_pos",
                    )

            if self.expect_perceived_object_errors:
                position_noise = self._vector(
                    perceived["pos_noise"], object_location, "pos_noise"
                )
                speed_noise = self._number(
                    perceived["spd_noise"], object_location, "spd_noise"
                )
                heading_noise = self._number(
                    perceived["hed_noise"], object_location, "hed_noise"
                )
                if position_noise and any(value != 0.0 for value in position_noise):
                    self.nonzero_sensor_errors["object.pos"] += 1
                if speed_noise is not None and speed_noise != 0.0:
                    self.nonzero_sensor_errors["object.spd_noise"] += 1
                if heading_noise is not None and heading_noise != 0.0:
                    self.nonzero_sensor_errors["object.hed_noise"] += 1

    def _validate_cpm_ego_consistency(self) -> None:
        for key, received in self.transmissions["cpm"].items():
            ego = self.ego.get(key)
            if ego is None:
                self.error(f"cpm:{key[0]}:{key[1]}", "received CPM has no ego snapshot")
            elif ego != received:
                self.error(
                    f"cpm:{key[0]}:{key[1]}",
                    "received CPM source differs from its ego snapshot",
                )

    def _validate_cpm_timing(self) -> None:
        for sender_id, times in self.ego_times.items():
            ordered = sorted(times)
            for previous, current in zip(ordered, ordered[1:]):
                gap = current - previous
                if gap <= 0 or gap % 1_000_000_000 != 0:
                    self.error(
                        f"ego:{sender_id}",
                        f"CPM send gap {gap} ns is not a positive whole second",
                    )

    def _validate_expected_sensor_errors(self) -> None:
        if not self.expect_sensor_errors:
            return
        required = (
            "sender.pos",
            "sender.spd_noise",
            "sender.hed_noise",
            "receiver.pos",
            "receiver.spd_noise",
            "receiver.hed_noise",
        )
        if self.expect_perceived_object_errors:
            required += ("object.pos", "object.spd_noise", "object.hed_noise")
        for field in required:
            if self.nonzero_sensor_errors[field] == 0:
                self.error("sensor-errors", f"{field} is zero throughout the dataset")

    def _canonical_source(
        self, message: dict[str, Any], stream: str
    ) -> dict[str, Any]:
        fields = CPM_SOURCE_FIELDS if stream == "cpm" else CPM_SOURCE_FIELDS[:-1]
        return {field: message.get(field) for field in fields}

    def _integer(self, value: Any, location: str, field: str) -> int | None:
        if isinstance(value, bool):
            self.error(location, f"{field} is boolean, expected integer")
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            self.error(location, f"{field} is not an integer")
            return None
        try:
            if float(value) != parsed:
                raise ValueError
        except (TypeError, ValueError):
            self.error(location, f"{field} is not an integer")
            return None
        return parsed

    def _number(self, value: Any, location: str, field: str) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            self.error(location, f"{field} is not numeric")
            return None
        if not math.isfinite(parsed):
            self.error(location, f"{field} is not finite")
            return None
        return parsed

    def _vector(
        self, value: Any, location: str, field: str
    ) -> tuple[float, float, float] | None:
        if not isinstance(value, str):
            self.error(location, f"{field} is not a comma-separated string")
            return None
        parts = value.split(",")
        if len(parts) != 3:
            self.error(location, f"{field} does not contain three components")
            return None
        parsed = tuple(self._number(part, location, field) for part in parts)
        if any(component is None for component in parsed):
            return None
        return parsed  # type: ignore[return-value]

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": str(self.root),
            "valid": self.counts["errors"] == 0,
            "errors": self.counts["errors"],
            "error_examples": self.errors,
            "files": {
                stream: self.counts[f"{stream}_files"]
                for stream in ("cam", "cpm", "ego")
            },
            "messages": {
                stream: self.counts[f"{stream}_messages"]
                for stream in ("cam", "cpm", "ego")
            },
            "unique_transmissions": {
                "cam": len(self.transmissions["cam"]),
                "cpm": len(self.transmissions["cpm"]),
            },
            "perceived_objects": self.counts["perceived_objects"],
            "nonzero_sensor_errors": dict(sorted(self.nonzero_sensor_errors.items())),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject malformed or already-attacked CAM/CPM output."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Raw simulation output containing cam/, cpm/, and ego/ directories.",
    )
    parser.add_argument(
        "--expect-sensor-errors",
        action="store_true",
        help="Require nonzero sender and receiver sensor-error realizations.",
    )
    parser.add_argument(
        "--expect-perceived-object-errors",
        action="store_true",
        help="Require CPM perceived-object pos/speed/heading error fields.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validator = DatasetValidator(
        args.dataset,
        args.expect_sensor_errors,
        args.expect_perceived_object_errors,
    )
    validator.validate()
    print(json.dumps(validator.summary(), indent=2, sort_keys=True))
    return 0 if validator.counts["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
