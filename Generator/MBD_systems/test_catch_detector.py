import unittest
import json
import tempfile
from pathlib import Path

import pandas as pd

from catch_profiles import load_catch_profile
from data_processing import calculate_metrics, load_catch_messages, perform_catch_checks


class StubChecks:
    def __init__(self, range_factor=1.0, intersection_factor=1.0, position_factor=1.0):
        self.params = load_catch_profile("urban-low")
        self.range_factor = range_factor
        self.intersection_factor = intersection_factor
        self.position_factor = position_factor
        self.intersection_calls = 0

    def range_plausibility_check(self, *args):
        return self.range_factor

    def position_plausibility_check(self, *args):
        return self.position_factor

    def speed_plausibility_check(self, *args):
        return 1.0

    def position_consistency_check(self, *args):
        return 1.0

    def speed_consistency_check(self, *args):
        return 1.0

    def position_speed_consistency_check(self, *args):
        return 1.0

    def position_heading_consistency_check(self, *args):
        return 1.0

    def intersection_check(self, *args):
        self.intersection_calls += 1
        return self.intersection_factor


def message(sender_id, alias, seconds, position=0.0):
    time_ns = int(seconds * 1_000_000_000)
    return {
        "rcvTime": time_ns,
        "sendTime": time_ns,
        "sender_id": sender_id,
        "sender_alias": alias,
        "messageID": f"cam_{sender_id}_{seconds}",
        "attacker": 0,
        "prediction": 0,
        "receiver_pos_lat": 100.0,
        "receiver_pos_lon": 0.0,
        "receiver_pos_alt": 0.0,
        "receiver_pos_lat_noise": 1.0,
        "receiver_pos_lon_noise": 1.0,
        "receiver_pos_alt_noise": 0.0,
        "receiver_spd": 0.0,
        "receiver_spd_noise": 0.1,
        "receiver_acl": 0.0,
        "receiver_acl_noise": 0.1,
        "receiver_hed": 0.0,
        "receiver_hed_noise": 1.0,
        "receiver_driversProfile": "NORMAL",
        "sender_pos_lat": position,
        "sender_pos_lon": 0.0,
        "sender_pos_alt": 0.0,
        "sender_pos_lat_noise": 1.0,
        "sender_pos_lon_noise": 1.0,
        "sender_pos_alt_noise": 0.0,
        "sender_spd": 1.0,
        "sender_spd_noise": 0.1,
        "sender_acl": 0.0,
        "sender_acl_noise": 0.1,
        "sender_hed": 0.0,
        "sender_hed_noise": 1.0,
        "sender_distance_to_road_edge": 1.0,
        "sender_driversProfile": "NORMAL",
    }


class CatchProfileTests(unittest.TestCase):
    def test_urban_low_profile(self):
        params = load_catch_profile("urban-low")
        self.assertEqual(params.MAX_PLAUSIBLE_RANGE, 336.568)
        self.assertEqual(params.MAX_DELTA_INTERSECTION, 4.697)


class CatchIdentityTests(unittest.TestCase):
    def test_no_pos_check_disables_only_road_edge_check(self):
        checks = StubChecks(position_factor=0.0)
        checks.params.POSITION_PLAUSIBILITY_ENABLED = False
        result = perform_catch_checks(
            pd.DataFrame([message("veh_1", 10, 0)]), checks, use_alias=True
        )

        self.assertEqual(result.iloc[0]["check_position_plausibility_check"], 1.0)
        self.assertEqual(result.iloc[0]["check_range_plausibility"], 1.0)
        self.assertEqual(result.iloc[0]["prediction"], 0)

    def test_type_zero_keeps_history_across_alias_change(self):
        rows = [message("veh_1", 10, 0), message("veh_1", 20, 1)]
        result = perform_catch_checks(pd.DataFrame(rows), StubChecks())

        self.assertIn("check_position_consistency_check", result.columns)
        self.assertNotIn(
            "position_consistency_check", result.iloc[1]["catch_skipped_checks"]
        )

    def test_alias_grace_keeps_stateless_checks_active(self):
        checks = StubChecks(range_factor=0.0)
        result = perform_catch_checks(
            pd.DataFrame([message("veh_1", 10, 0)]), checks, use_alias=True
        )

        self.assertTrue(result.iloc[0]["catch_alias_grace"])
        self.assertEqual(result.iloc[0]["prediction"], 1)
        self.assertIn("intersection_check", result.iloc[0]["catch_skipped_checks"])

    def test_temporal_checks_start_on_second_alias_message(self):
        rows = [message("veh_1", 10, 0), message("veh_1", 10, 1)]
        result = perform_catch_checks(
            pd.DataFrame(rows), StubChecks(), use_alias=True
        )

        self.assertNotIn(
            "position_consistency_check", result.iloc[1]["catch_skipped_checks"]
        )
        self.assertTrue(result.iloc[1]["catch_alias_grace"])

    def test_intersection_starts_after_grace(self):
        checks = StubChecks(intersection_factor=0.0)
        rows = [
            message("veh_1", 10, 0),
            message("veh_2", 20, 4.8, position=20.0),
            message("veh_1", 10, 5.0),
        ]
        result = perform_catch_checks(
            pd.DataFrame(rows), checks, use_alias=True
        )

        self.assertFalse(result.iloc[2]["catch_alias_grace"])
        self.assertEqual(checks.intersection_calls, 1)
        self.assertEqual(result.iloc[2]["prediction"], 1)


class CatchInputTests(unittest.TestCase):
    def test_cam_and_cpm_sender_messages_share_one_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            cam_path = Path(directory) / "cam.json"
            cpm_path = Path(directory) / "cpm.json"
            cam_path.write_text(json.dumps([message("veh_1", 10, 0.5)]))
            cpm_path.write_text(json.dumps([
                message("veh_1", 10, 0),
                message("veh_1", 10, 1),
            ]))

            frame = load_catch_messages(cam_path, cpm_path)
            result = perform_catch_checks(frame, StubChecks(), use_alias=True)
            metrics = calculate_metrics(result)

        self.assertEqual(result["message_type"].tolist(), ["CPM", "CAM", "CPM"])
        self.assertEqual(metrics["cam_messages"], 1)
        self.assertEqual(metrics["cpm_messages"], 2)
        self.assertIn(
            "position_consistency_check", result.iloc[1]["catch_skipped_checks"]
        )
        self.assertNotIn(
            "position_consistency_check", result.iloc[2]["catch_skipped_checks"]
        )


if __name__ == "__main__":
    unittest.main()
