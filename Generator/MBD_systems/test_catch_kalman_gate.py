import json
import tempfile
import unittest
from pathlib import Path

from data_structures import Parameters
from kalman_detector import CamCpmKalmanDetector, CamOnlyKalmanDetector, decision


def message(message_type="CAM", sender_x=0.0):
    return {
        "type": message_type,
        "messageID": f"{message_type.lower()}_1",
        "rcvTime": 1,
        "sendTime": 1,
        "sender_id": "veh_1",
        "sender_alias": 10,
        "attacker": 0,
        "receiver": {"pos": "0,0,0"},
        "sender": {"pos": f"{sender_x},0,0"},
        "perceivedObjects": [{}],
    }


class StubCamDetector(CamOnlyKalmanDetector):
    def __init__(self, catch_prediction):
        super().__init__(Parameters(MAX_PLAUSIBLE_RANGE=100.0))
        self.catch_prediction = catch_prediction
        self.kalman_calls = 0

    def catch_messages(self, messages):
        messages = messages.copy()
        messages["catch_prediction"] = self.catch_prediction
        messages["catch_failed_checks"] = [
            ["range_plausibility"] if self.catch_prediction else []
            for _ in range(len(messages))
        ]
        return messages

    def process_cam(self, cam, ego_snapshots):
        self.kalman_calls += 1
        return decision(True, "stub_accept", None, None, None)


class StubCpmDetector(CamCpmKalmanDetector):
    def __init__(self):
        super().__init__(Parameters())
        self.object_calls = 0

    def catch_messages(self, messages):
        messages = messages.copy()
        messages["catch_prediction"] = 1
        messages["catch_failed_checks"] = [["range_plausibility"]]
        return messages

    def process_perceived_objects(self, cpm):
        self.object_calls += 1
        return super().process_perceived_objects(cpm)


class CatchKalmanGateTests(unittest.TestCase):
    def run_cam(self, detector):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "veh_1.json"
            path.write_text(json.dumps([message()]), encoding="utf-8")
            return detector.process_receiver(path, None)

    def test_catch_rejection_skips_kalman(self):
        detector = StubCamDetector(1)
        result = self.run_cam(detector).iloc[0]

        self.assertEqual(detector.kalman_calls, 0)
        self.assertTrue(result["kalman_skipped"])
        self.assertIsNone(result["kalman_prediction"])
        self.assertEqual(result["reason"], "catch_reject")

    def test_catch_pass_runs_existing_kalman_path(self):
        detector = StubCamDetector(0)
        result = self.run_cam(detector).iloc[0]

        self.assertEqual(detector.kalman_calls, 1)
        self.assertFalse(result["kalman_skipped"])
        self.assertEqual(result["kalman_prediction"], 0)
        self.assertEqual(result["reason"], "stub_accept")

    def test_rejected_cpm_skips_perceived_objects(self):
        detector = StubCpmDetector()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "veh_1.json"
            path.write_text(json.dumps([message("CPM")]), encoding="utf-8")
            result = detector.process_receiver(None, path, None).iloc[0]

        self.assertEqual(detector.object_calls, 0)
        self.assertEqual(result["cpm_objects_source_rejected"], 1)

    def test_wireless_margin_uses_catch_range(self):
        detector = CamOnlyKalmanDetector(
            Parameters(MAX_PLAUSIBLE_RANGE=100.0)
        )
        self.assertTrue(detector.within_wireless_margin(message(sender_x=60)))
        self.assertFalse(detector.within_wireless_margin(message(sender_x=151)))


if __name__ == "__main__":
    unittest.main()
