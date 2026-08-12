from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run_week10", HERE / "run_week10.py")
assert SPEC and SPEC.loader
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


class Week10PipelineTests(unittest.TestCase):
    def test_manifest_has_exact_conference_matrix(self) -> None:
        manifest = PIPELINE.load_manifest()
        PIPELINE.validate_manifest(manifest)
        self.assertEqual(manifest["active_simulation_seeds"], ["1"])
        self.assertEqual(manifest["attack_seeds"], [1, 2, 3])
        self.assertEqual(len(manifest["scenarios"]), 4)

    def test_stream_finalizer_is_atomic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "veh_1.json"
            path.write_text('{"sendTime":1}\n{"sendTime":2}\n', encoding="utf-8")
            self.assertEqual(PIPELINE.finalize_streamed_file(path), 2)
            self.assertEqual(json.loads(path.read_text()), [{"sendTime": 1}, {"sendTime": 2}])
            self.assertEqual(PIPELINE.finalize_streamed_file(path), 2)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_sender_state_normalization_handles_cam_strings(self) -> None:
        cam = {
            "pos": "1.0,2.0,0.0", "pos_noise": "0.1,0.2,0.0",
            "spd": "3.0", "spd_noise": "0.3", "acl": "4.0",
            "acl_noise": "0.4", "hed": "5.0", "hed_noise": "0.5",
            "driversProfile": "NORMAL",
        }
        cpm = dict(cam)
        for field in ("spd", "spd_noise", "acl", "acl_noise", "hed", "hed_noise"):
            cpm[field] = float(cpm[field])
        self.assertEqual(
            PIPELINE.normalized_sender_state(cam),
            PIPELINE.normalized_sender_state(cpm),
        )

    def test_week10_application_has_no_unseeded_randomness(self) -> None:
        java = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (HERE / "CamApp/src/main/java").rglob("*.java")
        )
        self.assertNotIn("Math.random", java)
        self.assertNotIn("new Random()", java)
        self.assertIn("new SensorErrorModel(getRandom())", java)

    def test_attack_seed_and_output_directory_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "clean"
            source.mkdir()
            records = []
            for index in range(20):
                station = {
                    "pos": f"{index}.0,{index + 1}.0,0.0",
                    "pos_noise": "0.0,0.0,0.0",
                    "spd": "10.0",
                    "spd_noise": "0.0",
                    "acl": "0.0",
                    "acl_noise": "0.0",
                    "hed": "90.0",
                    "hed_noise": "0.0",
                    "driversProfile": "NORMAL",
                }
                receiver = dict(station)
                receiver["just_entered_communication_zone"] = 0
                records.append({
                    "type": "CAM", "rcvTime": index + 1, "sendTime": index + 1,
                    "sender_id": f"veh_{index}", "sender_alias": 1000 + index,
                    "messageID": f"cam_{index}", "just_entered_communication_zone": 0,
                    "receiver": receiver, "sender": station,
                })
            (source / "receiver.json").write_text(json.dumps(records), encoding="utf-8")

            outputs = [root / "seed7-a", root / "seed7-b", root / "seed8"]
            seeds = [7, 7, 8]
            for output, seed in zip(outputs, seeds):
                subprocess.run([
                    str(PIPELINE.ATTACK_PYTHON), str(PIPELINE.ATTACK_GENERATOR),
                    str(source), "constantPositionOffset", str(root / "unused.sumocfg"),
                    "--seed", str(seed), "--output-dir", str(output),
                ], check=True, stdout=subprocess.DEVNULL)
            first = (outputs[0] / "receiver.json").read_bytes()
            self.assertEqual(first, (outputs[1] / "receiver.json").read_bytes())
            self.assertNotEqual(first, (outputs[2] / "receiver.json").read_bytes())
            self.assertEqual(json.loads((source / "receiver.json").read_text()), records)


if __name__ == "__main__":
    unittest.main()
