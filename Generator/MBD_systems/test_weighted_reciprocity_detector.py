import unittest

from weighted_reciprocity_detector import (
    CPM_SENSOR_RANGE_M,
    NIS_THRESHOLD,
    EdgeEvidence,
    MaintainedTrustReciprocityDetector,
    MaintainedVehicleTrust,
    NoAnonymousMaintainedTrustDetector,
    VehicleTrust,
    WeightedReciprocityDetector,
    distance_opportunity,
    edge_evidence,
    maintained_trust_pair_score,
    nis_confidence,
    pair_score,
)


class WeightedReciprocityMathTests(unittest.TestCase):
    def test_nis_confidence_uses_gate_domain_without_clamping(self):
        self.assertEqual(nis_confidence(0.0), 1.0)
        self.assertEqual(nis_confidence(NIS_THRESHOLD), 0.5)

    def test_distance_opportunity_is_clamped(self):
        self.assertEqual(distance_opportunity(-1.0), 1.0)
        self.assertEqual(distance_opportunity(0.0), 1.0)
        self.assertEqual(distance_opportunity(CPM_SENSOR_RANGE_M), 0.0)
        self.assertEqual(distance_opportunity(CPM_SENSOR_RANGE_M + 1.0), 0.0)

    def test_pair_score_cases(self):
        inbound = EdgeEvidence(0.0, 0.0, 1.0, 1.0, 0.81)
        outbound = EdgeEvidence(0.0, 0.0, 1.0, 0.5, 0.25)
        self.assertAlmostEqual(pair_score(inbound, outbound), 0.9)
        self.assertEqual(pair_score(inbound, None), 0.81)
        self.assertEqual(pair_score(None, outbound), -0.25)
        self.assertEqual(pair_score(None, None), 0.0)

    def test_maintained_trust_failure_uses_weight_without_extra_opportunity(self):
        outbound = EdgeEvidence(0.0, 0.0, 1.0, 0.5, 0.25)

        self.assertEqual(pair_score(None, outbound), -0.25)
        self.assertEqual(maintained_trust_pair_score(None, outbound), -0.5)

    def test_edge_weight_combines_distance_and_confidence(self):
        evidence = edge_evidence(NIS_THRESHOLD, CPM_SENSOR_RANGE_M / 2.0)
        self.assertEqual(evidence.confidence, 0.5)
        self.assertEqual(evidence.opportunity, 0.5)
        self.assertEqual(evidence.weight, 0.25)


class WeightedReciprocityStateTests(unittest.TestCase):
    def setUp(self):
        self.detector = WeightedReciprocityDetector()
        self.detector.current_bucket = 10
        self.detector.trust = {1: VehicleTrust(), 2: VehicleTrust()}

    def test_quarantined_counterpart_cannot_affect_other_vehicle(self):
        self.detector.trust[1].accepted = False
        self.detector.trust[1].scores.extend([-1.0, 0.0, 0.0])
        self.detector.bucket_edges[(1, 2)] = edge_evidence(0.0, 10.0)

        result = self.detector.close_current_bucket()

        self.assertLess(result["scores"][1], 0.0)
        self.assertEqual(result["scores"][2], 0.0)


class RollingAverageStateTests(unittest.TestCase):
    def test_uses_available_scores_from_start(self):
        detector = WeightedReciprocityDetector()
        state = VehicleTrust()

        state.scores.append(0.0)
        self.assertTrue(detector.history_is_accepted(state))

        state.scores.clear()
        state.scores.append(-1.0)
        self.assertFalse(detector.history_is_accepted(state))

        state.scores.append(3.0)
        self.assertTrue(detector.history_is_accepted(state))
        self.assertEqual(detector.rolling_score(state), 1.0)

    def test_reentry_uses_three_score_average(self):
        detector = WeightedReciprocityDetector()
        state = VehicleTrust(accepted=False)
        state.scores.extend([-2.0, 1.0, 0.0])
        self.assertFalse(detector.history_is_accepted(state))

        state.scores.append(0.0)
        self.assertTrue(detector.history_is_accepted(state))


class MaintainedTrustStateTests(unittest.TestCase):
    def setUp(self):
        self.detector = MaintainedTrustReciprocityDetector()
        self.detector.current_bucket = 10
        self.detector.trust = {
            1: MaintainedVehicleTrust(),
            2: MaintainedVehicleTrust(),
        }

    def test_updates_trust_with_normalized_interval_score(self):
        self.detector.bucket_edges[(1, 2)] = edge_evidence(0.0, 0.0)

        result = self.detector.close_current_bucket()

        self.assertEqual(result["evidence_counts"][1], 1)
        self.assertEqual(result["normalized_scores"][1], -1.0)
        self.assertAlmostEqual(self.detector.trust[1].score, -1.0 / 3.0)
        self.assertFalse(self.detector.trust[1].accepted)

    def test_no_evidence_does_not_change_trust_or_state(self):
        self.detector.trust[1] = MaintainedVehicleTrust(
            accepted=False, score=-0.25
        )

        result = self.detector.close_current_bucket()

        self.assertIsNone(result["normalized_scores"][1])
        self.assertEqual(self.detector.trust[1].score, -0.25)
        self.assertFalse(self.detector.trust[1].accepted)

    def test_positive_evidence_allows_reentry_at_zero(self):
        self.detector.trust[1] = MaintainedVehicleTrust(
            accepted=False, score=-0.25
        )
        self.detector.bucket_edges[(2, 1)] = edge_evidence(0.0, 0.0)

        self.detector.close_current_bucket()

        self.assertEqual(self.detector.trust[1].score, 0.0)
        self.assertTrue(self.detector.trust[1].accepted)


class NoAnonymousTrackTests(unittest.TestCase):
    def test_unmatched_cpm_object_does_not_create_track(self):
        detector = NoAnonymousMaintainedTrustDetector()

        self.assertFalse(detector.add_anonymous_object_track({}))
        self.assertEqual(detector.tracks, [])


if __name__ == "__main__":
    unittest.main()
