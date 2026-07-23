"""Legacy constant-velocity Kalman filter with fixed position/speed gates."""

from dataclasses import dataclass

from filterpy.kalman import KalmanFilter
import numpy as np

from Generator.MBD_systems.complex_kalman_types.kalman_filter import (
    GateResult,
    measurement_from_cam,
    transition_matrix,
)


POSITION_THRESHOLD_M = 20.0
SPEED_THRESHOLD_MPS = 10.0
INITIAL_COVARIANCE = np.diag([25.0, 25.0, 9.0, 9.0])
MEASUREMENT_NOISE = np.diag([9.0, 9.0, 4.0, 4.0])


def fixed_gate(predicted_state: np.ndarray, cam: dict, dt: float) -> GateResult:
    measurement = measurement_from_cam(cam)
    innovation = measurement - predicted_state
    position_error = float(np.linalg.norm(innovation[0:2]))
    speed_error = float(abs(np.linalg.norm(predicted_state[2:4]) - np.linalg.norm(measurement[2:4])))
    accepted = position_error <= POSITION_THRESHOLD_M and speed_error <= SPEED_THRESHOLD_MPS
    return GateResult(
        accepted=accepted,
        nis=float("nan"),
        nis_threshold=float("nan"),
        innovation=innovation,
        position_error=position_error,
        speed_error=speed_error,
        dt=dt,
        innovation_covariance_trace=float("nan"),
        innovation_covariance_condition=float("nan"),
    )


def measurement_gate(reference_cam: dict, candidate_cam: dict) -> GateResult:
    return fixed_gate(measurement_from_cam(reference_cam), candidate_cam, 0.0)


@dataclass
class KalmanTrack:
    station_id: str
    station_alias: int
    filter: KalmanFilter
    last_update_time: int

    @classmethod
    def from_cam(cls, cam: dict):
        kf = KalmanFilter(dim_x=4, dim_z=4)
        kf.x = measurement_from_cam(cam)
        kf.P = INITIAL_COVARIANCE.copy()
        kf.H = np.eye(4)
        kf.R = MEASUREMENT_NOISE.copy()
        return cls(str(cam["sender_id"]), int(cam.get("sender_alias", 0)), kf, int(cam["rcvTime"]))

    def predicted_state(self, time_ns: int):
        dt = max(0.0, (int(time_ns) - self.last_update_time) / 1_000_000_000.0)
        return transition_matrix(dt) @ self.filter.x, dt

    def gate(self, cam: dict) -> GateResult:
        state, dt = self.predicted_state(int(cam["rcvTime"]))
        return fixed_gate(state, cam, dt)

    def predict_to(self, time_ns: int):
        dt = max(0.0, (int(time_ns) - self.last_update_time) / 1_000_000_000.0)
        if dt == 0.0:
            return
        self.filter.F = transition_matrix(dt)
        # Preserve the original fixed-gate filter's simple process-noise behavior.
        self.filter.Q = np.eye(4) * max(dt, 1e-3)
        self.filter.predict()
        self.last_update_time = int(time_ns)

    def update_from_cam(self, cam: dict):
        self.predict_to(int(cam["rcvTime"]))
        self.filter.R = MEASUREMENT_NOISE.copy()
        self.filter.update(measurement_from_cam(cam))
        self.station_alias = int(cam.get("sender_alias", self.station_alias))
