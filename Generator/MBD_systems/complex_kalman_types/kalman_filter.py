from dataclasses import dataclass
import math

from filterpy.kalman import KalmanFilter
import numpy as np


# Kalman model follows FilterPy's standard predict/update and white-noise setup:
# https://github.com/rlabbe/filterpy#basic-use
# The four-dimensional gate uses the 95% chi-square quantile (4 DoF).
NIS_THRESHOLD_95 = 9.487729036781154
PROCESS_ACCEL_STD_MPS2 = 3.1
# Sensor-model parameters come from SensorErrorModel.java, not realized *_noise fields.
POSITION_STD_M = 5.0 / math.sqrt(3.0)
RELATIVE_SPEED_STD = 0.00016
HEADING_UNIFORM_BOUND_DEG = 20.0
MIN_VELOCITY_VARIANCE = 1e-6


def parse_position(pos_string: str) -> np.ndarray:
    parts = str(pos_string).split(",")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def velocity_from_cam(cam: dict) -> np.ndarray:
    sender = cam["sender"]
    speed = float(sender["spd"])
    heading_rad = math.radians(float(sender["hed"]))
    return np.array([speed * math.sin(heading_rad), speed * math.cos(heading_rad)], dtype=float)


def measurement_from_cam(cam: dict) -> np.ndarray:
    position = parse_position(cam["sender"]["pos"])
    velocity = velocity_from_cam(cam)
    return np.array([position[0], position[1], velocity[0], velocity[1]], dtype=float)


def measurement_covariance(cam: dict) -> np.ndarray:
    """Build R from configured sensor-error distributions, never realized noise fields."""
    speed = abs(float(cam["sender"]["spd"]))
    heading = math.radians(float(cam["sender"]["hed"]))
    speed_std = max(speed * RELATIVE_SPEED_STD, math.sqrt(MIN_VELOCITY_VARIANCE))
    heading_std = math.radians(
        (HEADING_UNIFORM_BOUND_DEG / math.sqrt(3.0)) * math.exp(-0.1 * speed)
    )

    # Propagate speed/heading uncertainty into correlated vx/vy uncertainty with
    # the Jacobian of [v sin(h), v cos(h)] with respect to [v, h].
    jacobian = np.array([
        [math.sin(heading), speed * math.cos(heading)],
        [math.cos(heading), -speed * math.sin(heading)],
    ])
    polar_covariance = np.diag([speed_std ** 2, heading_std ** 2])
    velocity_covariance = jacobian @ polar_covariance @ jacobian.T
    velocity_covariance += np.eye(2) * MIN_VELOCITY_VARIANCE

    covariance = np.zeros((4, 4), dtype=float)
    covariance[0:2, 0:2] = np.eye(2) * POSITION_STD_M ** 2
    covariance[2:4, 2:4] = velocity_covariance
    return covariance


def transition_matrix(dt: float) -> np.ndarray:
    return np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def process_covariance(dt: float, acceleration_std: float = PROCESS_ACCEL_STD_MPS2) -> np.ndarray:
    # Discrete white-acceleration process noise for state [x, y, vx, vy].
    # This is the 2-D equivalent of FilterPy's Q_discrete_white_noise(dim=2).
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    return acceleration_std ** 2 * np.array([
        [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
        [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
        [dt3 / 2.0, 0.0, dt2, 0.0],
        [0.0, dt3 / 2.0, 0.0, dt2],
    ])


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    nis: float
    nis_threshold: float
    innovation: np.ndarray
    position_error: float
    speed_error: float
    dt: float
    innovation_covariance_trace: float
    innovation_covariance_condition: float


def gate_from_prediction(predicted_state: np.ndarray, predicted_covariance: np.ndarray,
                         cam: dict, dt: float, threshold: float = NIS_THRESHOLD_95) -> GateResult:
    measurement = measurement_from_cam(cam)
    innovation = measurement - predicted_state
    innovation_covariance = predicted_covariance + measurement_covariance(cam)
    position_error = float(np.linalg.norm(innovation[0:2]))
    speed_error = float(abs(np.linalg.norm(predicted_state[2:4]) - np.linalg.norm(measurement[2:4])))

    try:
        if not np.all(np.isfinite(innovation_covariance)) or not np.all(np.isfinite(innovation)):
            raise np.linalg.LinAlgError("non-finite innovation covariance")
        # NIS/Mahalanobis gate: nu.T S^-1 nu. Solve instead of explicitly
        # inverting S for numerical stability. Reference derivation:
        # https://users.cecs.anu.edu.au/~Jonghyuk.Kim/pdf/KimPhDWeb.pdf (Sec. 4.2.7)
        solved = np.linalg.solve(innovation_covariance, innovation)
        nis = float(innovation.T @ solved)
        condition = float(np.linalg.cond(innovation_covariance))
        valid = math.isfinite(nis) and nis >= 0.0 and math.isfinite(condition)
    except np.linalg.LinAlgError:
        nis = float("inf")
        condition = float("inf")
        valid = False

    return GateResult(
        accepted=valid and nis <= threshold,
        nis=nis,
        nis_threshold=threshold,
        innovation=innovation,
        position_error=position_error,
        speed_error=speed_error,
        dt=dt,
        innovation_covariance_trace=float(np.trace(innovation_covariance)),
        innovation_covariance_condition=condition,
    )


def measurement_gate(reference_cam: dict, candidate_cam: dict,
                     threshold: float = NIS_THRESHOLD_95) -> GateResult:
    """NIS gate between two independent measurements at the same receiver time."""
    reference = measurement_from_cam(reference_cam)
    return gate_from_prediction(
        reference, measurement_covariance(reference_cam), candidate_cam, 0.0, threshold
    )


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
        kf.P = measurement_covariance(cam)
        kf.H = np.eye(4)
        kf.R = measurement_covariance(cam)
        kf.F = np.eye(4)
        kf.Q = np.zeros((4, 4))
        return cls(str(cam["sender_id"]), int(cam.get("sender_alias", 0)), kf, int(cam["rcvTime"]))

    def predicted_state_and_covariance(self, time_ns: int):
        # Candidate gating must predict both x and P without mutating the track.
        dt = max(0.0, (int(time_ns) - self.last_update_time) / 1_000_000_000.0)
        transition = transition_matrix(dt)
        process_noise = process_covariance(dt)
        predicted_state = transition @ self.filter.x
        predicted_covariance = transition @ self.filter.P @ transition.T + process_noise
        return predicted_state, predicted_covariance, dt

    def gate(self, cam: dict) -> GateResult:
        state, covariance, dt = self.predicted_state_and_covariance(int(cam["rcvTime"]))
        return gate_from_prediction(state, covariance, cam, dt)

    def predict_to(self, time_ns: int):
        dt = max(0.0, (int(time_ns) - self.last_update_time) / 1_000_000_000.0)
        if dt == 0.0:
            return
        self.filter.F = transition_matrix(dt)
        self.filter.Q = process_covariance(dt)
        self.filter.predict()
        self.last_update_time = int(time_ns)

    def update_from_cam(self, cam: dict):
        self.predict_to(int(cam["rcvTime"]))
        self.filter.R = measurement_covariance(cam)
        self.filter.update(measurement_from_cam(cam))
        self.station_alias = int(cam.get("sender_alias", self.station_alias))
