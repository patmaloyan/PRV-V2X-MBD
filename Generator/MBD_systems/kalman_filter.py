from dataclasses import dataclass
import math

from filterpy.kalman import KalmanFilter
import numpy as np


# State and measurement vector: [x, y, vx, vy].
# A new track has absolute GNSS uncertainty from the persistent ±5 m bias.
INITIAL_STATE_COVARIANCE = np.diag([
    25.0 / 3.0,
    25.0 / 3.0,
    0.25,
    0.25,
])
# Once a track exists, the persistent bias is part of its coordinate offset.
# Update noise therefore models the changing error between reports. These
# variances come from benign noCheat2 successive-error measurements.
SENSOR_MEASUREMENT_COVARIANCE = np.diag([
    0.01,
    0.01,
    0.0036,
    0.0036,
])
# CPM objects and the target track are expressed through two different
# vehicles' GNSS frames, so their independent persistent biases both apply.
CPM_ASSOCIATION_COVARIANCE = np.diag([
    50.0 / 3.0,
    50.0 / 3.0,
    0.25,
    0.25,
])
# Continuous white-acceleration noise intensity, calibrated on benign noCheat2
# innovations so the 99% NIS threshold rejects about 1% of established tracks.
PROCESS_NOISE_INTENSITY = 3.6


@dataclass(frozen=True)
class KalmanDeviation:
    position_error: float
    speed_error: float
    nis: float


def parse_position(pos_string: str) -> np.ndarray:
    parts = str(pos_string).split(",")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def velocity_from_cam(cam: dict) -> np.ndarray:
    sender = cam["sender"]
    speed = float(sender["spd"])
    heading_rad = math.radians(float(sender["hed"]))
    return np.array([speed * math.sin(heading_rad), speed * math.cos(heading_rad)], dtype=float)


def acceleration_from_cam(cam: dict) -> np.ndarray:
    sender = cam["sender"]
    acceleration = float(sender.get("acl", 0.0) or 0.0)
    heading_rad = math.radians(float(sender["hed"]))
    return np.array([
        acceleration * math.sin(heading_rad),
        acceleration * math.cos(heading_rad),
    ], dtype=float)


def measurement_from_cam(cam: dict) -> np.ndarray:
    pos = parse_position(cam["sender"]["pos"])
    velocity = velocity_from_cam(cam)
    return np.array([pos[0], pos[1], velocity[0], velocity[1]], dtype=float)


def motion_matrices(dt: float):
    transition = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    control = np.array([
        [0.5 * dt**2, 0.0],
        [0.0, 0.5 * dt**2],
        [dt, 0.0],
        [0.0, dt],
    ])
    # Continuous white-acceleration noise allows benign jerk and turning instead
    # of forcing position and velocity process errors to be perfectly coupled.
    process_noise = PROCESS_NOISE_INTENSITY**2 * np.array([
        [dt**3 / 3.0, 0.0, dt**2 / 2.0, 0.0],
        [0.0, dt**3 / 3.0, 0.0, dt**2 / 2.0],
        [dt**2 / 2.0, 0.0, dt, 0.0],
        [0.0, dt**2 / 2.0, 0.0, dt],
    ])
    return transition, control, process_noise


@dataclass
class KalmanTrack:
    station_alias: int
    filter: KalmanFilter
    last_acceleration: np.ndarray
    last_update_time: int
    last_accepted_time: int

    @classmethod
    def from_cam(cls, cam: dict, initial_covariance: np.ndarray):
        # Step 1: Initialize a new track from the first CAM measurement.
        pos = parse_position(cam["sender"]["pos"])
        velocity = velocity_from_cam(cam)
        acceleration = acceleration_from_cam(cam)
        kf = KalmanFilter(dim_x=4, dim_z=4)
        kf.x = np.array([pos[0], pos[1], velocity[0], velocity[1]], dtype=float)
        kf.P = initial_covariance.copy()  # Starting uncertainty for a new vehicle track.
        kf.H = np.eye(4)
        time_ns = int(cam["rcvTime"])
        return cls(
            station_alias=int(cam.get("sender_alias", 0)),
            filter=kf,
            last_acceleration=acceleration,
            last_update_time=time_ns,
            last_accepted_time=time_ns,
        )

    def predict_candidate(self, time_ns: int):
        dt = max(0.0, (int(time_ns) - self.last_update_time) / 1_000_000_000.0)
        if dt == 0:
            return self.filter.x.copy(), self.filter.P.copy()

        transition, control, process_noise = motion_matrices(dt)
        predicted_state = (
            transition @ self.filter.x + control @ self.last_acceleration
        )
        predicted_covariance = (
            transition @ self.filter.P @ transition.T + process_noise
        )
        return predicted_state, predicted_covariance

    def predict_to(self, time_ns: int):
        # Step 2: Predict position and velocity at the new receive time.
        dt = max(0.0, (int(time_ns) - self.last_update_time) / 1_000_000_000.0)
        if dt == 0:
            return

        self.filter.F, self.filter.B, self.filter.Q = motion_matrices(dt)
        self.filter.predict(u=self.last_acceleration)
        self.last_update_time = int(time_ns)

    def deviation_against_cam(self, cam: dict, measurement_noise: np.ndarray):
        # Step 3: Score the innovation without changing the accepted track.
        predicted_state, predicted_covariance = self.predict_candidate(
            int(cam["rcvTime"])
        )
        measurement = measurement_from_cam(cam)
        innovation = measurement - predicted_state
        innovation_covariance = predicted_covariance + measurement_noise
        nis = float(
            innovation @ np.linalg.solve(innovation_covariance, innovation)
        )
        return KalmanDeviation(
            position_error=float(np.linalg.norm(innovation[0:2])),
            speed_error=float(np.linalg.norm(innovation[2:4])),
            nis=nis,
        )

    def update_from_cam(self, cam: dict, measurement_noise: np.ndarray):
        # Step 4: Correct the predicted track with the accepted CAM measurement.
        self.predict_to(int(cam["rcvTime"]))
        self.last_accepted_time = int(cam["rcvTime"])
        measurement = measurement_from_cam(cam)
        self.filter.R = measurement_noise
        self.filter.update(measurement)
        self.last_acceleration = acceleration_from_cam(cam)
        self.station_alias = int(cam.get("sender_alias", self.station_alias))

    def reset_from_cam(self, cam: dict, initial_covariance: np.ndarray):
        fresh = self.from_cam(cam, initial_covariance)
        self.filter = fresh.filter
        self.last_acceleration = fresh.last_acceleration
        self.last_update_time = fresh.last_update_time
        self.last_accepted_time = fresh.last_accepted_time
        self.station_alias = fresh.station_alias
