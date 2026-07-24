from dataclasses import dataclass
import math

from filterpy.kalman import KalmanFilter
import numpy as np


# State and measurement vector: [x, y, vx, vy].
# Fixed approximation of SensorErrorModel.java:
# position is uniform ±5 m; velocity variance includes its heading error.
SENSOR_MEASUREMENT_COVARIANCE = np.diag([
    25.0 / 3.0,
    25.0 / 3.0,
    0.25,
    0.25,
])


def parse_position(pos_string: str) -> np.ndarray:
    parts = str(pos_string).split(",")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def velocity_from_cam(cam: dict) -> np.ndarray:
    sender = cam["sender"]
    speed = float(sender["spd"])
    heading_rad = math.radians(float(sender["hed"]))
    return np.array([speed * math.sin(heading_rad), speed * math.cos(heading_rad)], dtype=float)


@dataclass
class KalmanTrack:
    station_id: str
    station_alias: int
    filter: KalmanFilter
    last_update_time: int
    last_seen_time: int

    @classmethod
    def from_cam(cls, cam: dict, initial_covariance: np.ndarray):
        # Step 1: Initialize a new track from the first CAM measurement.
        pos = parse_position(cam["sender"]["pos"])
        velocity = velocity_from_cam(cam)
        kf = KalmanFilter(dim_x=4, dim_z=4)
        kf.x = np.array([pos[0], pos[1], velocity[0], velocity[1]], dtype=float)
        kf.P = initial_covariance.copy()  # Starting uncertainty for a new vehicle track.
        kf.H = np.eye(4)
        time_ns = int(cam["rcvTime"])
        return cls(str(cam["sender_id"]), int(cam.get("sender_alias", 0)), kf, time_ns, time_ns)

    def predict_state(self, time_ns: int):
        state = self.filter.x.copy()
        dt = max(0.0, (int(time_ns) - self.last_update_time) / 1_000_000_000.0)
        state[0] += state[2] * dt
        state[1] += state[3] * dt
        return state

    def predict_to(self, time_ns: int):
        # Step 2: Predict position and velocity at the new receive time.
        dt = max(0.0, (int(time_ns) - self.last_update_time) / 1_000_000_000.0)
        if dt == 0:
            return

        self.filter.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.filter.Q = np.eye(4) * max(dt, 1e-3)  # Small process noise for motion uncertainty.
        self.filter.predict()
        self.last_update_time = int(time_ns)

    def errors_against_cam(self, cam: dict):
        # Step 3: Compare the prediction with the incoming CAM.
        predicted = self.predict_state(int(cam["rcvTime"]))
        pos = parse_position(cam["sender"]["pos"])
        velocity = velocity_from_cam(cam)
        pos_error = float(np.linalg.norm(predicted[0:2] - pos[0:2]))
        speed_error = float(abs(np.linalg.norm(predicted[2:4]) - np.linalg.norm(velocity)))
        return pos_error, speed_error

    def update_from_cam(self, cam: dict, measurement_noise: np.ndarray):
        # Step 4: Correct the predicted track with the accepted CAM measurement.
        self.predict_to(int(cam["rcvTime"]))
        self.last_seen_time = int(cam["rcvTime"])
        pos = parse_position(cam["sender"]["pos"])
        velocity = velocity_from_cam(cam)
        measurement = np.array([pos[0], pos[1], velocity[0], velocity[1]], dtype=float)
        self.filter.R = measurement_noise
        self.filter.update(measurement)
        self.station_alias = int(cam.get("sender_alias", self.station_alias))
