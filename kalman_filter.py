import numpy as np


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
        dt: time step
        u_x: acceleration in x-direction
        u_y: acceleration in y-direction
        std_acc: standard deviation of the acceleration
        x_std_meas: standard deviation of the measurement in x-direction
        y_std_meas: standard deviation of the measurement in y-direction
        """
        self.dt = dt

        # State transition model
        self.F = np.array([[1, dt, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]])

        # Control-input model
        self.B = np.array([[0.5 * dt ** 2, 0],
                           [dt, 0],
                           [0, 0.5 * dt ** 2],
                           [0, dt]])

        # Measurement model
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])

        # Process noise covariance matrix
        self.Q = np.array([[(dt ** 4) / 4, (dt ** 3) / 2, 0, 0],
                           [(dt ** 3) / 2, dt ** 2, 0, 0],
                           [0, 0, (dt ** 4) / 4, (dt ** 3) / 2],
                           [0, 0, (dt ** 3) / 2, dt ** 2]]) * std_acc ** 2

        # Measurement noise covariance matrix
        self.R = np.array([[x_std_meas ** 2, 0],
                           [0, y_std_meas ** 2]])

        # Initial state covariance matrix
        self.P = np.eye(4)

        # Initial state vector
        self.x = np.array([[0], [0], [0], [0]])

    def predict(self, u):
        """
        Predict the next state
        u: control input vector
        """
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        """
        Update the state with the new measurement
        z: measurement vector
        """
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)
        return self.x
