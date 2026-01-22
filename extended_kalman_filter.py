"""
Extended Kalman Filter for tracking 2D point motion with velocity estimation.
Models the quasi-periodic motion of the heart surface.
Tuned for temporal smoothing to reduce flicker.
"""

import numpy as np


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for 2D position tracking with velocity estimation.
    
    State vector: [x, y, vx, vy]
        - x, y: 2D position in image coordinates
        - vx, vy: velocity components
    
    Tuned for smoothing optical flow measurements to eliminate flicker.
    """
    
    def __init__(self, initial_position, process_noise=10.0, measurement_noise=20.0, dt=0.033):
        """
        Initialize the Extended Kalman Filter.
        
        Args:
            initial_position: Initial [x, y] position
            process_noise: Process noise coefficient (higher = more smoothing)
            measurement_noise: Measurement noise coefficient (higher = less trust in measurements)
            dt: Time step between frames (default ~30 fps)
        """
        self.dt = dt
        
        # State vector: [x, y, vx, vy]
        self.x = np.array([
            initial_position[0],
            initial_position[1],
            0.0,  # Initial velocity x
            0.0   # Initial velocity y
        ], dtype=np.float32)
        
        # State covariance matrix (start with lower uncertainty)
        self.P = np.eye(4, dtype=np.float32) * 10.0
        
        # Process noise covariance matrix
        q = process_noise
        self.Q = np.array([
            [q * dt**4 / 4, 0, q * dt**3 / 2, 0],
            [0, q * dt**4 / 4, 0, q * dt**3 / 2],
            [q * dt**3 / 2, 0, q * dt**2, 0],
            [0, q * dt**3 / 2, 0, q * dt**2]
        ], dtype=np.float32)
        
        # Measurement noise covariance matrix
        r = measurement_noise
        self.R = np.array([
            [r, 0],
            [0, r]
        ], dtype=np.float32)
        
        # Measurement matrix (we only observe position, not velocity)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Track number of prediction-only steps
        self.prediction_only_steps = 0
        self.max_prediction_steps = 30
        
    def predict(self):
        """
        Prediction step: Estimate next state based on motion model.
        Uses constant velocity model.
        """
        # State transition matrix (constant velocity model)
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Predict state
        self.x = F @ self.x
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
        self.prediction_only_steps += 1
        
        return self.get_position()
    
    def update(self, measurement):
        """
        Update step: Incorporate new measurement (optical flow detection).
        
        Args:
            measurement: Observed [x, y] position from optical flow
        """
        if measurement is None:
            return self.predict()
        
        # Convert measurement to numpy array
        z = np.array(measurement, dtype=np.float32)
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + K @ y
        
        # Update covariance estimate
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        # Reset prediction counter
        self.prediction_only_steps = 0
        
        return self.get_position()
    
    def get_position(self):
        """Get current estimated position [x, y]."""
        return self.x[:2].copy()
    
    def get_velocity(self):
        """Get current estimated velocity [vx, vy]."""
        return self.x[2:].copy()
    
    def get_state(self):
        """Get full state vector [x, y, vx, vy]."""
        return self.x.copy()
    
    def is_tracking_lost(self):
        """Check if tracking is likely lost."""
        return self.prediction_only_steps > self.max_prediction_steps
    
    def get_uncertainty(self):
        """Get position uncertainty (standard deviation in x and y)."""
        return np.sqrt(np.diag(self.P)[:2])
    
    def adjust_noise_parameters(self, process_noise=None, measurement_noise=None):
        """Dynamically adjust noise parameters."""
        if process_noise is not None:
            q = process_noise
            dt = self.dt
            self.Q = np.array([
                [q * dt**4 / 4, 0, q * dt**3 / 2, 0],
                [0, q * dt**4 / 4, 0, q * dt**3 / 2],
                [q * dt**3 / 2, 0, q * dt**2, 0],
                [0, q * dt**3 / 2, 0, q * dt**2]
            ], dtype=np.float32)
        
        if measurement_noise is not None:
            r = measurement_noise
            self.R = np.array([
                [r, 0],
                [0, r]
            ], dtype=np.float32)
