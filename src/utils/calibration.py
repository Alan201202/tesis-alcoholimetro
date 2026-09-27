# calibration.py
import time
import numpy as np

class Calibrator:
    def __init__(self, calibration_time=5.0):
        self.calibration_time = calibration_time
        self.start_time = None
        self.baseline_list = []
        self.is_calibrated = False
        self.ear_base = 0.25

    def start(self):
        self.start_time = time.time()

    def process(self, current_ear):
        if self.is_calibrated:
            return True, self.ear_base

        elapsed_time = time.time() - self.start_time
        if elapsed_time < self.calibration_time:
            self.baseline_list.append(current_ear)
            return False, elapsed_time
        else:
            self.ear_base = float(np.mean(self.baseline_list))
            self.is_calibrated = True
            return True, self.ear_base