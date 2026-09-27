# src/engines/feature_extractor.py
import numpy as np
from collections import deque
import time

class TemporalFeatureExtractor:
    def __init__(self, window_seconds=60.0, fps=30.0):
        self.window_seconds = window_seconds
        self.max_samples = int(window_seconds * fps)
        
        # Buffers temporales
        self.ear_history = deque(maxlen=self.max_samples)
        self.blink_timestamps = deque()
        self.gaze_history = deque(maxlen=self.max_samples)
        self.pitch_history = deque(maxlen=self.max_samples)

    def update(self, ear, gaze_jitter, pitch, is_blink):
        """
        Agrega la muestra del fotograma actual a las ventanas temporales.
        """
        current_time = time.time()
        self.ear_history.append(ear)
        self.gaze_history.append(gaze_jitter)
        self.pitch_history.append(pitch)

        if is_blink:
            self.blink_timestamps.append(current_time)

        # Limpiar marcas de tiempo de parpadeos fuera de la ventana de 60s
        while self.blink_timestamps and (current_time - self.blink_timestamps[0] > self.window_seconds):
            self.blink_timestamps.popleft()

    def get_perclos(self, ear_threshold):
        """
        Calcula PERCLOS: Porcentaje de fotogramas donde el EAR cae por debajo del umbral.
        """
        if not self.ear_history:
            return 0.0
        
        closed_frames = sum(1 for ear in self.ear_history if ear < ear_threshold)
        return float(closed_frames / len(self.ear_history)) * 100.0

    def get_blink_frequency(self):
        """
        Retorna la frecuencia de parpadeos por minuto (BPM).
        """
        return len(self.blink_timestamps)

    def get_feature_vector(self, ear_threshold):
        """
        Retorna el vector condensado de características temporales listo para ML.
        """
        perclos = self.get_perclos(ear_threshold)
        bpm = self.get_blink_frequency()
        mean_gaze_jitter = float(np.mean(self.gaze_history)) if self.gaze_history else 0.0
        pitch_std = float(np.std(self.pitch_history)) if self.pitch_history else 0.0
        mean_ear = float(np.mean(self.ear_history)) if self.ear_history else 0.0

        return {
            'PERCLOS': round(perclos, 2),
            'BPM': bpm,
            'Mean_EAR': round(mean_ear, 4),
            'Mean_Gaze_Jitter': round(mean_gaze_jitter, 6),
            'Pitch_STD': round(pitch_std, 4)
        }