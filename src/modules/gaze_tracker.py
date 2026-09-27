# src/modules/gaze_tracker.py
import numpy as np
from collections import deque
from src.config import (LEFT_IRIS, RIGHT_IRIS, 
                        LEFT_EYE_CORNER_L, LEFT_EYE_CORNER_R, 
                        RIGHT_EYE_CORNER_L, RIGHT_EYE_CORNER_R)

class GazeTracker:
    def __init__(self, buffer_size=30):
        # Mantiene las últimas N posiciones (X, Y) para calcular la varianza/jitter
        self.gaze_history = deque(maxlen=buffer_size)

    def _get_center(self, iris_landmarks, landmarks_list, w, h):
        """Calcula el centro geométrico del iris."""
        coords = np.array([[landmarks_list[idx].x * w, landmarks_list[idx].y * h] for idx in iris_landmarks])
        return np.mean(coords, axis=0)

    def calculate_gaze_ratio(self, landmarks_list, iris_indices, corner_left_idx, corner_right_idx, w, h):
        """
        Calcula la posición relativa del iris dentro de la cavidad ocular (0.0 a 1.0)
        """
        iris_center = self._get_center(iris_indices, landmarks_list, w, h)
        
        c_left = np.array([landmarks_list[corner_left_idx].x * w, landmarks_list[corner_left_idx].y * h])
        c_right = np.array([landmarks_list[corner_right_idx].x * w, landmarks_list[corner_right_idx].y * h])

        # Ancho total del ojo en pixeles
        eye_width = np.linalg.norm(c_right - c_left)
        if eye_width == 0:
            return 0.5, iris_center

        # Distancia del centro del iris a la esquina izquierda
        dist_x = np.linalg.norm(iris_center - c_left)
        ratio_x = dist_x / eye_width

        return ratio_x, iris_center

    def process(self, landmarks_list, w, h, left_ear, right_ear):
        """
        Procesa la oculometría completa. Retorna varianza del movimiento y asimetría palpebral.
        """
        lx_ratio, l_center = self.calculate_gaze_ratio(landmarks_list, LEFT_IRIS, LEFT_EYE_CORNER_L, LEFT_EYE_CORNER_R, w, h)
        rx_ratio, r_center = self.calculate_gaze_ratio(landmarks_list, RIGHT_IRIS, RIGHT_EYE_CORNER_L, RIGHT_EYE_CORNER_R, w, h)

        # Promedio del vector de centro de mirada
        avg_gaze_x = (lx_ratio + rx_ratio) / 2.0
        
        # Guardar en ventana deslizante
        self.gaze_history.append(avg_gaze_x)

        # 1. Métrica: Gaze Jitter (Varianza del movimiento ocular)
        if len(self.gaze_history) > 5:
            gaze_jitter = float(np.var(self.gaze_history))
        else:
            gaze_jitter = 0.0

        # 2. Métrica: Asimetría Palpebral (Diferencia de apertura entre ojos)
        eye_asymmetry = abs(left_ear - right_ear)

        return gaze_jitter, eye_asymmetry, (l_center, r_center)