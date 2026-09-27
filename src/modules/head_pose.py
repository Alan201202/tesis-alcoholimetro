# src/modules/head_pose.py
import cv2
import numpy as np
from collections import deque
from src.config import NOSE_TIP, CHIN, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, MOUTH_LEFT, MOUTH_RIGHT

class HeadPoseEstimator:
    def __init__(self, buffer_size=30):
        # Historial de ángulos para calcular la inestabilidad postural (varianza)
        self.pitch_history = deque(maxlen=buffer_size)
        self.yaw_history = deque(maxlen=buffer_size)
        
        # Modelo facial 3D canónico (en milímetros)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Punta de la nariz
            (0.0, -330.0, -65.0),        # Barbilla
            (-225.0, 170.0, -135.0),     # Esquina externa ojo izquierdo
            (225.0, 170.0, -135.0),      # Esquina externa ojo derecho
            (-150.0, -150.0, -125.0),    # Comisura izquierda de la boca
            (150.0, -150.0, -125.0)      # Comisura derecha de la boca
        ], dtype=np.float64)

    def process(self, landmarks_list, w, h):
        """
        Calcula los ángulos Pitch (Cabeceo), Yaw (Giro) y Roll (Inclinación).
        """
        # Extraer coordenadas 2D del frame
        image_points = np.array([
            (landmarks_list[NOSE_TIP].x * w, landmarks_list[NOSE_TIP].y * h),
            (landmarks_list[CHIN].x * w, landmarks_list[CHIN].y * h),
            (landmarks_list[LEFT_EYE_OUTER].x * w, landmarks_list[LEFT_EYE_OUTER].y * h),
            (landmarks_list[RIGHT_EYE_OUTER].x * w, landmarks_list[RIGHT_EYE_OUTER].y * h),
            (landmarks_list[MOUTH_LEFT].x * w, landmarks_list[MOUTH_LEFT].y * h),
            (landmarks_list[MOUTH_RIGHT].x * w, landmarks_list[MOUTH_RIGHT].y * h)
        ], dtype=np.float64)

        # Matriz intrínseca de cámara estimada
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        # Resolver el problema Perspective-n-Point (PnP)
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, 0.0

        # Convertir vector de rotación a matriz
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))

        # Descomponer la matriz para obtener los ángulos de Euler (Pitch, Yaw, Roll)
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]

        # Actualizar historial para varianza
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)

        # Inestabilidad/Varianza del movimiento del cuello
        head_instability = float(np.var(self.pitch_history) + np.var(self.yaw_history)) if len(self.pitch_history) > 5 else 0.0

        return pitch, yaw, roll, head_instability