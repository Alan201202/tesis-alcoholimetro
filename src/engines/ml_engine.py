# src/engines/ml_engine.py
import numpy as np

class MLEngine:
    def __init__(self, model_path=None):
        """
        En la Etapa 2 completa, aquí se carga el archivo 'model.pkl' entrenado.
        """
        self.model_path = model_path

    def predict_impairment_probability(self, ear, gaze_jitter, asymmetry, blink_duration, pitch, head_instability):
        """
        Vector Completo Módulos A, B y C:
        V = [EAR, Gaze_Jitter, Asimetría, Duración_Parpadeo, Pitch_Cabeza, Inestabilidad_Cabeza]
        """
        score = 0.0

        # 1. Módulo A: Parpadeo y Párpados
        if ear < 0.18:
            score += 0.25
        if blink_duration > 0.4:
            score += min(0.25, blink_duration * 0.4)

        # 2. Módulo B: Oculometría / Fijación
        if gaze_jitter > 0.015:
            score += min(0.25, gaze_jitter * 15.0)
        if asymmetry > 0.05:
            score += 0.10

        # 3. Módulo C: Pose Cefálica (Cabeceo e Inestabilidad de Cabeza)
        if pitch < -15.0:
            score += 0.25
        if head_instability > 12.0:
            score += 0.15

        probability = float(np.clip(score, 0.0, 1.0))
        return probability