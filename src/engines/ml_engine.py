# src/engines/ml_engine.py
import numpy as np
import pickle
import os

class MLEngine:
    def __init__(self, model_path="src/engines/model.pkl", weight_ml=0.65, weight_heuristic=0.35):
        self.model_path = model_path
        self.weight_ml = weight_ml
        self.weight_heuristic = weight_heuristic
        self.model = None
        self.load_model()

    def load_model(self):
        """Carga el modelo .pkl entrenado si existe."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"[ML Engine] Modelo ML '{self.model_path}' cargado correctamente. Modo HIBRIDO activo.")
            except Exception as e:
                print(f"[ML Engine] Error al cargar {self.model_path}: {e}. Operando en modo 100% heuristico.")
        else:
            print("[ML Engine] Modelo .pkl no encontrado. Operando en modo 100% heuristico.")

    def _calculate_heuristic_score(self, ear, gaze_jitter, asymmetry, blink_duration, pitch, head_instability):
        """Calcula el puntaje biomecánico basado en umbrales físicos (Módulos A, B, C)."""
        score = 0.0

        # Módulo A: Apertura Ocular y Duración de Parpadeo
        if ear < 0.18:
            score += 0.25
        if blink_duration > 0.4:
            score += min(0.25, blink_duration * 0.4)

        # Módulo B: Oculometría y Asimetría
        if gaze_jitter > 0.015:
            score += min(0.25, gaze_jitter * 15.0)
        if asymmetry > 0.05:
            score += 0.10

        # Módulo C: Pose Cefálica e Inestabilidad
        if pitch < -15.0:
            score += 0.25
        if head_instability > 12.0:
            score += 0.15

        return float(np.clip(score, 0.0, 1.0))

    def predict_impairment_probability(self, ear, gaze_jitter, asymmetry, blink_duration, pitch, head_instability, perclos=0.0, bpm=0, mean_ear=0.0, pitch_std=0.0):
        """
        Calcula la probabilidad combinada mediante Ensamble Híbrido.
        Retorna: (prob_final, prob_ml, prob_heuristica)
        """
        # 1. Puntuación del motor biomecánico
        p_heuristic = self._calculate_heuristic_score(
            ear, gaze_jitter, asymmetry, blink_duration, pitch, head_instability
        )

        # 2. Puntuación del modelo supervisado .pkl (si existe)
        p_ml = 0.0
        if self.model is not None:
            try:
                features = np.array([[ear, perclos, bpm, mean_ear, gaze_jitter, pitch, pitch_std]])
                p_ml = float(self.model.predict_proba(features)[0][1])
                
                # Fusión ponderada: 65% ML + 35% Biomecánica
                p_final = (self.weight_ml * p_ml) + (self.weight_heuristic * p_heuristic)
                return float(np.clip(p_final, 0.0, 1.0)), p_ml, p_heuristic
            except Exception:
                pass

        # Fallback cuando aún no se ha entrenado el .pkl
        return p_heuristic, 0.0, p_heuristic