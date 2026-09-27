# eye_dynamics.py
import numpy as np

def calculate_ear(eye_landmarks, landmarks_list, frame_w, frame_h):
    """
    Calcula el Eye Aspect Ratio (EAR) para un ojo.
    """
    coords = []
    for idx in eye_landmarks:
        lm = landmarks_list[idx]
        coords.append(np.array([lm.x * frame_w, lm.y * frame_h]))
    
    # Distancias verticales
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    # Distancia horizontal
    C = np.linalg.norm(coords[0] - coords[3])
    
    ear = (A + B) / (2.0 * C)
    return ear