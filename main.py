# main.py
import cv2
import time
import urllib.request
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import (
    LEFT_EYE, 
    RIGHT_EYE, 
    CALIBRATION_TIME_SEC, 
    EAR_THRESHOLD_FACTOR
)
from src.modules.eye_dynamics import calculate_ear
from src.utils.calibration import Calibrator
from src.modules.gaze_tracker import GazeTracker
from src.modules.head_pose import HeadPoseEstimator
from src.engines.ml_engine import MLEngine
from src.engines.feature_extractor import TemporalFeatureExtractor
from src.utils.data_logger import DataLogger

MODEL_PATH = "face_landmarker.task"
TEST_DURATION_SEC = 30  # Duración de la prueba de seguimiento ocular

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo 'face_landmarker.task'...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    calibrator = Calibrator(calibration_time=CALIBRATION_TIME_SEC)
    calibrator.start()
    
    gaze_tracker = GazeTracker(buffer_size=30)
    head_pose = HeadPoseEstimator(buffer_size=30)
    ml_engine = MLEngine()
    
    # Ventana temporal ajustada a 15.0s para respuesta inmediata
    temporal_extractor = TemporalFeatureExtractor(window_seconds=15.0, fps=30.0)
    logger = DataLogger(filename="data/dataset_features.csv")

    eye_closed = False
    closed_start_time = 0.0
    last_blink_duration = 0.0
    last_log_time = time.time()

    # Historiales para evaluación posterior
    history_ml = []
    history_heur = []
    history_final = []
    history_jitter = []

    test_started = False
    test_start_time = 0.0

    print("\n--- INICIANDO PRUEBA DE EVALUACIÓN: SEGUIMIENTO OCULAR Y NISTAGMO ---")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            left_ear = calculate_ear(LEFT_EYE, landmarks, w, h)
            right_ear = calculate_ear(RIGHT_EYE, landmarks, w, h)
            current_ear = (left_ear + right_ear) / 2.0

            gaze_jitter, asymmetry, (l_center, r_center) = gaze_tracker.process(landmarks, w, h, left_ear, right_ear)
            pitch, yaw, roll, head_instability = head_pose.process(landmarks, w, h)

            is_calibrated, calib_val = calibrator.process(current_ear)

            if not is_calibrated:
                remaining_calib = int(CALIBRATION_TIME_SEC - calib_val) + 1
                status_text = f"CALIBRANDO BASE... Sostén la mirada fija ({remaining_calib}s)"
                color = (0, 255, 255)
            else:
                if not test_started:
                    test_started = True
                    test_start_time = time.time()

                elapsed_test = time.time() - test_start_time
                remaining_test = max(0, int(TEST_DURATION_SEC - elapsed_test))

                ear_base = calib_val
                threshold_ear = ear_base * EAR_THRESHOLD_FACTOR

                is_blink_event = False
                if current_ear < threshold_ear:
                    if not eye_closed:
                        eye_closed = True
                        closed_start_time = time.time()
                else:
                    if eye_closed:
                        eye_closed = False
                        last_blink_duration = time.time() - closed_start_time
                        is_blink_event = True

                curr_closed_dur = (time.time() - closed_start_time) if eye_closed else 0.0

                temporal_extractor.update(current_ear, gaze_jitter, pitch, is_blink_event)
                feats = temporal_extractor.get_feature_vector(threshold_ear)

                prob_deterioro, p_ml, p_heur = ml_engine.predict_impairment_probability(
                    current_ear, gaze_jitter, asymmetry, max(curr_closed_dur, last_blink_duration), 
                    pitch, head_instability, feats['PERCLOS'], feats['BPM'], feats['Mean_EAR'], feats['Pitch_STD']
                )

                history_ml.append(p_ml)
                history_heur.append(p_heur)
                history_final.append(prob_deterioro)
                history_jitter.append(gaze_jitter)

                if time.time() - last_log_time >= 1.0:
                    logger.log(
                        current_ear, feats['PERCLOS'], feats['BPM'], feats['Mean_EAR'], 
                        gaze_jitter, pitch, feats['Pitch_STD'], prob_deterioro
                    )
                    last_log_time = time.time()

                if prob_deterioro >= 0.70:
                    status_text = f"ALERTA: INESTABILIDAD OCULAR / DETERIORO ({prob_deterioro*100:.1f}%)"
                    color = (0, 0, 255)
                elif prob_deterioro >= 0.35:
                    status_text = f"PRECAUCION: SEGUIMIENTO LENTO ({prob_deterioro*100:.1f}%)"
                    color = (0, 165, 255)
                else:
                    status_text = f"SEGUIMIENTO OPTIMO / SOBRIO ({prob_deterioro*100:.1f}%)"
                    color = (0, 255, 0)

                # Despliegue de datos en vivo
                cv2.putText(frame, f"PRUEBA DE SEGUIMIENTO OCULAR: {remaining_test}s", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Gaze Jitter (Inestabilidad): {gaze_jitter:.4f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"PERCLOS: {feats['PERCLOS']:.1f}% | BPM: {feats['BPM']}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Pitch STD (Inestabilidad Cefalica): {feats['Pitch_STD']:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.putText(frame, f"Model 1 (Biomecanica / Reglas): {p_heur*100:.1f}%", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, f"Model 2 (Machine Learning .pkl): {p_ml*100:.1f}%", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, f"DICTAMEN HIBRIDO EN VIVO: {prob_deterioro*100:.1f}%", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                # Guía visual para seguimiento
                cx = int(w / 2 + np.sin(time.time() * 2.0) * (w * 0.3))
                cy = int(h / 2)
                cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)
                cv2.putText(frame, "SIGUE EL PUNTO CON LA MIRADA", (cx - 100, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                if elapsed_test >= TEST_DURATION_SEC:
                    break

            cv2.rectangle(frame, (20, h - 50), (w - 20, h - 10), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.imshow("Alcoholimetro Visual - Test de Seguimiento Ocular", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- INFORME COMPARATIVO FINAL (PROMEDIO POR SEGUNDO Y EVENTOS CRÍTICOS) ---
    if history_final:
        fps_approx = 30
        
        # Agrupación por bloques de 1 segundo (30 cuadros)
        sec_heur = [np.mean(history_heur[i:i+fps_approx]) for i in range(0, len(history_heur), fps_approx)]
        sec_ml = [np.mean(history_ml[i:i+fps_approx]) for i in range(0, len(history_ml), fps_approx)]
        sec_final = [np.mean(history_final[i:i+fps_approx]) for i in range(0, len(history_final), fps_approx)]

        avg_heur = np.mean(sec_heur) * 100
        avg_ml = np.mean(sec_ml) * 100
        avg_final = np.mean(sec_final) * 100

        # Métrica de Severidad Pico (Percentil 85)
        peak_final = np.percentile(sec_final, 85) * 100
        
        seconds_alert = sum(1 for s in sec_final if s >= 0.70)
        seconds_warning = sum(1 for s in sec_final if 0.35 <= s < 0.70)

        print("\n==================================================")
        print("    INFORME DE EVALUACIÓN DE SEGUIMIENTO OCULAR   ")
        print("==================================================")
        print(f"Duración de la sesión: {len(sec_final)} segundos")
        print(f"Jitter Promedio de Mirada: {np.mean(history_jitter):.5f}\n")
        
        print(f"1. Promedio Biomecánico: {avg_heur:.2f}% | Pico Muestreado: {np.max(sec_heur)*100:.2f}%")
        print(f"2. Promedio Machine Learning: {avg_ml:.2f}% | Pico Muestreado: {np.max(sec_ml)*100:.2f}%")
        print(f"--------------------------------------------------")
        print(f"PROMEDIO HÍBRIDO GENERAL:      {avg_final:.2f}%")
        print(f"SEVERIDAD PICO SOSTENIDO (P85): {peak_final:.2f}%")
        print(f"Segundos en Alerta (Deterioro): {seconds_alert}s")
        print(f"Segundos en Precaución:         {seconds_warning}s\n")
        
        # Dictamen según eventos críticos detectados
        if peak_final >= 65.0 or seconds_alert >= 3:
            print("DICTAMEN FINAL: POSITIVO A DETERIORO POR ALCOHOL")
            print("Fundamento: Se detectaron eventos críticos o picos de inestabilidad ocular/postural.")
        elif peak_final >= 35.0 or seconds_warning >= 5:
            print("DICTAMEN FINAL: PRECAUCIÓN / INESTABILIDAD DETECTADA")
            print("Fundamento: Se registraron oscilaciones leves en el seguimiento o pestañeo.")
        else:
            print("DICTAMEN FINAL: NEGATIVO (Estado Sobrio / Normal)")
            print("Fundamento: Desempeño óptimo y sostenido durante toda la prueba.")
        print("==================================================\n")

if __name__ == "__main__":
    main()