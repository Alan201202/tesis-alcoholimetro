# main.py
import cv2
import time
import urllib.request
import os
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
from src.engines.feature_extractor import TemporalFeatureExtractor  # <--- NUEVO
from src.utils.data_logger import DataLogger                        # <--- NUEVO

MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo 'face_landmarker.task'...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
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
    
    # Instancias de Extracción Temporal y Logger
    temporal_extractor = TemporalFeatureExtractor(window_seconds=60.0, fps=30.0)
    logger = DataLogger(filename="data/dataset_features.csv")

    blink_count = 0
    eye_closed = False
    closed_start_time = 0.0
    last_blink_duration = 0.0
    last_log_time = time.time()

    print("--- SISTEMA A+B+C CON EXTRACCIÓN TEMPORAL Y REGISTRO EN CSV ---")

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

            # 1. Módulos A, B y C
            left_ear = calculate_ear(LEFT_EYE, landmarks, w, h)
            right_ear = calculate_ear(RIGHT_EYE, landmarks, w, h)
            current_ear = (left_ear + right_ear) / 2.0

            gaze_jitter, asymmetry, (l_center, r_center) = gaze_tracker.process(landmarks, w, h, left_ear, right_ear)
            pitch, yaw, roll, head_instability = head_pose.process(landmarks, w, h)

            # Calibración
            is_calibrated, calib_val = calibrator.process(current_ear)

            if not is_calibrated:
                remaining = int(CALIBRATION_TIME_SEC - calib_val) + 1
                status_text = f"CALIBRANDO BASE... ({remaining}s)"
                color = (0, 255, 255)
                prob_deterioro = 0.0
            else:
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
                        blink_count += 1
                        last_blink_duration = time.time() - closed_start_time
                        is_blink_event = True

                curr_closed_dur = (time.time() - closed_start_time) if eye_closed else 0.0

                # Actualizar extractor temporal (PERCLOS / BPM)
                temporal_extractor.update(current_ear, gaze_jitter, pitch, is_blink_event)
                feats = temporal_extractor.get_feature_vector(threshold_ear)

                # Predicción del Modelo
                prob_deterioro = ml_engine.predict_impairment_probability(
                    current_ear, gaze_jitter, asymmetry, max(curr_closed_dur, last_blink_duration), pitch, head_instability
                )

                # Guardar en CSV cada 1 segundo
                if time.time() - last_log_time >= 1.0:
                    logger.log(
                        current_ear, feats['PERCLOS'], feats['BPM'], feats['Mean_EAR'], 
                        gaze_jitter, pitch, feats['Pitch_STD'], prob_deterioro
                    )
                    last_log_time = time.time()

                # Estados
                if prob_deterioro >= 0.70:
                    status_text = f"ALERTA: EBRIEDAD / DETERIORO ({prob_deterioro*100:.1f}%)"
                    color = (0, 0, 255)
                elif prob_deterioro >= 0.35:
                    status_text = f"PRECAUCION: Inestabilidad ({prob_deterioro*100:.1f}%)"
                    color = (0, 165, 255)
                else:
                    status_text = f"ESTADO: Normal / Sobrio ({prob_deterioro*100:.1f}%)"
                    color = (0, 255, 0)

                # Overlay en pantalla de métricas avanzadas
                cv2.putText(frame, f"PERCLOS: {feats['PERCLOS']:.1f}%", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, f"BPM (Parpadeos/min): {feats['BPM']}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, f"Gaze Jitter: {gaze_jitter:.4f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, f"Pitch STD: {feats['Pitch_STD']:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, f"Prob. ML: {prob_deterioro*100:.1f}%", (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Barra inferior
            cv2.rectangle(frame, (20, h - 50), (w - 20, h - 10), (0, 0, 0), -1)
            cv2.putText(frame, status_text, (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.imshow("Sistema Biometrico - Extractor Avanzado de Features", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()