# app.py
import streamlit as st
import cv2
import time
import urllib.request
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Importaciones del backend
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

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Dashboard Biométrico Oculomotor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para Dashboard Médico / Biométrico
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1e222d;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #2e364f;
    }
    .status-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #1e222d;
        border-left: 5px solid #00d26a;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "face_landmarker.task"
TEST_DURATION_SEC = 30 

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)

download_model()

def get_hgn_target_position(elapsed, w, h):
    margin = int(w * 0.15)
    min_x = margin
    max_x = w - margin
    range_x = max_x - min_x
    center_y = int(h / 2)

    cycle_time = elapsed % 10.0
    
    if cycle_time < 3.0:
        ratio = cycle_time / 3.0
        cx = int(min_x + ratio * range_x)
        stage = "Persecución Suave (Der)"
    elif cycle_time < 5.0:
        cx = max_x
        stage = "Fijación Extrema (Der)"
    elif cycle_time < 8.0:
        ratio = (cycle_time - 5.0) / 3.0
        cx = int(max_x - ratio * range_x)
        stage = "Persecución Suave (Izq)"
    else:
        cx = min_x
        stage = "Fijación Extrema (Izq)"

    return cx, center_y, stage

def get_eval_tag(val, norm_max, warn_max):
    if val <= norm_max:
        return "OK (Normal)"
    elif val <= warn_max:
        return "PRECAUCIÓN"
    else:
        return "ALTERADO"

# --- ENCABEZADO Y CONTROL ---
st.title("👁️ Dashboard Biométrico de Evaluación Oculomotora")
st.caption("Sistema en tiempo real para análisis de nistagmo (HGN), dinámica palpebral y estabilidad cefálica.")

st.sidebar.header("🕹️ Panel de Control")
st.sidebar.markdown("---")
start_button = st.sidebar.button("▶️ Iniciar Evaluación (30s)", type="primary", use_container_width=True)
st.sidebar.info("💡 **Instrucciones:** Mantenga su rostro frente a la cámara y siga con la mirada la esfera roja sin mover la cabeza.")

if start_button:
    # Disposición en dos columnas para el Dashboard principal
    col_cam, col_metrics = st.columns([2, 1])

    with col_cam:
        st.subheader("📹 Captura y Seguimiento Ocular")
        frame_placeholder = st.empty()
        status_box = st.empty()

    with col_metrics:
        st.subheader("⚡ Métricas en Tiempo Real")
        m_perclos = st.metric("PERCLOS", "0.0%", delta_color="normal")
        m_bpm = st.metric("Frecuencia Parpadeo (BPM)", "0.0")
        m_jitter = st.metric("Nistagmo (Gaze Jitter)", "0.00000")
        m_pitch = st.metric("Inestabilidad Cefálica", "0.00°")
        
        st.markdown("**Nivel de Deterioro Integrado**")
        risk_bar = st.progress(0)

    # Contenedor para la gráfica en vivo
    st.divider()
    st.subheader("📈 Monitoreo Temporal Continuo")
    chart_placeholder = st.empty()

    # Inicialización del pipeline
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
    
    temporal_extractor = TemporalFeatureExtractor(window_seconds=15.0, fps=30.0)
    logger = DataLogger(filename="data/dataset_features.csv")

    eye_closed = False
    closed_start_time = 0.0
    last_blink_duration = 0.0
    last_log_time = time.time()

    history_ml = []
    history_heur = []
    history_final = []
    history_jitter = []
    history_perclos = []
    history_bpm = []
    history_pitch_std = []

    # Buffers para gráficas en vivo
    df_chart = pd.DataFrame(columns=["PERCLOS (%)", "Nistagmo (x1000)"])

    test_started = False
    test_start_time = 0.0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.error("Error al acceder a la fuente de video.")
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
                status_box.warning(f"⏳ **Calibrando línea base... ({remaining_calib}s)** Fijar mirada al frente.")
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

                if not eye_closed:
                    history_jitter.append(gaze_jitter)
                
                history_perclos.append(feats['PERCLOS'])
                history_bpm.append(feats['BPM'])
                history_pitch_std.append(feats['Pitch_STD'])

                history_ml.append(p_ml)
                history_heur.append(p_heur)
                history_final.append(prob_deterioro)

                # Actualizar dashboard dinámico
                m_perclos.metric("PERCLOS", f"{feats['PERCLOS']:.1f}%")
                m_bpm.metric("Frecuencia Parpadeo (BPM)", f"{feats['BPM']:.1f}")
                m_jitter.metric("Nistagmo (Gaze Jitter)", f"{gaze_jitter:.5f}")
                m_pitch.metric("Inestabilidad Cefálica", f"{feats['Pitch_STD']:.2f}°")
                risk_bar.progress(int(min(max(prob_deterioro, 0.0), 1.0) * 100))

                # Actualizar gráfica
                new_row = pd.DataFrame({
                    "PERCLOS (%)": [feats['PERCLOS']],
                    "Nistagmo (x1000)": [gaze_jitter * 1000]
                })
                df_chart = pd.concat([df_chart, new_row], ignore_index=True).tail(50)
                chart_placeholder.line_chart(df_chart)

                if time.time() - last_log_time >= 1.0:
                    logger.log(
                        current_ear, feats['PERCLOS'], feats['BPM'], feats['Mean_EAR'], 
                        gaze_jitter, pitch, feats['Pitch_STD'], prob_deterioro
                    )
                    last_log_time = time.time()

                cx, cy, stage_name = get_hgn_target_position(elapsed_test, w, h)
                status_box.info(f"🎯 **Fase HGN:** {stage_name} | ⏱️ **Tiempo:** {remaining_test}s")

                # Estímulo de movimiento en la cámara
                cv2.circle(frame, (cx, cy), 14, (0, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

                if elapsed_test >= TEST_DURATION_SEC:
                    break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    frame_placeholder.empty()
    status_box.success("🎉 **Evaluación biométrica finalizada.** Generando informe...")

    # --- DESPLIEGUE DEL INFORME FINAL ---
    if history_final:
        fps_approx = 30
        sec_heur = [np.mean(history_heur[i:i+fps_approx]) for i in range(0, len(history_heur), fps_approx)]
        sec_ml = [np.mean(history_ml[i:i+fps_approx]) for i in range(0, len(history_ml), fps_approx)]
        sec_final = [np.mean(history_final[i:i+fps_approx]) for i in range(0, len(history_final), fps_approx)]

        avg_heur = np.mean(sec_heur) * 100
        avg_ml = np.mean(sec_ml) * 100
        avg_final = np.mean(sec_final) * 100

        peak_p85 = np.percentile(sec_final, 85) * 100
        seconds_severe = sum(1 for s in sec_final if s >= 0.70)
        seconds_warning = sum(1 for s in sec_final if 0.35 <= s < 0.70)

        mean_jitter = np.mean(history_jitter) if history_jitter else 0.0
        mean_perclos = np.mean(history_perclos) if history_perclos else 0.0
        mean_bpm = np.mean(history_bpm) if history_bpm else 0.0
        mean_pitch_std = np.mean(history_pitch_std) if history_pitch_std else 0.0

        st.divider()
        st.subheader("📊 Reporte Consolidado por Motores Biométricos")

        table_data = [
            {
                "Eje de Evaluación": "1. Oculomotor (Nistagmo)",
                "Medición": f"{mean_jitter:.5f}",
                "Rango Fisiológico Normal": "0.0001 - 0.0035",
                "Diagnóstico": get_eval_tag(mean_jitter, 0.0035, 0.0065)
            },
            {
                "Eje de Evaluación": "2. Palpebral (PERCLOS)",
                "Medición": f"{mean_perclos:.1f}%",
                "Rango Fisiológico Normal": "0.0% - 15.0%",
                "Diagnóstico": get_eval_tag(mean_perclos, 15.0, 30.0)
            },
            {
                "Eje de Evaluación": "3. Dinámica de Parpadeo (BPM)",
                "Medición": f"{mean_bpm:.1f}",
                "Rango Fisiológico Normal": "10 - 25 BPM",
                "Diagnóstico": "OK (Normal)" if 10 <= mean_bpm <= 25 else "FUERA DE RANGO"
            },
            {
                "Eje de Evaluación": "4. Estabilidad Postural (Pitch)",
                "Medición": f"{mean_pitch_std:.2f}°",
                "Rango Fisiológico Normal": "0.00° - 3.00°",
                "Diagnóstico": get_eval_tag(mean_pitch_std, 3.0, 6.0)
            }
        ]

        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        st.subheader("🔢 Índices de Evaluación Integrados")
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        r_col1.metric("Índice Biomecánico (Reglas)", f"{avg_heur:.1f}%")
        r_col2.metric("Índice Estadístico (ML)", f"{avg_ml:.1f}%")
        r_col3.metric("Índice Final Integrado", f"{avg_final:.1f}%")
        r_col4.metric("Severidad Sostenida (P85)", f"{peak_p85:.1f}%")

        st.divider()
        st.subheader("🏁 Dictamen Técnico Final")

        if peak_p85 >= 65.0 or seconds_severe >= 3:
            st.error("🔴 **[ ESTADO: ALTO RIESGO / DETERIORO SEVERO ]**\n\n**Criterio:** Alteraciones oculomotoras significativas detectadas en la prueba de seguimiento HGN o inestabilidad postural crítica.")
        elif peak_p85 >= 35.0 or seconds_warning >= 5:
            st.warning("🟠 **[ ESTADO: PRECAUCIÓN / ANOMALÍA LEVE ]**\n\n**Criterio:** Leves variaciones registradas en el control de fijación ocular o fatiga palpebral moderada.")
        else:
            st.success("🟢 **[ ESTADO: SOBRIO / DESEMPEÑO NORMAL ]**\n\n**Criterio:** Todos los indicadores biométricos se mantuvieron dentro de los parámetros fisiológicos estándar.")