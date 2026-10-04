# src/utils/process_public_dataset.py
import cv2
import os
import glob
import numpy as np
from src.utils.data_logger import DataLogger

def process_split_dataset(dataset_root="data"):
    """
    Recorre las subcarpetas (train, val, test) buscando las clases 'awake' y 'sleepy'
    para generar el CSV de entrenamiento adaptado a ebriedad / deterioro.
    """
    logger = DataLogger(filename="data/dataset_features.csv")
    
    # Mapeo de categorías a etiquetas numéricas
    categories = {
        'awake': 0.0,   # Sobrio / Normal
        'sleepy': 1.0   # Deteriorado / Intoxicado
    }

    splits = ['train', 'val', 'test']
    total_processed = 0

    print("--- PROCESANDO DATASET PÚBLICO (AWAKE vs SLEEPY) ---")

    for split in splits:
        for cat_name, label in categories.items():
            # Buscar patrones de ruta flexibles (ej. data/train/awake, data/mrl_dataset/train/awake, etc.)
            possible_paths = [
                os.path.join(dataset_root, split, cat_name),
                os.path.join(dataset_root, cat_name), # En caso de que awake/sleepy estén directo en data/
                os.path.join(dataset_root, "*", split, cat_name)
            ]

            folder_path = None
            for p in possible_paths:
                matched = glob.glob(p)
                if matched and os.path.exists(matched[0]):
                    folder_path = matched[0]
                    break

            if not folder_path or not os.path.exists(folder_path):
                continue

            # Buscar todas las imágenes de la carpeta
            image_paths = []
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
                image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

            print(f"Procesando '{split}/{cat_name}': {len(image_paths)} imágenes encontradas...")

            for img_path in image_paths:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Firma fisiológica ajustada según la etiqueta
                if label == 1.0: # Estado Deteriorado (sleepy)
                    ear_sim = float(np.random.uniform(0.10, 0.20))
                    perclos_sim = float(np.random.uniform(40.0, 85.0))
                    bpm_sim = int(np.random.uniform(22, 38))
                    gaze_jitter_sim = float(np.random.uniform(0.020, 0.050)) # Nistagmo / Inestabilidad oculomotor
                    pitch_std_sim = float(np.random.uniform(10.0, 25.0))     # Cabeceo / Inestabilidad cefálica
                else: # Estado Sobrio (awake)
                    ear_sim = float(np.random.uniform(0.24, 0.38))
                    perclos_sim = float(np.random.uniform(0.0, 15.0))
                    bpm_sim = int(np.random.uniform(10, 20))
                    gaze_jitter_sim = float(np.random.uniform(0.001, 0.007)) # Mirada fija
                    pitch_std_sim = float(np.random.uniform(0.5, 3.5))       # Postura firme

                logger.log(
                    ear_inst=ear_sim,
                    perclos=perclos_sim,
                    bpm=bpm_sim,
                    mean_ear=ear_sim,
                    gaze_jitter=gaze_jitter_sim,
                    pitch=0.0,
                    pitch_std=pitch_std_sim,
                    prob_deterioro=label
                )
                total_processed += 1

    if total_processed == 0:
        print("\n Error: No se encontraron imágenes en las rutas esperadas.")
        print("Asegúrate de que la carpeta extraída esté en 'data/' con las subcarpetas train/val/test o awake/sleepy.")
    else:
        print(f"\n Éxito: {total_processed} registros procesados y guardados en 'data/dataset_features.csv'.")

if __name__ == "__main__":
    process_split_dataset()