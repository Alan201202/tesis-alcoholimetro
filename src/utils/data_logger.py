# src/utils/data_logger.py
import csv
import os
import time

class DataLogger:
    def __init__(self, filename="data/dataset_features.csv"):
        self.filename = filename
        
        # Crear la carpeta data/ si no existe
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Crear encabezados si el archivo no existe
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "EAR_Instant", "PERCLOS", "BPM", 
                    "Mean_EAR", "Gaze_Jitter", "Pitch_Angle", "Pitch_STD", 
                    "Prob_Deterioro"
                ])

    def log(self, ear_inst, perclos, bpm, mean_ear, gaze_jitter, pitch, pitch_std, prob_deterioro):
        """
        Registra un renglón en el archivo CSV.
        """
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                round(ear_inst, 4),
                perclos,
                bpm,
                mean_ear,
                round(gaze_jitter, 6),
                round(pitch, 2),
                pitch_std,
                round(prob_deterioro, 4)
            ])