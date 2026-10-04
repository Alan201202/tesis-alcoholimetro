# src/engines/train_model.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATASET_PATH = "data/dataset_features.csv"
MODEL_OUTPUT_PATH = "src/engines/model.pkl"

def train():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: No se encontró '{DATASET_PATH}'. Corre primero 'python src/utils/process_public_dataset.py'.")
        return

    print("--- ENTRENANDO MODELO DE ML PARA DETERIORO POR ALCOHOL ---")
    df = pd.read_csv(DATASET_PATH)

    # Variables de entrada y vector objetivo
    X = df[['EAR_Instant', 'PERCLOS', 'BPM', 'Mean_EAR', 'Gaze_Jitter', 'Pitch_Angle', 'Pitch_STD']]
    y = (df['Prob_Deterioro'] >= 0.50).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # Modelo Random Forest
    clf = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n Precisión del Modelo (Accuracy): {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=['Sobrio', 'Deteriorado']))

    # Guardar binario .pkl
    os.makedirs("src/engines", exist_ok=True)
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(clf, f)

    print(f"Modelo exportado correctamente en: '{MODEL_OUTPUT_PATH}'")

if __name__ == "__main__":
    train()