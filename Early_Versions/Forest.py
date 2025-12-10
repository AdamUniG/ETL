import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(r'C:\Users\adamv\ETL')
DATA_DIR = BASE_DIR / 'Data'
INPUT_FILE = DATA_DIR / 'master_advanced_morphology.csv' # Reading the NEW file
MODEL_OUTPUT_FILE = DATA_DIR / 'advanced_morphology_classifier.pkl'

# --- NEW FULL FEATURE LIST ---
FEATURE_COLS = [
    # Complexity / Shape
    'circularity', 
    'solidity', 
    'eccentricity', 
    'extent',
    # Size / Dimensions
    'area_px', 
    'aspect_ratio', 
    'major_axis', 
    'minor_axis'
]
TARGET_COL = 'Ground_Truth_Label'

def train_advanced_model():
    print("="*60)
    print("🌲 TRAINING RANDOM FOREST (ADVANCED MORPHOLOGY)")
    print("="*60)

    if not INPUT_FILE.exists():
        print(f"❌ File not found: {INPUT_FILE}\n   Run All_Together_Advanced_Morphology.py first!")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Optional: Drop rows where circularity is 0 (failed calc)
    df = df[df['circularity'] > 0]
    
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    clf = RandomForestClassifier(
        n_estimators=300, 
        max_depth=15, 
        class_weight='balanced', 
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    
    print("\n" + "="*40)
    print(f"ACCURACY: {accuracy_score(y_test, y_pred):.2%}")
    print("="*40)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Visualizing Feature Importance (Crucial step)
    print("\n🔍 FEATURE IMPORTANCE RANKING:")
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(importances)
    
    # Save
    joblib.dump(clf, MODEL_OUTPUT_FILE)
    print(f"\n✅ Model Saved: {MODEL_OUTPUT_FILE}")

if __name__ == '__main__':
    train_advanced_model()