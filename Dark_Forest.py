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
INPUT_FILE = DATA_DIR / 'master_final_dataset.csv'
MODEL_OUTPUT_FILE = DATA_DIR / 'final_cell_classifier.pkl'

# --- THE FULL FEATURE SET ---
FEATURE_COLS = [
    # 1. Calcium Dynamics (Behavior)
    'ca_std',          # Activity
    'ca_skew',         # Spiking vs Noise
    'ca_num_peaks',    # Event Rate
    'ca_energy',       # Total Output
    'ca_kurtosis',     # Spike Sharpness
    
    # 2. Morphology (Structure)
    'circularity',     # Round vs Spidery
    'solidity',        # Dense vs Branching
    'area_px',         # Size
    'aspect_ratio',    # Elongation
    'eccentricity',    # Linear-ness
    'extent'           # Box Fill
]
TARGET_COL = 'Ground_Truth_Label'

def train_final_model():
    print("="*60)
    print("🌲 TRAINING FINAL RANDOM FOREST (Bio-Unified)")
    print("="*60)

    if not INPUT_FILE.exists():
        print("❌ Dataset not found. Run All_Together_Final.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Clean data (remove failed calculations)
    df = df.fillna(0)
    
    print(f"Loaded {len(df)} cells.")
    print("Class Distribution:\n", df[TARGET_COL].value_counts())

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train (More trees = better stability for mixed features)
    clf = RandomForestClassifier(
        n_estimators=500, 
        max_depth=20, 
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    
    print("\n" + "-"*40)
    print(f"ACCURACY: {accuracy_score(y_test, y_pred):.2%}")
    print("-" * 40)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature Importance Plot
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis')
    plt.title('What Drives the Classification?')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'feature_importance_final.png')
    print(f"\n📊 Feature Importance saved to {DATA_DIR / 'feature_importance_final.png'}")
    print(importances.head(10))

    # Save Model
    joblib.dump(clf, MODEL_OUTPUT_FILE)
    print(f"\n✅ Final Model Saved: {MODEL_OUTPUT_FILE}")

if __name__ == '__main__':
    train_final_model()