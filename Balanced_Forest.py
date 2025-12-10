import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pathlib import Path

# --- CONFIG ---
warnings.simplefilter(action='ignore')
BASE_DIR = Path(r'C:\Users\adamv\ETL')
DATA_DIR = BASE_DIR / 'Data'
INPUT_FILE = DATA_DIR / 'master_final_dataset.csv'
RESULTS_FILE = DATA_DIR / 'sequential_results_package.pkl'

FEATURE_COLS = [
    'ca_energy', 'ca_std', 'ca_skew', 'ca_kurtosis', 'ca_num_peaks', 
    'circularity', 'solidity', 'area_px', 'aspect_ratio', 'extent'
]
TARGET_COL = 'Ground_Truth_Label'

def run_sequential_pipeline():
    print("="*60)
    print("🧬 RUNNING SEQUENTIAL CASCADE (Astro -> Oli/OPC)")
    print("="*60)

    # 1. Load Data
    if not INPUT_FILE.exists():
        print(f"❌ Error: {INPUT_FILE} not found.")
        return
    
    df = pd.read_csv(INPUT_FILE).fillna(0)
    # Filter for valid classes only
    df = df[df[TARGET_COL].isin(['OPC', 'Oli', 'Astro'])].copy()
    
    # 2. Global Split (Must keep test set pure for final evaluation)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    print(f"Total Training Cells: {len(X_train_full)}")
    print(f"Total Test Cells:     {len(X_test)}")

    # --- STEP 1: TRAIN ASTRO HUNTER ---
    # Target: Is it Astro (1) or Glia (0)?
    print("\n1️⃣ Training Stage 1: The Astrocyte Hunter...")
    
    # Create binary labels for Stage 1
    y_train_astro = y_train_full.apply(lambda x: 1 if x == 'Astro' else 0)
    
    clf_astro = RandomForestClassifier(
        n_estimators=300, 
        max_depth=10, 
        class_weight='balanced', # Crucial for rare Astros
        random_state=42, 
        n_jobs=-1
    )
    clf_astro.fit(X_train_full, y_train_astro)

    # --- STEP 2: TRAIN OLI/OPC DIFFERENTIATOR ---
    # Filter training data: Remove all Astros to focus Model B
    print("2️⃣ Training Stage 2: The Differentiation Specialist (Oli vs OPC)...")
    
    mask_glia = y_train_full != 'Astro'
    X_train_glia = X_train_full[mask_glia]
    y_train_glia = y_train_full[mask_glia] # Contains only 'OPC' and 'Oli'
    
    # Map to Binary: Oli=1, OPC=0
    y_train_oli_binary = y_train_glia.apply(lambda x: 1 if x == 'Oli' else 0)
    
    clf_oli = RandomForestClassifier(
        n_estimators=300, 
        max_depth=10, 
        class_weight='balanced', # Crucial for rare Olis
        random_state=42, 
        n_jobs=-1
    )
    clf_oli.fit(X_train_glia, y_train_oli_binary)

    # --- INFERENCE CASCADE ---
    print("\n⚙️ Running Cascade on Test Data...")
    
    # 1. Get Astro Probabilities
    prob_astro = clf_astro.predict_proba(X_test)[:, 1]
    
    # 2. Get Oli Probabilities (Model B runs on EVERYTHING, we filter later)
    prob_oli = clf_oli.predict_proba(X_test)[:, 1]
    
    final_preds = []
    
    # CASCADE LOGIC
    # Tunable thresholds
    THRESH_ASTRO = 0.40  # Lower = Catch more Astros
    THRESH_OLI = 0.45    # Lower = Catch more Olis
    
    for p_astro, p_oli in zip(prob_astro, prob_oli):
        # Step A: Is it an Astro?
        if p_astro >= THRESH_ASTRO:
            final_preds.append("Astro")
        # Step B: If not, is it an Oli?
        elif p_oli >= THRESH_OLI:
            final_preds.append("Oli")
        # Step C: If neither, it's an OPC
        else:
            final_preds.append("OPC")

    # --- PACKAGING ---
    results = {
        "dataset": {
            "classes": ['OPC', 'Astro', 'Oli'],
            "y_test": y_test,
            "y_pred": np.array(final_preds)
        },
        "metrics": {
            "report": classification_report(y_test, final_preds, zero_division=0)
        },
        "models": {
            "astro_model": clf_astro,
            "oli_model": clf_oli,
            "feature_names": FEATURE_COLS
        },
        # Save probs for ROC curves
        "probs": {
            "astro": prob_astro,
            "oli": prob_oli
        }
    }
    
    joblib.dump(results, RESULTS_FILE)
    print("\n" + "="*60)
    print(f"✅ DONE. Results saved to: {RESULTS_FILE.name}")
    print("   Run 'Visuals_Sequential.py' to generate the PI Report.")
    print("="*60)

if __name__ == '__main__':
    run_sequential_pipeline()