import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

# --- CONFIG ---
warnings.simplefilter(action='ignore')
BASE_DIR = Path(r'C:\Users\adamv\ETL')
DATA_DIR = BASE_DIR / 'Data'
INPUT_FILE = DATA_DIR / 'master_final_dataset.csv'
REPORT_DIR = DATA_DIR / 'Report_Images_Ensemble'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    'ca_energy', 'ca_std', 'ca_skew', 'ca_kurtosis', 'ca_num_peaks', 
    'circularity', 'solidity', 'area_px', 'aspect_ratio', 'extent'
]
TARGET_COL = 'Ground_Truth_Label'

def engineer_interaction_features(df):
    """Avenue 1: Create 'Physics' Features (Density, Regularity)"""
    # 1. Energy Density (Activity per pixel)
    # Avoid division by zero
    df['energy_density'] = df['ca_energy'] / (df['area_px'] + 1)
    
    # 2. Spike Density (Events per size)
    df['spike_density'] = df['ca_num_peaks'] / (df['area_px'] + 1)
    
    # 3. Shape Regularity (Combined roundness and solidity)
    df['shape_regularity'] = df['circularity'] * df['solidity']
    
    # Add new features to the list
    new_features = FEATURE_COLS + ['energy_density', 'spike_density', 'shape_regularity']
    return df, new_features

def run_ensemble_pipeline():
    print("="*60)
    print("⚖️ RUNNING VOTING ENSEMBLE (RF + SVM + Logistic)")
    print("="*60)

    # 1. Load
    if not INPUT_FILE.exists():
        print(f"❌ Error: {INPUT_FILE} not found.")
        return
    
    df = pd.read_csv(INPUT_FILE).fillna(0)
    valid_classes = ['OPC', 'Oli', 'Astro']
    df = df[df[TARGET_COL].isin(valid_classes)].copy()
    
    # 2. Feature Engineering
    df, full_features = engineer_interaction_features(df)
    print(f"Added Derived Features. Total Features: {len(full_features)}")

    X = df[full_features]
    y = df[TARGET_COL]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # 4. Scaling (Crucial for SVM/Logistic, not for RF)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # --- DEFINE THE COUNCIL ---
    
    # Member 1: The Balanced Forest (Good at Behavior)
    clf_rf = BalancedRandomForestClassifier(
        n_estimators=300, 
        sampling_strategy='all', 
        replacement=True,
        random_state=42
    )

    # Member 2: Weighted SVM (Good at Geometry/Shape)
    clf_svm = SVC(
        kernel='rbf', 
        class_weight='balanced', 
        probability=True, 
        random_state=42
    )

    # Member 3: Logistic Regression (The Baseline Anchor)
    clf_lr = LogisticRegression(
        class_weight='balanced', 
        max_iter=1000, 
        random_state=42
    )

    # The Voting Machine (Soft Voting = Average of probabilities)
    voting_clf = VotingClassifier(
        estimators=[
            ('balanced_rf', clf_rf), 
            ('svm', clf_svm), 
            ('log_reg', clf_lr)
        ],
        voting='soft',
        n_jobs=-1
    )

    print("\nTraining The Council...")
    # Note: SVM/LR need scaled data. RF doesn't care, but handles scaled data fine.
    voting_clf.fit(X_train_scaled, y_train)

    # --- EVALUATION ---
    print("Generating Consensus Predictions...")
    y_pred = voting_clf.predict(X_test_scaled)
    
    # Report
    print("\n" + "="*40)
    print("🏆 ENSEMBLE RESULTS")
    print("="*40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.1%}")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=voting_clf.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=voting_clf.classes_, yticklabels=voting_clf.classes_)
    plt.title('Ensemble Consensus Accuracy', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'Ensemble_Confusion_Matrix.png')
    print(f"📊 Matrix saved to {REPORT_DIR}")

    # --- CALCULATE PI SCORE ---
    # Indices helper
    classes = list(voting_clf.classes_)
    if 'Astro' in classes and 'Oli' in classes:
        idx_astro = classes.index('Astro')
        idx_oli = classes.index('Oli')
        
        astro_recall = cm[idx_astro, idx_astro] / cm[idx_astro, :].sum()
        oli_recall = cm[idx_oli, idx_oli] / cm[idx_oli, :].sum()
        pi_score = (astro_recall + oli_recall) / 2
        
        print("\n" + "*"*40)
        print(f"📈 RARE CELL DISCOVERY RATE: {pi_score:.1%}")
        print("*"*40)

if __name__ == '__main__':
    run_ensemble_pipeline()