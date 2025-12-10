import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from pathlib import Path

# --- CONFIG ---
warnings.simplefilter(action='ignore')
BASE_DIR = Path(r'C:\Users\adamv\ETL')
DATA_DIR = BASE_DIR / 'Data'
RESULTS_FILE = DATA_DIR / 'sequential_results_package.pkl'
REPORT_DIR = DATA_DIR / 'Report_Images_Sequential'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def visualize_sequential():
    print("="*60)
    print("📊 GENERATING PI REPORT (Sequential)")
    print("="*60)

    if not RESULTS_FILE.exists():
        print(f"❌ Error: {RESULTS_FILE} missing.")
        return
    
    data = joblib.load(RESULTS_FILE)
    
    classes = data['dataset']['classes'] # ['OPC', 'Astro', 'Oli']
    y_test = data['dataset']['y_test']
    y_pred = data['dataset']['y_pred']
    
    # --- 1. THE "SIMPLE NUMBER" FOR THE PI ---
    # We define success as "Rare Cell Detection Rate" (Recall of Oli + Astro)
    # This is more honest than "Accuracy" (which is dominated by OPCs)
    
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Calculate Recall per class
    # Format of CM: rows=True, cols=Predicted
    # Indices: OPC=0, Astro=1, Oli=2 (Based on alphabet sort usually, let's verify)
    # Actually, verify order:
    labels = classes # Explicitly ['OPC', 'Astro', 'Oli'] passed to cm
    
    # Astro Stats
    astro_tp = cm[1, 1]
    astro_total = np.sum(cm[1, :])
    astro_recall = astro_tp / astro_total if astro_total > 0 else 0
    
    # Oli Stats
    oli_tp = cm[2, 2]
    oli_total = np.sum(cm[2, :])
    oli_recall = oli_tp / oli_total if oli_total > 0 else 0
    
    # OPC Stats
    opc_tp = cm[0, 0]
    opc_total = np.sum(cm[0, :])
    opc_recall = opc_tp / opc_total if opc_total > 0 else 0

    # The PI Score: Average Recall of the Rare Classes
    discovery_score = (astro_recall + oli_recall) / 2
    overall_acc = accuracy_score(y_test, y_pred)

    print("\n" + "*"*40)
    print("📈 EXECUTIVE SUMMARY FOR PI")
    print("*"*40)
    print(f"Overall Model Accuracy:   {overall_acc:.1%}")
    print(f"Rare Cell Discovery Rate: {discovery_score:.1%} (The 'Simple Number')")
    print("-" * 30)
    print(f" > Astro Detection: {astro_recall:.1%} ({astro_tp}/{astro_total})")
    print(f" > Oli Detection:   {oli_recall:.1%} ({oli_tp}/{oli_total})")
    print(f" > OPC Accuracy:    {opc_recall:.1%}")
    print("*"*40)

    # --- GRAPH 1: CONFUSION MATRIX ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Sequential Model Accuracy', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / '1_Confusion_Matrix.png')
    plt.close()

    # --- GRAPH 2: DUAL ROC CURVES (The Specialists) ---
    plt.figure(figsize=(10, 8))
    
    # Astro ROC
    y_test_astro = (y_test == 'Astro').astype(int)
    probs_astro = data['probs']['astro']
    fpr_a, tpr_a, _ = roc_curve(y_test_astro, probs_astro)
    auc_a = auc(fpr_a, tpr_a)
    
    # Oli ROC
    # Note: For Oli ROC validation, we look at Oli vs Rest in the global set for simplicity
    y_test_oli = (y_test == 'Oli').astype(int)
    probs_oli = data['probs']['oli']
    fpr_o, tpr_o, _ = roc_curve(y_test_oli, probs_oli)
    auc_o = auc(fpr_o, tpr_o)

    plt.plot(fpr_a, tpr_a, lw=3, color='purple', label=f'Step 1: Astro Hunter (AUC={auc_a:.2f})')
    plt.plot(fpr_o, tpr_o, lw=3, color='orange', label=f'Step 2: Oli Hunter (AUC={auc_o:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Performance of Sequential Specialists', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(REPORT_DIR / '2_Specialist_ROC.png')
    plt.close()

    # --- GRAPH 3: FEATURE COMPARISON ---
    astro_imp = data['models']['astro_model'].feature_importances_
    oli_imp = data['models']['oli_model'].feature_importances_
    names = data['models']['feature_names']
    
    df_imp = pd.DataFrame({'Feature': names, 'Astro': astro_imp, 'Oli': oli_imp})
    df_melt = df_imp.melt(id_vars='Feature', var_name='Model', value_name='Importance')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x='Feature', y='Importance', hue='Model', 
                palette={'Astro': 'purple', 'Oli': 'orange'})
    plt.title('What Drives the Decision? (Feature Importance)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / '3_Feature_Comparison.png')
    plt.close()

    print(f"\n✅ Charts saved to: {REPORT_DIR}")

if __name__ == '__main__':
    visualize_sequential()