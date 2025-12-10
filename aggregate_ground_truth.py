import pandas as pd
import json
from collections import Counter
import numpy as np

# --- CONFIGURATION & BUSINESS RULES ---
RAW_LABELS_FILE = 'raw_biologist_labels.json' 
FINAL_APPROVED_FILE = 'high_confidence_ground_truth.csv'

VETO_USER_ID = 'Inbar'
MIN_TOTAL_INPUTS = 4
MIN_CONSENSUS_PERCENT = 0.74 # 75% or higher
VALID_CELL_TYPES = ['Oli', 'OPC', 'Astro'] 

def build_high_confidence_ground_truth():
    # 1. Load Raw Labels
    try:
        with open(RAW_LABELS_FILE, 'r') as f:
            raw_labels = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Raw label file '{RAW_LABELS_FILE}' not found. Run the Firebase export first.")
        return
    
    df_raw = pd.DataFrame(raw_labels)
    
    # Ensure 'created_at' is datetime for sorting the VETO user's last submission
    # Handle cases where created_at might be missing or not a standard format
    df_raw['created_at'] = pd.to_datetime(df_raw['created_at'], errors='coerce')
    
    # 2. Filter Invalid Labels (Rule 2)
    # Only keep the definitive cell types, removing Multiple, Unknown, Nothing, etc.
    df_filtered = df_raw[df_raw['label'].isin(VALID_CELL_TYPES)].copy()
    
    # 3. Aggregation Logic Function (Rule 3)
    def apply_confidence_rules(group):
        roi_id = group['roi_id_full'].iloc[0]

        # --- RULE A: VETO LOGIC ---
        if VETO_USER_ID in group['user_id'].values:
            # Find Inbar's latest, valid label
            inbar_submissions = group[group['user_id'] == VETO_USER_ID].sort_values(by='created_at', ascending=False)
            
            # Use the latest valid label from the VETO user.
            veto_label = inbar_submissions['label'].iloc[0]
            
            return pd.Series({
                'Ground_Truth_Label': veto_label,
                'Confidence': 1.0, # Veto is 100% confidence by definition
                'Source': f'VETO:{VETO_USER_ID}'
            })

        # --- RULE B: CONSENSUS LOGIC ---
        total_votes = len(group)
        
        if total_votes < MIN_TOTAL_INPUTS:
            return pd.Series({
                'Ground_Truth_Label': 'Rejected_Low_Votes',
                'Confidence': 0.0,
                'Source': f'REJECTED_Votes<{MIN_TOTAL_INPUTS}'
            })

        # Count votes for the remaining VALID_CELL_TYPES
        counts = group['label'].value_counts()
        winning_label = counts.index[0]
        winning_percent = counts.iloc[0] / total_votes
        
        if winning_percent >= MIN_CONSENSUS_PERCENT:
            return pd.Series({
                'Ground_Truth_Label': winning_label,
                'Confidence': round(winning_percent, 4),
                'Source': f'CONSENSUS:{counts.iloc[0]}/{total_votes}'
            })
        else:
            return pd.Series({
                'Ground_Truth_Label': 'Rejected_Low_Consensus',
                'Confidence': round(winning_percent, 4),
                'Source': f'REJECTED_Consensus<{MIN_CONSENSUS_PERCENT*100}%'
            })

    # Group by the full ROI ID (Rule 1) and apply the rules
    df_results = df_filtered.groupby('roi_id_full').apply(apply_confidence_rules).reset_index()

    # 4. Final Filter and Save
    df_approved = df_results[~df_results['Ground_Truth_Label'].str.startswith('Rejected')].copy()
    
    # Final cleanup of columns and renaming
    df_output = df_approved.rename(columns={'roi_id_full': 'Full_ROI_ID'})
    
    # Select columns for the final training file
    df_output = df_output[['Full_ROI_ID', 'Ground_Truth_Label', 'Confidence', 'Source']]
    
    df_output.to_csv(FINAL_APPROVED_FILE, index=False)
    
    # Report on final dataset statistics
    print("-" * 60)
    print("✅ High-Confidence Ground Truth Generated")
    print(f"Total ROIs Analyzed: {df_raw['roi_id_full'].nunique()}")
    print(f"Total Approved ROIs for Training: {len(df_output)}")
    print(f"Approved Label Distribution:")
    print(df_output['Ground_Truth_Label'].value_counts())
    print("-" * 60)

if __name__ == '__main__':
    # NOTE: You MUST run the Firebase export script first to create 'raw_biologist_labels.json'
    build_high_confidence_ground_truth()