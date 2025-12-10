import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(r'C:\Users\adamv\ETL')
DATA_DIR = BASE_DIR / 'Data'
MASTER_OUTPUT_FILE = DATA_DIR / 'master_morphology_features_poc.csv'
GROUND_TRUTH_FILE = DATA_DIR / 'high_confidence_ground_truth.csv'

# Scan prefixes to process
SCAN_PREFIXES = [
    '20250409_5_Glut_100uM',
    '20250409_1_ATP_100uM',
    '20250409_3_Glut_1mM',
]

def load_spatial_from_coco_json(prefix):
    """
    Parses a COCO format JSON file to extract spatial features.
    Handles both 'TIF' and 'TIFF' naming conventions.
    """
    print(f"-> Processing Scan: {prefix}...")
    
    # 1. Find the file (Handling spelling differences)
    # We look for files that start with the prefix and end in 'coco.json'
    # This handles both ..._TIF_VIDEO_coco.json and ..._TIFF_VIDEO_coco.json
    candidates = list(DATA_DIR.glob(f"{prefix}*coco.json"))
    
    if not candidates:
        print(f"   ❌ ERROR: No COCO JSON found for prefix: {prefix}")
        return pd.DataFrame()
    
    json_path = candidates[0]
    print(f"   📂 Loading JSON: {json_path.name}")

    # 2. Parse the JSON
    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"   ❌ JSON Read Error: {e}")
        return pd.DataFrame()

    annotations = coco_data.get('annotations', [])
    if not annotations:
        print("   ⚠️ JSON has no annotations. Skipping.")
        return pd.DataFrame()

    # 3. Extract Features
    # COCO structure: 'id', 'area', 'bbox': [x, y, width, height]
    extracted_rows = []
    
    for ann in annotations:
        roi_id = ann.get('id')
        bbox = ann.get('bbox', [0, 0, 0, 0])
        width = bbox[2]
        height = bbox[3]
        
        extracted_rows.append({
            'roi_id_index': roi_id,
            'area_px': ann.get('area', 0),
            'bbox_w': width,
            'bbox_h': height,
            # Centroid approximation (center of bbox)
            'cx': bbox[0] + (width / 2),
            'cy': bbox[1] + (height / 2),
            'scan_prefix': prefix
        })

    df = pd.DataFrame(extracted_rows)
    
    # 4. Final Feature Engineering
    if not df.empty:
        # Construct the Join Key
        # Note: We cast roi_id_index to string to match the Ground Truth format
        df['Full_ROI_ID'] = df['scan_prefix'] + '_component_' + df['roi_id_index'].astype(str)
        
        # Calculate Aspect Ratio
        df['aspect_ratio'] = np.where(
            df['bbox_h'] > 0, 
            df['bbox_w'] / df['bbox_h'], 
            0.0
        )
        
        # Contour threshold is often missing in COCO, set to 0 to prevent errors
        df['contour_thr'] = 0.0

    return df

def build_master_dataset():
    print("="*60)
    print("🚀 STARTING JSON-ONLY MORPHOLOGY MERGE")
    print(f"📂 Data Folder: {DATA_DIR}")
    print("="*60)

    # 1. Load Ground Truth
    if not GROUND_TRUTH_FILE.exists():
        print(f"❌ FATAL: Ground Truth not found at {GROUND_TRUTH_FILE}")
        return

    df_labels = pd.read_csv(GROUND_TRUTH_FILE)
    # Clean whitespace from IDs just in case
    if 'Full_ROI_ID' in df_labels.columns:
        df_labels['Full_ROI_ID'] = df_labels['Full_ROI_ID'].str.strip()
        
    print(f"✅ Loaded Ground Truth: {len(df_labels)} cells.")

    # 2. Process all scans from JSONs
    all_scans = []
    for prefix in SCAN_PREFIXES:
        df_scan = load_spatial_from_coco_json(prefix)
        if not df_scan.empty:
            all_scans.append(df_scan)
            print(f"   - Extracted {len(df_scan)} ROIs.")

    if not all_scans:
        print("❌ No data extracted from any JSONs. Stopping.")
        return

    df_all_features = pd.concat(all_scans, ignore_index=True)
    print("-" * 60)
    print(f"📊 Total JSON Features Extracted: {len(df_all_features)}")

    # 3. Master Merge
    # Inner join filters for only the cells that passed the biologists' veto/consensus
    df_master = pd.merge(
        df_labels, 
        df_all_features, 
        on='Full_ROI_ID', 
        how='inner'
    )

    # 4. Save and Report
    if df_master.empty:
        print("⚠️ WARNING: Merge Result is Empty!")
        print("   Possibility: ID Mismatch.")
        print(f"   Example Label ID: {df_labels['Full_ROI_ID'].iloc[0]}")
        print(f"   Example JSON ID:  {df_all_features['Full_ROI_ID'].iloc[0]}")
    else:
        # Reorder columns for cleanliness
        cols = [
            'Full_ROI_ID', 'Ground_Truth_Label', 'Confidence', 'Source',
            'scan_prefix', 'area_px', 'aspect_ratio', 'bbox_w', 'bbox_h', 'cx', 'cy'
        ]
        # Only keep columns that actually exist
        final_cols = [c for c in cols if c in df_master.columns]
        
        df_master[final_cols].to_csv(MASTER_OUTPUT_FILE, index=False)
        
        print("\n" + "="*60)
        print("✨ SUCCESS! JSON Morphology Dataset Created ✨")
        print(f"📁 Output: {MASTER_OUTPUT_FILE}")
        print(f"🔢 Training Samples: {len(df_master)}")
        print("="*60)
        print(df_master.head())

if __name__ == '__main__':
    build_master_dataset()