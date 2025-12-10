import pandas as pd
import numpy as np
import json
import math
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from skimage.draw import polygon
from skimage.measure import regionprops, label

# --- CONFIGURATION ---
BASE_DIR = Path(r'C:\Users\adamv\ETL')
DATA_DIR = BASE_DIR / 'Data'
MASTER_OUTPUT_FILE = DATA_DIR / 'master_final_dataset.csv'
GROUND_TRUTH_FILE = DATA_DIR / 'high_confidence_ground_truth.csv'

SCAN_PREFIXES = [
    '20250409_5_Glut_100uM',
    '20250409_1_ATP_100uM',
    '20250409_3_Glut_1mM',
]

def extract_calcium_features(trace):
    trace = np.array(trace)
    if len(trace) == 0: return {}
    
    std_dev = np.std(trace)
    mean_val = np.mean(trace)
    # Event Detection: 3x Std Dev above mean
    peaks, _ = find_peaks(trace, height=mean_val + (3 * std_dev))
    
    return {
        "ca_std": std_dev,
        "ca_max": np.max(trace),
        "ca_skew": skew(trace),
        "ca_kurtosis": kurtosis(trace),
        "ca_num_peaks": len(peaks),
        "ca_energy": np.sum(trace**2)
    }

def extract_morphology_features(segmentation_list, width, height):
    if not segmentation_list: return {}
    
    poly_flat = segmentation_list[0]
    if len(poly_flat) < 6: return {}
    
    x_points = poly_flat[0::2]
    y_points = poly_flat[1::2]
    max_x, max_y = int(max(x_points)), int(max(y_points))
    canvas_shape = (max_y + 5, max_x + 5)
    
    try:
        rr, cc = polygon(y_points, x_points, shape=canvas_shape)
        mask = np.zeros(canvas_shape, dtype=np.uint8)
        mask[rr, cc] = 1
        
        props_list = regionprops(label(mask))
        if not props_list: return {}
        props = props_list[0]

        perimeter = props.perimeter
        area = props.area
        # Circularity: 4*pi*Area / Perimeter^2
        circularity = (4 * math.pi * area) / (perimeter**2) if perimeter > 0 else 0

        return {
            "circularity": circularity,
            "solidity": props.solidity,
            "eccentricity": props.eccentricity,
            "extent": props.extent,
            "major_axis": props.major_axis_length,
            "minor_axis": props.minor_axis_length,
            "orientation": props.orientation
        }
    except Exception:
        return {}

def process_scan_final(prefix):
    print(f"-> Processing: {prefix}...")
    
    # 1. Find Files
    coco_files = list(DATA_DIR.glob(f"{prefix}*coco.json"))
    calc_files = list(DATA_DIR.glob(f"{prefix}*estimated_calcium.json"))
    
    if not coco_files or not calc_files:
        print(f"   ❌ Missing files for {prefix}. Skipping.")
        return pd.DataFrame()
        
    with open(coco_files[0], 'r') as f: coco_data = json.load(f)
    with open(calc_files[0], 'r') as f: calc_data = json.load(f)
    
    annotations = coco_data.get('annotations', [])
    extracted_rows = []
    
    # Debug Counters
    found_trace_count = 0
    missing_trace_count = 0

    for ann in annotations:
        coco_id = ann.get('id') # Integer, e.g. 1
        
        # --- THE FIX ---
        # 1. Convert COCO ID (1-based) to CaImAn Index (0-based)
        caiman_index = coco_id - 1
        
        # 2. Construct the Key (e.g., "component_0")
        target_key = f"component_{caiman_index}"
        
        # 3. Lookup
        calcium_trace = calc_data.get(target_key, [])
        
        # 4. Check & Extract
        if calcium_trace and len(calcium_trace) > 10:
            found_trace_count += 1
            calcium_feats = extract_calcium_features(calcium_trace)
        else:
            missing_trace_count += 1
            # Fill with zeros so we don't break the DataFrame, but mark as empty
            calcium_feats = {
                "ca_std": 0, "ca_max": 0, "ca_skew": 0, 
                "ca_kurtosis": 0, "ca_num_peaks": 0, "ca_energy": 0
            }

        # 5. Morphology
        bbox = ann.get('bbox', [0,0,0,0])
        morph_feats = extract_morphology_features(ann.get('segmentation', []), bbox[2], bbox[3])
        
        # 6. Build Row
        row = {
            'roi_id_index': coco_id,
            'scan_prefix': prefix,
            'Full_ROI_ID': f"{prefix}_component_{caiman_index}", # Matches Ground Truth ID format
            'area_px': ann.get('area', 0),
            'bbox_w': bbox[2],
            'bbox_h': bbox[3],
            'aspect_ratio': bbox[2]/bbox[3] if bbox[3]>0 else 0,
            **morph_feats,
            **calcium_feats
        }
        extracted_rows.append(row)

    print(f"   📊 Trace Match Report: Found {found_trace_count} | Missing {missing_trace_count}")
    if found_trace_count == 0:
        print(f"      ⚠️ WARNING: Check if calcium file keys really look like '{target_key}'")

    df = pd.DataFrame(extracted_rows)
    return df.fillna(0)

def build_final_dataset():
    print("="*60)
    print("🚀 RE-BUILDING DATASET WITH 'component_X' KEYS")
    print("="*60)

    if not GROUND_TRUTH_FILE.exists():
        print("❌ Ground Truth file missing.")
        return
        
    df_labels = pd.read_csv(GROUND_TRUTH_FILE)
    if 'Full_ROI_ID' in df_labels.columns:
        df_labels['Full_ROI_ID'] = df_labels['Full_ROI_ID'].str.strip()

    all_scans = []
    for prefix in SCAN_PREFIXES:
        df_scan = process_scan_final(prefix)
        if not df_scan.empty:
            all_scans.append(df_scan)

    if not all_scans:
        print("❌ No data processed.")
        return

    df_features = pd.concat(all_scans, ignore_index=True)
    
    # Inner Merge with Ground Truth
    df_master = pd.merge(df_labels, df_features, on='Full_ROI_ID', how='inner')
    
    # Output Columns
    final_cols = [
        'Full_ROI_ID', 'Ground_Truth_Label', 'Confidence',
        'ca_std', 'ca_skew', 'ca_num_peaks', 'ca_energy', 'ca_kurtosis',
        'circularity', 'solidity', 'area_px', 'aspect_ratio', 'extent', 'eccentricity'
    ]
    # Filter for columns that actually exist
    valid_cols = [c for c in final_cols if c in df_master.columns]
    
    df_master[valid_cols].to_csv(MASTER_OUTPUT_FILE, index=False)
    
    print("\n" + "="*60)
    print(f"✅ DONE. Saved to {MASTER_OUTPUT_FILE}")
    print(f"   Total Training Samples: {len(df_master)}")
    print("="*60)

if __name__ == '__main__':
    build_final_dataset()