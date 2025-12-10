import pandas as pd
import numpy as np
import json
import math
from pathlib import Path
from skimage.draw import polygon
from skimage.measure import regionprops, label

# --- CONFIGURATION ---
BASE_DIR = Path(r'C:\Users\adamv\ETL')
DATA_DIR = BASE_DIR / 'Data'
MASTER_OUTPUT_FILE = DATA_DIR / 'master_advanced_morphology.csv'
GROUND_TRUTH_FILE = DATA_DIR / 'high_confidence_ground_truth.csv'

SCAN_PREFIXES = [
    '20250409_5_Glut_100uM',
    '20250409_1_ATP_100uM',
    '20250409_3_Glut_1mM',
]

def calculate_advanced_features(segmentation_list, width, height):
    """
    Converts COCO polygon points into advanced shape metrics.
    """
    if not segmentation_list:
        return {}

    # 1. Flatten and Reshape Polygon
    # COCO stores as [x1, y1, x2, y2...] -> We need two lists: rows (y), cols (x)
    poly_flat = segmentation_list[0] # Take first polygon
    if len(poly_flat) < 6: return {} # Need at least 3 points for a triangle
    
    x_points = poly_flat[0::2]
    y_points = poly_flat[1::2]
    
    # 2. Create Binary Mask
    # We create a blank image (canvas) slightly larger than the cell to draw the shape
    max_x, max_y = int(max(x_points)), int(max(y_points))
    canvas_shape = (max_y + 5, max_x + 5)
    
    try:
        # Draw the polygon on the canvas
        rr, cc = polygon(y_points, x_points, shape=canvas_shape)
        mask = np.zeros(canvas_shape, dtype=np.uint8)
        mask[rr, cc] = 1
    except IndexError:
        return {} # Polygon out of bounds

    # 3. Calculate Properties using RegionProps
    # 'label(mask)' groups connected pixels (should be just 1 cell)
    props_list = regionprops(label(mask))
    if not props_list: return {}
    
    props = props_list[0] # Get the first (and likely only) region

    # 4. Extract Features
    # Circularity formula: (4 * pi * Area) / Perimeter^2
    perimeter = props.perimeter
    area = props.area
    circularity = (4 * math.pi * area) / (perimeter**2) if perimeter > 0 else 0

    return {
        "circularity": circularity,       # 1.0 = Perfect Circle, < 0.5 = Spidery/Complex
        "solidity": props.solidity,       # Area / Convex Hull Area (Density)
        "eccentricity": props.eccentricity, # 0 = Circle, 1 = Line segment
        "extent": props.extent,           # Area / Bounding Box Area
        "major_axis": props.major_axis_length,
        "minor_axis": props.minor_axis_length,
        "orientation": props.orientation
    }

def process_scan_advanced(prefix):
    print(f"-> Processing: {prefix}...")
    
    # Find JSON file (handling TIF/TIFF)
    candidates = list(DATA_DIR.glob(f"{prefix}*coco.json"))
    if not candidates:
        print(f"   ❌ JSON not found for {prefix}")
        return pd.DataFrame()
    
    json_path = candidates[0]
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    annotations = coco_data.get('annotations', [])
    if not annotations: return pd.DataFrame()

    extracted_rows = []
    
    for ann in annotations:
        # Basic Features
        roi_id = ann.get('id')
        bbox = ann.get('bbox', [0, 0, 0, 0])
        seg = ann.get('segmentation', [])
        
        # Advanced Calculation
        adv_feats = calculate_advanced_features(seg, bbox[2], bbox[3])
        
        # Merge
        row = {
            'roi_id_index': roi_id,
            'scan_prefix': prefix,
            'area_px': ann.get('area', 0),
            'bbox_w': bbox[2],
            'bbox_h': bbox[3],
            'cx': bbox[0] + (bbox[2]/2),
            'cy': bbox[1] + (bbox[3]/2),
            **adv_feats # Unpack dictionary of advanced features
        }
        extracted_rows.append(row)

    df = pd.DataFrame(extracted_rows)
    
    # Create Join Key
    if not df.empty:
        df['Full_ROI_ID'] = df['scan_prefix'] + '_component_' + df['roi_id_index'].astype(str)
        # Recalculate Aspect Ratio here for consistency
        df['aspect_ratio'] = np.where(df['bbox_h'] > 0, df['bbox_w'] / df['bbox_h'], 0)
        
        # Fill NaNs (for failed polygon calculations) with 0
        df = df.fillna(0)

    return df

def build_advanced_dataset():
    print("="*60)
    print("🚀 EXTRACTING ADVANCED MORPHOLOGY (POLYGONS)")
    print("="*60)

    # 1. Load Ground Truth
    if not GROUND_TRUTH_FILE.exists():
        print("❌ Ground Truth file missing.")
        return
    df_labels = pd.read_csv(GROUND_TRUTH_FILE)
    if 'Full_ROI_ID' in df_labels.columns:
        df_labels['Full_ROI_ID'] = df_labels['Full_ROI_ID'].str.strip()

    # 2. Process All Scans
    all_scans = []
    for prefix in SCAN_PREFIXES:
        df_scan = process_scan_advanced(prefix)
        if not df_scan.empty:
            all_scans.append(df_scan)
            print(f"   ✅ {prefix}: Extracted {len(df_scan)} polygons.")

    if not all_scans: return

    df_features = pd.concat(all_scans, ignore_index=True)

    # 3. Merge
    df_master = pd.merge(df_labels, df_features, on='Full_ROI_ID', how='inner')

    # 4. Save
    output_cols = [
        'Full_ROI_ID', 'Ground_Truth_Label', 'Confidence',
        'circularity', 'solidity', 'eccentricity', 'extent', # The New Power Features
        'area_px', 'aspect_ratio', 'major_axis', 'minor_axis', 'orientation'
    ]
    # Filter to keep only columns that exist (in case calculation failed entirely)
    final_cols = [c for c in output_cols if c in df_master.columns]
    
    df_master[final_cols].to_csv(MASTER_OUTPUT_FILE, index=False)
    
    print("\n" + "="*60)
    print(f"✨ DONE. Advanced Dataset Saved: {MASTER_OUTPUT_FILE}")
    print(f"   Samples: {len(df_master)}")
    print("="*60)
    print(df_master[['Ground_Truth_Label', 'circularity', 'solidity']].head())

if __name__ == '__main__':
    build_advanced_dataset()