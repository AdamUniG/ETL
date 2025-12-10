import firebase_admin
from firebase_admin import credentials, firestore
import json
import os

# --- CONFIGURATION ---
CREDENTIALS_FILE = 'serviceAccountKey.json' 
OUTPUT_FILE = 'raw_biologist_labels.json'
COLLECTION_NAME = 'labels_master'

def export_raw_labels():
    # 1. Initialize Firebase
    if not os.path.exists(CREDENTIALS_FILE):
        print(f"ERROR: Could not find {CREDENTIALS_FILE}")
        print("Please download your Service Account Key and ensure it is named 'serviceAccountKey.json'.")
        return

    try:
        cred = credentials.Certificate(CREDENTIALS_FILE)
        # Only initialize if not already done
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
    except Exception as e:
        print(f"ERROR initializing Firebase: {e}")
        return

    print(f"Fetching ALL raw documents from collection: '{COLLECTION_NAME}'...")
    
    # We will collect a list of dictionaries, one for every single submission
    raw_labels_list = []

    # 2. Query Data - Fetch ALL documents
    docs = db.collection(COLLECTION_NAME).stream()

    for doc in docs:
        data = doc.to_dict()
        
        # Ensure we have the minimum required fields
        if 'picture_id' in data and 'category' in data and 'user_id' in data:
            raw_labels_list.append({
                "roi_id_full": data['picture_id'], # e.g., "20250409_5_Glut_100uM_component_74"
                "user_id": data['user_id'],
                "label": data['category'],         # e.g., "Multiple" or "OPC"
                "created_at": data.get('created_at', None) # Keep timestamp for potential sorting
            })

    # 3. Export to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(raw_labels_list, f, indent=4, default=str) # default=str handles Firestore Timestamps

    print("=" * 60)
    print(f"✅ Export Complete: Exported {len(raw_labels_list)} raw label submissions to '{OUTPUT_FILE}'")
    print("=" * 60)

if __name__ == '__main__':
    export_raw_labels()