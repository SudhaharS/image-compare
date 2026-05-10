import os
import base64
import httpx
import sys
import uuid
import json
import math
from datetime import datetime

# Configuration
SERVICE_URL = os.environ.get("SERVICE_URL", "https://gemini-embedding-api-335234379342.us-central1.run.app")
ENDPOINT = f"{SERVICE_URL.rstrip('/')}/embed_image"
OUTPUT_DIR = "E:\ImageComparator\output"
SIMILARITY_THRESHOLD = 0.90  # Adjust this: 1.0 is exact, 0.95-0.98 is "near duplicate"

def calculate_cosine_similarity(v1, v2):
    """Calculates the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude1 = math.sqrt(sum(a * a for a in v1))
    magnitude2 = math.sqrt(sum(b * b for b in v2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def check_for_duplicates(new_vector):
    """Compares the new vector against all stored JSON files in the output folder."""
    if not os.path.exists(OUTPUT_DIR):
        return None

    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(OUTPUT_DIR, filename)
            try:
                with open(file_path, "r") as f:
                    stored_vector = json.load(f)
                
                similarity = calculate_cosine_similarity(new_vector, stored_vector)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    return filename, similarity
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")
    return None

def save_vector_to_file(vector, original_image_name):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    base_name = os.path.basename(original_image_name).split('.')[0]
    filename = f"vector_{base_name}_{timestamp}_{unique_id}.json"
    file_path = os.path.join(OUTPUT_DIR, filename)

    with open(file_path, "w") as f:
        json.dump(vector, f)
    print(f"✅ Unique image detected. Vector saved to: {file_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <path_to_image>")
        return

    image_path = sys.argv[1]
    
    # 1. Load and Encode Image
    with open(image_path, "rb") as f:
        b64_str = base64.b64encode(f.read()).decode('utf-8')

    print(f"Processing: {os.path.basename(image_path)}...")
    
    try:
        # 2. Get Vector from Cloud Run
        response = httpx.post(ENDPOINT, json={"image_base64": b64_str}, timeout=60.0)
        if response.status_code != 200:
            print(f"Error: {response.text}")
            return
            
        new_vector = response.json().get("vector", [])

        # 3. Duplicate Check Logic
        duplicate_info = check_for_duplicates(new_vector)

        if duplicate_info:
            existing_file, score = duplicate_info
            print(f"⚠️  POSSIBLE DUPLICATE DETECTED!")
            print(f"Matches existing entry: {existing_file}")
            print(f"Similarity Score: {score:.4f}")
            print("Action: File was NOT written to the output folder.")
        else:
            save_vector_to_file(new_vector, image_path)
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()