import requests
import os
import glob
import time

def verify_metadata_generation():
    url = "http://localhost:8000/v1/images/generations"
    payload = {
        "prompt": "test metadata generation",
        "store_local": True,
        "num_inference_steps": 4,
        "height": 256,
        "width": 256
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Request successful.")
    except Exception as e:
        print(f"Request failed: {e}")
        return

    # Check for recent JSON files in outputs
    outputs_dir = os.path.join(os.getcwd(), "outputs")
    json_files = glob.glob(os.path.join(outputs_dir, "image_*.json"))
    
    # Sort by modification time
    json_files.sort(key=os.path.getmtime, reverse=True)
    
    if not json_files:
        print("No JSON metadata files found in outputs/.")
        return

    latest_json = json_files[0]
    print(f"Found latest JSON file: {latest_json}")
    
    # Check if it was created just now (within last 10 seconds)
    if time.time() - os.path.getmtime(latest_json) < 10:
        print("PASS: Metadata file created successfully.")
        with open(latest_json, 'r') as f:
            print("Content:", f.read())
    else:
        print("FAIL: No new metadata file created (found old one).")

if __name__ == "__main__":
    verify_metadata_generation()
