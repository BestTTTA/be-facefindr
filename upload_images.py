import os
import requests
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

# API endpoint
API_URL = "https://facefindr.api.thetigerteamacademy.net/upload-to-db"

# Image directories
IMAGE_DIRS = [
    r"E:\BEST\Images-event-facefindr\สควค-1",
    r"E:\BEST\Images-event-facefindr\สควค-2",
    r"E:\BEST\Images-event-facefindr\สควค-3"
]

def upload_image(image_path):
    """Upload a single image to the API"""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(API_URL, files=files)
            
        if response.status_code == 200:
            result = response.json()
            return {
                'status': 'success',
                'file': image_path,
                'faces_found': result.get('faces_found', 0),
                'message': result.get('message', '')
            }
        else:
            return {
                'status': 'error',
                'file': image_path,
                'error': f'HTTP {response.status_code}: {response.text}'
            }
    except Exception as e:
        return {
            'status': 'error',
            'file': image_path,
            'error': str(e)
        }

def process_directory(directory):
    """Process all images in a directory"""
    image_files = []
    for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
        image_files.extend(list(Path(directory).glob(f'*{ext}')))
    return image_files

def main():
    # Collect all image files
    all_images = []
    for directory in IMAGE_DIRS:
        if os.path.exists(directory):
            all_images.extend(process_directory(directory))
        else:
            print(f"Directory not found: {directory}")

    if not all_images:
        print("No images found in the specified directories")
        return

    print(f"Found {len(all_images)} images to process")

    # Process images in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(upload_image, str(img)) for img in all_images]
        
        # Show progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Uploading images"):
            results.append(future.result())

    # Print summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print("\nUpload Summary:")
    print(f"Total images processed: {len(results)}")
    print(f"Successfully uploaded: {len(successful)}")
    print(f"Failed uploads: {len(failed)}")
    
    total_faces = sum(r['faces_found'] for r in successful)
    print(f"Total faces found: {total_faces}")

    # Print failed uploads if any
    if failed:
        print("\nFailed uploads:")
        for fail in failed:
            print(f"- {fail['file']}: {fail['error']}")

if __name__ == "__main__":
    main() 