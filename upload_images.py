import os
import requests
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

API_URL = "https://facefindr.api.thetigerteamacademy.net/upload-to-db"

IMAGE_DIRS = [
    r"E:\BEST\Images-event-facefindr\สควค-1",
    r"E:\BEST\Images-event-facefindr\สควค-2",
    r"E:\BEST\Images-event-facefindr\สควค-3"
]

def upload_image(image_path):
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
    print(f"\nProcessing directory: {directory}")
    

    ext_counts = {}
    for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:
        files = list(Path(directory).glob(f'*{ext}'))
        ext_counts[ext] = len(files)
        image_files.extend(files)
        print(f"Found {len(files)} files with extension {ext}")
    
    # Check for duplicate filenames (case-insensitive)
    filenames = [f.name.lower() for f in image_files]
    duplicates = set([x for x in filenames if filenames.count(x) > 1])
    if duplicates:
        print(f"Warning: Found {len(duplicates)} duplicate filenames (case-insensitive)")
        for dup in list(duplicates)[:5]:  # Show first 5 duplicates
            print(f"  - {dup}")
        if len(duplicates) > 5:
            print(f"  ... and {len(duplicates) - 5} more")
    
    return image_files

def main():
    all_images = []
    for directory in IMAGE_DIRS:
        if os.path.exists(directory):
            all_images.extend(process_directory(directory))
        else:
            print(f"Directory not found: {directory}")

    if not all_images:
        print("No images found in the specified directories")
        return

    print(f"\nTotal unique images found: {len(all_images)}")
    
    # Remove duplicates (case-insensitive)
    unique_images = []
    seen_names = set()
    for img in all_images:
        if img.name.lower() not in seen_names:
            seen_names.add(img.name.lower())
            unique_images.append(img)
    
    print(f"After removing duplicates: {len(unique_images)} images")
    
    if len(unique_images) != len(all_images):
        print("Removing duplicate files from processing...")
        all_images = unique_images

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(upload_image, str(img)) for img in all_images]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Uploading images"):
            results.append(future.result())

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print("\nUpload Summary:")
    print(f"Total images processed: {len(results)}")
    print(f"Successfully uploaded: {len(successful)}")
    print(f"Failed uploads: {len(failed)}")
    
    total_faces = sum(r['faces_found'] for r in successful)
    print(f"Total faces found: {total_faces}")

    if failed:
        print("\nFailed uploads:")
        for fail in failed:
            print(f"- {fail['file']}: {fail['error']}")

if __name__ == "__main__":
    main() 