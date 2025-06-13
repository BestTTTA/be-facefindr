from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import os
import uuid
import shutil
from pathlib import Path
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import asyncio

from facefindr import FaceFindr

app = FastAPI(title="FaceFindr API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server configuration
SERVER_BASE_URL = "https://facefindr.api.thetigerteamacademy.net"  # Updated domain

# Initialize FaceFindr
face_finder = FaceFindr(db_path="facedb.sqlite")

# Create directories for uploads and results
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Ensure directories have proper permissions
os.chmod(UPLOAD_DIR, 0o777)
os.chmod(RESULTS_DIR, 0o777)

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

async def process_image_for_search_async(image_path: str, tolerance: float = 0.4, max_results: int = 20) -> Dict:
    """Process a single image for searching asynchronously"""
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        executor,
        face_finder.search_image,
        image_path,
        tolerance,
        max_results
    )

    # Group results by confidence level
    grouped_results = {
        "exact_matches": [],  # confidence >= 0.99
        "high_matches": [],   # 0.95 <= confidence < 0.99
        "partial_matches": [] # confidence < 0.95
    }

    for face, confidence in results:
        # Create a copy of the matched image in results directory
        result_filename = f"result_{uuid.uuid4()}_{Path(face.image_path).name}"
        result_path = RESULTS_DIR / result_filename
        shutil.copy2(face.image_path, result_path)

        match_info = {
            "image_url": f"{SERVER_BASE_URL}/results/{result_filename}",
            "confidence": float(confidence),
            "original_path": face.image_path,
            "face_location": face.face_location,
            "filename": Path(face.image_path).name
        }

        # Group matches by confidence level
        if confidence >= 0.99:
            grouped_results["exact_matches"].append(match_info)
        elif confidence >= 0.95:
            grouped_results["high_matches"].append(match_info)
        else:
            grouped_results["partial_matches"].append(match_info)

    # Sort each group by confidence
    for group in grouped_results.values():
        group.sort(key=lambda x: x["confidence"], reverse=True)

    # Calculate statistics
    stats = {
        "total_matches": len(results),
        "exact_matches": len(grouped_results["exact_matches"]),
        "high_matches": len(grouped_results["high_matches"]),
        "partial_matches": len(grouped_results["partial_matches"])
    }

    return {
        "statistics": stats,
        "matches": grouped_results
    }

async def add_image_to_db_async(image_path: str) -> Dict:
    """Add a single image to the database asynchronously"""
    loop = asyncio.get_event_loop()
    faces_found = await loop.run_in_executor(
        executor,
        face_finder.add_image,
        image_path
    )
    return {"faces_found": faces_found, "image_path": image_path}


@app.post("/upload-to-db")
async def upload_image_to_database(file: UploadFile = File(...)):
    """Uploads an image and adds its faces to the database."""
    try:
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Add image to database
        db_add_result = await add_image_to_db_async(str(file_path))

        # Optionally, delete the uploaded file after processing if it's no longer needed in 'uploads'
        # if file_path.exists():
        #     os.remove(file_path)

        return {
            "status": "success",
            "message": f"Image uploaded to database. Found {db_add_result['faces_found']} faces.",
            "faces_found": db_add_result['faces_found'],
            "original_file": file.filename,
            "stored_path": str(file_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image to database: {str(e)}")

# --- MODIFIED ENDPOINT FOR SEARCHING ---
@app.post("/search-image")
async def search_uploaded_image(file: UploadFile = File(...)):
    """Uploads an image, searches for similar faces, and returns results."""
    # Save uploaded file temporarily for searching
    file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the image for search
        results = await process_image_for_search_async(str(file_path))

        return {
            "status": "success",
            "message": f"Search complete. Found {results['statistics']['total_matches']} total matches.",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search image: {str(e)}")
    finally:
        # Clean up the temporarily uploaded file after searching
        if file_path.exists():
            os.remove(file_path)

@app.post("/upload-multiple")
async def upload_multiple_images(files: List[UploadFile] = File(...)):
    """Uploads and adds multiple images to the database in parallel."""
    try:
        tasks = []
        uploaded_info = []
        for file in files:
            file_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            tasks.append(add_image_to_db_async(str(file_path)))
            uploaded_info.append({"filename": file.filename, "path": str(file_path)})

        results = await asyncio.gather(*tasks)

        total_faces_found = sum(r["faces_found"] for r in results)

        return {
            "status": "success",
            "message": f"Successfully uploaded {len(files)} images to database. Total faces found: {total_faces_found}",
            "uploaded_files": uploaded_info,
            "faces_per_image": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload multiple images: {str(e)}")

@app.get("/results/{filename}")
async def get_result_image(filename: str):
    """Get a processed result image"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    try:
        return FileResponse(file_path, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        stats = face_finder.get_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)