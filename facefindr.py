import os
import sys
import time
import json
import pickle
import hashlib
import argparse
import logging
import sqlite3
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import cv2
from PIL import Image, ImageDraw, ExifTags
from tqdm import tqdm

import insightface
from insightface.app import FaceAnalysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FaceFindr")

class FaceEmbedding:
    def __init__(self, face_id: str, image_path: str, embedding: np.ndarray, 
                 face_location: tuple, timestamp: float, metadata: Dict = None):
        self.face_id = face_id
        self.image_path = image_path
        self.embedding = embedding
        self.face_location = face_location
        self.timestamp = timestamp
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        return {
            "face_id": self.face_id,
            "image_path": self.image_path,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "face_location": self.face_location,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FaceEmbedding':
        embedding = np.array(data["embedding"]) if data["embedding"] is not None else None
        return cls(
            data["face_id"],
            data["image_path"],
            embedding,
            tuple(data["face_location"]),
            data["timestamp"],
            data["metadata"]
        )

# Database handler
class FaceDatabase:
    def __init__(self, db_path: str = "facedb.sqlite"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                image_path TEXT PRIMARY KEY,
                file_hash TEXT,
                last_modified REAL,
                processed BOOLEAN DEFAULT 0,
                metadata TEXT
            )
            ''')
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                face_id TEXT PRIMARY KEY,
                image_path TEXT,
                embedding BLOB,
                face_location TEXT,
                timestamp REAL,
                metadata TEXT,
                FOREIGN KEY (image_path) REFERENCES images(image_path)
            )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_path ON faces(image_path)')
            conn.commit()
    
    def add_image(self, image_path: str, file_hash: str, last_modified: float, metadata: Dict = None):
        last_modified = float(last_modified)
        metadata = convert_numpy(metadata or {})
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR REPLACE INTO images (image_path, file_hash, last_modified, processed, metadata) VALUES (?, ?, ?, ?, ?)',
                (image_path, file_hash, last_modified, False, json.dumps(metadata))
            )
            conn.commit()
            
    def add_face(self, face: FaceEmbedding):
        # แก้จุดนี้ แปลงทุกค่าใน face_location ให้เป็น int ปกติ
        face_location = [int(x) for x in face.face_location]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR REPLACE INTO faces (face_id, image_path, embedding, face_location, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?)',
                (
                    face.face_id,
                    face.image_path,
                    pickle.dumps(face.embedding),
                    json.dumps(face_location),   # ใช้ face_location ที่แปลงแล้ว
                    face.timestamp,
                    json.dumps(face.metadata)
                )
            )
            conn.commit()
    
    def mark_image_processed(self, image_path: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE images SET processed = 1 WHERE image_path = ?', (image_path,))
            conn.commit()
    
    def get_unprocessed_images(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT image_path FROM images WHERE processed = 0')
            return [row[0] for row in cursor.fetchall()]
    
    def get_all_face_embeddings(self) -> List[FaceEmbedding]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT face_id, image_path, embedding, face_location, timestamp, metadata FROM faces')
            faces = []
            for row in cursor.fetchall():
                face_id, image_path, embedding_blob, face_location_str, timestamp, metadata_str = row
                embedding = pickle.loads(embedding_blob)
                face_location = tuple(json.loads(face_location_str))
                metadata = json.loads(metadata_str)
                face = FaceEmbedding(
                    face_id=face_id,
                    image_path=image_path,
                    embedding=embedding,
                    face_location=face_location,
                    timestamp=timestamp,
                    metadata=metadata
                )
                faces.append(face)
            return faces
    
    def get_image_info(self, image_path: str) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT file_hash, last_modified, processed, metadata FROM images WHERE image_path = ?', (image_path,))
            row = cursor.fetchone()
            if row:
                file_hash, last_modified, processed, metadata_str = row
                return {
                    "image_path": image_path,
                    "file_hash": file_hash,
                    "last_modified": last_modified,
                    "processed": bool(processed),
                    "metadata": json.loads(metadata_str)
                }
            return None
    
    def get_faces_by_image(self, image_path: str) -> List[FaceEmbedding]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT face_id, image_path, embedding, face_location, timestamp, metadata FROM faces WHERE image_path = ?', (image_path,))
            faces = []
            for row in cursor.fetchall():
                face_id, image_path, embedding_blob, face_location_str, timestamp, metadata_str = row
                embedding = pickle.loads(embedding_blob)
                face_location = tuple(json.loads(face_location_str))
                metadata = json.loads(metadata_str)
                face = FaceEmbedding(
                    face_id=face_id,
                    image_path=image_path,
                    embedding=embedding,
                    face_location=face_location,
                    timestamp=timestamp,
                    metadata=metadata
                )
                faces.append(face)
            return faces
    
    def delete_image_and_faces(self, image_path: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM faces WHERE image_path = ?', (image_path,))
            cursor.execute('DELETE FROM images WHERE image_path = ?', (image_path,))
            conn.commit()
    
    def get_image_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM images')
            return cursor.fetchone()[0]
    
    def get_face_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM faces')
            return cursor.fetchone()[0]

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    return obj
# ImageProcessor using RetinaFace/InsightFace
class ImageProcessor:
    def __init__(self, face_db: FaceDatabase, num_workers: int = None):
        self.face_db = face_db
        self.num_workers = num_workers or os.cpu_count()
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        self.retinapp = FaceAnalysis(name="buffalo_l")
        self.retinapp.prepare(ctx_id=0, det_size=(640, 640))  # 0 = GPU, -1 = CPU
    
    def _calculate_file_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    def _extract_image_metadata(self, image_path: str) -> Dict:
        try:
            with Image.open(image_path) as img:
                metadata = {}
                metadata['width'] = img.width
                metadata['height'] = img.height
                metadata['format'] = img.format
                metadata['mode'] = img.mode
                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif():
                    exif = {
                        ExifTags.TAGS.get(tag, tag): value
                        for tag, value in img._getexif().items()
                        if tag in ExifTags.TAGS
                    }
                    for key, value in exif.items():
                        if isinstance(value, (bytes, bytearray)):
                            exif_data[key] = "binary_data"
                        elif isinstance(value, datetime):
                            exif_data[key] = value.isoformat()
                        else:
                            try:
                                json.dumps({key: value})
                                exif_data[key] = value
                            except:
                                exif_data[key] = str(value)
                    metadata['exif'] = exif_data
                return metadata
        except Exception as e:
            logger.warning(f"ไม่สามารถอ่านเมตาดาต้าจาก {image_path}: {e}")
            return {}
    
    def process_image(self, image_path: str) -> List[FaceEmbedding]:
        """
        Detects faces in an image and adds their embeddings to the database.
        Returns a list of FaceEmbedding objects found.
        """
        try:
            file_hash = self._calculate_file_hash(image_path)
            last_modified = os.path.getmtime(image_path)
            existing_info = self.face_db.get_image_info(image_path)
            
            # Check if image already processed and up-to-date
            if existing_info and existing_info["file_hash"] == file_hash and existing_info["processed"]:
                logger.debug(f"ข้ามภาพที่ประมวลผลไปแล้ว: {image_path}")
                return self.face_db.get_faces_by_image(image_path)
            
            # If not processed or file changed, add/update image entry and process
            metadata = self._extract_image_metadata(image_path)
            metadata = convert_numpy(metadata) 
            self.face_db.add_image(image_path, file_hash, last_modified, metadata)
            
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                logger.warning(f"ไม่สามารถอ่านไฟล์ภาพ {image_path}")
                return []
            
            faces = self.retinapp.get(img_cv)
            if not faces:
                logger.debug(f"ไม่พบใบหน้าในภาพ: {image_path}")
                self.face_db.mark_image_processed(image_path)
                return []
            
            faces_out = []
            timestamp = time.time()
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = face.bbox.astype(int)
                # เปลี่ยนเป็น (top, right, bottom, left)
                top, right, bottom, left = y1, x2, y2, x1
                embedding = face.embedding  # 512-dim vector
                face_id = f"{os.path.basename(image_path)}_{i}_{timestamp}"
                
                face_emb = FaceEmbedding(
                    face_id=face_id,
                    image_path=image_path,
                    embedding=embedding,
                    face_location=(top, right, bottom, left),
                    timestamp=timestamp,
                    metadata={"index": i}
                )
                self.face_db.add_face(face_emb)
                faces_out.append(face_emb)
            
            self.face_db.mark_image_processed(image_path)
            return faces_out
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการประมวลผลภาพ {image_path}: {e}")
            return []
    
    def scan_directory(self, directory: str, recursive: bool = True) -> int:
        directory = os.path.abspath(directory)
        logger.info(f"เริ่มสแกนไดเร็กทอรี: {directory}")
        image_paths = []
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if Path(file_path).suffix.lower() in self.valid_extensions:
                        image_paths.append(file_path)
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and Path(file_path).suffix.lower() in self.valid_extensions:
                    image_paths.append(file_path)
        logger.info(f"พบไฟล์ภาพทั้งหมด {len(image_paths)} ไฟล์")
        processed_count = 0
        with tqdm(total=len(image_paths), desc="กำลังประมวลผลภาพ") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(self.process_image, path): path for path in image_paths}
                for future in concurrent.futures.as_completed(futures):
                    path = futures[future]
                    try:
                        faces = future.result()
                        if faces: # If any faces were found and added
                            logger.debug(f"พบใบหน้า {len(faces)} ใบหน้าในภาพ {path}")
                        # We count processed images, not just images with faces for the scan summary
                        processed_count += 1 
                    except Exception as e:
                        logger.error(f"เกิดข้อผิดพลาดในการประมวลผล {path}: {e}")
                    pbar.update(1)
        logger.info(f"ประมวลผลเสร็จสิ้น: ประมวลผลภาพทั้งหมด {processed_count}/{len(image_paths)} ไฟล์")
        return processed_count

# Cosine similarity สำหรับ embedding
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

# FaceFinder ที่ใช้ cosine similarity
class FaceFinder:
    def __init__(self, face_db: FaceDatabase, processor: ImageProcessor):
        self.face_db = face_db
        self.processor = processor
    
    def search_face(self, query_image: str, tolerance: float = 0.4, max_results: int = 20) -> List[Tuple[FaceEmbedding, float]]:
        # For searching, we just need to process the query image to get its face embeddings.
        # We don't need to add it to the database permanently.
        query_faces_temp = self.processor.process_image(query_image) 
        
        # After processing, delete the image and its faces from the DB if it was added temporarily for search
        # This assumes process_image adds it to DB, which it does.
        # This cleanup is important for a dedicated 'search' endpoint.
        # If query_image is *already* in the DB from a previous 'index' operation,
        # then this will re-add and re-process it.
        # A more robust solution for search would be to have a separate 'detect_faces_only' method
        # that doesn't interact with the DB for transient query images.
        
        if not query_faces_temp:
            logger.warning(f"ไม่พบใบหน้าในภาพค้นหา: {query_image}")
            return []
        
        # We only take the first detected face for the query
        # If you need to search for multiple faces in the query image, you'll need to loop here.
        query_embedding = query_faces_temp[0].embedding
        
        db_faces = self.face_db.get_all_face_embeddings()
        if not db_faces:
            logger.warning("ไม่พบใบหน้าในฐานข้อมูล")
            return []
        
        scores = []
        for db_face in db_faces:
            # Ensure embeddings are not None before comparison
            if db_face.embedding is not None and query_embedding is not None:
                sim = cosine_similarity(query_embedding, db_face.embedding)
                if sim >= tolerance:
                    scores.append((db_face, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:max_results]
    
    def display_results(self, results: List[Tuple[FaceEmbedding, float]], output_dir: str = None):
        if not results:
            print("❌ ไม่พบผลลัพธ์ที่ตรงกัน")
            return
        print(f"✅ พบใบหน้าที่ตรงกัน {len(results)} รายการ:")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        for i, (face, confidence) in enumerate(results, 1):
            img_path = face.image_path
            location = face.face_location
            print(f"{i}. similarity: {confidence:.3f}")
            print(f"   ไฟล์: {img_path}")
            try:
                image = Image.open(img_path)
                draw = ImageDraw.Draw(image)
                top, right, bottom, left = location
                #draw.rectangle([(left, top), (right, bottom)], outline="red", width=3) # Commented out as it might not be desired for saved results
                if output_dir:
                    result_path = os.path.join(output_dir, f"result_{i}_{os.path.basename(img_path)}")
                    image.save(result_path)
                    print(f"   บันทึกผลลัพธ์ที่: {result_path}")
                # else:
                #     image.show() # Don't show if not in a desktop environment or for API
            except Exception as e:
                logger.error(f"ไม่สามารถแสดงภาพ {img_path}: {e}")

class FaceFindr:
    def __init__(self, db_path: str = "facedb.sqlite", num_workers: int = None, default_tolerance: float = 0.4):
        self.db_path = db_path
        self.num_workers = num_workers
        self.default_tolerance = default_tolerance
        self.face_db = FaceDatabase(db_path)
        self.processor = ImageProcessor(self.face_db, num_workers)
        self.finder = FaceFinder(self.face_db, self.processor)
    
    def index_directory(self, directory: str, recursive: bool = True) -> int:
        """
        Scans a directory, processes images, and adds faces to the database.
        Returns the count of images that were processed (even if no faces were found).
        """
        return self.processor.scan_directory(directory, recursive)
    
    def add_image(self, image_path: str) -> int:
        """
        Detects faces in a single image and adds them to the database.
        Returns the number of faces found and added from this image.
        """
        faces_found = self.processor.process_image(image_path)
        return len(faces_found)

    def search_image(self, query_image: str, tolerance: float = None, 
                     max_results: int = 30, output_dir: str = None) -> List[Tuple[FaceEmbedding, float]]:
        """
        Searches for faces in the database similar to faces found in the query image.
        Returns a list of (FaceEmbedding, confidence) tuples.
        Note: The query_image is processed on the fly and its faces are NOT permanently added to the DB.
        """
        tolerance = tolerance or self.default_tolerance
        results = self.finder.search_face(query_image, tolerance, max_results)
        # The display_results is mostly for CLI, for API you'd return the raw results
        # self.finder.display_results(results, output_dir) 
        
        # Important: If FaceFinder.search_face calls process_image, it will temporarily add the query image to DB.
        # We need to ensure it's removed immediately after searching, as it's a 'query' image, not an 'indexed' image.
        # This assumes process_image adds it and search_face doesn't explicitly remove it.
        # If `search_face` ensures temporary processing without DB write, this cleanup isn't needed here.
        # Given your `process_image` always writes to DB, this cleanup is necessary for a pure 'search' endpoint.
        self.face_db.delete_image_and_faces(query_image)
        logger.info(f"Cleaned up temporary query image: {query_image} from DB.")

        return results
    
    def get_database_stats(self) -> Dict:
        """
        Returns statistics about the database.
        """
        return {
            "total_images": self.face_db.get_image_count(),
            "total_faces": self.face_db.get_face_count(),
            "db_path": self.db_path,
        }
    
    def delete_image_from_db(self, image_path: str) -> bool:
        """
        Deletes an image and all its associated faces from the database.
        Returns True if successful, False otherwise.
        """
        try:
            self.face_db.delete_image_and_faces(image_path)
            logger.info(f"Deleted image {image_path} and its faces from the database.")
            return True
        except Exception as e:
            logger.error(f"Error deleting image {image_path} from database: {e}")
            return False