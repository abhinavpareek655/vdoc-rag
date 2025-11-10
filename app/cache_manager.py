import os
import hashlib
import json
import shutil

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _hash_file(path: str) -> str:
    """Compute SHA256 fingerprint for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_cache_path(pdf_path: str) -> str:
    fid = _hash_file(pdf_path)
    return os.path.join(CACHE_DIR, f"{fid}.json")


def save_chunks_to_cache(pdf_path: str, chunks) -> str:
    path = get_cache_path(pdf_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    return path


def load_chunks_from_cache(pdf_path: str):
    path = get_cache_path(pdf_path)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def clear_cache() -> bool:
    """Delete all cached JSON files and recreate cache directory."""
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    return True
