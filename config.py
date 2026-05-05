import os

BASE_DIR = os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
IMAGE_ROOT_DIR = os.environ.get("IMAGES_DIR", os.path.join(BASE_DIR, "images"))
PORT = int(os.environ.get("PORT", 5000))
THUMBNAIL_SIZE = int(os.environ.get("THUMBNAIL_SIZE", 300))
CACHE_DIR = os.path.join(BASE_DIR, "cache", "thumbnails")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
