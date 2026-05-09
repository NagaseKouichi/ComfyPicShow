import os
import io
import re
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, send_file, abort, jsonify, request, redirect, make_response
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo

from config import IMAGE_ROOT_DIR, PORT, THUMBNAIL_SIZE, CACHE_DIR, BASE_DIR, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, MEDIA_EXTENSIONS

app = Flask(__name__)
os.makedirs(CACHE_DIR, exist_ok=True)


def get_safe_path(rel_path):
    """Resolve a relative path within IMAGE_ROOT_DIR, preventing directory traversal."""
    abs_path = os.path.normpath(os.path.join(IMAGE_ROOT_DIR, rel_path))
    if not abs_path.startswith(os.path.normpath(IMAGE_ROOT_DIR)):
        abort(403)
    return abs_path


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def is_video_file(filename):
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTENSIONS


def is_media_file(filename):
    return os.path.splitext(filename)[1].lower() in MEDIA_EXTENSIONS


def get_thumbnail_path(img_path):
    rel = os.path.relpath(img_path, IMAGE_ROOT_DIR)
    hash_name = hashlib.md5(rel.encode()).hexdigest() + ".jpg"
    return os.path.join(CACHE_DIR, hash_name)


def generate_thumbnail(img_path):
    """Generate and cache a thumbnail. Returns the thumbnail file path."""
    thumb_path = get_thumbnail_path(img_path)
    if os.path.exists(thumb_path):
        return thumb_path

    try:
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # respect EXIF orientation
        img.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.LANCZOS)

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img.save(thumb_path, "JPEG", quality=85)
        return thumb_path
    except Exception:
        return None


def _detect_gpu_hwaccel():
    """Detect available GPU and return ffmpeg hwaccel flags. Cached after first call."""
    if hasattr(_detect_gpu_hwaccel, "_cache"):
        return _detect_gpu_hwaccel._cache

    hwaccel = []
    try:
        # Check NVIDIA
        nv = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        if nv.returncode == 0:
            hwaccel = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            _detect_gpu_hwaccel._cache = hwaccel
            return hwaccel
    except Exception:
        pass

    try:
        # Check VAAPI (AMD/Intel)
        va = subprocess.run(["vainfo"], capture_output=True, timeout=5)
        if va.returncode == 0:
            hwaccel = ["-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"]
            _detect_gpu_hwaccel._cache = hwaccel
            return hwaccel
    except Exception:
        pass

    try:
        # Check Intel QSV
        qsv = subprocess.run(["intel_gpu_top", "-L"], capture_output=True, timeout=5)
        if qsv.returncode == 0:
            hwaccel = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
            _detect_gpu_hwaccel._cache = hwaccel
            return hwaccel
    except Exception:
        pass

    _detect_gpu_hwaccel._cache = []
    return []


def generate_video_thumbnail(video_path):
    """Generate a thumbnail from a video file using ffmpeg with GPU acceleration if available."""
    thumb_path = get_thumbnail_path(video_path)
    if os.path.exists(thumb_path):
        return thumb_path

    try:
        hwaccel = _detect_gpu_hwaccel()
        scale_filter = f"scale={THUMBNAIL_SIZE}:{THUMBNAIL_SIZE}:force_original_aspect_ratio=decrease,pad={THUMBNAIL_SIZE}:{THUMBNAIL_SIZE}:(ow-iw)/2:(oh-ih)/2"
        cmd = ["ffmpeg", "-y"] + hwaccel + ["-ss", "1", "-i", video_path, "-vframes", "1", "-vf", scale_filter, thumb_path]
        result = subprocess.run(cmd, capture_output=True, timeout=15)

        # If GPU decode failed, retry without hwaccel
        if not os.path.exists(thumb_path) and hwaccel:
            cmd = ["ffmpeg", "-y", "-ss", "1", "-i", video_path, "-vframes", "1", "-vf", scale_filter, thumb_path]
            subprocess.run(cmd, capture_output=True, timeout=15)

        if os.path.exists(thumb_path):
            return thumb_path
        return None
    except Exception:
        return None


def extract_video_metadata(video_path):
    """Extract ComfyUI metadata from a video file.

    Checks: embedded ffprobe tags, and a companion PNG with the same base name.
    """
    try:
        # Check embedded tags via ffprobe
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})

        raw = {}
        for key in ("prompt", "workflow", "parameters"):
            if key in tags:
                try:
                    raw[key] = json.loads(tags[key])
                except (json.JSONDecodeError, TypeError):
                    raw[key] = tags[key]

        # If no embedded metadata, check companion PNG
        if not raw:
            base = os.path.splitext(video_path)[0]
            for ext in (".png", ".PNG"):
                png_path = base + ext
                if os.path.isfile(png_path):
                    return extract_comfyui_metadata(png_path)

        if not raw:
            return None

        result = {}
        raw_prompt = raw.get("prompt")
        if raw_prompt and isinstance(raw_prompt, dict):
            parsed = _parse_comfyui_prompt(raw_prompt)
            if parsed:
                result["parsed"] = parsed

        # Also store the raw data
        result.update(raw)
        return result if result else None

    except Exception:
        return None


def get_video_info(video_path):
    """Extract video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)

        fmt = data.get("format", {})
        duration = float(fmt.get("duration", 0))
        size_bytes = int(fmt.get("size", 0))

        video_stream = None
        audio_stream = None
        for stream in data.get("streams", []):
            if stream["codec_type"] == "video" and video_stream is None:
                video_stream = stream
            elif stream["codec_type"] == "audio" and audio_stream is None:
                audio_stream = stream

        width = video_stream.get("width", 0) if video_stream else 0
        height = video_stream.get("height", 0) if video_stream else 0
        has_audio = audio_stream is not None

        # Extract FPS from r_frame_rate (e.g. "30000/1001" -> 29.97)
        fps_display = "?"
        if video_stream:
            fps_str = video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate", "")
            if "/" in fps_str:
                parts = fps_str.split("/")
                try:
                    fps = float(parts[0]) / float(parts[1])
                    fps_display = f"{fps:.2f}"
                except (ValueError, ZeroDivisionError):
                    pass
            elif fps_str:
                try:
                    fps_display = f"{float(fps_str):.2f}"
                except ValueError:
                    pass

        minutes = int(duration // 60)
        seconds = int(duration % 60)
        duration_display = f"{minutes}:{seconds:02d}"

        return {
            "width": width,
            "height": height,
            "duration": duration,
            "duration_display": duration_display,
            "has_audio": has_audio,
            "size_bytes": size_bytes,
            "size_display": format_file_size(size_bytes),
            "fps_display": fps_display,
        }
    except Exception:
        return {"width": 0, "height": 0, "duration": 0, "duration_display": "?", "has_audio": False, "size_bytes": 0, "size_display": "?", "fps_display": "?"}


def _is_comfyui_node_format(prompt_data):
    """Check if prompt data follows the ComfyUI node-based format."""
    if not isinstance(prompt_data, dict):
        return False
    for v in prompt_data.values():
        if isinstance(v, dict) and "class_type" in v:
            return True
    return False


def _strip_weight(tag):
    """Remove weight notation (:number or :number.number) from the end of a tag."""
    return re.sub(r":\d+(\.\d+)?$", "", tag).strip()


def _strip_matching_outer_parens(text):
    """Remove matching outer parentheses layers from text."""
    s = text.strip()
    while len(s) >= 2 and s[0] == "(":
        depth = 0
        matched = True
        for ch in s[1:-1]:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                matched = False
                break
        if depth != 0:
            matched = False
        if not matched:
            break
        s = s[1:-1].strip()
    return s


def _split_prompt_tags(text):
    """Split a ComfyUI prompt string into individual tags.

    Strips outer parentheses, splits by top-level commas, and recurses
    into remaining parenthesized groups so that emphasis tokens like
    ((tag1, tag2:1.3)) are broken apart.
    """
    if not text or not text.strip():
        return []

    text = _strip_matching_outer_parens(text)
    if not text:
        return []

    # Split by commas at depth 0 (top level of remaining text)
    tags = []
    current = ""
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            current += ch + text[i + 1]
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1

        if ch == "," and depth == 0:
            tag = current.strip()
            if tag:
                tags.append(tag)
            current = ""
        else:
            current += ch
        i += 1

    tag = current.strip()
    if tag:
        tags.append(tag)

    # Recurse: any tag still wrapped in parens gets split further
    result = []
    for t in tags:
        if t.startswith("(") and t.endswith(")"):
            result.extend(_split_prompt_tags(t))
        else:
            result.append(_strip_weight(t))

    return result


def _parse_comfyui_prompt(prompt_data):
    """Parse ComfyUI node-based prompt JSON into a user-friendly structure.

    Returns a dict with:
      - passes: list of per-sampler dicts (positive, negative, tags, sampler params)
      - positive / negative / positive_tags / negative_tags: merged/legacy fields
    """
    if not _is_comfyui_node_format(prompt_data):
        return None

    # Collect all nodes
    all_nodes = {}
    for node_id, node in prompt_data.items():
        if not isinstance(node, dict):
            continue
        all_nodes[node_id] = node

    # Collect text nodes
    text_nodes = {}
    for node_id, node in all_nodes.items():
        ct = node.get("class_type", "")
        if "CLIPTextEncode" in ct:
            text = node.get("inputs", {}).get("text", "").strip()
            # Fallback: some video workflows store text in widgets_values
            if not text:
                wv = node.get("widgets_values", [])
                if wv and isinstance(wv[0], str):
                    text = wv[0].strip()
            if text:
                text_nodes[node_id] = text

    # Resolve a link target node ID
    def resolve_link(link):
        if isinstance(link, list) and len(link) > 0:
            return str(link[0])
        if isinstance(link, (int, str)):
            return str(link)
        return None

    # Recursively find the ultimate CLIPTextEncode node behind a link
    def find_text_source(link, field_hint=None):
        visited = set()
        current = resolve_link(link)
        while current and current not in visited:
            visited.add(current)
            tn = text_nodes.get(current)
            if tn:
                return tn
            intermediate = all_nodes.get(current)
            if intermediate and isinstance(intermediate, dict):
                # Try following further links from the intermediate node
                inputs = intermediate.get("inputs", {})
                # Prefer the same field name, fall back to any
                next_link = None
                if field_hint:
                    next_link = inputs.get(field_hint)
                if not next_link:
                    next_link = inputs.get("positive") or inputs.get("negative")
                if next_link:
                    current = resolve_link(next_link)
                    continue
            break
        return ""

    # Collect sampler nodes
    sampler_nodes = []
    for node_id, node in all_nodes.items():
        ct = node.get("class_type", "")
        if "Sampler" in ct or ct == "KSampler":
            inputs = node.get("inputs", {})
            sampler_nodes.append({
                "id": node_id,
                "inputs": inputs,
                "positive_link": resolve_link(inputs.get("positive")),
                "negative_link": resolve_link(inputs.get("negative")),
            })

    # Build per-sampler passes
    passes = []
    for i, sn in enumerate(sampler_nodes):
        inputs = sn["inputs"]
        positive_text = find_text_source(sn["positive_link"], "positive")
        negative_text = find_text_source(sn["negative_link"], "negative")

        pass_data = {"index": i}

        if positive_text:
            pass_data["positive"] = positive_text
            tags = []
            seen = set()
            for t in _split_prompt_tags(positive_text):
                key = t.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    tags.append(t)
            pass_data["positive_tags"] = tags

        if negative_text:
            pass_data["negative"] = negative_text
            tags = []
            seen = set()
            for t in _split_prompt_tags(negative_text):
                key = t.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    tags.append(t)
            pass_data["negative_tags"] = tags

        sampler_params = {}
        for key in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"):
            if key in inputs:
                sampler_params[key] = inputs[key]
        if sampler_params:
            pass_data["sampler"] = sampler_params

        passes.append(pass_data)

    # Extract global model/LoRA info
    models = []
    loras = []
    for node_id, node in all_nodes.items():
        ct = node.get("class_type", "")
        if "CheckpointLoader" in ct:
            ckpt = node.get("inputs", {}).get("ckpt_name", "")
            if ckpt:
                models.append(ckpt)
        elif "LoraLoader" in ct:
            lora = node.get("inputs", {}).get("lora_name", "")
            strength = node.get("inputs", {}).get("strength_model", 1)
            if lora:
                loras.append({"name": lora, "strength": strength})

    # Build merged/legacy fields for search and fallback
    all_positive = []
    all_negative = []
    for p in passes:
        if p.get("positive"):
            all_positive.append(p["positive"])
        if p.get("negative"):
            all_negative.append(p["negative"])

    result = {}
    if passes:
        result["passes"] = passes

    if all_positive:
        result["positive"] = "\n".join(all_positive)
        tags = []
        seen = set()
        for t in _split_prompt_tags(result["positive"]):
            key = t.strip().lower()
            if key and key not in seen:
                seen.add(key)
                tags.append(t)
        result["positive_tags"] = tags
    if all_negative:
        result["negative"] = "\n".join(all_negative)
        tags = []
        seen = set()
        for t in _split_prompt_tags(result["negative"]):
            key = t.strip().lower()
            if key and key not in seen:
                seen.add(key)
                tags.append(t)
        result["negative_tags"] = tags

    if models:
        result["model"] = models
    if loras:
        result["loras"] = loras

    return result


def extract_comfyui_metadata(img_path):
    """Extract ComfyUI metadata (prompt/workflow) from a PNG image."""
    if not img_path.lower().endswith(".png"):
        return None

    try:
        img = Image.open(img_path)
        info = img.info

        result = {}

        # ComfyUI stores workflow/prompt in PNG text chunks
        raw_prompt = None
        for key in ("prompt", "workflow", "parameters"):
            if key in info:
                try:
                    result[key] = json.loads(info[key])
                    if key == "prompt":
                        raw_prompt = result[key]
                except (json.JSONDecodeError, TypeError):
                    result[key] = info[key]
                    if key == "prompt":
                        raw_prompt = info[key]

        # Some tools store description/prompt in other fields
        if "Description" in info and "prompt" not in result:
            result["prompt"] = info["Description"]

        # Parse ComfyUI node-format prompt into structured data
        if raw_prompt and isinstance(raw_prompt, dict):
            parsed = _parse_comfyui_prompt(raw_prompt)
            if parsed:
                result["parsed"] = parsed

        return result if result else None
    except Exception:
        return None


def get_image_info(img_path):
    """Return image file information."""
    stat = os.stat(img_path)
    try:
        img = Image.open(img_path)
        width, height = img.size
        fmt = img.format
    except Exception:
        width, height, fmt = 0, 0, "Unknown"

    return {
        "filename": os.path.basename(img_path),
        "size_bytes": stat.st_size,
        "size_display": format_file_size(stat.st_size),
        "width": width,
        "height": height,
        "format": fmt,
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def format_file_size(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


FILE_INDEX_FILE = os.path.join(BASE_DIR, "cache", "file_index.json")


def _load_file_index():
    if not os.path.exists(FILE_INDEX_FILE):
        return {}
    try:
        with open(FILE_INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_file_index(index):
    try:
        with open(FILE_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False)
    except Exception:
        pass


def scan_directory(rel_path="", sort="name", order="asc", dir_sort=None, dir_order=None):
    """Scan a directory and return subdirectories and image files.

    Uses a file index cache to avoid re-scanning unchanged directories.
    """
    if dir_sort is None:
        dir_sort = sort
    if dir_order is None:
        dir_order = order
    abs_path = get_safe_path(rel_path)
    if not os.path.isdir(abs_path):
        abort(404)

    dir_mtime = os.path.getmtime(abs_path)
    index = _load_file_index()
    cache_key = rel_path if rel_path else "."

    # Check cache validity
    cached = index.get(cache_key)
    if cached and cached.get("_mtime") == dir_mtime:
        dirs = cached.get("dirs", [])
        files = cached.get("files", [])
        # Check for old cache format missing mtime on dirs
        if dirs and "mtime" not in dirs[0]:
            cached = None
    else:
        cached = None

    if cached is None:
        # Re-scan directory
        dirs = []
        files = []
        try:
            entries = sorted(os.scandir(abs_path), key=lambda e: e.name.lower())
        except OSError:
            abort(403)

        for entry in entries:
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": os.path.join(rel_path, entry.name), "mtime": entry.stat().st_mtime})
            elif entry.is_file() and is_media_file(entry.name):
                img_rel = os.path.join(rel_path, entry.name) if rel_path else entry.name
                files.append({
                    "name": entry.name,
                    "path": img_rel,
                    "mtime": entry.stat().st_mtime,
                    "is_video": is_video_file(entry.name),
                })

        index[cache_key] = {"_mtime": dir_mtime, "dirs": dirs, "files": files}
        _save_file_index(index)

    # Build image list with thumbnail check (lightweight, done on each request)
    images = []
    for f in files:
        abs_file = os.path.join(abs_path, f["name"])
        thumb_path = get_thumbnail_path(abs_file)
        images.append({
            "name": f["name"],
            "path": f["path"],
            "has_thumbnail": os.path.exists(thumb_path),
            "mtime": f["mtime"],
            "is_video": f.get("is_video", False),
        })

    # Sort images
    reverse = (order == "desc")
    if sort == "date":
        images.sort(key=lambda x: x["mtime"], reverse=reverse)
    else:
        images.sort(key=lambda x: x["name"].lower(), reverse=reverse)

    # Sort dirs
    dir_reverse = (dir_order == "desc")
    if dir_sort == "date":
        dirs.sort(key=lambda x: x["mtime"], reverse=dir_reverse)
    else:
        dirs.sort(key=lambda x: x["name"].lower(), reverse=dir_reverse)

    # Build breadcrumbs
    crumbs = [{"name": "Home", "path": ""}]
    if rel_path:
        parts = rel_path.replace("\\", "/").split("/")
        accum = ""
        for i, part in enumerate(parts):
            accum = os.path.join(accum, part) if accum else part
            crumbs.append({"name": part, "path": accum})

    return {"dirs": dirs, "images": images, "breadcrumbs": crumbs, "current_path": rel_path, "total": len(images), "dir_count": len(dirs)}


def paginate_images(images, page=1, per_page=30):
    """Slice an image list for the given page."""
    total = len(images)
    start = (page - 1) * per_page
    end = start + per_page
    page_images = images[start:end]
    has_more = end < total
    return page_images, total, has_more


# --- Favorites ---

FAVORITES_FILE = os.path.join(BASE_DIR, "cache", "favorites.json")


def load_favorites():
    if not os.path.exists(FAVORITES_FILE):
        return []
    try:
        with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_favorites(favs):
    try:
        with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
            json.dump(favs, f, ensure_ascii=False)
    except Exception:
        pass


# --- Notes ---

NOTES_FILE = os.path.join(BASE_DIR, "cache", "notes.json")


def load_notes():
    if not os.path.exists(NOTES_FILE):
        return {}
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_notes(notes):
    try:
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False)
    except Exception:
        pass


# --- Custom Lists ---

LISTS_FILE = os.path.join(BASE_DIR, "cache", "lists.json")


def load_lists():
    if not os.path.exists(LISTS_FILE):
        return {}
    try:
        with open(LISTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_lists(lists_data):
    try:
        with open(LISTS_FILE, "w", encoding="utf-8") as f:
            json.dump(lists_data, f, ensure_ascii=False)
    except Exception:
        pass


# --- Search ---

METADATA_CACHE_FILE = os.path.join(BASE_DIR, "cache", "metadata_cache.json")


def _load_metadata_cache():
    """Load the metadata cache from disk."""
    if not os.path.exists(METADATA_CACHE_FILE):
        return {}
    try:
        with open(METADATA_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_metadata_cache(cache):
    """Save the metadata cache to disk."""
    try:
        with open(METADATA_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass


def _get_cached_prompt_text(abs_path):
    """Get prompt text for a file, using cache if the file hasn't changed."""
    mtime = os.path.getmtime(abs_path)
    cache = _load_metadata_cache()
    rel = os.path.relpath(abs_path, IMAGE_ROOT_DIR)

    if rel in cache and cache[rel].get("_mtime") == mtime:
        return cache[rel].get("positive", ""), cache[rel].get("negative", "")

    if is_video_file(abs_path):
        metadata = extract_video_metadata(abs_path)
    else:
        metadata = extract_comfyui_metadata(abs_path)

    positive = ""
    negative = ""

    if metadata and "parsed" in metadata:
        positive = metadata["parsed"].get("positive", "")
        negative = metadata["parsed"].get("negative", "")
    elif metadata and "prompt" in metadata:
        # Handle flat JSON format: {"positive": "...", "negative": "..."}
        raw = metadata["prompt"]
        if isinstance(raw, dict) and not _is_comfyui_node_format(raw):
            for key in ("positive", "pos", "Positive"):
                if key in raw and isinstance(raw[key], str):
                    positive = raw[key]
                    break
            for key in ("negative", "neg", "Negative"):
                if key in raw and isinstance(raw[key], str):
                    negative = raw[key]
                    break

    cache[rel] = {
        "_mtime": mtime,
        "positive": positive,
        "negative": negative,
    }
    _save_metadata_cache(cache)
    return positive, negative


def search_images(query):
    """Search all images for prompts matching the query.

    Only searches against the pre-built metadata cache — does NOT trigger
    new metadata extraction during search. Files without cached metadata
    are matched by filename only.
    """
    if not query or len(query.strip()) < 1:
        return []

    query_lower = query.strip().lower()
    meta_cache = _load_metadata_cache()
    file_index = _load_file_index()
    results = []

    # Collect all known files from file index
    all_files = []
    for cache_key, entry in file_index.items():
        for f in entry.get("files", []):
            all_files.append(f)

    # If file index is empty, fall back to scanning (first-use scenario)
    if not all_files:
        for root, dirs, files in os.walk(IMAGE_ROOT_DIR):
            dirs.sort()
            for f in sorted(files):
                if is_media_file(f):
                    rel_path = os.path.relpath(os.path.join(root, f), IMAGE_ROOT_DIR)
                    all_files.append({"name": f, "path": rel_path, "is_video": is_video_file(f)})

    for f in all_files:
        rel_path = f["path"]
        abs_path = os.path.join(IMAGE_ROOT_DIR, rel_path)
        if not os.path.isfile(abs_path):
            continue

        # Check prompt from cache only (no extraction)
        prompt_match = False
        matched_text = ""
        cached = meta_cache.get(rel_path, {})
        positive = cached.get("positive", "")
        negative = cached.get("negative", "")

        if positive and query_lower in positive.lower():
            prompt_match = True
            idx = positive.lower().index(query_lower)
            start = max(0, idx - 60)
            end = min(len(positive), idx + len(query) + 60)
            snippet = positive[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(positive):
                snippet = snippet + "..."
            matched_text = snippet

        if not prompt_match and negative and query_lower in negative.lower():
            prompt_match = True
            idx = negative.lower().index(query_lower)
            start = max(0, idx - 60)
            end = min(len(negative), idx + len(query) + 60)
            snippet = negative[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(negative):
                snippet = snippet + "..."
            matched_text = snippet

        if prompt_match:
            thumb_path = get_thumbnail_path(abs_path)
            results.append({
                "name": f["name"],
                "path": rel_path,
                "has_thumbnail": os.path.exists(thumb_path),
                "prompt_match": True,
                "matched_text": matched_text,
                "mtime": os.path.getmtime(abs_path),
                "is_video": f.get("is_video", False),
            })

    return results


def sort_results(results, sort="name", order="asc"):
    """Sort a list of image result dicts by name or date."""
    reverse = (order == "desc")
    if sort == "date":
        results.sort(key=lambda x: x.get("mtime", 0), reverse=reverse)
    else:
        results.sort(key=lambda x: x["name"].lower(), reverse=reverse)
    return results


# --- Routes ---

PER_PAGE = 30


@app.route("/")
def index():
    sort = request.args.get("sort", "name")
    order = request.args.get("order", "asc")
    dir_sort = request.args.get("dir_sort", sort)
    dir_order = request.args.get("dir_order", order)
    page = request.args.get("page", 1, type=int)
    is_ajax = request.args.get("ajax", "0") == "1"
    return browse("", sort, order, page, is_ajax, dir_sort, dir_order)


@app.route("/browse/")
@app.route("/browse/<path:subpath>")
def browse(subpath="", sort=None, order=None, page=None, is_ajax=None, dir_sort=None, dir_order=None):
    if sort is None:
        sort = request.args.get("sort", "name")
    if order is None:
        order = request.args.get("order", "asc")
    if dir_sort is None:
        dir_sort = request.args.get("dir_sort", sort)
    if dir_order is None:
        dir_order = request.args.get("dir_order", order)
    if page is None:
        page = request.args.get("page", 1, type=int)
    if is_ajax is None:
        is_ajax = request.args.get("ajax", "0") == "1"

    data = scan_directory(subpath, sort=sort, order=order, dir_sort=dir_sort, dir_order=dir_order)
    all_images = data["images"]
    page_images, total, has_more = paginate_images(all_images, page=page, per_page=PER_PAGE)

    if is_ajax:
        return jsonify({"images": page_images, "total": total, "page": page, "has_more": has_more})

    data["images"] = page_images
    data["sort"] = sort
    data["order"] = order
    data["dir_sort"] = dir_sort
    data["dir_order"] = dir_order
    data["page"] = page
    data["has_more"] = has_more
    data["total"] = total
    return render_template("index.html", **data)


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    sort = request.args.get("sort", "name")
    order = request.args.get("order", "asc")
    page = request.args.get("page", 1, type=int)
    is_ajax = request.args.get("ajax", "0") == "1"

    results = search_images(query) if query else []
    sort_results(results, sort=sort, order=order)
    page_results, total, has_more = paginate_images(results, page=page, per_page=PER_PAGE)

    if is_ajax:
        return jsonify({"images": page_results, "total": total, "page": page, "has_more": has_more})

    breadcrumbs = [{"name": "Home", "path": ""}, {"name": f"搜索: {query}" if query else "搜索", "path": ""}]
    return render_template(
        "search.html",
        query=query,
        results=page_results,
        breadcrumbs=breadcrumbs,
        sort=sort,
        order=order,
        page=page,
        has_more=has_more,
        total=total,
    )


@app.route("/image/<path:imgpath>")
def image_detail(imgpath):
    """Redirect to index with hash so modal auto-opens."""
    return redirect("/#/image/" + imgpath)


@app.route("/fragment/detail/<path:imgpath>")
def fragment_detail(imgpath):
    """Return just the detail panel HTML for modal display."""
    abs_path = get_safe_path(imgpath)
    if not os.path.isfile(abs_path):
        return "", 404

    if is_video_file(abs_path):
        info = get_video_info(abs_path)
        info["filename"] = os.path.basename(imgpath)
        info["format"] = os.path.splitext(imgpath)[1].upper().lstrip(".")
        stat = os.stat(abs_path)
        info["modified"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        metadata = extract_video_metadata(abs_path)
        is_video = True
    else:
        info = get_image_info(abs_path)
        metadata = extract_comfyui_metadata(abs_path)
        is_video = False

    # Write prompt text to metadata cache for search
    if metadata and "parsed" in metadata:
        p = metadata["parsed"]
        positive = p.get("positive", "")
        negative = p.get("negative", "")
    elif metadata and "prompt" in metadata and isinstance(metadata["prompt"], dict):
        raw = metadata["prompt"]
        if not _is_comfyui_node_format(raw):
            positive = raw.get("positive", "") or raw.get("pos", "") or raw.get("Positive", "")
            negative = raw.get("negative", "") or raw.get("neg", "") or raw.get("Negative", "")
        else:
            positive = ""
            negative = ""
    else:
        positive = ""
        negative = ""

    if positive or negative:
        cache = _load_metadata_cache()
        cache[imgpath] = {"_mtime": os.path.getmtime(abs_path), "positive": positive, "negative": negative}
        _save_metadata_cache(cache)

    return render_template(
        "_detail_panel.html",
        image=info,
        metadata=metadata,
        imgpath=imgpath,
        is_video=is_video,
    )


@app.route("/thumbnail/<path:imgpath>")
def thumbnail(imgpath):
    abs_path = get_safe_path(imgpath)
    if not os.path.isfile(abs_path):
        abort(404)

    if is_video_file(abs_path):
        thumb_path = generate_video_thumbnail(abs_path)
    else:
        thumb_path = generate_thumbnail(abs_path)

    if thumb_path is None:
        abort(404)

    return send_file(thumb_path, mimetype="image/jpeg")


@app.route("/api/image-info/<path:imgpath>")
def api_image_info(imgpath):
    abs_path = get_safe_path(imgpath)
    if not os.path.isfile(abs_path):
        return jsonify({"error": "Not found"}), 404

    info = get_image_info(abs_path)
    metadata = extract_comfyui_metadata(abs_path)
    return jsonify({"info": info, "metadata": metadata})


@app.route("/raw/<path:imgpath>")
def raw_image(imgpath):
    abs_path = get_safe_path(imgpath)
    if not os.path.isfile(abs_path):
        abort(404)

    file_size = os.path.getsize(abs_path)
    range_header = request.headers.get("Range")

    if range_header:
        # Handle Range request for video streaming
        range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if range_match:
            start = int(range_match.group(1))
            end_str = range_match.group(2)
            end = int(end_str) - 1 if end_str else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            with open(abs_path, "rb") as f:
                f.seek(start)
                data = f.read(length)

            resp = make_response(data, 206)
            resp.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            resp.headers["Accept-Ranges"] = "bytes"
            resp.headers["Content-Length"] = str(length)
            resp.headers["Content-Type"] = "application/octet-stream"
            return resp

    resp = make_response(send_file(abs_path))
    resp.headers["Accept-Ranges"] = "bytes"
    return resp


# --- Favorites routes ---

@app.route("/favorites")
def favorites_page():
    favs = load_favorites()
    sort = request.args.get("sort", "name")
    order = request.args.get("order", "asc")
    page = request.args.get("page", 1, type=int)
    is_ajax = request.args.get("ajax", "0") == "1"

    images = []
    for rel_path in favs:
        abs_path = os.path.join(IMAGE_ROOT_DIR, rel_path)
        if os.path.isfile(abs_path):
            thumb_path = get_thumbnail_path(abs_path)
            images.append({
                "name": os.path.basename(rel_path),
                "path": rel_path,
                "has_thumbnail": os.path.exists(thumb_path),
                "mtime": os.path.getmtime(abs_path),
                "is_video": is_video_file(rel_path),
            })
    # Sort
    reverse = (order == "desc")
    if sort == "date":
        images.sort(key=lambda x: x["mtime"], reverse=reverse)
    else:
        images.sort(key=lambda x: x["name"].lower(), reverse=reverse)

    page_images, total, has_more = paginate_images(images, page=page, per_page=PER_PAGE)

    if is_ajax:
        return jsonify({"images": page_images, "total": total, "page": page, "has_more": has_more})

    breadcrumbs = [{"name": "Home", "path": ""}, {"name": "收藏", "path": ""}]
    return render_template("favorites.html", images=page_images, breadcrumbs=breadcrumbs, sort=sort, order=order, fav_count=len(page_images), total=total, page=page, has_more=has_more)


@app.route("/api/favorite/<path:imgpath>", methods=["POST"])
def toggle_favorite(imgpath):
    abs_path = get_safe_path(imgpath)
    if not os.path.isfile(abs_path):
        return jsonify({"error": "Not found"}), 404

    favs = load_favorites()
    if imgpath in favs:
        favs.remove(imgpath)
        status = False
    else:
        favs.append(imgpath)
        status = True
    save_favorites(favs)
    return jsonify({"favorited": status, "path": imgpath})


@app.route("/api/favorite/<path:imgpath>", methods=["GET"])
def check_favorite(imgpath):
    favs = load_favorites()
    return jsonify({"favorited": imgpath in favs})


# --- Notes routes ---

@app.route("/api/note/<path:imgpath>", methods=["GET"])
def get_note(imgpath):
    notes = load_notes()
    return jsonify({"text": notes.get(imgpath, "")})


@app.route("/api/note/<path:imgpath>", methods=["POST"])
def save_note(imgpath):
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    notes = load_notes()
    if text.strip():
        notes[imgpath] = text.strip()
    else:
        notes.pop(imgpath, None)
    save_notes(notes)
    return jsonify({"saved": True})


# --- Custom Lists routes ---

@app.route("/api/lists", methods=["GET"])
def get_lists():
    """Return all lists with name and item count."""
    lists_data = load_lists()
    result = []
    for lid, lst in lists_data.items():
        result.append({"id": lid, "name": lst.get("name", lid), "count": len(lst.get("items", []))})
    return jsonify(result)


@app.route("/api/lists", methods=["POST"])
def create_list():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400
    lists_data = load_lists()
    existing = [int(k) for k in lists_data.keys() if k.isdigit()]
    lid = str(max(existing) + 1 if existing else 1)
    lists_data[lid] = {"name": name, "items": []}
    save_lists(lists_data)
    return jsonify({"id": lid, "name": name, "count": 0})


@app.route("/api/lists/<lid>", methods=["DELETE"])
def delete_list(lid):
    lists_data = load_lists()
    if lid in lists_data:
        del lists_data[lid]
        save_lists(lists_data)
    return jsonify({"deleted": True})


@app.route("/api/lists/<lid>/rename", methods=["POST"])
def rename_list(lid):
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400
    lists_data = load_lists()
    lst = lists_data.get(lid)
    if not lst:
        return jsonify({"error": "List not found"}), 404
    lst["name"] = name
    save_lists(lists_data)
    return jsonify({"id": lid, "name": name})


@app.route("/lists/<lid>")
def view_list(lid):
    lists_data = load_lists()
    lst = lists_data.get(lid)
    if not lst:
        abort(404)

    images = []
    for rel_path in lst.get("items", []):
        abs_path = os.path.join(IMAGE_ROOT_DIR, rel_path)
        if os.path.isfile(abs_path):
            thumb_path = get_thumbnail_path(abs_path)
            images.append({
                "name": os.path.basename(rel_path),
                "path": rel_path,
                "has_thumbnail": os.path.exists(thumb_path),
                "mtime": os.path.getmtime(abs_path),
                "is_video": is_video_file(rel_path),
            })

    total = len(images)
    breadcrumbs = [{"name": "Home", "path": ""}, {"name": lst["name"], "path": ""}]
    resp = make_response(render_template("lists.html", images=images, breadcrumbs=breadcrumbs, list_name=lst["name"], list_id=lid, page=1, has_more=False, total=total, all_paths=lst.get("items", [])))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/lists/<lid>/items", methods=["POST"])
def add_to_list(lid):
    data = request.get_json(silent=True) or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "Path required"}), 400
    lists_data = load_lists()
    lst = lists_data.get(lid)
    if not lst:
        return jsonify({"error": "List not found"}), 404
    items = lst.get("items", [])
    if path not in items:
        items.append(path)
        lst["items"] = items
        save_lists(lists_data)
    return jsonify({"added": True, "count": len(items)})


@app.route("/api/lists/<lid>/items", methods=["DELETE"])
def remove_from_list(lid):
    path = request.args.get("path", "").strip()
    if not path:
        return jsonify({"error": "Path required"}), 400
    lists_data = load_lists()
    lst = lists_data.get(lid)
    if not lst:
        return jsonify({"error": "List not found"}), 404
    items = lst.get("items", [])
    if path in items:
        items.remove(path)
        lst["items"] = items
        save_lists(lists_data)
    return jsonify({"removed": True, "count": len(items)})


@app.route("/api/lists/<lid>/reorder", methods=["POST"])
def reorder_list(lid):
    data = request.get_json(silent=True) or {}
    items = data.get("items", [])
    lists_data = load_lists()
    lst = lists_data.get(lid)
    if not lst:
        return jsonify({"error": "List not found"}), 404
    lst["items"] = items
    save_lists(lists_data)
    return jsonify({"reordered": True})


@app.route("/api/lists/for-item/<path:imgpath>")
def lists_for_item(imgpath):
    lists_data = load_lists()
    result = []
    for lid, lst in lists_data.items():
        if imgpath in lst.get("items", []):
            result.append({"id": lid, "name": lst.get("name", lid)})
    return jsonify(result)


# --- Delete file ---

@app.route("/api/file/<path:imgpath>", methods=["DELETE"])
def delete_file(imgpath):
    abs_path = get_safe_path(imgpath)
    if not os.path.isfile(abs_path):
        return jsonify({"error": "Not found"}), 404

    try:
        os.remove(abs_path)
    except OSError as e:
        return jsonify({"error": str(e)}), 500

    # Clean related cache data
    thumb_path = get_thumbnail_path(abs_path)
    if os.path.exists(thumb_path):
        try:
            os.remove(thumb_path)
        except OSError:
            pass

    # Metadata cache
    cache = _load_metadata_cache()
    if imgpath in cache:
        del cache[imgpath]
        _save_metadata_cache(cache)

    # Favorites
    favs = load_favorites()
    if imgpath in favs:
        favs.remove(imgpath)
        save_favorites(favs)

    # Notes
    notes = load_notes()
    if imgpath in notes:
        del notes[imgpath]
        save_notes(notes)

    # Lists
    lists_data = load_lists()
    changed = False
    for lst in lists_data.values():
        if imgpath in lst.get("items", []):
            lst["items"].remove(imgpath)
            changed = True
    if changed:
        save_lists(lists_data)

    # File index (invalidate)
    index = _load_file_index()
    index.clear()
    _save_file_index(index)

    return jsonify({"deleted": True, "path": imgpath})


# --- Cleanup ---

@app.route("/api/cleanup", methods=["POST"])
def cleanup_orphaned_data():
    """Remove cache and data entries for files that no longer exist."""
    result = {"thumbnails": 0, "metadata": 0, "favorites": 0, "notes": 0, "lists": 0, "index": 0}

    # Find all existing media files
    valid_paths = set()
    for root, dirs, files in os.walk(IMAGE_ROOT_DIR):
        for f in files:
            if is_media_file(f):
                valid_paths.add(os.path.relpath(os.path.join(root, f), IMAGE_ROOT_DIR))

    # Clean thumbnails
    thumb_dir = CACHE_DIR
    if os.path.isdir(thumb_dir):
        for f in os.listdir(thumb_dir):
            if f.endswith(".jpg"):
                thumb_path = os.path.join(thumb_dir, f)
                # Can't easily map thumbnail hash back to original path,
                # so check all thumbnails for orphaned files by timestamp
                pass  # Thumbnails use MD5 hash, hard to reverse-map
        # Alternative: walk through metadata cache to know which hashes are valid
        meta_cache = _load_metadata_cache()
        valid_hashes = set()
        for rel_path in meta_cache:
            hash_name = hashlib.md5(rel_path.encode()).hexdigest() + ".jpg"
            valid_hashes.add(hash_name)
        # Also add hashes for valid files not in metadata cache
        for rel_path in valid_paths:
            hash_name = hashlib.md5(rel_path.encode()).hexdigest() + ".jpg"
            valid_hashes.add(hash_name)
        if os.path.isdir(thumb_dir):
            for f in os.listdir(thumb_dir):
                if f.endswith(".jpg") and f not in valid_hashes:
                    try:
                        os.remove(os.path.join(thumb_dir, f))
                        result["thumbnails"] += 1
                    except OSError:
                        pass

    # Clean metadata cache
    meta_cache = _load_metadata_cache()
    stale = [k for k in meta_cache if k not in valid_paths]
    for k in stale:
        del meta_cache[k]
        result["metadata"] += 1
    if stale:
        _save_metadata_cache(meta_cache)

    # Clean file index
    index = _load_file_index()
    # Rebuild fresh (directory mtimes will trigger re-scan anyway)
    index.clear()
    _save_file_index(index)
    result["index"] = 1 if os.path.exists(FILE_INDEX_FILE) else 0

    # Clean favorites
    favs = load_favorites()
    new_favs = [p for p in favs if p in valid_paths]
    result["favorites"] = len(favs) - len(new_favs)
    if result["favorites"]:
        save_favorites(new_favs)

    # Clean notes
    notes = load_notes()
    stale_notes = [k for k in notes if k not in valid_paths]
    for k in stale_notes:
        del notes[k]
        result["notes"] += 1
    if stale_notes:
        save_notes(notes)

    # Clean lists
    lists_data = load_lists()
    for lid, lst in lists_data.items():
        old_count = len(lst.get("items", []))
        lst["items"] = [p for p in lst.get("items", []) if p in valid_paths]
        result["lists"] += old_count - len(lst["items"])
    if result["lists"]:
        save_lists(lists_data)

    return jsonify(result)


@app.route("/api/scan-all", methods=["POST"])
def scan_all():
    """Walk all media files and generate thumbnails + extract metadata."""
    count = 0
    for root, dirs, files in os.walk(IMAGE_ROOT_DIR):
        for f in files:
            if not is_media_file(f):
                continue
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, IMAGE_ROOT_DIR)

            # Generate thumbnail
            if is_video_file(abs_path):
                generate_video_thumbnail(abs_path)
            else:
                generate_thumbnail(abs_path)

            # Extract metadata into cache
            _get_cached_prompt_text(abs_path)
            count += 1

    return jsonify({"scanned": count})


# --- Novels ---

NOVELS_FILE = os.path.join(BASE_DIR, "cache", "novels.json")


def load_novels():
    if not os.path.exists(NOVELS_FILE):
        return {}
    try:
        with open(NOVELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_novels(data):
    try:
        with open(NOVELS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


@app.route("/novels")
def novels_page():
    return render_template("novels.html")


@app.route("/api/novels", methods=["GET"])
def api_get_novels():
    data = load_novels()
    result = []
    for nid, n in data.items():
        result.append({
            "id": nid,
            "title": n.get("title", ""),
            "cover": n.get("cover", ""),
            "chapter_count": len(n.get("chapters", [])),
            "created_at": n.get("created_at", ""),
        })
    return jsonify(result)


@app.route("/api/novels", methods=["POST"])
def api_create_novel():
    body = request.get_json(silent=True) or {}
    title = body.get("title", "").strip()
    cover = body.get("cover", "").strip()
    if not title:
        return jsonify({"error": "Title required"}), 400
    data = load_novels()
    existing = [int(k) for k in data.keys() if k.isdigit()]
    nid = str(max(existing) + 1 if existing else 1)
    data[nid] = {
        "id": nid,
        "title": title,
        "cover": cover,
        "created_at": datetime.now().isoformat(),
        "chapters": [],
    }
    save_novels(data)
    return jsonify({"id": nid, "title": title, "cover": cover, "chapter_count": 0})


@app.route("/api/novels/<nid>", methods=["GET"])
def api_get_novel(nid):
    data = load_novels()
    novel = data.get(nid)
    if not novel:
        return jsonify({"error": "Not found"}), 404
    return jsonify(novel)


@app.route("/api/novels/<nid>", methods=["DELETE"])
def api_delete_novel(nid):
    data = load_novels()
    if nid in data:
        del data[nid]
        save_novels(data)
    return jsonify({"deleted": True})


@app.route("/api/novels/<nid>/chapters", methods=["POST"])
def api_add_chapter(nid):
    body = request.get_json(silent=True) or {}
    title = body.get("title", "").strip()
    content = body.get("content", "").strip()
    if not title:
        return jsonify({"error": "Title required"}), 400
    data = load_novels()
    novel = data.get(nid)
    if not novel:
        return jsonify({"error": "Novel not found"}), 404
    chapters = novel.get("chapters", [])
    update_id = body.get("chapter_id", "").strip() if body.get("chapter_id") else ""
    if update_id:
        # Update existing chapter
        cid = update_id
        for i, ch in enumerate(chapters):
            if ch["id"] == update_id:
                chapters[i] = {"id": cid, "title": title, "content": content, "created_at": datetime.now().isoformat()}
                break
    else:
        existing = [int(c["id"]) for c in chapters] if chapters else []
        cid = str(max(existing) + 1 if existing else 1)
        chapters.append({"id": cid, "title": title, "content": content, "created_at": datetime.now().isoformat()})
    novel["chapters"] = chapters
    save_novels(data)
    return jsonify({"id": cid, "title": title})


@app.route("/api/novels/<nid>/chapters/<cid>", methods=["GET"])
def api_get_chapter(nid, cid):
    data = load_novels()
    novel = data.get(nid)
    if not novel:
        return jsonify({"error": "Not found"}), 404
    chapter = None
    next_id = None
    for i, ch in enumerate(novel.get("chapters", [])):
        if ch["id"] == cid:
            chapter = ch
            if i < len(novel["chapters"]) - 1:
                next_id = novel["chapters"][i + 1]["id"]
            break
    if not chapter:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"title": chapter["title"], "content": chapter["content"], "next_id": next_id})


@app.route("/api/novels/<nid>/chapters/<cid>", methods=["DELETE"])
def api_delete_chapter(nid, cid):
    data = load_novels()
    novel = data.get(nid)
    if not novel:
        return jsonify({"error": "Not found"}), 404
    novel["chapters"] = [c for c in novel.get("chapters", []) if c["id"] != cid]
    save_novels(data)
    return jsonify({"deleted": True})


@app.route("/novels/<nid>/chapters/<cid>")
def chapter_read_page(nid, cid):
    data = load_novels()
    novel = data.get(nid)
    if not novel:
        abort(404)
    chapter = None
    for ch in novel.get("chapters", []):
        if ch["id"] == cid:
            chapter = ch
            break
    if not chapter:
        abort(404)
    prev_id = None
    next_id = None
    for i, ch in enumerate(novel["chapters"]):
        if ch["id"] == cid:
            if i > 0:
                prev_id = novel["chapters"][i - 1]["id"]
            if i < len(novel["chapters"]) - 1:
                next_id = novel["chapters"][i + 1]["id"]
            break
    return render_template("chapter.html",
                          novel=novel, chapter=chapter,
                          nid=nid, cid=cid,
                          prev_id=prev_id, next_id=next_id)


@app.route("/novels/<nid>/edit-chapter")
@app.route("/novels/<nid>/edit-chapter/<cid>")
def chapter_edit_page(nid, cid=None):
    data = load_novels()
    novel = data.get(nid)
    if not novel:
        abort(404)
    chapter = None
    if cid:
        for ch in novel.get("chapters", []):
            if ch["id"] == cid:
                chapter = ch
                break
    return render_template("chapter_edit.html", novel=novel, chapter=chapter, nid=nid, cid=cid or "")


@app.route("/api/images/all")
def api_all_images():
    """Return all image/video paths for cover selection."""
    images = []
    if os.path.exists(FILE_INDEX_FILE):
        file_index = _load_file_index()
        for cache_key, entry in file_index.items():
            for f in entry.get("files", []):
                thumb_path = get_thumbnail_path(os.path.join(IMAGE_ROOT_DIR, f["path"]))
                images.append({
                    "name": f["name"],
                    "path": f["path"],
                    "has_thumbnail": os.path.exists(thumb_path),
                    "is_video": f.get("is_video", False),
                })
    if not images:
        # Fallback: scan images directory
        for root, dirs, files in os.walk(IMAGE_ROOT_DIR):
            for f in sorted(files):
                if is_media_file(f):
                    rel_path = os.path.relpath(os.path.join(root, f), IMAGE_ROOT_DIR)
                    thumb_path = get_thumbnail_path(os.path.join(root, f))
                    images.append({
                        "name": f,
                        "path": rel_path,
                        "has_thumbnail": os.path.exists(thumb_path),
                        "is_video": is_video_file(f),
                    })
    return jsonify(images)


if __name__ == "__main__":
    print(f"Image root directory: {IMAGE_ROOT_DIR}")
    print(f"Server starting at http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
