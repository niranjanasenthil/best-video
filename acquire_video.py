import sys
import os
import json
import subprocess
import urllib.request
import urllib.error
import yt_dlp

def validate_url(url):
    """
    Validate that the URL is well-formed and uses HTTP or HTTPS.
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string."
    
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return False, "Invalid URL scheme. URL must start with http:// or https://"
    
    return True, None

def get_video_metadata_ffmpeg(file_path):
    """
    Extract exact video metadata using ffmpeg media stream probe.
    """
    try:
        cmd = ["ffmpeg", "-i", file_path]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, timeout=15)
        output = result.stderr

        duration = None
        resolution = None
        fps = None
        vcodec = None
        acodec = None

        for line in output.splitlines():
            if "Duration:" in line:
                parts = line.split(",")
                for p in parts:
                    if "Duration:" in p:
                        dur_str = p.split("Duration:")[1].strip()
                        try:
                            h, m, s = dur_str.split(":")
                            duration = round(float(h)*3600 + float(m)*60 + float(s), 2)
                        except Exception:
                            duration = dur_str
            if "Stream #" in line and "Video:" in line:
                parts = line.split("Video:")[1].split(",")
                if len(parts) > 0:
                    vcodec = parts[0].strip().split()[0]
                for p in parts:
                    p_str = p.strip()
                    if "x" in p_str:
                        subparts = p_str.split()[0]
                        if "x" in subparts and subparts.replace("x", "").isdigit():
                            resolution = subparts
                    if "fps" in p_str:
                        try:
                            fps = float(p_str.split("fps")[0].strip())
                        except Exception:
                            pass
            if "Stream #" in line and "Audio:" in line:
                parts = line.split("Audio:")[1].split(",")
                if len(parts) > 0:
                    acodec = parts[0].strip().split()[0]

        return {
            "duration_seconds": duration,
            "resolution": resolution,
            "fps": fps,
            "video_codec": vcodec,
            "audio_codec": acodec
        }
    except Exception:
        return {}

def direct_download_file(url, target_temp_path):
    """
    Fallback method to stream raw video files directly over HTTP/HTTPS.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': '*/*'
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response, open(target_temp_path, 'wb') as out_file:
        chunk_size = 1024 * 1024 # 1MB chunks
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)

def acquire_video(url, output_filename="input_video.mp4"):
    """
    Video Acquisition Agent core logic:
    1. Validates reachability
    2. Downloads highest available quality
    3. Preserves original video & audio streams
    4. Saves as input_video.mp4
    5. Returns local file path & full metadata (or structured error on failure)
    """
    url = url.strip()
    is_valid, err_msg = validate_url(url)
    if not is_valid:
        return {
            "status": "error",
            "error_type": "INVALID_URL",
            "reason": err_msg,
            "url": url
        }

    output_dir = os.path.dirname(os.path.abspath(output_filename))
    target_path = os.path.abspath(output_filename)

    # Clean up any existing target file
    if os.path.exists(target_path):
        try:
            os.remove(target_path)
        except Exception:
            pass

    # Standard options for yt-dlp to grab highest quality while preserving streams
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(output_dir, 'temp_input_video.%(ext)s'),
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': False,
        'concurrent_fragment_downloads': 4,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    download_success = False
    info_extracted = {}
    download_error_detail = ""

    # Attempt 1: Download via yt-dlp (handles 1000+ platforms & formats)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_extracted = ydl.extract_info(url, download=True)
            download_success = True
    except Exception as e:
        download_error_detail = str(e)
        # Attempt 2: Direct HTTP streaming fallback for direct video URLs (.mp4, .mkv, .mov, etc.)
        try:
            temp_fallback_path = os.path.join(output_dir, 'temp_input_video.mp4')
            direct_download_file(url, temp_fallback_path)
            download_success = True
        except Exception as direct_e:
            return {
                "status": "error",
                "error_type": "DOWNLOAD_FAILED",
                "reason": f"Acquisition failed via standard engines. Primary error: {download_error_detail}. Direct download error: {str(direct_e)}",
                "url": url
            }

    # Locate downloaded file in output directory
    temp_files = [
        f for f in os.listdir(output_dir) 
        if f.startswith('temp_input_video')
    ]

    if not temp_files:
        return {
            "status": "error",
            "error_type": "FILE_NOT_FOUND",
            "reason": "Downloaded stream could not be saved to disk.",
            "url": url
        }

    temp_filepath = os.path.join(output_dir, temp_files[0])

    # Convert/Remux to mp4 container without re-encoding unless streams are incompatible
    try:
        if temp_filepath.endswith('.mp4'):
            if temp_filepath != target_path:
                if os.path.exists(target_path):
                    os.remove(target_path)
                os.rename(temp_filepath, target_path)
        else:
            # Stream copy (-c copy) preserves exact video & audio bitstreams
            remux_cmd = [
                "ffmpeg", "-y", "-i", temp_filepath,
                "-c", "copy", target_path
            ]
            remux_res = subprocess.run(remux_cmd, capture_output=True, text=True)
            if remux_res.returncode != 0:
                # Re-encode only if mp4 container stream copy fails due to container incompatibility
                reencode_cmd = [
                    "ffmpeg", "-y", "-i", temp_filepath,
                    "-c:v", "libx264", "-c:a", "aac", target_path
                ]
                subprocess.run(reencode_cmd, capture_output=True, text=True)

            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
    except Exception as conv_err:
        return {
            "status": "error",
            "error_type": "REMUX_FAILED",
            "reason": f"Failed to finalize video file into MP4 container: {str(conv_err)}",
            "url": url
        }

    if not os.path.exists(target_path):
        return {
            "status": "error",
            "error_type": "FILE_SAVE_FAILED",
            "reason": f"Target file '{output_filename}' not found after acquisition.",
            "url": url
        }

    # 4. Extract Final File Metadata
    file_size_bytes = os.path.getsize(target_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    ffmpeg_meta = get_video_metadata_ffmpeg(target_path)

    duration = info_extracted.get('duration') or ffmpeg_meta.get('duration_seconds') or "Unknown"
    width = info_extracted.get('width')
    height = info_extracted.get('height')
    if width and height:
        resolution = f"{width}x{height}"
    else:
        resolution = ffmpeg_meta.get('resolution') or "Unknown"

    fps = info_extracted.get('fps') or ffmpeg_meta.get('fps') or "Unknown"
    vcodec = info_extracted.get('vcodec') or ffmpeg_meta.get('video_codec') or "Unknown"
    acodec = info_extracted.get('acodec') or ffmpeg_meta.get('audio_codec') or "Unknown"

    return {
        "status": "success",
        "local_file_path": target_path,
        "metadata": {
            "file_name": os.path.basename(target_path),
            "file_size": f"{file_size_mb} MB ({file_size_bytes} bytes)",
            "file_size_bytes": file_size_bytes,
            "duration": f"{duration} seconds" if isinstance(duration, (int, float)) else str(duration),
            "duration_seconds": duration,
            "resolution": resolution,
            "fps": fps,
            "codec": f"Video: {vcodec}, Audio: {acodec}",
            "video_codec": vcodec,
            "audio_codec": acodec
        }
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "error_type": "MISSING_URL",
            "reason": "No video URL provided. Usage: python acquire_video.py <VIDEO_URL>"
        }, indent=2))
        sys.exit(1)

    video_url = sys.argv[1]
    result = acquire_video(video_url)
    print(json.dumps(result, indent=2))
