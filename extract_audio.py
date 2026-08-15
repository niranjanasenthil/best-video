import os
import sys
import json
import subprocess
import wave

def get_audio_metadata_wave(file_path):
    """
    Extract accurate metadata from an uncompressed WAV file using Python's standard wave library.
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth() # bytes per sample (e.g., 2 = 16-bit)
            n_frames = wf.getnframes()
            duration = round(n_frames / float(sample_rate), 2)
            bit_depth = sample_width * 8

            return {
                "sample_rate": sample_rate,
                "channels": channels,
                "channel_layout": "mono" if channels == 1 else ("stereo" if channels == 2 else f"{channels} channels"),
                "duration_seconds": duration,
                "bit_depth": bit_depth,
                "audio_format": f"PCM {bit_depth}-bit LE WAV"
            }
    except Exception as e:
        return {}

def extract_audio(input_video="input_video.mp4", output_audio="audio.wav", channels=None, sample_rate=None):
    """
    Audio Processing Agent core function:
    Extracts high-quality uncompressed WAV audio from video file preserving speech quality.
    """
    input_path = os.path.abspath(input_video)
    output_path = os.path.abspath(output_audio)

    if not os.path.exists(input_path):
        return {
            "status": "error",
            "error_type": "FILE_NOT_FOUND",
            "reason": f"Input video file '{input_video}' does not exist.",
            "file_path": input_path
        }

    # Clean up existing audio output if present
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass

    # Build ffmpeg extraction command:
    # -vn: disable video recording
    # -acodec pcm_s16le: lossless PCM 16-bit uncompressed WAV output to avoid compression artifacts or clipping
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le"]

    if channels:
        ffmpeg_cmd.extend(["-ac", str(channels)])
    
    if sample_rate:
        ffmpeg_cmd.extend(["-ar", str(sample_rate)])

    ffmpeg_cmd.append(output_path)

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            return {
                "status": "error",
                "error_type": "FFMPEG_EXTRACTION_FAILED",
                "reason": result.stderr,
                "input_video": input_path
            }

        if not os.path.exists(output_path):
            return {
                "status": "error",
                "error_type": "OUTPUT_FILE_MISSING",
                "reason": f"Extracted audio file '{output_audio}' was not found after processing.",
                "input_video": input_path
            }

        file_size_bytes = os.path.getsize(output_path)
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

        wave_meta = get_audio_metadata_wave(output_path)

        return {
            "status": "success",
            "local_file_path": output_path,
            "metadata": {
                "file_name": os.path.basename(output_path),
                "file_size": f"{file_size_mb} MB ({file_size_bytes} bytes)",
                "file_size_bytes": file_size_bytes,
                "duration": f"{wave_meta.get('duration_seconds')} seconds",
                "duration_seconds": wave_meta.get('duration_seconds'),
                "sample_rate": f"{wave_meta.get('sample_rate')} Hz",
                "sample_rate_hz": wave_meta.get('sample_rate'),
                "channels": wave_meta.get('channels'),
                "channel_layout": wave_meta.get('channel_layout'),
                "bit_depth": f"{wave_meta.get('bit_depth')}-bit",
                "format": wave_meta.get('audio_format')
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "error_type": "AUDIO_PROCESSING_ERROR",
            "reason": str(e),
            "input_video": input_path
        }

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "input_video.mp4"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "audio.wav"
    
    result = extract_audio(input_file, output_file)
    print(json.dumps(result, indent=2))
