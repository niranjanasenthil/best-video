import os
import subprocess
import json
from groq import Groq

# Environment Adaptation: Set GROQ API Key in environment
os.environ["GROQ_API_KEY"] = "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"

def preprocess_audio(input_path, output_path="processed_audio.wav"):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16kHz
        "-c:a", "pcm_s16le",
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def transcribe_audio(audio_path, model="whisper-large-v3", language="ta"):
    """
    model options:
      "whisper-large-v3"        -> best accuracy (recommended for Tamil)
      "whisper-large-v3-turbo"  -> faster, slightly lower accuracy
    language:
      "ta"  -> force Tamil
      None  -> auto-detect (better for Tamil+English mixed speech)
    """
    client = Groq()  # reads GROQ_API_KEY from environment

    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model=model,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
            language=language,
            temperature=0.0
        )

    raw_segments = []
    for seg in transcription.segments:
        raw_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })

    raw_words = []
    for w in transcription.words:
        raw_words.append({
            "word": w["word"],
            "start": w["start"],
            "end": w["end"]
        })

    detected_language = getattr(transcription, "language", language or "ta")
    return raw_segments, raw_words, detected_language

def group_into_chunks(raw_segments, raw_words, min_chunk=10.0, max_chunk=15.0):
    chunks = []
    current_text = []
    chunk_start = None
    chunk_end = None

    def collect_words(start, end):
        return [
            {"word": w["word"], "start": round(w["start"], 2), "end": round(w["end"], 2)}
            for w in raw_words if w["start"] >= start and w["end"] <= end
        ]

    for seg in raw_segments:
        if chunk_start is None:
            chunk_start = seg["start"]

        duration = seg["end"] - chunk_start

        if duration > max_chunk and current_text:
            chunks.append({
                "start": round(chunk_start, 2),
                "end": round(chunk_end, 2),
                "text": " ".join(current_text).strip(),
                "words": collect_words(chunk_start, chunk_end)
            })
            current_text = []
            chunk_start = seg["start"]

        current_text.append(seg["text"])
        chunk_end = seg["end"]

        if (chunk_end - chunk_start) >= min_chunk:
            chunks.append({
                "start": round(chunk_start, 2),
                "end": round(chunk_end, 2),
                "text": " ".join(current_text).strip(),
                "words": collect_words(chunk_start, chunk_end)
            })
            current_text = []
            chunk_start = None
            chunk_end = None

    if current_text:
        chunks.append({
            "start": round(chunk_start, 2),
            "end": round(chunk_end, 2),
            "text": " ".join(current_text).strip(),
            "words": collect_words(chunk_start, chunk_end)
        })

    return chunks

LANGUAGE_NAMES = {"ta": "Tamil", "en": "English"}

def save_output(chunks, language, output_path="output.json"):
    result = {
        "language": LANGUAGE_NAMES.get(language, language),
        "segments": chunks
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return output_path

def run_pipeline(input_path, model="whisper-large-v3", language="ta"):
    processed_path = preprocess_audio(input_path)
    raw_segments, raw_words, detected_language = transcribe_audio(processed_path, model, language)
    chunks = group_into_chunks(raw_segments, raw_words, min_chunk=10.0, max_chunk=15.0)
    output_path = save_output(chunks, detected_language)
    return output_path

if __name__ == "__main__":
    input_path = "audio.wav"
    output_path = run_pipeline(input_path, model="whisper-large-v3", language="ta")
    print(f"OUTPUT_SAVED:{output_path}")
