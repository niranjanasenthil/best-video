import os
import sys
import json
import time
import subprocess
from groq import Groq

# Set GROQ API Key
os.environ["GROQ_API_KEY"] = "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"

def load_transcript(transcript_path="output.json"):
    """
    Load transcript JSON file. Supports output.json and transcript.json schema.
    """
    if not os.path.exists(transcript_path):
        if os.path.exists("transcript.json"):
            transcript_path = "transcript.json"
        else:
            raise FileNotFoundError(f"Transcript file '{transcript_path}' not found.")

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def extract_all_words(transcript_data):
    """
    Extract a flattened list of all word objects with exact timestamps.
    """
    all_words = []
    segments = transcript_data.get("segments", [])
    
    for seg in segments:
        words = seg.get("words", [])
        if words:
            for w in words:
                all_words.append({
                    "word": w.get("word", "").strip(),
                    "start": round(float(w.get("start", 0.0)), 2),
                    "end": round(float(w.get("end", 0.0)), 2)
                })
        else:
            seg_text = seg.get("text", "").strip()
            if seg_text:
                all_words.append({
                    "word": seg_text,
                    "start": round(float(seg.get("start", 0.0)), 2),
                    "end": round(float(seg.get("end", 0.0)), 2)
                })

    return all_words

def generate_candidates(all_words, min_dur=30.0, max_dur=60.0, step_dur=15.0):
    """
    Generate clean, distinct overlapping candidate windows.
    Preferred duration: 35s - 60s. Step: ~15s for efficient coverage.
    """
    if not all_words:
        return []

    total_duration = all_words[-1]["end"] - all_words[0]["start"]

    # Short video fallback
    if total_duration <= min_dur:
        return [{
            "candidate_id": 0,
            "start": all_words[0]["start"],
            "end": all_words[-1]["end"],
            "duration": round(all_words[-1]["end"] - all_words[0]["start"], 2),
            "text": " ".join([w["word"] for w in all_words]).strip(),
            "words": all_words
        }]

    candidates = []
    cand_id = 0
    num_words = len(all_words)
    
    last_added_start = -999.0

    for i in range(num_words):
        start_time = all_words[i]["start"]
        if start_time - last_added_start < step_dur:
            continue

        for j in range(i, num_words):
            end_time = all_words[j]["end"]
            dur = end_time - start_time

            if min_dur <= dur <= max_dur:
                cand_text = " ".join([w["word"] for w in all_words[i:j+1]]).strip()
                candidates.append({
                    "candidate_id": cand_id,
                    "start": start_time,
                    "end": end_time,
                    "duration": round(dur, 2),
                    "text": cand_text,
                    "words": all_words[i:j+1]
                })
                cand_id += 1
                last_added_start = start_time
                break # Move to next sliding window start

    if not candidates:
        cand_text = " ".join([w["word"] for w in all_words]).strip()
        candidates.append({
            "candidate_id": 0,
            "start": all_words[0]["start"],
            "end": all_words[-1]["end"],
            "duration": round(total_duration, 2),
            "text": cand_text,
            "words": all_words
        })

    return candidates

def stage1_candidate_ranking(client, candidates, batch_size=12):
    """
    Stage 1: Process candidate windows in small batches to ensure fast, robust LLM ranking.
    """
    if len(candidates) == 1:
        return [{
            "candidate_id": 0,
            "score": 95.0,
            "reason": "Single candidate window covering full speech."
        }]

    all_rankings = []

    # Process in batches
    for b in range(0, len(candidates), batch_size):
        batch = candidates[b : b + batch_size]

        prompt = f"""You are an expert video content intelligence editor.
Analyze the following transcript candidates and evaluate each on:
1. Entertainment value
2. Humor / punchline quality
3. Emotional impact
4. Surprise / unexpected moment
5. Curiosity
6. Story payoff
7. Information value
8. Memorable quality
9. Audience retention potential
10. Standalone quality
11. Context completeness
12. Viral / short-form potential

Candidates:
{json.dumps([{ 'candidate_id': c['candidate_id'], 'text': c['text'], 'duration': c['duration'], 'start': c['start'], 'end': c['end'] } for c in batch], ensure_ascii=False, indent=2)}

Return ONLY JSON format:
{{
  "rankings": [
    {{
      "candidate_id": 0,
      "score": 85.0,
      "reason": "Evaluation summary"
    }}
  ]
}}
"""
        # Retry loop for API resilience
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                result = json.loads(response.choices[0].message.content)
                rankings = result.get("rankings", [])
                all_rankings.extend(rankings)
                break
            except Exception as err:
                if attempt == 2:
                    # Fallback default score if batch fails
                    for c in batch:
                        all_rankings.append({
                            "candidate_id": c["candidate_id"],
                            "score": 50.0,
                            "reason": f"Fallback evaluation due to network retry limit: {str(err)}"
                        })
                else:
                    time.sleep(2)

    return all_rankings

def stage2_final_judge(client, top_candidates):
    """
    Stage 2: Final Judge compares top candidates to select the ONE best moment.
    """
    if len(top_candidates) == 1:
        return {
            "selected_candidate": top_candidates[0]["candidate_id"],
            "reason": "Selected as the strongest standalone short-form content moment.",
            "confidence": 0.98
        }

    prompt = f"""You are the final editor for a top short-form content channel.
Compare these top candidate moments and select the ONE moment that would make the strongest standalone short-form clip from this entire video.

Question to answer:
"If I could keep only ONE moment from this entire video to represent the video and maximize viewer interest, which moment would I choose?"

Top Candidate Moments:
{json.dumps([{ 'candidate_id': c['candidate_id'], 'text': c['text'], 'start': c['start'], 'end': c['end'], 'stage1_score': c.get('stage1_score', 90) } for c in top_candidates], ensure_ascii=False, indent=2)}

Return ONLY a JSON object:
{{
  "selected_candidate": 0,
  "reason": "Detailed explanation of why this moment is the best clip",
  "confidence": 0.95
}}
"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as err:
            if attempt == 2:
                return {
                    "selected_candidate": top_candidates[0]["candidate_id"],
                    "reason": f"Highest scoring stage 1 candidate selected (Judge fallback: {str(err)}).",
                    "confidence": 0.90
                }
            time.sleep(2)

def select_best_content(transcript_path="output.json", output_json="best_content.json"):
    """
    Main LLM-Based Best Content Selection Pipeline.
    """
    transcript_data = load_transcript(transcript_path)
    all_words = extract_all_words(transcript_data)

    if not all_words:
        raise ValueError("No spoken words found in transcript.")

    candidates = generate_candidates(all_words)
    client = Groq(timeout=60.0, max_retries=3)

    # Stage 1: Candidate Ranking
    rankings = stage1_candidate_ranking(client, candidates)

    # Attach stage 1 scores to candidates
    score_map = {r.get("candidate_id"): float(r.get("score", 50.0)) for r in rankings}
    for c in candidates:
        c["stage1_score"] = score_map.get(c["candidate_id"], 50.0)

    # Sort candidates by score
    sorted_candidates = sorted(candidates, key=lambda x: x["stage1_score"], reverse=True)
    top_candidates = sorted_candidates[:5]

    # Stage 2: Final Judge
    final_decision = stage2_final_judge(client, top_candidates)

    selected_id = final_decision.get("selected_candidate", top_candidates[0]["candidate_id"])
    selected_cand = next((c for c in candidates if c["candidate_id"] == selected_id), top_candidates[0])

    # Context-Aware Boundary Refinement (Ensuring exact word timestamp alignment)
    final_start = selected_cand["words"][0]["start"]
    final_end = selected_cand["words"][-1]["end"]

    best_result = {
        "start": final_start,
        "end": final_end,
        "text": selected_cand["text"],
        "reason": final_decision.get("reason", "Selected by LLM semantic evaluation as the best moment."),
        "score": selected_cand.get("stage1_score", 95.0),
        "confidence": final_decision.get("confidence", 0.98)
    }

    # Save best_content.json
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(best_result, f, ensure_ascii=False, indent=2)

    # Trigger FFmpeg clipping via video_editing_agent.py
    clip_cmd = [
        "python", "video_editing_agent.py",
        str(final_start),
        str(final_end),
        "input_video.mp4",
        "best_clip.mp4"
    ]
    subprocess.run(clip_cmd, check=True)

    return best_result

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    t_file = sys.argv[1] if len(sys.argv) > 1 else "output.json"
    res = select_best_content(t_file)
    print(json.dumps(res, indent=2, ensure_ascii=False))
