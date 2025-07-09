import logging
from typing import Any
from utils.gemini_llm import gemini_generate
import ast
import re
from transformers import pipeline
import soundfile as sf
import os

def clean_llm_output(text):
    return re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip(), flags=re.MULTILINE).strip()

def safe_literal_eval(text):
    # Try normal parse
    try:
        return ast.literal_eval(text)
    except Exception:
        # Try to auto-close brackets
        if text.count('[') > text.count(']'):
            text = text + (']' * (text.count('[') - text.count(']')))
        if text.count('{') > text.count('}'):
            text = text + ('}' * (text.count('{') - text.count('}')))
        try:
            return ast.literal_eval(text)
        except Exception as e:
            logging.error(f"[VocalAgent] Failed to parse LLM output after auto-fix: {e}")
            return []

def notes_array_to_dicts(notes_array):
    return [
        {'pitch': n[0], 'start': n[1], 'end': n[2], 'velocity': n[3]}
        for n in notes_array if isinstance(n, (list, tuple)) and len(n) == 4
    ]

def vocal_agent(state: Any) -> Any:
    logging.info("[VocalAgent] Generating vocals with Gemini LLM (if enabled).")
    try:
        if not state.get('vocals'):
            logging.info("[VocalAgent] Vocals not enabled, skipping.")
            return state
        lyrics = state.get('lyrics', '')
        if not lyrics.strip():
            logging.info("[VocalAgent] No lyrics provided, skipping vocal generation.")
            return state
        structure = state.get('structure', {})
        melody_onsets = structure.get('melody_onsets', [])
        artist_profile = state.get('artist_profile', {})
        artist = state.get('user_input', {}).get('artist', '')
        vocal_style = artist_profile.get('vocal_style', 'expressive pop/rock vocals')
        profile_str = f" in the style of {artist}: {artist_profile}" if artist and artist_profile else ""
        sections = state.get('sections', [])
        section_str = ""
        if sections:
            section_str = "The song has the following sections:\n"
            for s in sections:
                section_str += f"- {s['name'].capitalize()} (lines {s['start']}-{s['end']})\n"
            section_str += "Vary the vocal melody and delivery for each section. Add transitions at section boundaries. "
        prompt = (
            f"Generate a vocal melody track for a {state.get('genre')} song, "
            f"mood: {state.get('mood')}, tempo: {state.get('tempo')} BPM, "
            f"duration: {state.get('duration')} minutes. "
            f"Align vocal melody to these melody note onsets (in seconds): {melody_onsets}. "
            f"Base the vocal melody and delivery on the following artist's vocal style: {vocal_style}. "
            f"{section_str}Output ONLY a valid Python list of lists, where each sublist is [pitch, start, end, velocity] in that order. Do not include any explanation or extra text."
        )
        vocal_text = gemini_generate(prompt)
        logging.info(f"[VocalAgent] Raw Gemini output: {vocal_text}")
        cleaned = clean_llm_output(vocal_text)
        notes_array = safe_literal_eval(cleaned)
        notes = notes_array_to_dicts(notes_array)
        # Optionally, quantize note start times to melody_onsets
        for n in notes:
            if melody_onsets:
                n['start'] = min(melody_onsets, key=lambda t: abs(t - n['start']))
        # Warn if notes do not cover the full song duration
        duration = state.get('duration', 2)
        if notes:
            max_end = max(n['end'] for n in notes)
            if max_end < duration * 60 - 1:
                logging.warning(f"[VocalAgent] LLM notes end at {max_end:.2f}s, which is shorter than song duration {duration*60:.2f}s.")
        vocal_track = {
            'name': 'Vocals',
            'program': 54,
            'is_drum': False,
            'notes': notes
        }
        state['vocals_track'] = {'vocal_track': vocal_track}
        logging.info("[VocalAgent] Vocals generated and aligned.")

        # --- TTS VOCALS PATCH ---
        try:
            tts = pipeline("text-to-speech", "suno/bark")
            tts_out = tts(lyrics)
            audio = tts_out["audio"]
            sr = tts_out["sampling_rate"]
            os.makedirs("output", exist_ok=True)
            sf.write("output/vocals.wav", audio, sr)
            logging.info("[VocalAgent] Realistic vocals generated with suno/bark and saved to output/vocals.wav.")
            state['tts_vocals_path'] = "output/vocals.wav"
        except Exception as e:
            logging.error(f"[VocalAgent] TTS generation failed: {e}")
    except Exception as e:
        logging.error(f"[VocalAgent] Error: {e}")
    return state 