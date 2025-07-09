import logging
from typing import Any
from utils.gemini_llm import gemini_generate
import ast
import math
import re

def quantize_time(time, step=0.25):
    return round(time / step) * step

def clean_llm_output(text):
    # Remove code block markers and leading/trailing whitespace
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
            logging.error(f"[MelodyAgent] Failed to parse LLM output after auto-fix: {e}")
            return []

def melody_agent(state: Any) -> Any:
    logging.info("[MelodyAgent] Generating melody with Gemini LLM.")
    try:
        artist_profile = state.get('artist_profile', {})
        artist = state.get('user_input', {}).get('artist', '')
        profile_str = f" in the style of {artist}: {artist_profile}" if artist and artist_profile else ""
        sections = state.get('sections', [])
        section_str = ""
        if sections:
            section_str = "The song has the following sections:\n"
            for s in sections:
                section_str += f"- {s['name'].capitalize()} (lines {s['start']}-{s['end']})\n"
            section_str += "Vary the melody, rhythm, and intensity for each section. Add fills and transitions at section boundaries. "
        melody_style = artist_profile.get('melody_style', 'catchy, singable pop/rock melodies')
        prompt = (
            f"Generate a melody for a {state.get('genre')} song, "
            f"mood: {state.get('mood')}, tempo: {state.get('tempo')} BPM, "
            f"duration: {state.get('duration')} minutes, instruments: {', '.join(state.get('instruments', []))}. "
            f"Base the melody on the following artist's melody style: {melody_style}. "
            f"{section_str}Output ONLY a valid Python list of note dictionaries with pitch, start, end, and velocity. Do not include any explanation or extra text."
        )
        melody_text = gemini_generate(prompt)
        logging.info(f"[MelodyAgent] Raw Gemini output: {melody_text}")
        cleaned = clean_llm_output(melody_text)
        notes = safe_literal_eval(cleaned)
        # Quantize note onsets and collect bar start times
        tempo = state.get('tempo', 120)
        duration = state.get('duration', 2)
        beats_per_bar = 4
        seconds_per_beat = 60.0 / tempo
        total_beats = int((duration * 60) / seconds_per_beat)
        bar_times = [quantize_time(i * beats_per_bar * seconds_per_beat) for i in range(math.ceil(total_beats / beats_per_bar))]
        note_onsets = sorted(set([quantize_time(n['start']) for n in notes])) if notes else []
        # Warn if notes do not cover the full song duration
        if notes:
            max_end = max(n['end'] for n in notes)
            if max_end < duration * 60 - 1:
                logging.warning(f"[MelodyAgent] LLM notes end at {max_end:.2f}s, which is shorter than song duration {duration*60:.2f}s.")
        melody_track = {
            'name': 'Melody',
            'program': 0,
            'is_drum': False,
            'notes': notes
        }
        state['melody'] = {'melody_track': melody_track}
        state['structure'] = {
            'melody_onsets': note_onsets,
            'bar_times': bar_times,
            'tempo': tempo,
            'duration': duration
        }
        logging.info("[MelodyAgent] Melody and structure generated.")
    except Exception as e:
        logging.error(f"[MelodyAgent] Error: {e}")
    return state 