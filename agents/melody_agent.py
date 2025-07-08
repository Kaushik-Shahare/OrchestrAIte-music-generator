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

def melody_agent(state: Any) -> Any:
    logging.info("[MelodyAgent] Generating melody with Gemini LLM.")
    try:
        prompt = (
            f"Generate a simple melody in MIDI note format for a {state.get('genre')} song, "
            f"mood: {state.get('mood')}, tempo: {state.get('tempo')} BPM, "
            f"duration: {state.get('duration')} minutes, instruments: {', '.join(state.get('instruments', []))}. "
            "Output ONLY a valid Python list of note dictionaries with pitch, start, end, and velocity. Do not include any explanation or extra text."
        )
        melody_text = gemini_generate(prompt)
        logging.info(f"[MelodyAgent] Raw Gemini output: {melody_text}")
        cleaned = clean_llm_output(melody_text)
        try:
            notes = ast.literal_eval(cleaned)
        except Exception as e:
            logging.error(f"[MelodyAgent] Failed to parse Gemini output: {e}")
            notes = []
        # Quantize note onsets and collect bar start times
        tempo = state.get('tempo', 120)
        duration = state.get('duration', 2)
        beats_per_bar = 4
        seconds_per_beat = 60.0 / tempo
        total_beats = int((duration * 60) / seconds_per_beat)
        bar_times = [quantize_time(i * beats_per_bar * seconds_per_beat) for i in range(math.ceil(total_beats / beats_per_bar))]
        note_onsets = sorted(set([quantize_time(n['start']) for n in notes])) if notes else []
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