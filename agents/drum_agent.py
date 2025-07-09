import logging
from typing import Any
from utils.gemini_llm import gemini_generate
import ast

def notes_array_to_dicts(notes_array):
    return [
        {'pitch': n[0], 'start': n[1], 'end': n[2], 'velocity': n[3]}
        for n in notes_array if isinstance(n, (list, tuple)) and len(n) == 4
    ]

def drum_agent(state: Any) -> Any:
    logging.info("[DrumAgent] Adding drum and percussion tracks with Gemini LLM.")
    try:
        prompt = (
            f"Generate a drum and percussion track for a {state.get('genre')} song, "
            f"mood: {state.get('mood')}, tempo: {state.get('tempo')} BPM, "
            f"duration: {state.get('duration')} minutes. "
            f"Align drum patterns to these bar start times (in seconds): {bar_times}. "
            f"Base the drum patterns, fills, and sound on the following artist's drum style: {drum_style}. "
            f"{section_str}Output ONLY a valid Python list of lists, where each sublist is [pitch, start, end, velocity] in that order. Do not include any explanation or extra text."
        )
        drum_text = gemini_generate(prompt)
        logging.info(f"[DrumAgent] Raw Gemini output: {drum_text}")
        cleaned = clean_llm_output(drum_text)
        notes_array = safe_literal_eval(cleaned)
        notes = notes_array_to_dicts(notes_array)
        drum_track = {
            'name': 'Drums',
            'program': 0,
            'is_drum': True,
            'notes': notes
        }
        state['drum_tracks'] = {'drum_track': drum_track}
        logging.info("[DrumAgent] Drum tracks added.")
    except Exception as e:
        logging.error(f"[DrumAgent] Error: {e}")
    return state 