import logging
from typing import Any
from utils.gemini_llm import gemini_generate
import ast

def drum_agent(state: Any) -> Any:
    logging.info("[DrumAgent] Adding drum and percussion tracks with Gemini LLM.")
    try:
        prompt = (
            f"Generate a drum and percussion track for a {state.get('genre')} song, "
            f"mood: {state.get('mood')}, tempo: {state.get('tempo')} BPM, "
            f"duration: {state.get('duration')} minutes. "
            "Output as a Python list of note dicts (pitch, start, end, velocity) for a drum track."
        )
        drum_text = gemini_generate(prompt)
        notes = ast.literal_eval(drum_text)
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