import logging
from typing import Any
from utils.gemini_llm import gemini_generate
import ast
import re

def clean_llm_output(text):
    return re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip(), flags=re.MULTILINE).strip()

def drum_agent(state: Any) -> Any:
    logging.info("[DrumAgent] Adding drum and percussion tracks with Gemini LLM.")
    try:
        structure = state.get('structure', {})
        bar_times = structure.get('bar_times', [])
        artist_profile = state.get('artist_profile', {})
        artist = state.get('user_input', {}).get('artist', '')
        profile_str = f" in the style of {artist}: {artist_profile}" if artist and artist_profile else ""
        prompt = (
            f"Generate a drum and percussion track{profile_str} for a {state.get('genre')} song, "
            f"mood: {state.get('mood')}, tempo: {state.get('tempo')} BPM, "
            f"duration: {state.get('duration')} minutes. "
            f"Align drum patterns to these bar start times (in seconds): {bar_times}. "
            "Output ONLY a valid Python list of note dicts (pitch, start, end, velocity) for a drum track. Do not include any explanation or extra text."
        )
        drum_text = gemini_generate(prompt)
        logging.info(f"[DrumAgent] Raw Gemini output: {drum_text}")
        cleaned = clean_llm_output(drum_text)
        try:
            notes = ast.literal_eval(cleaned)
        except Exception as e:
            logging.error(f"[DrumAgent] Failed to parse Gemini output: {e}")
            notes = []
        # Optionally, quantize drum note start times to bar_times
        for n in notes:
            n['start'] = min(bar_times, key=lambda t: abs(t - n['start'])) if bar_times else n['start']
        drum_track = {
            'name': 'Drums',
            'program': 0,
            'is_drum': True,
            'notes': notes
        }
        state['drum_tracks'] = {'drum_track': drum_track}
        logging.info("[DrumAgent] Drum tracks added and aligned.")
    except Exception as e:
        logging.error(f"[DrumAgent] Error: {e}")
    return state 