import logging
from typing import Any
from utils.gemini_llm import gemini_generate
import ast
import re

def clean_llm_output(text):
    return re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip(), flags=re.MULTILINE).strip()

def chord_agent(state: Any) -> Any:
    logging.info("[ChordAgent] Generating chords with Gemini LLM.")
    try:
        structure = state.get('structure', {})
        melody_onsets = structure.get('melody_onsets', [])
        artist_profile = state.get('artist_profile', {})
        artist = state.get('user_input', {}).get('artist', '')
        profile_str = f" in the style of {artist}: {artist_profile}" if artist and artist_profile else ""
        prompt = (
            f"Generate a chord progression{profile_str} for a {state.get('genre')} song, "
            f"mood: {state.get('mood')}, tempo: {state.get('tempo')} BPM, "
            f"duration: {state.get('duration')} minutes. "
            f"Align chord changes to these melody note onsets (in seconds): {melody_onsets}. "
            "Output ONLY a valid Python list of chord note dictionaries with pitch, start, end, and velocity. Do not include any explanation or extra text."
        )
        chord_text = gemini_generate(prompt)
        logging.info(f"[ChordAgent] Raw Gemini output: {chord_text}")
        cleaned = clean_llm_output(chord_text)
        try:
            notes = ast.literal_eval(cleaned)
        except Exception as e:
            logging.error(f"[ChordAgent] Failed to parse Gemini output: {e}")
            notes = []
        # Optionally, quantize chord start times to melody_onsets
        for n in notes:
            n['start'] = min(melody_onsets, key=lambda t: abs(t - n['start'])) if melody_onsets else n['start']
        chord_track = {
            'name': 'Chords',
            'program': 24,
            'is_drum': False,
            'notes': notes
        }
        state['chords'] = {'chord_track': chord_track}
        logging.info("[ChordAgent] Chords generated and aligned.")
    except Exception as e:
        logging.error(f"[ChordAgent] Error: {e}")
    return state 