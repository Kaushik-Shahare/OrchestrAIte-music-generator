import logging
from typing import Any
from utils.gemini_llm import gemini_generate
import ast
import re

def clean_llm_output(text):
    return re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip(), flags=re.MULTILINE).strip()

def vocal_agent(state: Any) -> Any:
    logging.info("[VocalAgent] Generating vocals with Gemini LLM (if enabled).")
    try:
        if state.get('vocals'):
            structure = state.get('structure', {})
            melody_onsets = structure.get('melody_onsets', [])
            artist_profile = state.get('artist_profile', {})
            artist = state.get('user_input', {}).get('artist', '')
            profile_str = f" in the style of {artist}: {artist_profile}" if artist and artist_profile else ""
            prompt = (
                f"Generate a vocal melody track{profile_str} for a {state.get('genre')} song, "
                f"mood: {state.get('mood')}, tempo: {state.get('tempo')} BPM, "
                f"duration: {state.get('duration')} minutes. "
                f"Align vocal melody to these melody note onsets (in seconds): {melody_onsets}. "
                "Output ONLY a valid Python list of note dicts (pitch, start, end, velocity) for a vocal track. Do not include any explanation or extra text."
            )
            vocal_text = gemini_generate(prompt)
            logging.info(f"[VocalAgent] Raw Gemini output: {vocal_text}")
            cleaned = clean_llm_output(vocal_text)
            try:
                notes = ast.literal_eval(cleaned)
            except Exception as e:
                logging.error(f"[VocalAgent] Failed to parse Gemini output: {e}")
                notes = []
            # Optionally, quantize note start times to melody_onsets
            for n in notes:
                if melody_onsets:
                    n['start'] = min(melody_onsets, key=lambda t: abs(t - n['start']))
            vocal_track = {
                'name': 'Vocals',
                'program': 54,
                'is_drum': False,
                'notes': notes
            }
            state['vocals_track'] = {'vocal_track': vocal_track}
            logging.info("[VocalAgent] Vocals generated and aligned.")
        else:
            logging.info("[VocalAgent] Vocals not enabled, skipping.")
    except Exception as e:
        logging.error(f"[VocalAgent] Error: {e}")
    return state 