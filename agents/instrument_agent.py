import logging
from typing import Any
from utils.gemini_llm import gemini_generate
import ast
import re

def clean_llm_output(text):
    return re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip(), flags=re.MULTILINE).strip()

def instrument_agent(state: Any) -> Any:
    logging.info("[InstrumentAgent] Layering instruments with Gemini LLM.")
    try:
        structure = state.get('structure', {})
        melody_onsets = structure.get('melody_onsets', [])
        bar_times = structure.get('bar_times', [])
        artist_profile = state.get('artist_profile', {})
        artist = state.get('user_input', {}).get('artist', '')
        profile_str = f" in the style of {artist}: {artist_profile}" if artist and artist_profile else ""
        prompt = (
            f"Generate additional instrument tracks{profile_str} for a {state.get('genre')} song, "
            f"mood: {state.get('mood')}, tempo: {state.get('tempo')} BPM, "
            f"duration: {state.get('duration')} minutes, instruments: {', '.join(state.get('instruments', []))}. "
            f"Align instrument note start times to these melody onsets or bar start times (in seconds): {melody_onsets} or {bar_times}. "
            "Output ONLY a valid Python list of track dicts, each with name, program, is_drum, and a list of note dicts (pitch, start, end, velocity). Do not include any explanation or extra text."
        )
        instrument_text = gemini_generate(prompt)
        logging.info(f"[InstrumentAgent] Raw Gemini output: {instrument_text}")
        cleaned = clean_llm_output(instrument_text)
        try:
            tracks = ast.literal_eval(cleaned)
        except Exception as e:
            logging.error(f"[InstrumentAgent] Failed to parse Gemini output: {e}")
            tracks = []
        # Optionally, quantize note start times
        for track in tracks:
            for n in track.get('notes', []):
                if melody_onsets:
                    n['start'] = min(melody_onsets, key=lambda t: abs(t - n['start']))
                elif bar_times:
                    n['start'] = min(bar_times, key=lambda t: abs(t - n['start']))
        state['instrument_tracks'] = {'instrument_tracks': tracks}
        logging.info("[InstrumentAgent] Instrument tracks added and aligned.")
    except Exception as e:
        logging.error(f"[InstrumentAgent] Error: {e}")
    return state 