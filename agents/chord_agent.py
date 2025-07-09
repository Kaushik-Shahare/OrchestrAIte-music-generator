import logging
from typing import Any
from utils.gemini_llm import gemini_generate
import ast
import re

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
            logging.error(f"[ChordAgent] Failed to parse LLM output after auto-fix: {e}")
            return []

def notes_array_to_dicts(notes_array):
    return [
        {'pitch': n[0], 'start': n[1], 'end': n[2], 'velocity': n[3]}
        for n in notes_array if isinstance(n, (list, tuple)) and len(n) == 4
    ]

def chord_agent(state: Any) -> Any:
    logging.info("[ChordAgent] Generating chords with Gemini LLM.")
    try:
        structure = state.get('structure', {})
        melody_onsets = structure.get('melody_onsets', [])
        artist_profile = state.get('artist_profile', {})
        artist = state.get('user_input', {}).get('artist', '')
        chord_style = artist_profile.get('chord_style', 'standard pop/rock chord voicings')
        profile_str = f" in the style of {artist}: {artist_profile}" if artist and artist_profile else ""
        sections = state.get('sections', [])
        section_str = ""
        if sections:
            section_str = "The song has the following sections:\n"
            for s in sections:
                section_str += f"- {s['name'].capitalize()} (lines {s['start']}-{s['end']})\n"
            section_str += "Vary the chords and voicings for each section. Add transitions at section boundaries. "
        prompt = (
            f"Generate a chord progression for a {state.get('genre')} song, "
            f"mood: {state.get('mood')}, tempo: {state.get('tempo')} BPM, "
            f"duration: {state.get('duration')} minutes. "
            f"Align chord changes to these melody note onsets (in seconds): {melody_onsets}. "
            f"Base the chord progression, voicings, and rhythm on the following artist's chord style: {chord_style}. "
            f"Do NOT add synth, lead, arpeggio, or non-chord notes. Only generate chord notes for the chord track. "
            f"{section_str}Output ONLY a valid Python list of lists, where each sublist is [pitch, start, end, velocity] in that order. Do not include any explanation or extra text."
        )
        chord_text = gemini_generate(prompt)
        logging.info(f"[ChordAgent] Raw Gemini output: {chord_text}")
        cleaned = clean_llm_output(chord_text)
        notes_array = safe_literal_eval(cleaned)
        notes = notes_array_to_dicts(notes_array)
        # Optionally, quantize chord start times to melody_onsets
        for n in notes:
            n['start'] = min(melody_onsets, key=lambda t: abs(t - n['start'])) if melody_onsets else n['start']
        # Warn if notes do not cover the full song duration
        duration = state.get('duration', 2)
        if notes:
            max_end = max(n['end'] for n in notes)
            if max_end < duration * 60 - 1:
                logging.warning(f"[ChordAgent] LLM notes end at {max_end:.2f}s, which is shorter than song duration {duration*60:.2f}s.")
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