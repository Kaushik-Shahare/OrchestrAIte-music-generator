import logging
from utils.gemini_llm import gemini_generate
import ast
import re

def clean_llm_output(text):
    return re.sub(r'^```[a-zA-Z]*\n?|```$', '', text.strip(), flags=re.MULTILINE).strip()

def artist_style_agent(state: dict) -> dict:
    artist = state.get('user_input', {}).get('artist', '')
    if not artist:
        state['artist_profile'] = {}
        return state
    prompt = (
        f"Describe the musical style, arrangement, lyrical themes, and production techniques typical of {artist}. "
        "Output as a Python dict with keys: 'melody_style', 'chord_style', 'drum_style', 'vocal_style', 'lyric_themes', 'arrangement'. Do not include any explanation or extra text."
    )
    profile_text = gemini_generate(prompt)
    try:
        profile = ast.literal_eval(clean_llm_output(profile_text))
    except Exception as e:
        logging.error(f"[ArtistStyleAgent] Failed to parse Gemini output: {e}")
        profile = {}
    state['artist_profile'] = profile
    logging.info(f"[ArtistStyleAgent] Artist profile generated for {artist}.")
    return state 