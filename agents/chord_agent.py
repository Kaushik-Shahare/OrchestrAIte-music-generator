import logging
from typing import Any
from models.chord_model import ChordModel

def chord_agent(state: Any) -> Any:
    logging.info("[ChordAgent] Generating chords.")
    try:
        model = ChordModel()
        chord_data = model.generate_chords(
            melody=state.get('melody'),
            genre=state.get('genre'),
            mood=state.get('mood')
        )
        chord_track = {
            'name': 'Chords',
            'program': 24,  # Nylon Guitar by default
            'is_drum': False,
            'notes': chord_data.get('notes', [])
        }
        state['chords'] = {'chord_track': chord_track}
        logging.info("[ChordAgent] Chords generated.")
    except Exception as e:
        logging.error(f"[ChordAgent] Error: {e}")
    return state 