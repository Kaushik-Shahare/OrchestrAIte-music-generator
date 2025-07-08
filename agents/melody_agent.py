import logging
from typing import Any
from models.melody_model import MelodyModel

def melody_agent(state: Any) -> Any:
    logging.info("[MelodyAgent] Generating melody.")
    try:
        model = MelodyModel()
        melody_data = model.generate_melody(
            genre=state.get('genre'),
            mood=state.get('mood'),
            tempo=state.get('tempo'),
            duration=state.get('duration'),
            instruments=state.get('instruments')
        )
        # melody_data should include a list of notes and instrument info
        melody_track = {
            'name': 'Melody',
            'program': 0,  # Acoustic Grand Piano by default
            'is_drum': False,
            'notes': melody_data.get('notes', [])
        }
        state['melody'] = {'melody_track': melody_track}
        logging.info("[MelodyAgent] Melody generated.")
    except Exception as e:
        logging.error(f"[MelodyAgent] Error: {e}")
    return state 