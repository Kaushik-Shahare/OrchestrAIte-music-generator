import logging
from typing import Any

def drum_agent(state: Any) -> Any:
    logging.info("[DrumAgent] Adding drum and percussion tracks.")
    try:
        # Example: Add a simple drum pattern (kick and snare)
        tempo = state.get('tempo', 120)
        duration = state.get('duration', 2)
        beat_length = 60.0 / tempo
        total_beats = int(duration * tempo / 60)
        notes = []
        for i in range(total_beats):
            # Kick on every beat
            notes.append({
                'pitch': 36,  # General MIDI: Bass Drum
                'start': i * beat_length,
                'end': i * beat_length + 0.2,
                'velocity': 100
            })
            # Snare on every other beat
            if i % 2 == 1:
                notes.append({
                    'pitch': 38,  # General MIDI: Snare Drum
                    'start': i * beat_length,
                    'end': i * beat_length + 0.2,
                    'velocity': 100
                })
        drum_track = {
            'name': 'Drums',
            'program': 0,  # Program is ignored for drums
            'is_drum': True,
            'notes': notes
        }
        state['drum_tracks'] = {'drum_track': drum_track}
        logging.info("[DrumAgent] Drum tracks added.")
    except Exception as e:
        logging.error(f"[DrumAgent] Error: {e}")
    return state 