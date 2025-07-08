import logging
from typing import Any, List, Dict

class InstrumentModel:
    def __init__(self):
        logging.info("[InstrumentModel] Initialized.")

    def add_instruments(self, melody: Any, chords: Any, instruments: List[str]) -> Dict[str, Any]:
        logging.info(f"[InstrumentModel] Adding instruments: {instruments}")
        # Example: Add a simple bass and synth pad
        tracks = []
        # Bass
        bass_notes = []
        for i in range(4):
            bass_notes.append({
                'pitch': 36,  # C2
                'start': i * 1.0,
                'end': i * 1.0 + 1.0,
                'velocity': 90
            })
        tracks.append({
            'name': 'Bass',
            'program': 32,  # Acoustic Bass
            'is_drum': False,
            'notes': bass_notes
        })
        # Synth Pad
        pad_notes = []
        for i in range(4):
            pad_notes.append({
                'pitch': 60,  # C4
                'start': i * 1.0,
                'end': i * 1.0 + 1.0,
                'velocity': 60
            })
        tracks.append({
            'name': 'Pad',
            'program': 88,  # Pad 1 (new age)
            'is_drum': False,
            'notes': pad_notes
        })
        return {'tracks': tracks} 