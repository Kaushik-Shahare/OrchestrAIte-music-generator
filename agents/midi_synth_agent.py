import logging
from typing import Any
import os
from datetime import datetime
from utils.midi_utils import create_midi_file, save_midi_file

def midi_synth_agent(state: Any) -> Any:
    logging.info("[MIDISynthAgent] Combining tracks into MIDI file.")
    try:
        # Collect all tracks from state
        tracks = []
        # Melody
        melody = state.get('melody', {}).get('melody_track')
        if melody:
            tracks.append(melody)
        # Chords
        chords = state.get('chords', {}).get('chord_track')
        if chords:
            tracks.append(chords)
        # Instruments
        instrument_tracks = state.get('instrument_tracks', {}).get('instrument_tracks')
        if instrument_tracks:
            if isinstance(instrument_tracks, list):
                tracks.extend(instrument_tracks)
            else:
                tracks.append(instrument_tracks)
        # Drums
        drum_tracks = state.get('drum_tracks', {}).get('drum_track')
        if drum_tracks:
            tracks.append(drum_tracks)
        # Vocals (optional)
        vocals = state.get('vocals_track', {}).get('vocal_track')
        if vocals:
            tracks.append(vocals)
        # Fallback: if any agent returned a flat list of tracks
        if not tracks:
            for key in ['melody', 'chords', 'instrument_tracks', 'drum_tracks', 'vocals_track']:
                t = state.get(key)
                if isinstance(t, list):
                    tracks.extend(t)
        # Set tempo
        tempo = state.get('tempo', 120)
        midi_obj = create_midi_file(tracks, tempo=tempo)
        # Save MIDI file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        midi_path = os.path.join('output', f'song_{timestamp}.mid')
        save_midi_file(midi_obj, midi_path)
        state['midi_path'] = midi_path
        logging.info(f"[MIDISynthAgent] MIDI file created at {midi_path}.")
    except Exception as e:
        logging.error(f"[MIDISynthAgent] Error: {e}")
    return state 