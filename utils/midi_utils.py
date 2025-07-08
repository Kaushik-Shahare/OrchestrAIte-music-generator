import logging
from typing import Any, List, Dict
import pretty_midi
import os

def create_midi_file(tracks: List[Dict[str, Any]], tempo: int = 120) -> pretty_midi.PrettyMIDI:
    """
    Create a PrettyMIDI object from a list of track dicts.
    Each track dict should have: name, program, is_drum, notes (list of dicts: pitch, start, end, velocity)
    """
    logging.info("[MIDIUtils] Creating MIDI file from tracks.")
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for track in tracks:
        program = track.get('program', 0)
        is_drum = track.get('is_drum', False)
        name = track.get('name', 'Instrument')
        instrument = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
        for note in track.get('notes', []):
            n = pretty_midi.Note(
                velocity=note.get('velocity', 100),
                pitch=note['pitch'],
                start=note['start'],
                end=note['end']
            )
            instrument.notes.append(n)
        midi.instruments.append(instrument)
    return midi

def save_midi_file(midi_obj: pretty_midi.PrettyMIDI, path: str) -> None:
    logging.info(f"[MIDIUtils] Saving MIDI file to {path}.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    midi_obj.write(path)

def add_track_to_midi(midi_obj: pretty_midi.PrettyMIDI, track: Dict[str, Any]) -> pretty_midi.PrettyMIDI:
    logging.info("[MIDIUtils] Adding track to MIDI object.")
    program = track.get('program', 0)
    is_drum = track.get('is_drum', False)
    name = track.get('name', 'Instrument')
    instrument = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    for note in track.get('notes', []):
        n = pretty_midi.Note(
            velocity=note.get('velocity', 100),
            pitch=note['pitch'],
            start=note['start'],
            end=note['end']
        )
        instrument.notes.append(n)
    midi_obj.instruments.append(instrument)
    return midi_obj 