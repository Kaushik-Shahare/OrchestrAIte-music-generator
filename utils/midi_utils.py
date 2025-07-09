import logging
from typing import Any, List, Dict
import pretty_midi
import os
import re

# Helper function to convert note name (e.g., 'E3') to MIDI number
NOTE_NAME_TO_MIDI = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
def note_name_to_midi(note_name):
    match = re.match(r'^([A-Ga-g][#b]?)(-?\d+)$', note_name)
    if not match:
        return None
    name = match.group(1).capitalize()
    octave = int(match.group(2))
    if name not in NOTE_NAME_TO_MIDI:
        return None
    return 12 * (octave + 1) + NOTE_NAME_TO_MIDI[name]

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
            # Ensure pitch and velocity are valid MIDI integers, handle string values and note names
            raw_pitch = note['pitch']
            if isinstance(raw_pitch, str):
                try:
                    raw_pitch = float(raw_pitch)
                except ValueError:
                    midi_pitch = note_name_to_midi(raw_pitch)
                    if midi_pitch is None:
                        logging.warning(f"[MIDIUtils] Skipping note with invalid pitch: {raw_pitch}")
                        continue
                    raw_pitch = midi_pitch
            pitch = int(round(raw_pitch))
            pitch = max(0, min(127, pitch))
            raw_velocity = note.get('velocity', 100)
            if isinstance(raw_velocity, str):
                raw_velocity = float(raw_velocity)
            velocity = int(round(raw_velocity))
            velocity = max(0, min(127, velocity))
            n = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
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
        # Ensure pitch and velocity are valid MIDI integers, handle string values and note names
        raw_pitch = note['pitch']
        if isinstance(raw_pitch, str):
            try:
                raw_pitch = float(raw_pitch)
            except ValueError:
                midi_pitch = note_name_to_midi(raw_pitch)
                if midi_pitch is None:
                    logging.warning(f"[MIDIUtils] Skipping note with invalid pitch: {raw_pitch}")
                    continue
                raw_pitch = midi_pitch
        pitch = int(round(raw_pitch))
        pitch = max(0, min(127, pitch))
        raw_velocity = note.get('velocity', 100)
        if isinstance(raw_velocity, str):
            raw_velocity = float(raw_velocity)
        velocity = int(round(raw_velocity))
        velocity = max(0, min(127, velocity))
        n = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=note['start'],
            end=note['end']
        )
        instrument.notes.append(n)
    midi_obj.instruments.append(instrument)
    return midi_obj 