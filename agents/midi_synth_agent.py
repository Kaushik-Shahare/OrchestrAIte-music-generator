import logging
from typing import Any
import os
from datetime import datetime
from utils.midi_utils import create_midi_file, save_midi_file
from utils.state_utils import validate_agent_return, safe_state_update

def midi_synth_agent(state: Any) -> Any:
    """
    Combine only requested instrument tracks into MIDI file with strict filtering.
    """
    logging.info("[MIDISynthAgent] Combining tracks into MIDI file with strict instrument filtering.")
    
    try:
        # Get requested instruments from user input
        requested_instruments = state.get('instruments', ['piano'])
        lead_instrument = state.get('lead_instrument', '')
        backing_instruments = state.get('backing_instruments', [])
        vocals_enabled = state.get('vocals', False)
        
        # Create comprehensive list of allowed instruments
        allowed_instruments = set()
        for instr in requested_instruments:
            allowed_instruments.add(instr.lower().strip())
        
        if lead_instrument:
            allowed_instruments.add(lead_instrument.lower().strip())
        
        for instr in backing_instruments:
            allowed_instruments.add(instr.lower().strip())
        
        # Always allow drums and melody track
        allowed_instruments.add('drums')
        allowed_instruments.add('melody')
        allowed_instruments.add('chords')
        
        logging.info(f"[MIDISynthAgent] Allowed instruments: {allowed_instruments}")
        
        # Collect all tracks from state with strict filtering
        tracks = []
        
        # Always include melody track (core musical element)
        melody = state.get('melody', {}).get('melody_track')
        if melody and melody.get('notes'):
            tracks.append(melody)
            logging.info("[MIDISynthAgent] Added melody track")
        
        # Always include chord track (harmonic foundation)
        chords = state.get('chords', {}).get('chord_track')
        if chords and chords.get('notes'):
            tracks.append(chords)
            logging.info("[MIDISynthAgent] Added chord track")
        
        # Filter instrument tracks strictly
        instrument_tracks = state.get('instrument_tracks', {}).get('instrument_tracks', [])
        if instrument_tracks:
            for track in instrument_tracks:
                track_name = track.get('name', '').lower()
                track_instrument = track.get('instrument', '').lower()
                
                # Check if this track matches any allowed instrument
                is_allowed = False
                for allowed_instr in allowed_instruments:
                    if (allowed_instr in track_name or 
                        allowed_instr in track_instrument or
                        allowed_instr == track_instrument):
                        is_allowed = True
                        break
                
                if is_allowed and track.get('notes'):
                    tracks.append(track)
                    logging.info(f"[MIDISynthAgent] Added instrument track: {track.get('name', 'Unknown')}")
                else:
                    logging.info(f"[MIDISynthAgent] FILTERED OUT unwanted track: {track.get('name', 'Unknown')}")
        
        # Always include drums if present
        drum_tracks = state.get('drum_tracks', {}).get('drum_track')
        if drum_tracks and drum_tracks.get('notes'):
            tracks.append(drum_tracks)
            logging.info("[MIDISynthAgent] Added drum track")
        
        # Include vocals only if explicitly enabled
        if vocals_enabled:
            vocals = state.get('vocals_track', {}).get('vocal_track')
            if vocals and vocals.get('notes'):
                tracks.append(vocals)
                logging.info("[MIDISynthAgent] Added vocal track")
        
        # Validate tracks have content
        valid_tracks = []
        for track in tracks:
            notes = track.get('notes', [])
            if notes and len(notes) > 0:
                valid_tracks.append(track)
            else:
                logging.warning(f"[MIDISynthAgent] Skipping empty track: {track.get('name', 'Unknown')}")
        
        if not valid_tracks:
            logging.error("[MIDISynthAgent] No valid tracks found!")
            return safe_state_update(state, {'midi_path': None}, "MIDISynthAgent")
        
        # Set tempo and create MIDI
        tempo = state.get('tempo', 120)
        midi_obj = create_midi_file(valid_tracks, tempo=tempo)
        
        # Save MIDI file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('output', exist_ok=True)
        midi_path = os.path.join('output', f'song_{timestamp}.mid')
        save_midi_file(midi_obj, midi_path)
        
        logging.info(f"[MIDISynthAgent] MIDI file created at {midi_path} with {len(valid_tracks)} tracks.")
        
        # Log track summary
        track_summary = []
        for track in valid_tracks:
            track_name = track.get('name', 'Unknown')
            note_count = len(track.get('notes', []))
            track_summary.append(f"{track_name} ({note_count} notes)")
        
        logging.info(f"[MIDISynthAgent] Final tracks: {', '.join(track_summary)}")
        
        return safe_state_update(state, {'midi_path': midi_path}, "MIDISynthAgent")
        
    except Exception as e:
        logging.error(f"[MIDISynthAgent] Error: {e}")
        return safe_state_update(state, {'midi_path': None}, "MIDISynthAgent") 