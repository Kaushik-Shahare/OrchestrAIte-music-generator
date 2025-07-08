import logging
from typing import Any
import os

def export_agent(state: Any) -> Any:
    logging.info("[ExportAgent] Exporting generated files to output directory.")
    try:
        midi_path = state.get('midi_path')
        mp3_path = state.get('mp3_path')
        if midi_path and os.path.exists(midi_path):
            logging.info(f"[ExportAgent] MIDI generated: {midi_path}")
            print(f"\n🎵 MIDI generated: {midi_path}")
        else:
            logging.warning("[ExportAgent] MIDI file missing.")
        if mp3_path and os.path.exists(mp3_path):
            logging.info(f"[ExportAgent] MP3 generated: {mp3_path}")
            print(f"🎶 MP3 generated: {mp3_path}\n")
        else:
            logging.warning("[ExportAgent] MP3 file missing.")
        state['result'] = {'midi': midi_path, 'mp3': mp3_path}
    except Exception as e:
        logging.error(f"[ExportAgent] Error: {e}")
    return state 