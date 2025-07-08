import logging
from typing import Any
import os
from utils.audio_utils import midi_to_wav, wav_to_mp3

def audio_renderer_agent(state: Any) -> Any:
    logging.info("[AudioRendererAgent] Rendering audio from MIDI.")
    try:
        midi_path = state.get('midi_path')
        if not midi_path or not os.path.exists(midi_path):
            raise FileNotFoundError("MIDI file not found for audio rendering.")
        base = os.path.splitext(os.path.basename(midi_path))[0]
        wav_path = os.path.join('output', f'{base}.wav')
        mp3_path = os.path.join('output', f'{base}.mp3')
        # Convert MIDI to WAV
        midi_to_wav(midi_path, wav_path)
        # Convert WAV to MP3
        wav_to_mp3(wav_path, mp3_path)
        state['mp3_path'] = mp3_path
        logging.info(f"[AudioRendererAgent] MP3 file created at {mp3_path}.")
    except Exception as e:
        logging.error(f"[AudioRendererAgent] Error: {e}")
    return state 