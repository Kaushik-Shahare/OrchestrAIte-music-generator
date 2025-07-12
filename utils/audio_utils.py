import logging
from typing import Any
import os
import subprocess

try:
    import fluidsynth
except ImportError:
    fluidsynth = None

try:
    import soundfile as sf
except ImportError:
    sf = None

def midi_to_wav(midi_path: str, wav_path: str, soundfont: str = None) -> None:
    logging.info(f"[AudioUtils] Converting {midi_path} to WAV at {wav_path}.")
    
    # Try multiple soundfont locations
    soundfont_paths = [
        '/usr/share/sounds/sf2/FluidR3_GM.sf2',  # Linux
        '/System/Library/Components/CoreAudio.component/Contents/Resources/gs_instruments.dls',  # macOS
        '/usr/local/share/soundfonts/default.sf2',  # Homebrew
        'FluidR3_GM.sf2'  # Local file
    ]
    
    if soundfont:
        soundfont_paths.insert(0, soundfont)
    
    # Find available soundfont
    selected_soundfont = None
    for sf in soundfont_paths:
        if os.path.exists(sf):
            selected_soundfont = sf
            break
    
    if not selected_soundfont:
        # Use fluidsynth command line as fallback
        logging.warning("[AudioUtils] No soundfont found, trying command-line fluidsynth")
        try:
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            cmd = ['fluidsynth', '-ni', '-g', '1', '-F', wav_path, midi_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info(f"[AudioUtils] WAV file created at {wav_path} using command-line fluidsynth.")
            return
        except subprocess.CalledProcessError as e:
            logging.error(f"[AudioUtils] Command-line fluidsynth failed: {e}")
            raise
    
    # Use Python fluidsynth
    if fluidsynth is None:
        raise ImportError("pyfluidsynth is required for MIDI to WAV conversion.")
    
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    try:
        fs = fluidsynth.Synth()
        # Check which parameters fs.start accepts
        if hasattr(fs, 'start') and 'filename' in fs.start.__code__.co_varnames:
            # New API: fs.start(driver="file", filename=wav_path)
            fs.start(driver="file", filename=wav_path)
        else:
            # Old API: fs.start() then fs.setting() calls
            fs.start()
            if hasattr(fs, 'setting'):
                fs.setting('synth.gain', 0.5)
        
        sfid = fs.sfload(selected_soundfont)
        fs.program_select(0, sfid, 0, 0)
        fs.midi_file_play(midi_path)
        
        # Make sure we flush any buffered audio data
        if hasattr(fs, 'get_samples'):
            fs.get_samples(fs.midi_file_get_duration(midi_path) * 44100 * 2)
            
        fs.delete()
    except Exception as e:
        logging.error(f"[AudioUtils] Error using Python fluidsynth: {e}")
        # Fallback to command-line fluidsynth
        try:
            cmd = ['fluidsynth', '-ni', '-g', '1', '-F', wav_path, selected_soundfont, midi_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logging.info(f"[AudioUtils] WAV file created at {wav_path} using command-line fluidsynth fallback.")
        except subprocess.CalledProcessError as e:
            logging.error(f"[AudioUtils] Command-line fluidsynth failed: {e}")
            raise
    logging.info(f"[AudioUtils] WAV file created at {wav_path} using soundfont {selected_soundfont}.")

def wav_to_mp3(wav_path: str, mp3_path: str) -> None:
    logging.info(f"[AudioUtils] Converting {wav_path} to MP3 at {mp3_path} using ffmpeg.")
    os.makedirs(os.path.dirname(mp3_path), exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-i', wav_path, '-codec:a', 'libmp3lame', '-qscale:a', '2', mp3_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"[AudioUtils] MP3 file created at {mp3_path}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"[AudioUtils] ffmpeg error: {e.stderr.decode()}")
        raise

def save_audio_file(audio_obj: Any, path: str, samplerate: int = 44100) -> None:
    logging.info(f"[AudioUtils] Saving audio file to {path}.")
    if sf is None:
        raise ImportError("soundfile is required for saving audio files.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio_obj, samplerate)
    logging.info(f"[AudioUtils] Audio file saved at {path}.") 