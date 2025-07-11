#!/usr/bin/env python3
"""
Auto-play generated music files
Usage: python auto_play.py [file_path]
"""

import sys
import os
import subprocess
import glob
from pathlib import Path

def find_latest_music_file():
    """Find the most recently generated music file."""
    output_dir = Path("output")
    if not output_dir.exists():
        return None
    
    # Look for MP3 files first, then MIDI
    mp3_files = list(output_dir.glob("*.mp3"))
    midi_files = list(output_dir.glob("*.mid"))
    
    all_files = mp3_files + midi_files
    if not all_files:
        return None
    
    # Return the most recent file
    latest_file = max(all_files, key=lambda f: f.stat().st_mtime)
    return latest_file

def play_file(file_path):
    """Play a music file using the system default player."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"🎵 Playing: {file_path.name}")
    
    try:
        if sys.platform.startswith('darwin'):  # macOS
            subprocess.run(['open', str(file_path)], check=True)
        elif sys.platform.startswith('win'):  # Windows
            subprocess.run(['start', str(file_path)], shell=True, check=True)
        else:  # Linux
            subprocess.run(['xdg-open', str(file_path)], check=True)
        
        print(f"✅ Successfully opened: {file_path.name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to play file: {e}")
        
        # Fallback for MIDI files
        if file_path.suffix.lower() == '.mid':
            try:
                print("🎹 Trying fallback MIDI player...")
                if Path("play_midi.py").exists():
                    subprocess.run([sys.executable, "play_midi.py", str(file_path)], check=True)
                    print("✅ MIDI playback successful")
                    return True
            except Exception:
                pass
        
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    if len(sys.argv) > 1:
        # Play specific file
        file_path = sys.argv[1]
        play_file(file_path)
    else:
        # Play latest generated file
        latest_file = find_latest_music_file()
        if latest_file:
            print(f"🔍 Found latest file: {latest_file.name}")
            play_file(latest_file)
        else:
            print("❌ No music files found in output/ directory")
            print("🎵 Generate some music first using main.py")

if __name__ == "__main__":
    main()
