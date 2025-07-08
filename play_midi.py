import sys
import time

try:
    import pygame
except ImportError:
    print("pygame is required. Install it with: pip install pygame")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python play_midi.py <path_to_midi_file>")
    sys.exit(1)

midi_path = sys.argv[1]

pygame.init()
try:
    pygame.mixer.init()
    print(f"Loading MIDI file: {midi_path}")
    pygame.mixer.music.load(midi_path)
    print("Playing... Press Ctrl+C to stop.")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.5)
except Exception as e:
    print(f"Error playing MIDI file: {e}")
finally:
    pygame.mixer.music.stop()
    pygame.quit() 