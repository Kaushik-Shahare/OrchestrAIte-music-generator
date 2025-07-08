import argparse
import logging
from graph.composer_graph import composer_app

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Multi-Agent Music Composer\n\nExample usage:\n  python main.py --genre jazz --mood happy --tempo 120 --duration 2 --instruments piano,guitar --vocals false --lyrics 'Your lyrics here'",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--genre', type=str, default='pop', help='Music genre (e.g., jazz, pop, rock)')
    parser.add_argument('--mood', type=str, default='happy', help='Mood (e.g., happy, sad, energetic)')
    parser.add_argument('--tempo', type=int, default=120, help='Tempo in BPM')
    parser.add_argument('--duration', type=int, default=2, help='Duration in minutes')
    parser.add_argument('--instruments', type=str, default='piano', help='Comma-separated list of instruments')
    parser.add_argument('--vocals', type=str, default='false', help='Enable vocals (true/false)')
    parser.add_argument('--lyrics', type=str, default='', help='Lyrics for the vocal track (optional)')
    args = parser.parse_args()

    # Parse instruments and vocals
    instruments = [i.strip() for i in args.instruments.split(',')]
    vocals = args.vocals.lower() in ['true', '1', 'yes', 'y']

    # Build initial state as a dict
    user_input = {
        'genre': args.genre,
        'mood': args.mood,
        'tempo': args.tempo,
        'duration': args.duration,
        'instruments': instruments,
        'vocals': vocals,
        'lyrics': args.lyrics
    }
    state = {'user_input': user_input}

    try:
        logging.info("[Main] Starting music generation pipeline...")
        result = composer_app.invoke(state)
        output = result.get('result', {})
        if output:
            print(f"\n🎵 MIDI generated: {output.get('midi')}")
            print(f"🎶 MP3 generated: {output.get('mp3')}\n")
        else:
            print("\n[Main] Music generation failed. Check logs for details.\n")
    except Exception as e:
        logging.error(f"[Main] Error: {e}")
        print(f"\n[Main] Error: {e}\n")

if __name__ == '__main__':
    main() 