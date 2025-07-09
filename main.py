import argparse
import logging
from graph.composer_graph import composer_app

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Multi-Agent Music Composer\n\nExample usage:\n  python main.py --genre jazz --mood melancholic --tempo 95 --duration 3 --instruments piano,guitar,bass,drums --vocals true --lyrics 'Your story here' --artist 'Billie Eilish' --song_structure 'verse,chorus,verse,chorus,bridge,chorus' --harmonic_complexity medium --rhythmic_complexity high --dynamic_range wide --song_theme 'lost love' --reference_songs 'Bad Guy,Ocean Eyes'",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Basic musical parameters
    parser.add_argument('--genre', type=str, default='pop', help='Music genre (e.g., jazz, pop, rock, hip-hop, electronic, folk, country, r&b)')
    parser.add_argument('--subgenre', type=str, default='', help='Subgenre for more specificity (e.g., trap, indie-rock, neo-soul, lo-fi)')
    parser.add_argument('--mood', type=str, default='happy', help='Primary mood (e.g., happy, melancholic, energetic, dreamy, aggressive, peaceful)')
    parser.add_argument('--secondary_mood', type=str, default='', help='Secondary mood for complexity (e.g., nostalgic, hopeful, mysterious)')
    parser.add_argument('--tempo', type=int, default=120, help='Tempo in BPM (60-200)')
    parser.add_argument('--duration', type=int, default=2, help='Duration in minutes (1-10)')
    
    # Instrumentation and arrangement
    parser.add_argument('--instruments', type=str, default='piano', help='Comma-separated list of instruments')
    parser.add_argument('--lead_instrument', type=str, default='', help='Primary lead instrument (e.g., guitar, piano, synth, violin)')
    parser.add_argument('--backing_instruments', type=str, default='', help='Comma-separated backing instruments')
    parser.add_argument('--instrument_description', type=str, default='', help='Detailed playing style instructions')
    
    # Song structure and composition
    parser.add_argument('--song_structure', type=str, default='verse,chorus,verse,chorus', help='Song structure (e.g., intro,verse,chorus,verse,chorus,bridge,chorus,outro)')
    parser.add_argument('--key_signature', type=str, default='C major', help='Key signature (e.g., C major, Am minor, F# major)')
    parser.add_argument('--time_signature', type=str, default='4/4', help='Time signature (e.g., 4/4, 3/4, 6/8, 7/8)')
    parser.add_argument('--harmonic_complexity', type=str, default='medium', help='Harmonic complexity: simple, medium, complex')
    parser.add_argument('--rhythmic_complexity', type=str, default='medium', help='Rhythmic complexity: simple, medium, complex')
    
    # Dynamics and production
    parser.add_argument('--dynamic_range', type=str, default='medium', help='Dynamic range: narrow, medium, wide')
    parser.add_argument('--production_style', type=str, default='modern', help='Production style (e.g., vintage, modern, lo-fi, polished, raw)')
    parser.add_argument('--effects', type=str, default='', help='Desired effects (e.g., reverb, delay, distortion, chorus)')
    
    # Vocals and lyrics
    parser.add_argument('--vocals', type=str, default='false', help='Enable vocals (true/false)')
    parser.add_argument('--vocal_style', type=str, default='', help='Vocal style (e.g., smooth, raspy, powerful, whispery, belting)')
    parser.add_argument('--lyrics', type=str, default='', help='Lyrics for the vocal track')
    parser.add_argument('--song_theme', type=str, default='', help='Song theme/subject (e.g., love, heartbreak, celebration, social commentary)')
    parser.add_argument('--lyrical_style', type=str, default='', help='Lyrical style (e.g., storytelling, abstract, conversational, poetic)')
    
    # Artist and reference context
    parser.add_argument('--artist', type=str, default='', help='Artist/style to emulate')
    parser.add_argument('--reference_songs', type=str, default='', help='Comma-separated reference songs for style inspiration')
    parser.add_argument('--era', type=str, default='', help='Musical era (e.g., 60s, 80s, 90s, 2000s, 2010s, current)')
    parser.add_argument('--influences', type=str, default='', help='Musical influences (comma-separated artists/genres)')
    
    # Advanced musical concepts
    parser.add_argument('--modulations', type=str, default='', help='Desired key modulations (e.g., up_half_step, relative_minor)')
    parser.add_argument('--special_techniques', type=str, default='', help='Special techniques (e.g., polyrhythm, odd_meters, modal_interchange)')
    parser.add_argument('--emotional_arc', type=str, default='', help='Emotional journey (e.g., sad_to_hopeful, calm_to_explosive, mysterious_to_resolved)')
    
    args = parser.parse_args()

    # Parse instruments and vocals
    instruments = [i.strip() for i in args.instruments.split(',')]
    backing_instruments = [i.strip() for i in args.backing_instruments.split(',')] if args.backing_instruments else []
    reference_songs = [s.strip() for s in args.reference_songs.split(',')] if args.reference_songs else []
    influences = [i.strip() for i in args.influences.split(',')] if args.influences else []
    song_structure = [s.strip() for s in args.song_structure.split(',')]
    vocals = args.vocals.lower() in ['true', '1', 'yes', 'y']

    # Build comprehensive state
    user_input = {
        # Basic parameters
        'genre': args.genre,
        'subgenre': args.subgenre,
        'mood': args.mood,
        'secondary_mood': args.secondary_mood,
        'tempo': args.tempo,
        'duration': args.duration,
        
        # Instrumentation
        'instruments': instruments,
        'lead_instrument': args.lead_instrument,
        'backing_instruments': backing_instruments,
        'instrument_description': args.instrument_description,
        
        # Song structure
        'song_structure': song_structure,
        'key_signature': args.key_signature,
        'time_signature': args.time_signature,
        'harmonic_complexity': args.harmonic_complexity,
        'rhythmic_complexity': args.rhythmic_complexity,
        
        # Production
        'dynamic_range': args.dynamic_range,
        'production_style': args.production_style,
        'effects': args.effects,
        
        # Vocals
        'vocals': vocals,
        'vocal_style': args.vocal_style,
        'lyrics': args.lyrics,
        'song_theme': args.song_theme,
        'lyrical_style': args.lyrical_style,
        
        # Artist context
        'artist': args.artist,
        'reference_songs': reference_songs,
        'era': args.era,
        'influences': influences,
        
        # Advanced concepts
        'modulations': args.modulations,
        'special_techniques': args.special_techniques,
        'emotional_arc': args.emotional_arc
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