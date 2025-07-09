import logging
from typing import Any, Dict
from utils.lyrics_utils import parse_lyrics_sections

def user_input_agent(state: Any) -> Any:
    """
    Parse comprehensive user input parameters and build detailed musical context.
    """
    logging.info("[UserInputAgent] Parsing comprehensive user input parameters.")
    
    # Extract all user input parameters
    params = state.get('user_input', {})
    
    # Basic musical parameters
    genre = params.get('genre', 'pop')
    subgenre = params.get('subgenre', '')
    mood = params.get('mood', 'happy')
    secondary_mood = params.get('secondary_mood', '')
    tempo = int(params.get('tempo', 120))
    duration = int(params.get('duration', 2))
    
    # Instrumentation
    instruments = params.get('instruments', ['piano'])
    if isinstance(instruments, str):
        instruments = [i.strip() for i in instruments.split(',')]
    lead_instrument = params.get('lead_instrument', '')
    backing_instruments = params.get('backing_instruments', [])
    if isinstance(backing_instruments, str) and backing_instruments:
        backing_instruments = [i.strip() for i in backing_instruments.split(',')]
    
    # Song structure and theory
    song_structure = params.get('song_structure', ['verse', 'chorus', 'verse', 'chorus'])
    if isinstance(song_structure, str):
        song_structure = [s.strip() for s in song_structure.split(',')]
    key_signature = params.get('key_signature', 'C major')
    time_signature = params.get('time_signature', '4/4')
    harmonic_complexity = params.get('harmonic_complexity', 'medium')
    rhythmic_complexity = params.get('rhythmic_complexity', 'medium')
    
    # Production and dynamics
    dynamic_range = params.get('dynamic_range', 'medium')
    production_style = params.get('production_style', 'modern')
    effects = params.get('effects', '')
    
    # Vocals and lyrics
    vocals = bool(params.get('vocals', False))
    vocal_style = params.get('vocal_style', '')
    lyrics = params.get('lyrics', '')
    song_theme = params.get('song_theme', '')
    lyrical_style = params.get('lyrical_style', '')
    
    # Artist and reference context
    artist = params.get('artist', '')
    reference_songs = params.get('reference_songs', [])
    if isinstance(reference_songs, str) and reference_songs:
        reference_songs = [s.strip() for s in reference_songs.split(',')]
    era = params.get('era', '')
    influences = params.get('influences', [])
    if isinstance(influences, str) and influences:
        influences = [i.strip() for i in influences.split(',')]
    
    # Advanced concepts
    modulations = params.get('modulations', '')
    special_techniques = params.get('special_techniques', '')
    emotional_arc = params.get('emotional_arc', '')
    instrument_description = params.get('instrument_description', '')
    
    # Store all parameters in state
    state.update({
        'genre': genre,
        'subgenre': subgenre,
        'mood': mood,
        'secondary_mood': secondary_mood,
        'tempo': tempo,
        'duration': duration,
        'instruments': instruments,
        'lead_instrument': lead_instrument,
        'backing_instruments': backing_instruments,
        'song_structure': song_structure,
        'key_signature': key_signature,
        'time_signature': time_signature,
        'harmonic_complexity': harmonic_complexity,
        'rhythmic_complexity': rhythmic_complexity,
        'dynamic_range': dynamic_range,
        'production_style': production_style,
        'effects': effects,
        'vocals': vocals,
        'vocal_style': vocal_style,
        'lyrics': lyrics,
        'song_theme': song_theme,
        'lyrical_style': lyrical_style,
        'artist': artist,
        'reference_songs': reference_songs,
        'era': era,
        'influences': influences,
        'modulations': modulations,
        'special_techniques': special_techniques,
        'emotional_arc': emotional_arc,
        'instrument_description': instrument_description
    })
    
    # Parse sections from lyrics if present
    if lyrics:
        sections = parse_lyrics_sections(lyrics)
        if sections:
            state['lyrical_sections'] = sections
            logging.info(f"[UserInputAgent] Parsed lyrical sections: {sections}")
    
    # Create structured sections from song_structure
    if song_structure:
        section_duration = (duration * 60) / len(song_structure)
        structured_sections = []
        for i, section_name in enumerate(song_structure):
            start_time = i * section_duration
            end_time = (i + 1) * section_duration
            structured_sections.append({
                'name': section_name,
                'start_time': start_time,
                'end_time': end_time,
                'order': i
            })
        state['structured_sections'] = structured_sections
        logging.info(f"[UserInputAgent] Created structured sections: {[s['name'] for s in structured_sections]}")
    
    # Create musical context summary
    full_genre = f"{genre} {subgenre}".strip() if subgenre else genre
    full_mood = f"{mood} with {secondary_mood} undertones".strip() if secondary_mood else mood
    
    musical_context = {
        'full_genre': full_genre,
        'full_mood': full_mood,
        'complexity_profile': {
            'harmonic': harmonic_complexity,
            'rhythmic': rhythmic_complexity,
            'dynamic': dynamic_range
        },
        'has_detailed_structure': bool(song_structure and len(song_structure) > 2),
        'has_artist_reference': bool(artist),
        'has_song_references': bool(reference_songs),
        'has_influences': bool(influences),
        'has_advanced_concepts': bool(modulations or special_techniques or emotional_arc)
    }
    state['musical_context'] = musical_context
    
    logging.info(f"[UserInputAgent] Comprehensive context: genre={full_genre}, mood={full_mood}, tempo={tempo}, "
                f"duration={duration}, instruments={instruments}, vocals={vocals}, artist={artist}")
    
    return state 