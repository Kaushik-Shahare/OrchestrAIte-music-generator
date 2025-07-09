import logging
from utils.gemini_llm import gemini_generate
import ast
import re

def clean_llm_output(text):
    return re.sub(r'^```[a-zA-Z]*\n?|```$', '', text.strip(), flags=re.MULTILINE).strip()

def artist_style_agent(state: dict) -> dict:
    """
    Generate comprehensive artist style profile with detailed musical characteristics.
    """
    logging.info("[ArtistStyleAgent] Generating comprehensive artist style profile.")
    
    artist = state.get('artist', '')
    reference_songs = state.get('reference_songs', [])
    influences = state.get('influences', [])
    era = state.get('era', '')
    genre = state.get('genre', 'pop')
    
    if not artist and not reference_songs and not influences:
        state['artist_profile'] = {}
        logging.info("[ArtistStyleAgent] No artist context provided, skipping style analysis.")
        return state
    
    # Build comprehensive prompt for style analysis
    context_parts = []
    if artist:
        context_parts.append(f"Primary artist: {artist}")
    if reference_songs:
        context_parts.append(f"Reference songs: {', '.join(reference_songs)}")
    if influences:
        context_parts.append(f"Musical influences: {', '.join(influences)}")
    if era:
        context_parts.append(f"Era/period: {era}")
    
    context_string = "; ".join(context_parts)
    
    prompt = f"""
    Analyze the musical style characteristics for: {context_string}
    
    Consider the {genre} genre context and provide a detailed analysis covering:
    
    1. MELODY: Scale choices, interval patterns, melodic contour, phrasing, ornamentation
    2. HARMONY: Chord progressions, voicings, extensions, modal interchange, key centers
    3. RHYTHM: Groove patterns, syncopation, polyrhythms, swing vs straight feel
    4. ARRANGEMENT: Instrumentation, layering, countermelodies, call-and-response
    5. PRODUCTION: Sound palette, effects usage, mixing approach, stereo field
    6. DYNAMICS: Volume changes, build-ups, breakdowns, tension/release patterns
    7. FORM: Song structures, transitions, intros/outros, bridge characteristics
    8. VOCAL: Delivery style, range, techniques, harmonies (if applicable)
    9. SIGNATURE_ELEMENTS: Unique characteristics, recognizable patterns, innovations
    
    Output as a Python dictionary with these exact keys:
    'melody_characteristics', 'harmonic_language', 'rhythmic_elements', 'arrangement_style', 
    'production_techniques', 'dynamic_approach', 'structural_elements', 'vocal_characteristics', 
    'signature_sounds', 'tempo_preferences', 'key_preferences', 'instrumentation_style'
    
    Make each value a detailed string describing the specific musical elements.
    Do not include any explanation or extra text outside the dictionary.
    """
    
    try:
        profile_text = gemini_generate(prompt)
        cleaned_text = clean_llm_output(profile_text)
        
        # Try to parse the response
        try:
            profile = ast.literal_eval(cleaned_text)
        except Exception as e:
            logging.warning(f"[ArtistStyleAgent] Failed to parse structured response, using fallback: {e}")
            # Fallback: create a basic profile from the text
            profile = {
                'melody_characteristics': 'Expressive melodic lines with characteristic phrasing',
                'harmonic_language': 'Contemporary chord progressions with some complexity',
                'rhythmic_elements': 'Solid groove foundation with subtle variations',
                'arrangement_style': 'Layered arrangement with clear instrument roles',
                'production_techniques': 'Modern production with balanced dynamics',
                'dynamic_approach': 'Varied dynamics with effective build-ups',
                'structural_elements': 'Traditional song forms with creative transitions',
                'vocal_characteristics': 'Distinctive vocal delivery and style',
                'signature_sounds': 'Recognizable sonic characteristics',
                'tempo_preferences': 'Moderate to uptempo preferences',
                'key_preferences': 'Major and minor key flexibility',
                'instrumentation_style': 'Genre-appropriate instrumentation choices'
            }
        
        # Create style summary for quick reference
        style_summary = f"Musical style analysis for {context_string}: "
        if artist:
            style_summary += f"{artist}'s signature sound combines "
        style_summary += f"{profile.get('melody_characteristics', 'melodic elements')} with "
        style_summary += f"{profile.get('harmonic_language', 'harmonic approach')} and "
        style_summary += f"{profile.get('rhythmic_elements', 'rhythmic foundation')}."
        
        profile['style_summary'] = style_summary
        profile['context_source'] = context_string
        
        state['artist_profile'] = profile
        logging.info(f"[ArtistStyleAgent] Comprehensive style profile generated for: {context_string}")
        
    except Exception as e:
        logging.error(f"[ArtistStyleAgent] Error generating style profile: {e}")
        state['artist_profile'] = {}
    
    return state 