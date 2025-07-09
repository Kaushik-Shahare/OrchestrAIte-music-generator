import logging
from typing import Any, Dict
from utils.gemini_llm import gemini_generate
import ast
import re

def clean_llm_output(text):
    return re.sub(r'^```[a-zA-Z]*\n?|```$', '', text.strip(), flags=re.MULTILINE).strip()

def musical_director_agent(state: Dict) -> Dict:
    """
    Acts as a musical director, creating cohesive artistic vision and detailed arrangement plans.
    This agent ensures the music tells a story and has emotional depth rather than being monotonous.
    """
    logging.info("[MusicalDirectorAgent] Creating comprehensive musical vision and arrangement plan.")
    
    try:
        # Extract comprehensive context
        musical_context = state.get('musical_context', {})
        artist_profile = state.get('artist_profile', {})
        structured_sections = state.get('structured_sections', [])
        
        # Basic parameters
        genre = musical_context.get('full_genre', state.get('genre', 'pop'))
        mood = musical_context.get('full_mood', state.get('mood', 'happy'))
        tempo = state.get('tempo', 120)
        duration = state.get('duration', 2)
        
        # Advanced context
        song_theme = state.get('song_theme', '')
        emotional_arc = state.get('emotional_arc', '')
        lyrics = state.get('lyrics', '')
        instruments = state.get('instruments', ['piano'])
        
        # Create comprehensive musical vision
        vision_prompt = f"""
        You are a world-class musical director creating a detailed artistic vision for a {genre} song.
        
        BASIC PARAMETERS:
        - Genre: {genre}
        - Mood: {mood}  
        - Tempo: {tempo} BPM
        - Duration: {duration} minutes
        - Instruments: {', '.join(instruments)}
        
        ARTISTIC CONTEXT:
        {f'- Song Theme: {song_theme}' if song_theme else ''}
        {f'- Emotional Journey: {emotional_arc}' if emotional_arc else ''}
        {f'- Lyrics Context: {lyrics[:200]}...' if lyrics else ''}
        {f'- Artist Style: {artist_profile.get("style_summary", "")}' if artist_profile.get("style_summary") else ''}
        
        Create a comprehensive musical vision that includes:
        
        1. EMOTIONAL_NARRATIVE: How should the song's emotional intensity develop over time?
        2. DYNAMIC_ARCHITECTURE: How should volume, energy, and complexity change throughout?
        3. TEXTURAL_EVOLUTION: How should the arrangement density and instrumentation evolve?
        4. RHYTHMIC_JOURNEY: How should rhythmic complexity and groove change?
        5. HARMONIC_PROGRESSION: How should harmonic complexity and tension develop?
        6. MELODIC_CHARACTER: What should be the melodic personality and development?
        7. PRODUCTION_VISION: What sonic characteristics should define this song?
        8. SECTION_PERSONALITIES: Unique character for each section
        9. TRANSITION_STRATEGIES: How to connect sections meaningfully
        10. CLIMAX_POINTS: Where and how should musical peaks occur?
        
        Output as a Python dictionary with these exact keys.
        Each value should be a detailed string describing the musical approach.
        This will guide all other agents to create cohesive, non-monotonous music.
        
        Do not include any explanation or extra text outside the dictionary.
        """
        
        try:
            vision_text = gemini_generate(vision_prompt)
            cleaned_vision = clean_llm_output(vision_text)
            musical_vision = ast.literal_eval(cleaned_vision)
        except Exception as e:
            logging.warning(f"[MusicalDirectorAgent] Failed to parse vision, using fallback: {e}")
            # Create fallback vision
            musical_vision = {
                'emotional_narrative': f'Build from {mood} foundation with gradual emotional development',
                'dynamic_architecture': 'Start moderate, build to strong climax, then resolve',
                'textural_evolution': 'Begin sparse, add layers progressively, peak complexity in chorus',
                'rhythmic_journey': 'Establish solid groove, add complexity in sections, return to foundation',
                'harmonic_progression': 'Start simple, add color and tension, resolve satisfyingly',
                'melodic_character': 'Memorable, singable themes with expressive development',
                'production_vision': f'Clean, modern {genre} production with appropriate space and depth',
                'section_personalities': 'Distinct character for each section while maintaining cohesion',
                'transition_strategies': 'Smooth, musical transitions that enhance the narrative flow',
                'climax_points': 'Strategic peaks that serve the emotional story'
            }
        
        # Create detailed section-by-section arrangement plan
        if structured_sections:
            section_plan_prompt = f"""
            Based on this musical vision: {musical_vision}
            
            Create a detailed arrangement plan for each section:
            {[f"{s['name']} ({s['start_time']:.1f}s-{s['end_time']:.1f}s)" for s in structured_sections]}
            
            For each section, specify:
            - ENERGY_LEVEL (1-10 scale)
            - COMPLEXITY_LEVEL (simple/medium/complex)
            - DOMINANT_INSTRUMENTS (which instruments lead)
            - RHYTHMIC_APPROACH (groove characteristics)
            - HARMONIC_APPROACH (chord complexity)
            - MELODIC_FOCUS (melodic character)
            - DYNAMIC_LEVEL (soft/medium/loud/very_loud)
            - TEXTURAL_DENSITY (sparse/medium/dense)
            - EMOTIONAL_CHARACTER (feeling/mood for this section)
            
            Output as a Python dictionary where keys are section names and values are dictionaries
            with the above characteristics.
            """
            
            try:
                plan_text = gemini_generate(section_plan_prompt)
                cleaned_plan = clean_llm_output(plan_text)
                section_arrangement_plan = ast.literal_eval(cleaned_plan)
            except Exception as e:
                logging.warning(f"[MusicalDirectorAgent] Failed to parse section plan: {e}")
                # Create basic fallback plan
                section_arrangement_plan = {}
                for section in structured_sections:
                    section_name = section['name']
                    if 'verse' in section_name.lower():
                        energy = 5
                        complexity = 'medium'
                        dynamic = 'medium'
                    elif 'chorus' in section_name.lower():
                        energy = 8
                        complexity = 'medium'
                        dynamic = 'loud'
                    elif 'bridge' in section_name.lower():
                        energy = 6
                        complexity = 'complex'
                        dynamic = 'medium'
                    else:
                        energy = 6
                        complexity = 'medium'
                        dynamic = 'medium'
                    
                    section_arrangement_plan[section_name] = {
                        'energy_level': energy,
                        'complexity_level': complexity,
                        'dynamic_level': dynamic,
                        'emotional_character': f'{mood} with section-appropriate variation'
                    }
        else:
            section_arrangement_plan = {}
        
        # Create master arrangement guidelines
        arrangement_guidelines = {
            'overall_vision': musical_vision,
            'section_plans': section_arrangement_plan,
            'instrumentation_strategy': {
                'lead_instrument_focus': instruments[0] if instruments else 'piano',
                'supporting_instruments': instruments[1:] if len(instruments) > 1 else [],
                'arrangement_density': 'progressive' if len(instruments) > 2 else 'focused',
                'rhythmic_foundation': 'solid groove with creative variations',
                'harmonic_support': 'sophisticated but not overwhelming',
                'melodic_interaction': 'call and response between instruments'
            },
            'production_guidelines': {
                'sonic_palette': f'{genre}-appropriate timbres with artistic depth',
                'dynamic_range': state.get('dynamic_range', 'medium'),
                'spatial_design': 'wide stereo field with clear instrument placement',
                'temporal_pacing': 'carefully controlled musical time and space'
            }
        }
        
        # Store comprehensive arrangement plan
        state['musical_vision'] = musical_vision
        state['arrangement_guidelines'] = arrangement_guidelines
        state['section_arrangement_plan'] = section_arrangement_plan
        
        # Create summary for other agents
        director_summary = f"""
        MUSICAL DIRECTOR'S VISION:
        - Emotional Arc: {musical_vision.get('emotional_narrative', 'Dynamic emotional development')}
        - Dynamic Journey: {musical_vision.get('dynamic_architecture', 'Progressive build with strategic peaks')}
        - Textural Evolution: {musical_vision.get('textural_evolution', 'Thoughtful layering and space')}
        - Production Character: {musical_vision.get('production_vision', 'Professional and artistic')}
        
        This song should tell a musical story, NOT be monotonous or repetitive.
        Each section should have distinct personality while maintaining overall cohesion.
        """
        
        state['director_summary'] = director_summary
        
        logging.info("[MusicalDirectorAgent] Comprehensive musical vision and arrangement plan created.")
        logging.info(f"[MusicalDirectorAgent] Vision includes {len(musical_vision)} artistic elements and "
                    f"{len(section_arrangement_plan)} section-specific plans.")
        
    except Exception as e:
        logging.error(f"[MusicalDirectorAgent] Error: {e}")
        # Provide minimal fallback
        state['musical_vision'] = {'emotional_narrative': 'Create engaging musical story'}
        state['arrangement_guidelines'] = {}
        state['director_summary'] = "Create dynamic, non-monotonous music with artistic depth."
    
    return state
