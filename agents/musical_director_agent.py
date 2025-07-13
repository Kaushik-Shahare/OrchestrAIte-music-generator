import logging
from typing import Any, Dict
from utils.gemini_llm import gemini_generate
from utils.state_utils import validate_agent_return, safe_state_update
import json
import re

def safe_json_parse(text):
    """Safely parse JSON with multiple fallback attempts"""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logging.warning(f"Initial JSON parse failed: {e}")
        
        # Try to fix common issues
        try:
            # Remove trailing commas
            fixed_text = re.sub(r',\s*}', '}', text)
            fixed_text = re.sub(r',\s*]', ']', fixed_text)
            # Remove control characters
            fixed_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', fixed_text)
            # Remove comments
            fixed_text = re.sub(r'//.*?\n', '\n', fixed_text)
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            # Try to extract just the inner content
            try:
                # Look for the main JSON structure
                match = re.search(r'({[^{}]*(?:{[^{}]*}[^{}]*)*})', text, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
            
            # Final fallback - return None
            logging.error(f"All JSON parsing attempts failed for text: {text[:200]}...")
            return None

def clean_llm_output(text):
    """Clean LLM output and extract JSON if present"""
    # Remove code blocks
    cleaned = re.sub(r'^```[a-zA-Z]*\n?|```$', '', text.strip(), flags=re.MULTILINE).strip()
    
    # Try to extract JSON from the text
    json_match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
    if json_match:
        json_text = json_match.group(1)
        # Clean up common LLM JSON issues
        json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
        json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas in arrays
        json_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_text)  # Remove control characters
        return json_text
    
    return cleaned

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
        
        Output as a JSON object with these exact keys:
        {{
            "emotional_narrative": "How should the song's emotional intensity develop over time?",
            "dynamic_architecture": "How should volume, energy, and complexity change throughout?",
            "textural_evolution": "How should the arrangement density and instrumentation evolve?",
            "rhythmic_journey": "How should rhythmic complexity and groove change?",
            "harmonic_progression": "How should harmonic complexity and tension develop?",
            "melodic_character": "What should be the melodic personality and development?",
            "production_vision": "What sonic characteristics should define this song?",
            "section_personalities": "Unique character for each section",
            "transition_strategies": "How to connect sections meaningfully",
            "climax_points": "Where and how should musical peaks occur?"
        }}
        
        Each value should be a detailed string describing the musical approach.
        Respond with ONLY a valid JSON object - no extra text.
        """
        
        try:
            vision_text = gemini_generate(vision_prompt)
            cleaned_vision = clean_llm_output(vision_text)
            musical_vision = safe_json_parse(cleaned_vision)
            
            # Ensure we have a valid dictionary
            if not isinstance(musical_vision, dict):
                raise ValueError("Parsed result is not a dictionary")
                
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
            {', '.join([f"{s['name']} ({s['start_time']:.1f}s-{s['end_time']:.1f}s)" for s in structured_sections])}
            
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
            
            Output as a JSON object where keys are section names and values are JSON objects
            with the above characteristics.
            
            Example format:
            {{
                "verse_1": {{
                    "energy_level": 5,
                    "complexity_level": "medium",
                    "dominant_instruments": ["piano", "bass"],
                    "rhythmic_approach": "steady groove",
                    "harmonic_approach": "simple triads",
                    "melodic_focus": "memorable theme",
                    "dynamic_level": "medium",
                    "textural_density": "medium",
                    "emotional_character": "contemplative"
                }}
            }}
            
            Respond with ONLY a valid JSON object.
            """
            
            try:
                plan_text = gemini_generate(section_plan_prompt)
                cleaned_plan = clean_llm_output(plan_text)
                section_arrangement_plan = safe_json_parse(cleaned_plan)
                
                # Ensure we have a valid dictionary
                if not isinstance(section_arrangement_plan, dict):
                    raise ValueError("Parsed section plan is not a dictionary")
                    
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
        
        logging.info("[MusicalDirectorAgent] Comprehensive musical vision and arrangement plan created.")
        logging.info(f"[MusicalDirectorAgent] Vision includes {len(musical_vision)} artistic elements and "
                    f"{len(section_arrangement_plan)} section-specific plans.")
        
        # Use safe state update to store comprehensive arrangement plan
        return safe_state_update(state, {
            'musical_vision': musical_vision,
            'arrangement_guidelines': arrangement_guidelines,
            'section_arrangement_plan': section_arrangement_plan,
            'director_summary': director_summary
        }, "MusicalDirectorAgent")
        
    except Exception as e:
        logging.error(f"[MusicalDirectorAgent] Error: {e}")
        # Provide minimal fallback
        fallback_data = {
            'musical_vision': {'emotional_narrative': 'Create engaging musical story'},
            'arrangement_guidelines': {},
            'director_summary': "Create dynamic, non-monotonous music with artistic depth."
        }
        return safe_state_update(state, fallback_data, "MusicalDirectorAgent")
