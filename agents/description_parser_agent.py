#!/usr/bin/env python3
"""
Description Parser Agent
Converts natural language song descriptions into structured musical parameters
"""

import logging
import re
from typing import Any, Dict, List, Optional
from utils.gemini_llm import gemini_generate
import json

def description_parser_agent(state: Dict) -> Dict:
    """
    Parse natural language song description and extract musical parameters
    
    Takes a free-form description like "Create an upbeat jazz song with piano and saxophone,
    similar to Bill Evans, around 3 minutes long" and converts it to structured parameters.
    """
    logging.info("[DescriptionParserAgent] 🎵 Parsing natural language song description...")
    
    # Get the description from user input
    description = state.get('user_input', {}).get('description', '')
    if not description:
        # Check if description is at the top level
        description = state.get('description', '')
    
    if not description:
        logging.info("[DescriptionParserAgent] No description provided, skipping parsing")
        return state
    
    logging.info(f"[DescriptionParserAgent] 📝 Processing description: '{description[:100]}...'")
    
    try:
        # Use LLM to extract structured information from the description
        parsed_params = parse_description_with_llm(description)
        
        # Merge the parsed parameters with existing user input
        if 'user_input' not in state:
            state['user_input'] = {}
        
        # Add the original description
        state['user_input']['original_description'] = description
        state['user_input']['parsed_from_description'] = True
        
        # Update state with parsed parameters (only if not already specified)
        for key, value in parsed_params.items():
            if key not in state['user_input'] or not state['user_input'].get(key):
                state['user_input'][key] = value
                logging.info(f"[DescriptionParserAgent] ✅ Extracted {key}: {value}")
        
        # Create a summary of what was extracted
        extraction_summary = create_extraction_summary(parsed_params)
        state['description_extraction_summary'] = extraction_summary
        
        logging.info(f"[DescriptionParserAgent] 🎯 Successfully parsed description into {len(parsed_params)} parameters")
        
    except Exception as e:
        logging.error(f"[DescriptionParserAgent] ❌ Error parsing description: {e}")
        # Add error info but don't fail - let the system use defaults
        state['description_parsing_error'] = str(e)
    
    return state

def parse_description_with_llm(description: str) -> Dict:
    """
    Use Gemini LLM to extract structured musical parameters from description
    """
    logging.info("[DescriptionParserAgent] 🤖 Using LLM to parse musical description...")
    
    prompt = f"""
    You are a professional music producer and composer. Parse this song description and extract specific musical parameters.
    
    DESCRIPTION: "{description}"
    
    Extract the following information if mentioned (return "null" if not specified):
    
    1. GENRE: Main musical genre (e.g., "jazz", "rock", "pop", "classical", "blues", "country", "electronic", "hip-hop")
    2. SUBGENRE: Specific subgenre if mentioned (e.g., "bebop", "progressive rock", "synthpop")
    3. MOOD: Overall emotional feel (e.g., "happy", "sad", "energetic", "relaxed", "mysterious", "dramatic")
    4. TEMPO: Suggested BPM or descriptive tempo (e.g., "120", "fast", "slow", "moderate", "upbeat")
    5. DURATION: Song length in minutes (e.g., "3", "4.5") 
    6. INSTRUMENTS: List of mentioned instruments (e.g., ["piano", "saxophone", "drums", "guitar"])
    7. ARTIST: Any artist name mentioned as reference (e.g., "Bill Evans", "The Beatles")
    8. VOCAL_STYLE: Type of vocals if mentioned (e.g., "smooth", "raspy", "operatic", "rap")
    9. KEY_SIGNATURE: Musical key if specified (e.g., "C major", "A minor")
    10. TIME_SIGNATURE: Time signature if mentioned (e.g., "4/4", "3/4", "6/8")
    11. SONG_STRUCTURE: Mentioned song parts (e.g., ["verse", "chorus", "bridge"])
    12. SPECIAL_TECHNIQUES: Any special musical techniques mentioned
    13. ERA: Time period or decade if mentioned (e.g., "1960s", "modern", "vintage")
    14. PRODUCTION_STYLE: Production approach if mentioned (e.g., "raw", "polished", "lo-fi")
    15. HARMONIC_COMPLEXITY: Complexity level inferred from description ("simple", "medium", "complex")
    16. RHYTHMIC_COMPLEXITY: Rhythm complexity inferred ("simple", "medium", "complex")
    17. SONG_THEME: Thematic content or story if mentioned
    18. EMOTIONAL_ARC: How emotions should develop through the song
    19. REFERENCE_SONGS: Any specific songs mentioned as references
    20. INFLUENCES: Musical influences or styles mentioned
    
    Return ONLY a valid JSON object with these keys. Use descriptive values, not just single words.
    If something isn't mentioned, use null for that field.
    
    Example output format:
    {{
        "genre": "jazz",
        "subgenre": "bebop",
        "mood": "energetic and improvisational",
        "tempo": "fast",
        "duration": "4",
        "instruments": ["piano", "saxophone", "bass", "drums"],
        "artist": "Charlie Parker",
        "vocal_style": null,
        "key_signature": null,
        "time_signature": "4/4",
        "song_structure": ["intro", "head", "solos", "head", "outro"],
        "special_techniques": "improvisation and complex harmonies",
        "era": "1940s",
        "production_style": null,
        "harmonic_complexity": "complex",
        "rhythmic_complexity": "complex",
        "song_theme": null,
        "emotional_arc": "building energy through solos",
        "reference_songs": null,
        "influences": ["bebop", "swing"]
    }}
    """
    
    try:
        response = gemini_generate(prompt)
        logging.info(f"[DescriptionParserAgent] 📥 LLM response: {response[:200]}...")
        
        # Try to parse JSON from the response
        # Sometimes the LLM includes extra text, so we need to extract just the JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed_data = json.loads(json_str)
            
            # Clean up and validate the parsed data
            cleaned_data = clean_parsed_data(parsed_data)
            
            logging.info(f"[DescriptionParserAgent] ✅ Successfully parsed {len(cleaned_data)} parameters")
            return cleaned_data
        else:
            logging.error("[DescriptionParserAgent] ❌ No valid JSON found in LLM response")
            return {}
            
    except json.JSONDecodeError as e:
        logging.error(f"[DescriptionParserAgent] ❌ JSON parsing error: {e}")
        # Try to extract individual fields with regex as fallback
        return extract_with_regex_fallback(description)
    except Exception as e:
        logging.error(f"[DescriptionParserAgent] ❌ LLM parsing error: {e}")
        return extract_with_regex_fallback(description)

def clean_parsed_data(data: Dict) -> Dict:
    """
    Clean and validate the parsed data from LLM
    """
    cleaned = {}
    
    # Remove null values and clean up strings
    for key, value in data.items():
        if value is None or value == "null":
            continue
            
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in ['null', 'none', 'not mentioned', 'not specified', '']:
                continue
                
        # Convert tempo descriptions to approximate BPM
        if key == 'tempo' and isinstance(value, str):
            tempo_mapping = {
                'very slow': 60,
                'slow': 80,
                'moderate': 100,
                'medium': 120,
                'upbeat': 140,
                'fast': 160,
                'very fast': 180
            }
            for desc, bpm in tempo_mapping.items():
                if desc in value.lower():
                    value = bpm
                    break
        
        # Ensure instruments is a list
        if key == 'instruments' and isinstance(value, str):
            value = [inst.strip() for inst in value.split(',')]
        
        # Ensure song_structure is a list
        if key == 'song_structure' and isinstance(value, str):
            value = [part.strip() for part in value.split(',')]
        
        # Convert duration to integer if possible
        if key == 'duration':
            try:
                value = int(float(value))
            except (ValueError, TypeError):
                pass
        
        cleaned[key] = value
    
    return cleaned

def extract_with_regex_fallback(description: str) -> Dict:
    """
    Fallback method to extract basic parameters using regex patterns
    """
    logging.info("[DescriptionParserAgent] 🔧 Using regex fallback for parameter extraction")
    
    extracted = {}
    desc_lower = description.lower()
    
    # Extract genre
    genres = ['jazz', 'rock', 'pop', 'classical', 'blues', 'country', 'electronic', 'hip-hop', 'folk', 'r&b', 'metal', 'punk']
    for genre in genres:
        if genre in desc_lower:
            extracted['genre'] = genre
            break
    
    # Extract tempo keywords
    if 'fast' in desc_lower or 'upbeat' in desc_lower:
        extracted['tempo'] = 140
    elif 'slow' in desc_lower:
        extracted['tempo'] = 80
    elif 'moderate' in desc_lower:
        extracted['tempo'] = 120
    
    # Extract mood keywords
    moods = ['happy', 'sad', 'energetic', 'relaxed', 'dramatic', 'peaceful', 'aggressive', 'romantic']
    for mood in moods:
        if mood in desc_lower:
            extracted['mood'] = mood
            break
    
    # Extract duration (look for "X minutes" or "X min")
    duration_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)', desc_lower)
    if duration_match:
        extracted['duration'] = int(float(duration_match.group(1)))
    
    # Extract common instruments
    instruments = ['piano', 'guitar', 'drums', 'bass', 'saxophone', 'violin', 'trumpet', 'flute', 'cello']
    found_instruments = []
    for instrument in instruments:
        if instrument in desc_lower:
            found_instruments.append(instrument)
    if found_instruments:
        extracted['instruments'] = found_instruments
    
    # Extract artist names (look for "like X" or "similar to X")
    artist_match = re.search(r'(?:like|similar to|in the style of)\s+([A-Z][a-z\s]+(?:[A-Z][a-z]+)?)', description)
    if artist_match:
        extracted['artist'] = artist_match.group(1).strip()
    
    logging.info(f"[DescriptionParserAgent] 🔧 Regex fallback extracted {len(extracted)} parameters")
    return extracted

def create_extraction_summary(parsed_params: Dict) -> str:
    """
    Create a human-readable summary of what was extracted from the description
    """
    summary_parts = []
    
    if parsed_params.get('genre'):
        genre_part = f"Genre: {parsed_params['genre']}"
        if parsed_params.get('subgenre'):
            genre_part += f" ({parsed_params['subgenre']})"
        summary_parts.append(genre_part)
    
    if parsed_params.get('mood'):
        summary_parts.append(f"Mood: {parsed_params['mood']}")
    
    if parsed_params.get('tempo'):
        summary_parts.append(f"Tempo: {parsed_params['tempo']}")
    
    if parsed_params.get('duration'):
        summary_parts.append(f"Duration: {parsed_params['duration']} minutes")
    
    if parsed_params.get('instruments'):
        instruments = parsed_params['instruments']
        if isinstance(instruments, list):
            summary_parts.append(f"Instruments: {', '.join(instruments)}")
        else:
            summary_parts.append(f"Instruments: {instruments}")
    
    if parsed_params.get('artist'):
        summary_parts.append(f"Artist Reference: {parsed_params['artist']}")
    
    if parsed_params.get('key_signature'):
        summary_parts.append(f"Key: {parsed_params['key_signature']}")
    
    if parsed_params.get('song_structure'):
        structure = parsed_params['song_structure']
        if isinstance(structure, list):
            summary_parts.append(f"Structure: {' → '.join(structure)}")
        else:
            summary_parts.append(f"Structure: {structure}")
    
    if not summary_parts:
        return "No specific musical parameters were extracted from the description."
    
    return "Extracted: " + " | ".join(summary_parts)

# Enhanced version that can handle complex descriptions
def advanced_description_parser_agent(state: Dict) -> Dict:
    """
    Advanced version that can handle more complex descriptions with multiple sentences
    and extract emotional arcs, production details, etc.
    """
    logging.info("[DescriptionParserAgent] 🎼 Advanced parsing for complex description...")
    
    description = state.get('user_input', {}).get('description', '') or state.get('description', '')
    
    if not description:
        return state
    
    # Split description into sentences for better analysis
    sentences = re.split(r'[.!?]+', description)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    logging.info(f"[DescriptionParserAgent] 📋 Analyzing {len(sentences)} sentences...")
    
    try:
        # Create a more detailed prompt for complex descriptions
        detailed_prompt = f"""
        You are an expert music producer analyzing a detailed song description. Break down this description thoroughly:
        
        FULL DESCRIPTION: "{description}"
        
        SENTENCES TO ANALYZE:
        {chr(10).join([f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)])}
        
        Extract comprehensive musical parameters and return as JSON:
        
        {{
            "basic_info": {{
                "genre": "main genre",
                "subgenre": "specific subgenre if mentioned", 
                "mood": "primary emotional tone",
                "secondary_mood": "additional emotional elements",
                "tempo": "BPM number or descriptive tempo",
                "duration": "length in minutes",
                "key_signature": "musical key if specified",
                "time_signature": "time signature if mentioned"
            }},
            "instrumentation": {{
                "instruments": ["list", "of", "instruments"],
                "lead_instrument": "primary featured instrument",
                "backing_instruments": ["supporting", "instruments"],
                "instrument_description": "how instruments should sound/be played"
            }},
            "structure_and_style": {{
                "song_structure": ["verse", "chorus", "etc"],
                "production_style": "recording/production approach",
                "harmonic_complexity": "simple/medium/complex",
                "rhythmic_complexity": "simple/medium/complex",
                "special_techniques": "any special musical techniques",
                "effects": "reverb, distortion, etc."
            }},
            "vocals": {{
                "vocals": true/false,
                "vocal_style": "singing style if mentioned",
                "lyrical_style": "approach to lyrics",
                "song_theme": "thematic content"
            }},
            "references": {{
                "artist": "primary artist reference",
                "reference_songs": ["specific", "songs"],
                "era": "time period or decade",
                "influences": ["musical", "influences"]
            }},
            "emotional_journey": {{
                "emotional_arc": "how emotions develop through song",
                "dynamic_range": "quiet to loud progression",
                "section_moods": {{"intro": "mood", "verse": "mood", "chorus": "mood"}}
            }}
        }}
        
        Be thorough but only include information that's actually mentioned or strongly implied.
        Use null for unspecified fields.
        """
        
        response = gemini_generate(detailed_prompt)
        
        # Parse the detailed response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            detailed_data = json.loads(json_str)
            
            # Flatten the nested structure for compatibility
            flattened_params = flatten_detailed_params(detailed_data)
            
            # Update state with the detailed parameters
            if 'user_input' not in state:
                state['user_input'] = {}
            
            state['user_input']['original_description'] = description
            state['user_input']['advanced_parsing'] = True
            
            for key, value in flattened_params.items():
                if key not in state['user_input'] or not state['user_input'].get(key):
                    state['user_input'][key] = value
            
            # Create detailed summary
            state['advanced_extraction_summary'] = create_detailed_summary(detailed_data)
            
            logging.info(f"[DescriptionParserAgent] 🎯 Advanced parsing extracted {len(flattened_params)} parameters")
        
    except Exception as e:
        logging.error(f"[DescriptionParserAgent] ❌ Advanced parsing failed: {e}")
        # Fall back to basic parsing
        return description_parser_agent(state)
    
    return state

def flatten_detailed_params(detailed_data: Dict) -> Dict:
    """
    Flatten the nested detailed parameters structure
    """
    flattened = {}
    
    for category, params in detailed_data.items():
        if isinstance(params, dict):
            for key, value in params.items():
                if value is not None and value != "null":
                    flattened[key] = value
        else:
            flattened[category] = params
    
    return flattened

def create_detailed_summary(detailed_data: Dict) -> str:
    """
    Create a comprehensive summary of advanced parsing results
    """
    summary_sections = []
    
    # Basic info
    basic = detailed_data.get('basic_info', {})
    if basic:
        basic_parts = []
        for key, value in basic.items():
            if value and value != "null":
                basic_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        if basic_parts:
            summary_sections.append("BASIC INFO: " + " | ".join(basic_parts))
    
    # Instrumentation
    instrumentation = detailed_data.get('instrumentation', {})
    if instrumentation:
        inst_parts = []
        for key, value in instrumentation.items():
            if value and value != "null":
                if isinstance(value, list):
                    inst_parts.append(f"{key.replace('_', ' ').title()}: {', '.join(value)}")
                else:
                    inst_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        if inst_parts:
            summary_sections.append("INSTRUMENTATION: " + " | ".join(inst_parts))
    
    # References
    references = detailed_data.get('references', {})
    if references:
        ref_parts = []
        for key, value in references.items():
            if value and value != "null":
                if isinstance(value, list):
                    ref_parts.append(f"{key.replace('_', ' ').title()}: {', '.join(value)}")
                else:
                    ref_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        if ref_parts:
            summary_sections.append("REFERENCES: " + " | ".join(ref_parts))
    
    return "\n".join(summary_sections) if summary_sections else "Advanced parsing completed with basic parameters."
