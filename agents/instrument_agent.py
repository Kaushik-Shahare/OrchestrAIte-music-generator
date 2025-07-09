import logging
from typing import Any
from utils.gemini_llm import gemini_generate
import ast
import re

def clean_llm_output(text):
    return re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip(), flags=re.MULTILINE).strip()

def safe_literal_eval(text):
    try:
        return ast.literal_eval(text)
    except Exception:
        if text.count('[') > text.count(']'):
            text = text + (']' * (text.count('[') - text.count(']')))
        if text.count('{') > text.count('}'):
            text = text + ('}' * (text.count('{') - text.count('}')))
        try:
            return ast.literal_eval(text)
        except Exception as e:
            logging.error(f"[InstrumentAgent] Failed to parse LLM output after auto-fix: {e}")
            return []

def notes_array_to_dicts(notes_array):
    return [
        {'pitch': n[0], 'start': n[1], 'end': n[2], 'velocity': n[3]}
        for n in notes_array if isinstance(n, (list, tuple)) and len(n) == 4
    ]

def get_instrument_program(instrument_name):
    """Map instrument names to MIDI program numbers."""
    instrument_map = {
        'piano': 0, 'electric_piano': 4, 'guitar': 24, 'electric_guitar': 29,
        'acoustic_guitar': 24, 'bass': 33, 'electric_bass': 33, 'violin': 40,
        'viola': 41, 'cello': 42, 'trumpet': 56, 'trombone': 57, 'saxophone': 64,
        'sax': 64, 'flute': 73, 'clarinet': 71, 'organ': 16, 'synth': 80,
        'synthesizer': 80, 'strings': 48, 'pad': 88, 'lead': 80
    }
    
    instrument_lower = instrument_name.lower()
    for key, program in instrument_map.items():
        if key in instrument_lower:
            return program
    return 0  # Default to piano

def instrument_agent(state: Any) -> Any:
    """
    Generate sophisticated multi-instrument arrangements with style-specific playing techniques.
    """
    logging.info("[InstrumentAgent] Creating sophisticated instrument arrangements.")
    
    try:
        # Extract comprehensive context
        structure = state.get('structure', {})
        melody_onsets = structure.get('melody_onsets', [])
        bar_times = structure.get('bar_times', [])
        
        artist_profile = state.get('artist_profile', {})
        musical_context = state.get('musical_context', {})
        structured_sections = state.get('structured_sections', [])
        
        # Basic parameters
        genre = musical_context.get('full_genre', state.get('genre', 'pop'))
        mood = musical_context.get('full_mood', state.get('mood', 'happy'))
        tempo = state.get('tempo', 120)
        duration = state.get('duration', 2)
        
        # Instrumentation
        instruments = state.get('instruments', ['piano'])
        lead_instrument = state.get('lead_instrument', '')
        backing_instruments = state.get('backing_instruments', [])
        instrument_description = state.get('instrument_description', '')
        
        # Artist-specific arrangement characteristics
        arrangement_style = artist_profile.get('arrangement_style', 'balanced arrangement with clear instrument roles')
        instrumentation_style = artist_profile.get('instrumentation_style', 'genre-appropriate instrumentation choices')
        style_summary = artist_profile.get('style_summary', '')
        
        # Combine all instruments
        all_instruments = list(set(instruments + backing_instruments))
        if lead_instrument and lead_instrument not in all_instruments:
            all_instruments.append(lead_instrument)
        
        if not all_instruments:
            logging.warning("[InstrumentAgent] No instruments specified, skipping.")
            state['instrument_tracks'] = {'instrument_tracks': []}
            return state
        
        # Create individual tracks for each instrument
        instrument_tracks = []
        
        for i, instrument in enumerate(all_instruments):
            if instrument.lower() in ['drums', 'percussion']:
                continue  # Skip drums - handled by drum agent
            
            # Determine instrument role
            is_lead = (instrument == lead_instrument or 
                      (not lead_instrument and i == 0))
            is_backing = instrument in backing_instruments
            
            role = "lead" if is_lead else "backing" if is_backing else "supporting"
            
            # Section-specific instructions
            section_instructions = ""
            if structured_sections and len(structured_sections) > 2:
                section_instructions = f"\nCREATE {role.upper()} PARTS FOR EACH SECTION:\n"
                for section in structured_sections:
                    section_start = section['start_time']
                    section_end = section['end_time']
                    section_name = section['name'].upper()
                    
                    if 'verse' in section['name'].lower():
                        if is_lead:
                            approach = "melodic development, moderate complexity, supporting vocal line"
                        else:
                            approach = "rhythmic accompaniment, harmonic support, restrained playing"
                    elif 'chorus' in section['name'].lower():
                        if is_lead:
                            approach = "prominent melodic lines, increased energy, memorable hooks"
                        else:
                            approach = "fuller arrangements, stronger rhythmic drive, harmonic richness"
                    elif 'bridge' in section['name'].lower():
                        approach = "contrasting material, unique voicings, creative arrangements"
                    elif 'intro' in section['name'].lower():
                        approach = "establishing character, building anticipation, thematic introduction"
                    elif 'outro' in section['name'].lower():
                        approach = "concluding material, possible solo elements, resolution"
                    else:
                        approach = "section-appropriate instrumental contribution"
                    
                    section_instructions += f"- {section_name} ({section_start:.1f}s-{section_end:.1f}s): {approach}\n"
            
            # Build instrument-specific prompt
            context_description = f"""
            Create {instrument} parts for {genre} with {mood} character.
            
            INSTRUMENT ROLE: {role.upper()}
            ARRANGEMENT CONTEXT: {arrangement_style}
            INSTRUMENTATION STYLE: {instrumentation_style}
            {f'ARTIST CONTEXT: {style_summary}' if style_summary else ''}
            {f'PERFORMANCE INSTRUCTIONS: {instrument_description}' if instrument_description else ''}
            
            MUSICAL PARAMETERS:
            - Tempo: {tempo} BPM
            - Duration: {duration} minutes  
            - Key Context: Available from chord progression
            - Time Signature: {structure.get('time_signature', '4/4')}
            """
            
            # Instrument-specific playing techniques enhanced by genre
            technique_guidance = ""
            instrument_lower = instrument.lower()
            
            if 'guitar' in instrument_lower:
                if 'metal' in genre.lower() or 'rock' in genre.lower():
                    if is_lead:
                        technique_guidance = """Use aggressive power chord riffs, fast scalar runs, and heavy distortion effects. 
                        Include palm-muted sections, chromatic passages, and octave displacement. 
                        Use pitch range 40-75 (low E to high D). Velocity 90-120 for metal aggression.
                        Add technical solos with sweep picking, tapping, and string bending."""
                    else:
                        technique_guidance = """Provide heavy rhythm guitar with power chords and palm muting. 
                        Use syncopated patterns and heavy low-end emphasis. 
                        Pitch range 28-55 (drop tuning range). Velocity 85-110 for rhythm support."""
                elif 'jazz' in genre.lower():
                    technique_guidance = """Use sophisticated jazz chord voicings and single-note lines. 
                    Include extended chords, chromatic approach tones, and swing feel. 
                    Pitch range 40-80. Velocity 65-85 for jazz blend."""
                elif 'blues' in genre.lower():
                    technique_guidance = """Emphasize blue notes, string bending, and slide techniques. 
                    Use pentatonic scales and call-and-response phrasing. 
                    Pitch range 40-75. Velocity 70-95 for blues expression."""
                else:  # Pop/other
                    if is_lead:
                        technique_guidance = """Use melodic lines, tasteful solos, and accessible riffs. 
                        Pitch range 45-80. Velocity 75-100 for pop presence."""
                    else:
                        technique_guidance = """Provide chord support with strumming patterns and arpeggios. 
                        Pitch range 40-75. Velocity 70-90 for accompaniment."""
                        
            elif 'piano' in instrument_lower:
                if 'jazz' in genre.lower():
                    technique_guidance = """Use sophisticated jazz voicings, walking bass lines in left hand, 
                    and bebop-influenced right hand lines. Include block chords and comping patterns. 
                    Full piano range 21-108. Velocity 60-90 for jazz dynamics."""
                elif 'classical' in genre.lower():
                    technique_guidance = """Use proper classical technique with clear voice leading, 
                    balanced hand coordination, and appropriate pedaling effects. 
                    Full range 21-108. Velocity 50-100 for classical expression."""
                elif 'blues' in genre.lower():
                    technique_guidance = """Use blues scales, boogie-woogie left hand patterns, 
                    and blues chord progressions. Include grace notes and blue note emphasis. 
                    Range 28-96. Velocity 70-100 for blues character."""
                else:  # Pop/rock
                    if is_lead:
                        technique_guidance = """Create memorable melodic lines and fills in upper register. 
                        Range 60-96. Velocity 75-100 for melodic presence."""
                    else:
                        technique_guidance = """Provide harmonic support with chord voicings and bass lines. 
                        Range 28-84. Velocity 65-85 for accompaniment."""
                        
            elif 'bass' in instrument_lower:
                if 'metal' in genre.lower() or 'rock' in genre.lower():
                    technique_guidance = """Create powerful, driving bass lines following guitar riffs. 
                    Use aggressive attack and low-end emphasis. Include fast passages and chromatic runs. 
                    Range 28-50 (4-string bass). Velocity 85-110 for metal power."""
                elif 'jazz' in genre.lower():
                    technique_guidance = """Use walking bass lines with sophisticated harmonic movement. 
                    Include passing tones and chord extensions. 
                    Range 28-60. Velocity 65-85 for jazz groove."""
                elif 'funk' in genre.lower():
                    technique_guidance = """Create syncopated, rhythmically complex patterns with ghost notes. 
                    Use slapping and popping techniques. 
                    Range 28-55. Velocity 70-100 for funk groove."""
                else:  # Pop/other
                    technique_guidance = """Provide solid foundation with clear root movement and melodic interest. 
                    Range 28-55. Velocity 70-90 for supportive bass."""
                    
            else:  # Other instruments
                technique_guidance = f"""Use idiomatic {instrument} techniques appropriate for {genre} style. 
                Include genre-specific articulations, phrasing, and expressive techniques. 
                Use appropriate range and dynamics for the instrument and style."""
            
            final_prompt = f"""
            {context_description}
            
            {section_instructions}
            
            PLAYING TECHNIQUE:
            {technique_guidance}
            
            TIMING COORDINATION:
            Align musical phrases to melody onsets: {melody_onsets[:10]}...
            Consider bar boundaries: {bar_times[:8]}...
            
            REQUIREMENTS:
            1. Generate {instrument} parts for FULL {duration} minutes ({duration * 60} seconds)
            2. Create appropriate range and voicing for the instrument
            3. Include musical rests and phrase breaks
            4. Use velocity 70-110 for {role} role dynamics
            5. Coordinate with existing melody and chord structure
            6. Add stylistic ornamentation and expression where appropriate
            
            OUTPUT FORMAT:
            Generate ONLY a valid Python list of lists: [[pitch, start_time, end_time, velocity], ...]
            
            PITCH VALUES: MIDI numbers appropriate for {instrument} range
            TIME VALUES: Precise seconds aligned to musical structure  
            VELOCITY VALUES: 70-110 for dynamic expression
            
            NO explanations, NO text, ONLY the list.
            """
            
            logging.info(f"[InstrumentAgent] Generating {role} {instrument} track...")
            
            try:
                instrument_text = gemini_generate(final_prompt)
                cleaned = clean_llm_output(instrument_text)
                notes_array = safe_literal_eval(cleaned)
                notes = notes_array_to_dicts(notes_array)
                
                # Post-process notes
                if notes:
                    # Sort and validate
                    notes.sort(key=lambda n: n['start'])
                    
                    for note in notes:
                        # Quantize timing
                        note['start'] = round(note['start'] * 4) / 4  # Quarter-note grid
                        note['end'] = round(note['end'] * 4) / 4
                        
                        # Ensure minimum duration
                        if note['end'] <= note['start']:
                            note['end'] = note['start'] + 0.5
                        
                        # Validate velocity for role
                        if is_lead:
                            note['velocity'] = max(80, min(110, note['velocity']))
                        else:
                            note['velocity'] = max(70, min(95, note['velocity']))
                    
                    # Check coverage
                    max_end = max(n['end'] for n in notes)
                    coverage_ratio = max_end / (duration * 60)
                    
                    logging.info(f"[InstrumentAgent] {instrument} ({role}): {len(notes)} notes, "
                                f"{coverage_ratio:.1%} coverage")
                else:
                    logging.warning(f"[InstrumentAgent] No notes generated for {instrument}")
                    notes = []
                
                # Create track
                track = {
                    'name': f"{instrument.title()} ({role.title()})",
                    'program': get_instrument_program(instrument),
                    'is_drum': False,
                    'notes': notes,
                    'instrument': instrument,
                    'role': role
                }
                
                instrument_tracks.append(track)
                
            except Exception as e:
                logging.error(f"[InstrumentAgent] Error generating {instrument} track: {e}")
                continue
        
        state['instrument_tracks'] = {'instrument_tracks': instrument_tracks}
        logging.info(f"[InstrumentAgent] Generated {len(instrument_tracks)} instrument tracks")
        
    except Exception as e:
        logging.error(f"[InstrumentAgent] Error: {e}")
        state['instrument_tracks'] = {'instrument_tracks': []}
    
    return state 