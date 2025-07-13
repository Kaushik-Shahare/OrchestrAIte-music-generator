import logging
from typing import Any
from utils.gemini_llm import gemini_generate
from agents.rag_midi_reference_agent import get_rag_patterns_for_instrument
from utils.state_utils import validate_agent_return, safe_state_update
import ast
import re

def clean_llm_output(text):
    return re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip(), flags=re.MULTILINE).strip()

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
    """Map instrument names to MIDI program numbers with genre-appropriate sounds."""
    instrument_map = {
        'piano': 0, 'electric_piano': 4, 'guitar': 24, 
        'electric_guitar': 30,  # Overdriven Guitar for metal/rock
        'distortion_guitar': 30, 'overdriven_guitar': 29,
        'acoustic_guitar': 24, 'bass': 33, 'electric_bass': 34,  # Electric Bass (pick)
        'violin': 40, 'viola': 41, 'cello': 42, 'trumpet': 56, 
        'trombone': 57, 'saxophone': 64, 'sax': 64, 'flute': 73, 
        'clarinet': 71, 'organ': 16, 'synth': 80, 'synthesizer': 80, 
        'strings': 48, 'pad': 88, 'lead': 80
    }
    
    instrument_lower = instrument_name.lower()
    
    # Enhanced guitar mapping for metal/rock
    if 'electric' in instrument_lower and 'guitar' in instrument_lower:
        return 30  # Overdriven Guitar - much more aggressive sound
    elif 'guitar' in instrument_lower and any(word in instrument_lower for word in ['metal', 'rock', 'distort']):
        return 30  # Overdriven Guitar
    
    for key, program in instrument_map.items():
        if key in instrument_lower:
            return program
    return 0  # Default to piano

def get_instrument_pitch_range(instrument_name, genre="pop"):
    """Get appropriate pitch range for each instrument based on its physical capabilities."""
    instrument_lower = instrument_name.lower()
    
    # Guitar ranges - critical for metal/rock
    if 'electric_guitar' in instrument_lower or 'guitar' in instrument_lower:
        if 'metal' in genre.lower() or 'rock' in genre.lower():
            return {
                'low': 40,   # E2 (low E string)
                'high': 75,  # Eb5 (high frets on high E string)
                'power_chord_range': (28, 55),  # Drop tuning power chords
                'lead_range': (50, 80),         # Lead guitar range
                'rhythm_range': (28, 60)        # Rhythm guitar range
            }
        else:
            return {'low': 40, 'high': 77}  # Standard guitar range
    
    # Bass guitar - very important for low end
    elif 'bass' in instrument_lower:
        return {
            'low': 28,   # E1 (4-string bass low E)
            'high': 50,  # D3 (high fret on G string)
            'fundamental_range': (28, 43),  # Main bass register
            'slap_range': (33, 50)          # Slap bass range
        }
    
    # Piano/Keyboard ranges by register
    elif 'piano' in instrument_lower:
        return {
            'low': 21,    # A0 (piano lowest note)
            'high': 108,  # C8 (piano highest note)
            'bass_range': (21, 48),      # Left hand/bass
            'tenor_range': (48, 60),     # Low melody
            'alto_range': (60, 72),      # Mid melody
            'soprano_range': (72, 96)    # High melody/lead
        }
    
    # Drums - specific MIDI note mappings
    elif 'drum' in instrument_lower:
        return {
            'kick': (35, 36),
            'snare': (37, 40),
            'hihat': (42, 46),
            'crash': (49, 57),
            'ride': (51, 59),
            'toms': (43, 50)
        }
    
    # Wind instruments
    elif 'trumpet' in instrument_lower:
        return {'low': 58, 'high': 82}   # Bb3 to Bb5
    elif 'saxophone' in instrument_lower or 'sax' in instrument_lower:
        return {'low': 49, 'high': 81}   # Db3 to A5 (tenor sax)
    elif 'flute' in instrument_lower:
        return {'low': 60, 'high': 96}   # C4 to C7
    
    # Strings
    elif 'violin' in instrument_lower:
        return {'low': 55, 'high': 91}   # G3 to G6
    elif 'viola' in instrument_lower:
        return {'low': 48, 'high': 84}   # C3 to C6
    elif 'cello' in instrument_lower:
        return {'low': 36, 'high': 72}   # C2 to C5
    
    # Synthesizers - wide range but context dependent
    elif 'synth' in instrument_lower:
        if 'bass' in instrument_lower:
            return {'low': 24, 'high': 48}   # Bass synth
        elif 'lead' in instrument_lower:
            return {'low': 60, 'high': 96}   # Lead synth
        else:
            return {'low': 36, 'high': 84}   # General synth
    
    # Default range for unknown instruments
    else:
        return {'low': 48, 'high': 72}   # C3 to C5

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
        
        # NEW: Extract artist patterns from MIDI reference agent
        artist_patterns = state.get('artist_patterns', {})
        artist_instrument_instructions = state.get('artist_instrument_instructions', {})
        pattern_summary = state.get('pattern_summary', '')
        
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
        
        # Get artist patterns if available
        artist_patterns = state.get('artist_patterns', {})
        artist_instructions = state.get('artist_instrument_instructions', {})
        
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
            return safe_state_update(state, {'instrument_tracks': {'instrument_tracks': []}}, "InstrumentAgent")
        
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
            
            # Build instrument-specific prompt with REAL MIDI PATTERNS
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
            
            # Get RAG-specific patterns for this instrument
            rag_patterns_instruction = get_rag_patterns_for_instrument(state, instrument, role)
            context_description += f"\n\nRAG RETRIEVED PATTERNS:\n{rag_patterns_instruction}"
            
            # Get instrument pitch ranges
            pitch_range = get_instrument_pitch_range(instrument, genre)
            
            # Instrument-specific playing techniques enhanced by genre
            technique_guidance = ""
            instrument_lower = instrument.lower()
            
            if 'guitar' in instrument_lower:
                if 'metal' in genre.lower() or 'rock' in genre.lower():
                    # Get RAG patterns for reference
                    rag_patterns = state.get('rag_patterns', {})
                    
                    if is_lead:
                        technique_guidance = f"""ELECTRIC GUITAR LEAD - METAL/ROCK SPECIFICATIONS:
                        
                        CRITICAL: CREATE AGGRESSIVE, AUDIBLE ELECTRIC GUITAR SOUNDS!
                        
                        RAG PATTERNS APPLIED: Use the retrieved patterns above as exact templates.
                        
                        PITCH RANGES (ABSOLUTELY MANDATORY):
                        - Power Chords: {pitch_range.get('power_chord_range', (28, 55))[0]}-{pitch_range.get('power_chord_range', (28, 55))[1]} (crushing low riffs)
                        - Lead Lines: {pitch_range.get('lead_range', (50, 80))[0]}-{pitch_range.get('lead_range', (50, 80))[1]} (screaming solos)
                        - Main Riffs: 40-65 (the sweet spot for metal guitar)
                        
                        REQUIRED METAL GUITAR TECHNIQUES (IMPLEMENT ALL):
                        1. USE RAG-RETRIEVED PATTERNS FIRST - Apply the exact note sequences from similar songs above
                        
                        2. PALM-MUTED POWER CHORDS: Use pitches 28-50, duration 0.1-0.4s, velocity 105-127
                           Example pattern: [40, 0.0, 0.3, 120], [40, 0.5, 0.8, 115], [45, 1.0, 1.3, 120]
                        
                        3. FAST TREMOLO PICKING: Rapid repeated notes 45-65, velocity 100-120, duration 0.1-0.2s each
                           Example: [50, 0.0, 0.15, 110], [50, 0.15, 0.3, 110], [50, 0.3, 0.45, 110]
                        
                        4. AGGRESSIVE POWER CHORD PROGRESSIONS: Follow RAG patterns or use standard metal: [40,45,47,42]
                           
                        5. CHROMATIC METAL RUNS: Half-step sequences for brutality
                           Example: [40, 0.0, 0.2, 115], [41, 0.2, 0.4, 115], [42, 0.4, 0.6, 115]
                        
                        6. BREAKDOWN SECTIONS: Slow, heavy chords with maximum impact
                           Use pitches 28-45, duration 1.0-2.0s, velocity 120-127
                           
                        7. GUITAR HARMONICS AND BENDS: Include pitch variations within single notes
                        
                        VELOCITY REQUIREMENTS: 100-127 ONLY (metal must be LOUD and AGGRESSIVE)
                        TIMING PATTERNS: 
                        - Fast sections: 0.1-0.3s note durations
                        - Power chords: 0.3-1.0s durations  
                        - Breakdown: 1.0-2.0s durations
                        - Always align to beat grid: 0.0, 0.25, 0.5, 0.75, 1.0, 1.25...
                        
                        CRITICAL: This MUST sound like Metallica, not background music!
                        Generate CRUSHING, DRIVING metal guitar parts with proper aggression!
                        """
                    else:
                        technique_guidance = f"""ELECTRIC GUITAR RHYTHM - METAL/ROCK SPECIFICATIONS:
                        
                        PITCH RANGE: {pitch_range.get('rhythm_range', (28, 60))[0]}-{pitch_range.get('rhythm_range', (28, 60))[1]}
                        
                        RHYTHM GUITAR TECHNIQUES:
                        1. HEAVY POWER CHORDS: Root and fifth combinations in low register (28-50)
                        2. PALM-MUTED CHUGGING: Short, percussive chord hits with high velocity
                        3. SYNCOPATED RHYTHMS: Off-beat accents and complex timing
                        4. BREAKDOWN RIFFS: Slow, crushing chord progressions
                        
                        Velocity: 85-115, Duration: 0.2-1.5s for rhythm parts
                        """
                elif 'jazz' in genre.lower():
                    technique_guidance = f"""JAZZ GUITAR: Sophisticated chord voicings {pitch_range['low']}-{pitch_range['high']}, 
                    extended chords, chromatic approach tones, swing feel. Velocity 65-85."""
                elif 'blues' in genre.lower():
                    technique_guidance = f"""BLUES GUITAR: Blue notes, string bending, slide techniques {pitch_range['low']}-{pitch_range['high']}. 
                    Pentatonic scales, call-and-response phrasing. Velocity 70-95."""
                else:  # Pop/other
                    if is_lead:
                        technique_guidance = f"""POP GUITAR LEAD: Melodic lines, tasteful solos {pitch_range['low']}-{pitch_range['high']}. 
                        Velocity 75-100."""
                    else:
                        technique_guidance = f"""POP GUITAR RHYTHM: Chord strumming, arpeggios {pitch_range['low']}-{pitch_range['high']}. 
                        Velocity 70-90."""
                        
            elif 'piano' in instrument_lower:
                bass_range = pitch_range.get('bass_range', (21, 48))
                melody_range = pitch_range.get('soprano_range', (72, 96)) if is_lead else pitch_range.get('alto_range', (60, 72))
                
                if 'jazz' in genre.lower():
                    technique_guidance = f"""JAZZ PIANO: Sophisticated voicings, walking bass lines in left hand {bass_range[0]}-{bass_range[1]}, 
                    bebop lines in right hand {melody_range[0]}-{melody_range[1]}. Block chords and comping. Velocity 60-90."""
                elif 'classical' in genre.lower():
                    technique_guidance = f"""CLASSICAL PIANO: Proper voice leading {pitch_range['low']}-{pitch_range['high']}, 
                    balanced hands, pedaling effects. Velocity 50-100."""
                elif 'blues' in genre.lower():
                    technique_guidance = f"""BLUES PIANO: Blues scales, boogie-woogie left hand {bass_range[0]}-{bass_range[1]}, 
                    blues chord progressions. Grace notes and blue notes. Velocity 70-100."""
                else:  # Pop/rock
                    if is_lead:
                        technique_guidance = f"""PIANO LEAD: Melodic lines and fills in upper register {melody_range[0]}-{melody_range[1]}. 
                        Velocity 75-100."""
                    else:
                        technique_guidance = f"""PIANO ACCOMPANIMENT: Chord voicings {bass_range[0]}-{melody_range[1]}, 
                        bass lines in left hand. Velocity 65-85."""
                        
            elif 'bass' in instrument_lower:
                fundamental_range = pitch_range.get('fundamental_range', (28, 43))
                
                if 'metal' in genre.lower() or 'rock' in genre.lower():
                    technique_guidance = f"""METAL BASS: Powerful, driving bass lines {pitch_range['low']}-{pitch_range['high']} 
                    following guitar riffs. Aggressive attack, low-end emphasis. Include fast passages and chromatic runs. 
                    Focus on fundamental range {fundamental_range[0]}-{fundamental_range[1]}. 
                    Velocity 85-115 for metal power. Use short, punchy notes (0.2-0.8s) for palm-muted sections."""
                elif 'jazz' in genre.lower():
                    technique_guidance = f"""JAZZ BASS: Walking bass lines {pitch_range['low']}-{pitch_range['high']} 
                    with sophisticated harmonic movement. Include passing tones. Velocity 65-85."""
                elif 'funk' in genre.lower():
                    technique_guidance = f"""FUNK BASS: Syncopated, rhythmically complex patterns {pitch_range['low']}-{pitch_range['high']} 
                    with ghost notes. Slapping and popping techniques. Velocity 70-100."""
                else:  # Pop/other
                    technique_guidance = f"""POP BASS: Solid foundation {pitch_range['low']}-{pitch_range['high']} 
                    with clear root movement and melodic interest. Velocity 70-90."""
                    
            else:  # Other instruments
                technique_guidance = f"""INSTRUMENT RANGE: {pitch_range['low']}-{pitch_range['high']}. 
                Use idiomatic {instrument} techniques for {genre} style. Include genre-specific articulations and phrasing."""
            
            final_prompt = f"""
            {context_description}
            
            {section_instructions}
            
            INSTRUMENT: {instrument.upper()} - {role.upper()} ROLE
            GENRE: {genre.upper()}
            
            {technique_guidance}
            
            TIMING COORDINATION (CRITICAL FOR SYNC):
            - Melody onset times (sync points): {melody_onsets[:15]}...
            - Bar boundaries every {60/tempo*4:.2f}s: {[f"{t:.2f}s" for t in bar_times[:8]]}...
            - Beat grid (align all notes): {[f"{i*60/tempo:.2f}" for i in range(int(duration*tempo/60))]}...
            
            SYNCHRONIZATION REQUIREMENTS:
            1. START major musical phrases at melody onset times: {melody_onsets[:10]}
            2. ALIGN all note start times to quarter-beat grid (multiples of {60/tempo/4:.3f}s)
            3. COORDINATE with tempo {tempo} BPM - each beat = {60/tempo:.3f}s
            4. MATCH the energy and dynamics of the melody structure
            5. CREATE complementary parts that ENHANCE the melody, don't compete
            
            CRITICAL REQUIREMENTS FOR {instrument.upper()} IN {genre.upper()}:
            1. Generate {instrument} parts for FULL {duration} minutes ({duration * 60} seconds)
            2. ABSOLUTELY MANDATORY: Use ONLY pitches within the specified ranges - any note outside will be rejected
            3. Use the exact velocity ranges specified for {genre} character - this creates the genre sound
            4. Include musical rests and phrase breaks as specified
            5. Coordinate with existing melody and chord structure
            6. Add stylistic ornamentation and expression where appropriate
            7. Create GENRE-APPROPRIATE sound - {genre} should sound distinctly different from other genres
            8. For METAL/ROCK: Make it sound AGGRESSIVE and POWERFUL with proper guitar techniques
            9. CREATE VARIATION: Do NOT repeat the same patterns - build musical development
            10. SYNC to beat grid: Align notes to multiples of {60/tempo/4:.3f}s for timing sync
            
            ANTI-REPETITION REQUIREMENTS:
            - Use at least 3-4 different riff/phrase patterns throughout the song
            - Vary dynamics: include quiet breaks and loud power sections
            - Create build-ups: start simple, add complexity over time
            - Include variation techniques: inversions, rhythmic displacement, pitch variation
            - For solos: use different techniques in different sections (scales, arpeggios, power chords)
            
            MANDATORY PITCH RANGES (VIOLATIONS WILL BE AUTO-CORRECTED):
            - Electric guitar in metal: MUST use pitches 28-80 ONLY, focus on 28-55 for crushing power chords
            - Bass guitar: MUST use pitches 28-50 ONLY, emphasize 28-43 for deep fundamental bass
            - Piano lead: MUST use upper register 60-96 ONLY, no bass range conflicts
            - Piano accompaniment: MUST use 21-72 ONLY, separate bass (21-48) and harmony (48-72) clearly
            - Each instrument MUST stay in its designated register to avoid muddy mixing
            
            OUTPUT FORMAT:
            Generate ONLY a valid Python list of lists: [[pitch, start_time, end_time, velocity], ...]
            
            PITCH VALUES: MIDI numbers WITHIN THE SPECIFIED RANGES ABOVE
            TIME VALUES: Precise seconds aligned to musical structure  
            VELOCITY VALUES: Use the exact ranges specified above for {genre}
            DURATION VALUES: Use the specified durations for the playing style
            
            EXAMPLE FOR METAL ELECTRIC GUITAR:
            [[40, 0.0, 0.3, 110], [45, 0.3, 0.6, 115], [40, 0.6, 0.9, 110]] # Power chord riff
            
            NO explanations, NO text, ONLY the list.
            """
            
            logging.info(f"[InstrumentAgent] Generating {role} {instrument} track...")
            
            try:
                instrument_text = gemini_generate(final_prompt)
                cleaned = clean_llm_output(instrument_text)
                notes_array = safe_literal_eval(cleaned)
                notes = notes_array_to_dicts(notes_array)
                
                # Post-process notes with strict pitch range enforcement
                if notes:
                    # Sort and validate
                    notes.sort(key=lambda n: n['start'])
                    
                    # Get pitch range for validation
                    pitch_range = get_instrument_pitch_range(instrument, genre)
                    min_pitch = pitch_range.get('low', 21)
                    max_pitch = pitch_range.get('high', 108)
                    
                    # Special handling for guitars in metal
                    if 'guitar' in instrument.lower() and ('metal' in genre.lower() or 'rock' in genre.lower()):
                        if is_lead:
                            min_pitch = pitch_range.get('lead_range', (50, 80))[0]
                            max_pitch = pitch_range.get('lead_range', (50, 80))[1]
                        else:
                            min_pitch = pitch_range.get('rhythm_range', (28, 60))[0]
                            max_pitch = pitch_range.get('rhythm_range', (28, 60))[1]
                    elif 'bass' in instrument.lower():
                        min_pitch = pitch_range.get('fundamental_range', (28, 43))[0]
                        max_pitch = pitch_range.get('fundamental_range', (28, 43))[1]
                    elif 'piano' in instrument.lower():
                        if is_lead:
                            min_pitch = pitch_range.get('soprano_range', (72, 96))[0]
                            max_pitch = pitch_range.get('soprano_range', (72, 96))[1]
                        else:
                            min_pitch = pitch_range.get('bass_range', (21, 48))[0]
                            max_pitch = pitch_range.get('alto_range', (60, 72))[1]
                    
                    valid_notes = []
                    corrected_count = 0
                    
                    for note in notes:
                        # Quantize timing
                        note['start'] = round(note['start'] * 4) / 4  # Quarter-note grid
                        note['end'] = round(note['end'] * 4) / 4
                        
                        # Ensure minimum duration
                        if note['end'] <= note['start']:
                            note['end'] = note['start'] + 0.5
                        
                        # AGGRESSIVE pitch range enforcement
                        original_pitch = note['pitch']
                        
                        # For extreme violations, transpose to correct octave
                        if note['pitch'] < min_pitch:
                            # Move up to correct octave
                            while note['pitch'] < min_pitch:
                                note['pitch'] += 12
                            if note['pitch'] > max_pitch:
                                note['pitch'] = min_pitch + (original_pitch % 12)
                        elif note['pitch'] > max_pitch:
                            # Move down to correct octave  
                            while note['pitch'] > max_pitch:
                                note['pitch'] -= 12
                            if note['pitch'] < min_pitch:
                                note['pitch'] = max_pitch - (11 - (original_pitch % 12))
                        
                        # Final clamp
                        note['pitch'] = max(min_pitch, min(max_pitch, note['pitch']))
                        
                        if original_pitch != note['pitch']:
                            corrected_count += 1
                        
                        # Validate velocity for role and genre - MAXIMUM AGGRESSION for metal
                        if 'metal' in genre.lower() and 'guitar' in instrument.lower():
                            note['velocity'] = max(105, min(127, note['velocity']))  # EXTREME AGGRESSION
                        elif 'bass' in instrument.lower() and 'metal' in genre.lower():
                            note['velocity'] = max(100, min(127, note['velocity']))  # Heavy punchy bass
                        elif is_lead:
                            note['velocity'] = max(90, min(120, note['velocity']))  # Prominent lead
                        else:
                            note['velocity'] = max(80, min(110, note['velocity']))  # Strong backing
                        
                        valid_notes.append(note)
                    
                    if corrected_count > 0:
                        logging.warning(f"[InstrumentAgent] Corrected {corrected_count}/{len(notes)} pitches for {instrument} to stay in range {min_pitch}-{max_pitch}")
                    
                    notes = valid_notes
                    
                    # Check coverage
                    max_end = max(n['end'] for n in notes)
                    coverage_ratio = max_end / (duration * 60)
                    
                    logging.info(f"[InstrumentAgent] {instrument} ({role}): {len(notes)} notes, "
                                f"{coverage_ratio:.1%} coverage")
                    
                    # APPLY REAL MIDI PATTERNS to enhance the generated notes
                    rag_patterns = state.get('rag_patterns', {})
                    if rag_patterns:
                        notes = apply_rag_patterns_to_notes(notes, rag_patterns, instrument, role, duration)
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
        
        # Use safe state update
        logging.info(f"[InstrumentAgent] Generated {len(instrument_tracks)} instrument tracks")
        return safe_state_update(state, {'instrument_tracks': {'instrument_tracks': instrument_tracks}}, "InstrumentAgent")
        
    except Exception as e:
        logging.error(f"[InstrumentAgent] Error: {e}")
        return safe_state_update(state, {'instrument_tracks': {'instrument_tracks': []}}, "InstrumentAgent")

def apply_real_midi_patterns(generated_notes, playable_patterns, instrument, role, duration_minutes):
    """
    Apply real MIDI patterns to enhance generated notes.
    """
    try:
        if not playable_patterns or not generated_notes:
            return generated_notes
        
        logging.info(f"[InstrumentAgent] Applying real MIDI patterns to {instrument} ({role})")
        
        enhanced_notes = generated_notes.copy()
        duration_seconds = duration_minutes * 60
        
        # Apply segments first (most valuable patterns)
        if playable_patterns.get('segments') and len(generated_notes) > 0:
            segment = playable_patterns['segments'][0]  # Use first segment
            
            # Find matching instrument in segment
            matching_inst = None
            for seg_inst in segment.get('instruments', []):
                if instrument.lower().replace('_', ' ') in seg_inst.get('name', '').lower():
                    matching_inst = seg_inst
                    break
            
            if matching_inst and matching_inst.get('notes'):
                # Apply segment notes as inspiration
                segment_notes = []
                for note_data in matching_inst['notes'][:20]:  # First 20 notes
                    if len(note_data) >= 4:
                        pitch, start, duration, velocity = note_data[:4]
                        
                        # Convert to our format
                        segment_note = {
                            'pitch': int(pitch),
                            'start': float(start),
                            'end': float(start + duration),
                            'velocity': int(velocity)
                        }
                        segment_notes.append(segment_note)
                
                if segment_notes:
                    # Blend segment notes with generated notes
                    # Replace first 25% of generated notes with segment-inspired ones
                    blend_count = min(len(segment_notes), len(enhanced_notes) // 4)
                    
                    for i in range(blend_count):
                        if i < len(segment_notes):
                            # Scale timing to fit our duration
                            segment_note = segment_notes[i].copy()
                            time_scale = duration_seconds / 16.0  # Assume segment is ~16 seconds
                            segment_note['start'] *= time_scale
                            segment_note['end'] *= time_scale
                            
                            # Replace generated note with segment note
                            enhanced_notes[i] = segment_note
                    
                    logging.info(f"[InstrumentAgent] Applied {blend_count} segment notes to {instrument}")
        
        # Apply melodic sequences if it's a lead instrument
        if role == 'lead' and playable_patterns.get('melodies'):
            melody_pattern = playable_patterns['melodies'][0]
            intervals = melody_pattern.get('intervals', [])
            
            if intervals and len(enhanced_notes) > 10:
                # Apply intervals to modify existing melody
                start_idx = len(enhanced_notes) // 3  # Start from 1/3 through
                
                for i, interval in enumerate(intervals[:10]):  # Apply up to 10 intervals
                    note_idx = start_idx + i
                    if note_idx < len(enhanced_notes):
                        # Apply interval to modify pitch
                        enhanced_notes[note_idx]['pitch'] += interval
                        # Keep in reasonable range
                        enhanced_notes[note_idx]['pitch'] = max(24, min(108, enhanced_notes[note_idx]['pitch']))
                
                logging.info(f"[InstrumentAgent] Applied melodic intervals to {instrument}")
        
        # Apply chord progressions for backing instruments
        if role in ['backing', 'supporting'] and playable_patterns.get('chord_progressions'):
            chord_pattern = playable_patterns['chord_progressions'][0]
            chords = chord_pattern.get('chords', [])
            
            if chords and len(enhanced_notes) > 5:
                # Apply chord tones to some notes
                chord_idx = 0
                
                for i, note in enumerate(enhanced_notes):
                    if i % 4 == 0:  # Every 4th note
                        if chord_idx < len(chords):
                            chord = chords[chord_idx]
                            pitches = chord.get('pitches', [0, 4, 7])
                            
                            # Choose a chord tone
                            chosen_pitch_class = pitches[i % len(pitches)]
                            
                            # Apply to note (keeping octave similar)
                            current_octave = note['pitch'] // 12
                            new_pitch = current_octave * 12 + chosen_pitch_class
                            
                            # Adjust octave if needed
                            if abs(new_pitch - note['pitch']) > 6:  # More than tritone
                                if new_pitch > note['pitch']:
                                    new_pitch -= 12
                                else:
                                    new_pitch += 12
                            
                            enhanced_notes[i]['pitch'] = max(24, min(108, new_pitch))
                            chord_idx = (chord_idx + 1) % len(chords)
                
                logging.info(f"[InstrumentAgent] Applied chord progressions to {instrument}")
        
        # Apply rhythm patterns
        if playable_patterns.get('rhythm_patterns'):
            rhythm_pattern = playable_patterns['rhythm_patterns'][0]
            intervals = rhythm_pattern.get('intervals', [])
            
            if intervals and len(enhanced_notes) > 5:
                # Adjust note timing based on rhythm pattern
                interval_idx = 0
                current_time = 0.0
                
                for i, note in enumerate(enhanced_notes[:20]):  # First 20 notes
                    if interval_idx < len(intervals):
                        target_interval = intervals[interval_idx]
                        
                        # Adjust note timing
                        enhanced_notes[i]['start'] = current_time
                        enhanced_notes[i]['end'] = current_time + min(target_interval, 2.0)
                        
                        current_time += target_interval
                        interval_idx = (interval_idx + 1) % len(intervals)
                
                logging.info(f"[InstrumentAgent] Applied rhythm patterns to {instrument}")
        
        return enhanced_notes
        
    except Exception as e:
        logging.error(f"[InstrumentAgent] Error applying MIDI patterns to {instrument}: {e}")
        return generated_notes

def apply_rag_patterns_to_notes(notes, rag_patterns, instrument, role, duration):
    """
    Apply RAG-retrieved patterns to enhance generated notes with real MIDI data
    """
    try:
        if not rag_patterns or not notes:
            return notes
        
        # Get relevant patterns for this instrument
        segments = rag_patterns.get('segments', [])
        progressions = rag_patterns.get('progressions', [])
        
        enhanced_notes = []
        
        # Apply segment patterns if available
        for segment in segments[:2]:  # Use top 2 segments
            pattern_data = segment.get('pattern_data', {})
            if not pattern_data:
                continue
                
            pattern_instruments = pattern_data.get('instruments', [])
            
            # Find matching instrument patterns
            for pattern_inst in pattern_instruments:
                pattern_name = pattern_inst.get('name', '').lower()
                
                # Match instrument types
                if (instrument.lower() in pattern_name or 
                    pattern_name in instrument.lower() or
                    ('piano' in instrument.lower() and 'piano' in pattern_name) or
                    ('guitar' in instrument.lower() and 'guitar' in pattern_name) or
                    ('string' in instrument.lower() and any(s in pattern_name for s in ['violin', 'cello', 'string']))):
                    
                    # Apply pattern characteristics
                    pattern_notes = pattern_inst.get('notes', [])
                    if pattern_notes:
                        # Extract timing and velocity patterns
                        pattern_intervals = []
                        pattern_velocities = []
                        pattern_durations = []
                        
                        for note in pattern_notes[:20]:  # First 20 notes
                            if isinstance(note, dict):
                                pattern_velocities.append(note.get('velocity', 80))
                                pattern_durations.append(note.get('duration', 0.5))
                            elif isinstance(note, (list, tuple)) and len(note) >= 4:
                                pattern_velocities.append(note[3])
                                if len(note) >= 3:
                                    pattern_durations.append(note[2] - note[1])
                        
                        # Apply pattern characteristics to generated notes
                        if pattern_velocities and len(notes) > 0:
                            avg_pattern_velocity = sum(pattern_velocities) / len(pattern_velocities)
                            velocity_factor = avg_pattern_velocity / 80.0  # Normalize around 80
                            
                            # Adjust velocities based on pattern
                            for note in notes:
                                if isinstance(note, dict):
                                    note['velocity'] = int(min(127, max(20, note.get('velocity', 80) * velocity_factor)))
                                elif isinstance(note, list) and len(note) >= 4:
                                    note[3] = int(min(127, max(20, note[3] * velocity_factor)))
                        
                        # Apply rhythmic patterns if available
                        if pattern_durations and len(notes) > 1:
                            avg_pattern_duration = sum(pattern_durations) / len(pattern_durations)
                            duration_factor = max(0.1, min(2.0, avg_pattern_duration / 0.5))  # Reasonable bounds
                            
                            # Adjust note durations
                            for i, note in enumerate(notes):
                                if isinstance(note, dict):
                                    current_duration = note.get('end', note.get('start', 0) + 0.5) - note.get('start', 0)
                                    new_duration = current_duration * duration_factor
                                    note['end'] = note.get('start', 0) + new_duration
                                elif isinstance(note, list) and len(note) >= 3:
                                    current_duration = note[2] - note[1]
                                    new_duration = current_duration * duration_factor
                                    note[2] = note[1] + new_duration
        
        # Apply chord progression patterns
        for progression in progressions[:1]:  # Use top progression
            pattern_data = progression.get('pattern_data', {})
            chords = pattern_data.get('chords', [])
            
            if chords and role in ['lead', 'melody', 'harmony']:
                # Extract harmonic content to influence note choices
                chord_roots = []
                for chord in chords[:8]:  # First 8 chords
                    if isinstance(chord, dict):
                        root = chord.get('root', 60)
                        chord_roots.append(root % 12)  # Get root note class
                
                if chord_roots:
                    # Bias notes toward chord tones
                    for note in notes:
                        if isinstance(note, dict):
                            pitch = note.get('pitch', 60)
                        elif isinstance(note, list) and len(note) >= 1:
                            pitch = note[0]
                        else:
                            continue
                        
                        # Find closest chord root
                        note_class = pitch % 12
                        if note_class not in chord_roots:
                            # Find nearest chord tone
                            distances = [abs(note_class - root) for root in chord_roots]
                            min_distance = min(distances)
                            if min_distance <= 2:  # Within 2 semitones
                                closest_root = chord_roots[distances.index(min_distance)]
                                adjustment = closest_root - note_class
                                if abs(adjustment) > 6:  # Handle octave wrap
                                    adjustment = adjustment - 12 if adjustment > 0 else adjustment + 12
                                
                                new_pitch = pitch + adjustment
                                if 21 <= new_pitch <= 108:  # Piano range
                                    if isinstance(note, dict):
                                        note['pitch'] = new_pitch
                                    elif isinstance(note, list):
                                        note[0] = new_pitch
        
        logging.info(f"[RAG Pattern Application] Enhanced {len(notes)} notes for {instrument} using real MIDI patterns")
        return notes
        
    except Exception as e:
        logging.error(f"[RAG Pattern Application] Error applying patterns to {instrument}: {e}")
        return notes  # Return original notes if enhancement fails
