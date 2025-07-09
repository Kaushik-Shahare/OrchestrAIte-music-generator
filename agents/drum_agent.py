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
            logging.error(f"[DrumAgent] Failed to parse LLM output after auto-fix: {e}")
            return []

def notes_array_to_dicts(notes_array):
    return [
        {'pitch': n[0], 'start': n[1], 'end': n[2], 'velocity': n[3]}
        for n in notes_array if isinstance(n, (list, tuple)) and len(n) == 4
    ]

def drum_agent(state: Any) -> Any:
    """
    Generate dynamic, style-specific drum patterns with sophisticated rhythmic complexity.
    """
    logging.info("[DrumAgent] Generating sophisticated drum and percussion patterns.")
    
    try:
        # Extract comprehensive context
        structure = state.get('structure', {})
        bar_times = structure.get('bar_times', [])
        
        artist_profile = state.get('artist_profile', {})
        musical_context = state.get('musical_context', {})
        structured_sections = state.get('structured_sections', [])
        
        # Basic parameters
        genre = musical_context.get('full_genre', state.get('genre', 'pop'))
        mood = musical_context.get('full_mood', state.get('mood', 'happy'))
        tempo = state.get('tempo', 120)
        duration = state.get('duration', 2)
        time_signature = state.get('time_signature', '4/4')
        
        # Complexity and style
        rhythmic_complexity = state.get('rhythmic_complexity', 'medium')
        dynamic_range = state.get('dynamic_range', 'medium')
        emotional_arc = state.get('emotional_arc', '')
        instrument_description = state.get('instrument_description', '')
        
        # Artist-specific drum characteristics
        rhythmic_elements = artist_profile.get('rhythmic_elements', 'solid groove foundation with subtle variations')
        style_summary = artist_profile.get('style_summary', '')
        
        # Standard MIDI drum mapping for reference
        drum_map = {
            'kick': 36, 'snare': 38, 'hihat_closed': 42, 'hihat_open': 46,
            'crash': 49, 'ride': 51, 'tom_low': 45, 'tom_mid': 48, 'tom_high': 50,
            'kick_2': 35, 'snare_2': 40, 'cowbell': 56, 'tambourine': 54
        }
        
        # Complexity-based rhythm guidance
        complexity_guidance = {
            'simple': 'Basic kick-snare patterns, minimal hi-hat work, straightforward grooves',
            'medium': 'Varied kick patterns, ghost notes, hi-hat variations, moderate fills',
            'complex': 'Polyrhythmic patterns, advanced ghost notes, complex hi-hat work, sophisticated fills'
        }
        
        rhythm_instruction = complexity_guidance.get(rhythmic_complexity, complexity_guidance['medium'])
        
        # Dynamic range guidance
        dynamic_guidance = {
            'narrow': 'Consistent velocities (70-90), minimal dynamic variation',
            'medium': 'Moderate velocity range (60-100), some accents and dynamics',
            'wide': 'Full velocity range (40-127), dramatic accents, ghost notes, builds'
        }
        
        dynamic_instruction = dynamic_guidance.get(dynamic_range, dynamic_guidance['medium'])
        
        # Section-specific drum arrangement
        section_instructions = ""
        if structured_sections and len(structured_sections) > 2:
            section_instructions = "\nCREATE RHYTHMIC VARIATION FOR EACH SECTION:\n"
            for section in structured_sections:
                section_start = section['start_time']
                section_end = section['end_time']
                section_name = section['name'].upper()
                
                if 'verse' in section['name'].lower():
                    drum_approach = "solid foundation, moderate energy, subtle variations"
                elif 'chorus' in section['name'].lower():
                    drum_approach = "increased energy, stronger accents, fuller sound"
                elif 'bridge' in section['name'].lower():
                    drum_approach = "contrasting rhythm, possible breaks, unique patterns"
                elif 'intro' in section['name'].lower():
                    drum_approach = "building energy, establishing groove, gradual entrance"
                elif 'outro' in section['name'].lower():
                    drum_approach = "maintaining or reducing energy, possible fade or ending fill"
                else:
                    drum_approach = "section-appropriate rhythmic support"
                
                section_instructions += f"- {section_name} ({section_start:.1f}s-{section_end:.1f}s): {drum_approach}\n"
        
        # Build comprehensive prompt
        context_description = f"""
        Create {genre} drum patterns with {mood} rhythmic character.
        
        RHYTHMIC PARAMETERS:
        - Time Signature: {time_signature}
        - Tempo: {tempo} BPM
        - Duration: {duration} minutes
        - Rhythmic Complexity: {rhythmic_complexity}
        - Dynamic Range: {dynamic_range}
        - Rhythmic Style: {rhythmic_elements}
        
        STYLE CONTEXT:
        {f'- Artist Context: {style_summary}' if style_summary else ''}
        {f'- Emotional Journey: {emotional_arc}' if emotional_arc else ''}
        {f'- Performance Notes: {instrument_description}' if instrument_description else ''}
        """
        
        final_prompt = f"""
        {context_description}
        
        {section_instructions}
        
        RHYTHMIC APPROACH:
        {rhythm_instruction}
        
        DYNAMIC APPROACH:
        {dynamic_instruction}
        
        BAR STRUCTURE:
        Align major pattern changes to bar times: {bar_times[:12]}...
        
        GENRE-SPECIFIC DRUM REQUIREMENTS FOR {genre.upper()}:
        """
        
        # Add genre-specific drum guidance
        genre_lower = genre.lower()
        if 'metal' in genre_lower or 'rock' in genre_lower:
            final_prompt += """
        - Heavy kick drum patterns with double bass (36, 35)
        - Aggressive snare on beats 2 and 4 (38, 40)
        - Fast hi-hat work and crash accents (42, 46, 49)
        - Use toms for fills and transitions (45, 48, 50)
        - High velocity range: 90-127 for aggressive sound
        - Include blast beats and syncopated patterns
        - Add cowbell (56) for accent in some sections
        """
        elif 'jazz' in genre_lower:
            final_prompt += """
        - Swing feel with shuffle hi-hat patterns (42)
        - Subtle kick drum (36) with walking patterns
        - Cross-stick snare (37) and brushes technique
        - Ride cymbal emphasis (51) over hi-hat
        - Moderate velocity: 60-90 for jazz dynamics
        - Include polyrhythmic patterns and odd meters
        - Brushes and mallets for softer sections
        """
        elif 'electronic' in genre_lower or 'edm' in genre_lower:
            final_prompt += """
        - Four-on-the-floor kick patterns (36)
        - Electronic snare sounds (38, 40)
        - Constant hi-hat or electronic percussion (42, 44)
        - Build-ups with cymbal swells (49, 57)
        - High velocity: 100-127 for electronic punch
        - Include electronic percussion elements (54-81)
        - Rhythmic complexity with electronic timing
        """
        elif 'blues' in genre_lower:
            final_prompt += """
        - Shuffle feel with swing timing
        - Steady kick on 1 and 3 (36)
        - Backbeat snare on 2 and 4 (38)
        - Shuffle hi-hat pattern (42)
        - Medium velocity: 70-100 for bluesy feel
        - Include rim shots and ghost notes
        - Simple but groove-heavy patterns
        """
        elif 'folk' in genre_lower or 'country' in genre_lower:
            final_prompt += """
        - Simple, steady patterns supporting the song
        - Basic kick and snare foundation (36, 38)
        - Light hi-hat work (42)
        - Minimal fills, focus on groove
        - Moderate velocity: 65-85 for acoustic feel
        - Include brushes and light percussion
        - Natural, unforced rhythmic feel
        """
        else:  # Pop/other
            final_prompt += """
        - Steady four-to-the-floor or backbeat patterns
        - Clear kick (36) and snare (38) definition
        - Consistent hi-hat patterns (42)
        - Strategic fills at transitions
        - Balanced velocity: 75-100 for commercial sound
        - Include various percussion colors
        - Accessible rhythmic patterns
        """
        
        final_prompt += f"""
        
        TEMPO-SPECIFIC PATTERNS FOR {tempo} BPM:
        """
        
        # Add tempo-specific guidance
        if tempo < 80:
            final_prompt += """
        - Slower, more spacious drum patterns
        - Emphasis on quarter and half-note patterns
        - Allow for natural breathing in the rhythm
        - Use longer sustaining cymbal sounds
        """
        elif tempo < 120:
            final_prompt += """
        - Balanced rhythm with clear pulse
        - Mix of quarter and eighth note patterns
        - Moderate fill complexity
        - Good balance of space and activity
        """
        elif tempo < 160:
            final_prompt += """
        - Active patterns with driving energy
        - Eighth and sixteenth note subdivisions
        - More frequent fills and accents
        - Higher energy rhythmic drive
        """
        else:  # Very fast tempo
            final_prompt += """
        - Rapid-fire patterns with constant motion
        - Sixteenth note subdivisions prominent
        - Quick, punchy fills
        - Maximum rhythmic intensity
        """
        
        final_prompt += f"""
        
        DRUM KIT MAPPING (MIDI):
        - Kick Drum: 36 (primary), 35 (alternate)
        - Snare: 38 (main), 40 (rimshot), 37 (cross-stick)
        - Hi-Hat Closed: 42
        - Hi-Hat Open: 46
        - Hi-Hat Foot: 44
        - Crash Cymbal: 49, 57
        - Ride Cymbal: 51, 59
        - Toms: 45 (low), 48 (mid), 50 (high), 43 (floor)
        - Percussion: 54 (tambourine), 56 (cowbell), 58 (vibraslap)
        
        REQUIREMENTS:
        1. Generate drum patterns for the FULL {duration} minutes ({duration * 60} seconds)
        2. Create realistic drum kit patterns with proper kit piece usage
        3. Include fills at section transitions and natural musical moments
        4. Use appropriate velocity ranges for dynamics and ghost notes
        5. Ensure patterns lock with the tempo and time signature
        6. Add rhythmic interest without overwhelming the mix
        7. CREATE DISTINCTLY {genre.upper()} rhythmic character
        8. MATCH THE {mood.upper()} energy level
        
        OUTPUT FORMAT:
        Generate ONLY a valid Python list of lists: [[pitch, start_time, end_time, velocity], ...]
        
        PITCH VALUES: MIDI drum numbers (see mapping above)
        TIME VALUES: Precise seconds aligned to tempo grid
        VELOCITY VALUES: Use genre-appropriate ranges specified above
        NOTE DURATION: Short (0.1-0.3s) for most drum sounds
        
        NO explanations, NO text, ONLY the list.
        """
        
        logging.info(f"[DrumAgent] Generating drums with {len(bar_times)} bars, "
                    f"complexity: {rhythmic_complexity}, dynamics: {dynamic_range}")
        
        drum_text = gemini_generate(final_prompt)
        logging.info(f"[DrumAgent] Raw Gemini output preview: {drum_text[:200]}...")
        
        cleaned = clean_llm_output(drum_text)
        notes_array = safe_literal_eval(cleaned)
        notes = notes_array_to_dicts(notes_array)
        
        # Post-process drum notes
        if notes:
            # Sort by start time
            notes.sort(key=lambda n: n['start'])
            
            # Validate and adjust drum notes
            for note in notes:
                # Quantize to tight rhythmic grid (16th notes)
                sixteenth_duration = 60.0 / tempo / 4
                note['start'] = round(note['start'] / sixteenth_duration) * sixteenth_duration
                
                # Ensure appropriate drum note durations
                if note['end'] <= note['start']:
                    note['end'] = note['start'] + 0.1  # Short drum hits
                else:
                    note['end'] = min(note['end'], note['start'] + 0.5)  # Max duration
                
                # Validate drum MIDI range
                note['pitch'] = max(35, min(81, note['pitch']))  # Standard drum range
                
                # Ensure realistic velocity range
                note['velocity'] = max(40, min(127, note['velocity']))
            
            # Check coverage and density
            max_end = max(n['end'] for n in notes)
            target_duration = duration * 60
            coverage_ratio = max_end / target_duration
            
            # Calculate average hits per second
            hits_per_second = len(notes) / target_duration if target_duration > 0 else 0
            
            logging.info(f"[DrumAgent] Generated {len(notes)} drum hits covering {max_end:.1f}s "
                        f"({coverage_ratio:.1%} of target), {hits_per_second:.1f} hits/sec")
            
            if coverage_ratio < 0.8:
                logging.warning(f"[DrumAgent] Drum coverage is low ({coverage_ratio:.1%})")
            if hits_per_second > 20:
                logging.warning(f"[DrumAgent] Drum density may be too high ({hits_per_second:.1f} hits/sec)")
        else:
            logging.error("[DrumAgent] No valid drum notes generated!")
            notes = []
        
        drum_track = {
            'name': 'Drums',
            'program': 0,
            'is_drum': True,
            'notes': notes
        }
        
        state['drum_tracks'] = {'drum_track': drum_track}
        logging.info(f"[DrumAgent] Dynamic drum track generated with {len(notes)} hits")
        
    except Exception as e:
        logging.error(f"[DrumAgent] Error: {e}")
        # Provide fallback
        state['drum_tracks'] = {'drum_track': {'name': 'Drums', 'program': 0, 'is_drum': True, 'notes': []}}
    
    return state 