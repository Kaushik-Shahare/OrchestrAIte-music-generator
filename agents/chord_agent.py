import logging
from typing import Any
from utils.gemini_llm import gemini_generate
import ast
import re

def clean_llm_output(text):
    return re.sub(r"^```[a-zA-Z]*\n?|```$", "", text.strip(), flags=re.MULTILINE).strip()

def safe_literal_eval(text):
    # Try normal parse
    try:
        return ast.literal_eval(text)
    except Exception:
        # Try to auto-close brackets
        if text.count('[') > text.count(']'):
            text = text + (']' * (text.count('[') - text.count(']')))
        if text.count('{') > text.count('}'):
            text = text + ('}' * (text.count('{') - text.count('}')))
        try:
            return ast.literal_eval(text)
        except Exception as e:
            logging.error(f"[ChordAgent] Failed to parse LLM output after auto-fix: {e}")
            return []

def notes_array_to_dicts(notes_array):
    return [
        {'pitch': n[0], 'start': n[1], 'end': n[2], 'velocity': n[3]}
        for n in notes_array if isinstance(n, (list, tuple)) and len(n) == 4
    ]

def chord_agent(state: Any) -> Any:
    """
    Generate sophisticated harmonic progressions with style-specific voicings and complexity.
    """
    logging.info("[ChordAgent] Generating sophisticated chord progressions.")
    
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
        key_signature = state.get('key_signature', 'C major')
        time_signature = state.get('time_signature', '4/4')
        
        # Complexity and style
        harmonic_complexity = state.get('harmonic_complexity', 'medium')
        emotional_arc = state.get('emotional_arc', '')
        modulations = state.get('modulations', '')
        instrument_description = state.get('instrument_description', '')
        
        # Artist-specific harmonic characteristics
        harmonic_language = artist_profile.get('harmonic_language', 'contemporary chord progressions with tasteful extensions')
        arrangement_style = artist_profile.get('arrangement_style', 'balanced arrangement with clear harmonic support')
        style_summary = artist_profile.get('style_summary', '')
        
        # Build harmonic complexity guidance
        complexity_guidance = {
            'simple': 'Use basic triads, simple progressions (I-V-vi-IV), minimal extensions',
            'medium': 'Include 7th chords, some extensions (9ths, sus), secondary dominants, modal interchange',
            'complex': 'Advanced voicings, extended chords (9th, 11th, 13th), altered dominants, complex progressions, reharmonization'
        }
        
        harmonic_instruction = complexity_guidance.get(harmonic_complexity, complexity_guidance['medium'])
        
        # Section-specific harmonic planning
        section_instructions = ""
        if structured_sections and len(structured_sections) > 2:
            section_instructions = "\nCREATE HARMONIC VARIATION FOR EACH SECTION:\n"
            for section in structured_sections:
                section_start = section['start_time']
                section_end = section['end_time']
                section_name = section['name'].upper()
                
                if 'verse' in section['name'].lower():
                    harm_approach = "stable, foundational progressions, moderate voice leading"
                elif 'chorus' in section['name'].lower():
                    harm_approach = "stronger progressions, higher voicings, more energy"
                elif 'bridge' in section['name'].lower():
                    harm_approach = "contrasting harmony, possible key change, unique progressions"
                elif 'intro' in section['name'].lower():
                    harm_approach = "establishing key center, building harmonic foundation"
                elif 'outro' in section['name'].lower():
                    harm_approach = "resolving progressions, possible extension or fade harmony"
                else:
                    harm_approach = "section-appropriate harmonic support"
                
                section_instructions += f"- {section_name} ({section_start:.1f}s-{section_end:.1f}s): {harm_approach}\n"
        
        # Timing alignment strategy
        timing_strategy = ""
        if melody_onsets:
            timing_strategy = f"Align chord changes primarily to these melody onset times: {melody_onsets[:10]}... "
        if bar_times:
            timing_strategy += f"Also consider bar boundaries at: {bar_times[:8]}... "
        
        # Build comprehensive prompt
        context_description = f"""
        Create {genre} chord progressions with {mood} harmonic character.
        
        HARMONIC PARAMETERS:
        - Key: {key_signature}
        - Time Signature: {time_signature}
        - Tempo: {tempo} BPM  
        - Duration: {duration} minutes
        - Complexity Level: {harmonic_complexity}
        - Harmonic Language: {harmonic_language}
        
        STYLE CONTEXT:
        - Arrangement Approach: {arrangement_style}
        {f'- Artist Context: {style_summary}' if style_summary else ''}
        {f'- Emotional Journey: {emotional_arc}' if emotional_arc else ''}
        {f'- Modulation Instructions: {modulations}' if modulations else ''}
        {f'- Performance Notes: {instrument_description}' if instrument_description else ''}
        """
        
        final_prompt = f"""
        {context_description}
        
        {section_instructions}
        
        HARMONIC APPROACH:
        {harmonic_instruction}
        
        TIMING COORDINATION:
        {timing_strategy}
        
        GENRE-SPECIFIC CHORD REQUIREMENTS FOR {genre.upper()}:
        """
        
        # Add genre-specific chord guidance
        genre_lower = genre.lower()
        if 'metal' in genre_lower or 'rock' in genre_lower:
            final_prompt += """
        - Use power chords (root + 5th) as foundation
        - Include heavy low-end emphasis (E2-A3 range)
        - Add chromatic movement and diminished passing chords
        - Use drop tuning chord voicings when appropriate
        - Strong rhythmic emphasis on chord changes
        - Include suspended chords (sus2, sus4) for tension
        - Chord velocity: 85-110 for aggressive character
        """
        elif 'jazz' in genre_lower:
            final_prompt += """
        - Use extended chords (7ths, 9ths, 11ths, 13ths)
        - Include complex substitutions and alterations
        - Sophisticated voice leading with smooth connections
        - Use tritone substitutions and secondary dominants
        - Wide range chord voicings (C3-C6)
        - Subtle rhythmic placement with swing feel
        - Chord velocity: 65-85 for sophisticated blend
        """
        elif 'classical' in genre_lower:
            final_prompt += """
        - Use traditional triads and seventh chords
        - Clear tonal centers with functional harmony
        - Proper voice leading and chord inversions
        - Include modulations and tonicizations
        - Balanced chord voicings across registers
        - Chord velocity: 70-90 for balanced classical sound
        """
        elif 'electronic' in genre_lower or 'edm' in genre_lower:
            final_prompt += """
        - Use synthesizer-friendly chord stacks
        - Include wide-spread voicings for big sound
        - Rhythmic chord stabs and filter sweeps
        - Build-ups with layered harmonic content
        - Wide frequency range (C2-C7)
        - Chord velocity: 80-105 for electronic punch
        """
        elif 'blues' in genre_lower:
            final_prompt += """
        - Dominant 7th chords as primary harmony
        - 12-bar blues progression patterns
        - Include blue note harmony (♭3, ♭7)
        - Simple but effective chord voicings
        - Strong rhythmic emphasis on beat patterns
        - Chord velocity: 75-95 for bluesy character
        """
        elif 'folk' in genre_lower or 'country' in genre_lower:
            final_prompt += """
        - Simple triads and basic seventh chords
        - Open chord voicings with clear bass notes
        - Traditional folk/country progressions
        - Strumming-pattern friendly chord changes
        - Moderate range voicings (C3-C5)
        - Chord velocity: 70-85 for acoustic character
        """
        else:  # Pop/other
            final_prompt += """
        - Accessible major and minor triads with some extensions
        - Clear bass movement and memorable progressions
        - Modern pop chord voicings (not too complex)
        - Strategic use of inversions for smooth bass lines
        - Commercial-friendly harmonic content
        - Chord velocity: 75-90 for polished pop sound
        """
        
        final_prompt += f"""
        
        MOOD-SPECIFIC HARMONIC CHARACTER FOR {mood.upper()}:
        """
        
        # Add mood-specific harmonic guidance
        mood_lower = mood.lower()
        if 'aggressive' in mood_lower or 'angry' in mood_lower:
            final_prompt += """
        - Use dissonant intervals and chromatic harmony
        - Include diminished and augmented chords
        - Strong, accented chord attacks
        - Lower register emphasis for power
        """
        elif 'sad' in mood_lower or 'melancholic' in mood_lower:
            final_prompt += """
        - Emphasize minor keys and modal harmony
        - Use suspended chords for emotional tension
        - Include descending bass lines
        - Gentle chord voicings in mid-register
        """
        elif 'happy' in mood_lower or 'joyful' in mood_lower:
            final_prompt += """
        - Bright major key harmony
        - Uplifting chord progressions with clear resolution
        - Higher register chord voicings
        - Rhythmically active chord changes
        """
        elif 'mysterious' in mood_lower or 'dark' in mood_lower:
            final_prompt += """
        - Use diminished and half-diminished chords
        - Ambiguous tonal centers and chromatic movement
        - Lower register voicings for darkness
        - Sparse chord placement for mystery
        """
        else:  # Neutral/other moods
            final_prompt += """
        - Balanced major/minor harmony appropriate to context
        - Clear tonal center with some harmonic interest
        - Mid-register voicings for clarity
        - Moderate rhythmic activity
        """
        
        final_prompt += f"""
        
        REQUIREMENTS:
        1. Generate chord voicings for the FULL {duration} minutes ({duration * 60} seconds)
        2. Create ONLY chord tones - no melody, bass lines, or arpeggios
        3. Use appropriate voicing ranges as specified above
        4. Vary chord durations and voicings for musical interest
        5. Include strategic rests and chord releases
        6. Use velocity ranges specified above for the genre
        7. Create smooth voice leading between chord changes
        8. ENSURE HARMONIC CONTENT IS DISTINCTLY {genre.upper()} in character
        9. MATCH THE {mood.upper()} emotional character
        
        OUTPUT FORMAT:
        Generate ONLY a valid Python list of lists: [[pitch, start_time, end_time, velocity], ...]
        
        PITCH VALUES: MIDI numbers using the ranges specified above
        TIME VALUES: Precise seconds aligned to musical structure
        VELOCITY VALUES: Use the specified range for {genre} and {mood}
        CHORD VOICINGS: 3-6 simultaneous notes per chord
        
        NO explanations, NO text, ONLY the list.
        """
        
        logging.info(f"[ChordAgent] Generating chords with {len(melody_onsets)} melody onsets, "
                    f"{len(bar_times)} bars, complexity: {harmonic_complexity}")
        
        chord_text = gemini_generate(final_prompt)
        logging.info(f"[ChordAgent] Raw Gemini output preview: {chord_text[:200]}...")
        
        cleaned = clean_llm_output(chord_text)
        notes_array = safe_literal_eval(cleaned)
        notes = notes_array_to_dicts(notes_array)
        
        # Post-process chord notes
        if notes:
            # Sort by start time
            notes.sort(key=lambda n: n['start'])
            
            # Validate and adjust chord notes
            for note in notes:
                # Quantize to musical grid
                note['start'] = round(note['start'] * 4) / 4  # Quarter-note grid
                note['end'] = round(note['end'] * 4) / 4
                
                # Ensure minimum chord duration
                if note['end'] <= note['start']:
                    note['end'] = note['start'] + 1.0  # Minimum 1-second chords
                
                # Validate pitch range for chord voicings
                note['pitch'] = max(36, min(84, note['pitch']))  # C2 to C6
                
                # Adjust velocity for harmonic support role
                note['velocity'] = max(60, min(90, note['velocity']))
            
            # Optionally align to melody onsets for better coordination
            if melody_onsets:
                for note in notes:
                    # Find closest melody onset
                    closest_onset = min(melody_onsets, key=lambda t: abs(t - note['start']))
                    # Only snap if reasonably close (within 0.5 seconds)
                    if abs(note['start'] - closest_onset) <= 0.5:
                        note['start'] = closest_onset
            
            # Check coverage
            max_end = max(n['end'] for n in notes)
            target_duration = duration * 60
            coverage_ratio = max_end / target_duration
            
            logging.info(f"[ChordAgent] Generated {len(notes)} chord notes covering {max_end:.1f}s "
                        f"({coverage_ratio:.1%} of target {target_duration}s)")
            
            if coverage_ratio < 0.8:
                logging.warning(f"[ChordAgent] Chord coverage is low ({coverage_ratio:.1%})")
        else:
            logging.error("[ChordAgent] No valid chord notes generated!")
            notes = []
        
        chord_track = {
            'name': 'Chords',
            'program': 24,  # Electric Piano
            'is_drum': False,
            'notes': notes
        }
        
        state['chords'] = {'chord_track': chord_track}
        logging.info(f"[ChordAgent] Sophisticated chord progression generated with {len(notes)} notes")
        
    except Exception as e:
        logging.error(f"[ChordAgent] Error: {e}")
        # Provide fallback
        state['chords'] = {'chord_track': {'name': 'Chords', 'program': 24, 'is_drum': False, 'notes': []}}
    
    return state 