import logging
from typing import Any
from utils.gemini_llm import gemini_generate
from utils.state_utils import validate_agent_return, safe_state_update
import ast
import re
import json

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
        
        # Get RAG patterns if available
        rag_patterns = state.get('rag_patterns', {})
        rag_progressions = rag_patterns.get('progressions', [])
        rag_segments = rag_patterns.get('segments', [])
        rag_instructions = state.get('rag_instructions', '')
        
        logging.info(f"[ChordAgent] Found {len(rag_progressions)} RAG chord progression patterns and {len(rag_segments)} segments")
        
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
        
        # Create RAG-based chord patterns guidance
        rag_pattern_instructions = ""
        
        # Extract chord pattern data from RAG
        if rag_progressions or rag_segments:
            rag_pattern_instructions = "\n\nIMPORTANT: USE THESE RETRIEVED CHORD PATTERNS FROM REAL MUSIC:\n"
            
            # Process chord progressions
            if rag_progressions:
                rag_pattern_instructions += "\nCHORD PROGRESSIONS TO USE DIRECTLY:\n"
                for i, progression in enumerate(rag_progressions[:3]):  # Top 3 progressions
                    pattern_data = progression.get('pattern_data', {})
                    if isinstance(pattern_data, str):
                        try:
                            pattern_data = json.loads(pattern_data)
                        except:
                            pattern_data = {}
                    
                    # Get actual chord data
                    chords = pattern_data.get('chords', [])
                    source = progression.get('source_file', 'unknown')
                    
                    if chords and len(chords) > 0:
                        rag_pattern_instructions += f"\nProgression {i+1} (Source: {source}):\n"
                        
                        # Format chord data for clear instructions
                        chord_descriptions = []
                        for j, chord in enumerate(chords[:8]): # First 8 chords
                            root = chord.get('root', 0)
                            pitches = chord.get('pitches', [])
                            duration = chord.get('duration', 1.0)
                            
                            chord_descriptions.append(f"Chord {j+1}: Root={root}, Pitches={pitches}, Duration={duration:.1f}s")
                        
                        rag_pattern_instructions += "\n".join(chord_descriptions) + "\n"
                        rag_pattern_instructions += "APPLY THIS CHORD PROGRESSION DIRECTLY!\n"
            
            # Extract chord data from segments
            if rag_segments:
                extracted_chords = []
                
                for segment in rag_segments:
                    pattern_data = segment.get('pattern_data', {})
                    if isinstance(pattern_data, str):
                        try:
                            pattern_data = json.loads(pattern_data)
                        except:
                            pattern_data = {}
                    
                    # Look for chord data in the segment
                    if 'chord_sequence' in pattern_data:
                        chord_sequence = pattern_data['chord_sequence']
                        source = segment.get('source_file', 'unknown')
                        
                        if chord_sequence and len(chord_sequence) > 0:
                            extracted_chords.append({
                                'source': source,
                                'chords': chord_sequence[:8] # First 8 chords
                            })
                
                # If we found chord data in segments, add it to instructions
                if extracted_chords:
                    rag_pattern_instructions += "\nCHORDS FROM MUSICAL SEGMENTS:\n"
                    
                    for i, chord_data in enumerate(extracted_chords[:2]): # Top 2 segments with chords
                        rag_pattern_instructions += f"\nSegment Chords {i+1} (Source: {chord_data['source']}):\n"
                        
                        # Format chord data
                        chord_descriptions = []
                        for j, chord in enumerate(chord_data['chords']):
                            chord_descriptions.append(f"Chord {j+1}: {chord}")
                        
                        rag_pattern_instructions += "\n".join(chord_descriptions) + "\n"
                        rag_pattern_instructions += "INCORPORATE THESE CHORD PATTERNS!\n"
        
        final_prompt = f"""
        {context_description}
        
        {section_instructions}
        
        HARMONIC APPROACH:
        {harmonic_instruction}
        
        TIMING COORDINATION:
        {timing_strategy}
        
        {rag_pattern_instructions}
        
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
        
        # Create RAG-based chord patterns guidance
        rag_pattern_instructions = ""
        
        # Extract chord pattern data from RAG
        if rag_progressions or rag_segments:
            rag_pattern_instructions = "\n\nIMPORTANT: USE THESE RETRIEVED CHORD PATTERNS FROM REAL MUSIC:\n"
            
            # Process chord progressions
            if rag_progressions:
                rag_pattern_instructions += "\nCHORD PROGRESSIONS TO USE DIRECTLY:\n"
                for i, progression in enumerate(rag_progressions[:3]):  # Top 3 progressions
                    pattern_data = progression.get('pattern_data', {})
                    if isinstance(pattern_data, str):
                        try:
                            pattern_data = json.loads(pattern_data)
                        except:
                            pattern_data = {}
                    
                    # Get actual chord data
                    chords = pattern_data.get('chords', [])
                    source = progression.get('source_file', 'unknown')
                    
                    if chords and len(chords) > 0:
                        rag_pattern_instructions += f"\nProgression {i+1} (Source: {source}):\n"
                        
                        # Format chord data for clear instructions
                        chord_descriptions = []
                        for j, chord in enumerate(chords[:8]): # First 8 chords
                            root = chord.get('root', 0)
                            pitches = chord.get('pitches', [])
                            duration = chord.get('duration', 1.0)
                            
                            chord_descriptions.append(f"Chord {j+1}: Root={root}, Pitches={pitches}, Duration={duration:.1f}s")
                        
                        rag_pattern_instructions += "\n".join(chord_descriptions) + "\n"
                        rag_pattern_instructions += "APPLY THIS CHORD PROGRESSION DIRECTLY!\n"
            
            # Extract chord data from segments
            if rag_segments:
                extracted_chords = []
                
                for segment in rag_segments:
                    pattern_data = segment.get('pattern_data', {})
                    if isinstance(pattern_data, str):
                        try:
                            pattern_data = json.loads(pattern_data)
                        except:
                            pattern_data = {}
                    
                    # Look for chord data in the segment
                    if 'chord_sequence' in pattern_data:
                        chord_sequence = pattern_data['chord_sequence']
                        source = segment.get('source_file', 'unknown')
                        
                        if chord_sequence and len(chord_sequence) > 0:
                            extracted_chords.append({
                                'source': source,
                                'chords': chord_sequence[:8] # First 8 chords
                            })
                
                # If we found chord data in segments, add it to instructions
                if extracted_chords:
                    rag_pattern_instructions += "\nCHORDS FROM MUSICAL SEGMENTS:\n"
                    
                    for i, chord_data in enumerate(extracted_chords[:2]): # Top 2 segments with chords
                        rag_pattern_instructions += f"\nSegment Chords {i+1} (Source: {chord_data['source']}):\n"
                        
                        # Format chord data
                        chord_descriptions = []
                        for j, chord in enumerate(chord_data['chords']):
                            chord_descriptions.append(f"Chord {j+1}: {chord}")
                        
                        rag_pattern_instructions += "\n".join(chord_descriptions) + "\n"
                        rag_pattern_instructions += "INCORPORATE THESE CHORD PATTERNS!\n"
        
        final_prompt += rag_pattern_instructions
        
        logging.info(f"[ChordAgent] Generating chords with {len(melody_onsets)} melody onsets, "
                    f"{len(bar_times)} bars, complexity: {harmonic_complexity}")
        
        # Extract actual chord data from RAG patterns
        rag_chord_data = []
        chord_sequences = []
        
        # Process all available chord progressions from RAG
        for progression in rag_progressions:
            pattern_data = progression.get('pattern_data', {})
            if isinstance(pattern_data, str):
                try:
                    pattern_data = json.loads(pattern_data)
                except:
                    pattern_data = {}
            
            # Extract chord sequence
            chords = pattern_data.get('chords', [])
            if chords and len(chords) > 2:  # Only use if we have more than 2 chords
                chord_sequences.append({
                    'chords': chords,
                    'source': progression.get('source_file', 'unknown'),
                    'type': 'progression'
                })
                logging.info(f"[ChordAgent] Found usable chord progression with {len(chords)} chords")
        
        # Also check segments for chord sequences
        for segment in rag_segments:
            pattern_data = segment.get('pattern_data', {})
            if isinstance(pattern_data, str):
                try:
                    pattern_data = json.loads(pattern_data)
                except:
                    pattern_data = {}
            
            # Check for chord_sequence in segments
            if 'chord_sequence' in pattern_data:
                chord_seq = pattern_data['chord_sequence']
                if chord_seq and len(chord_seq) > 2:
                    chord_sequences.append({
                        'chords': chord_seq,
                        'source': segment.get('source_file', 'unknown'),
                        'type': 'segment'
                    })
                    logging.info(f"[ChordAgent] Found chord sequence in segment with {len(chord_seq)} chords")
        
        # Try to use the RAG chord data directly
        notes = []
        used_rag_directly = False
        
        if chord_sequences:
            try:
                logging.info(f"[ChordAgent] Using RAG chord progressions directly")
                
                # Merge sequences to create enough material for the duration
                all_chords = []
                for seq in chord_sequences:
                    all_chords.extend(seq['chords'])
                
                # If we don't have enough, repeat the patterns
                target_chord_count = (duration * 60) / 4  # Roughly 1 chord per 4 seconds
                while len(all_chords) < target_chord_count:
                    all_chords.extend(all_chords[:int(target_chord_count - len(all_chords))])
                
                # Convert chord data to notes
                current_time = 0.0
                for chord_data in all_chords:
                    # Different formats may exist
                    if isinstance(chord_data, dict):
                        # Format: {'root': 60, 'pitches': [60, 64, 67], 'duration': 2.0}
                        pitches = chord_data.get('pitches', [])
                        duration = chord_data.get('duration', 2.0)
                        
                        # If no pitches but we have root and chord type
                        if not pitches and 'root' in chord_data:
                            root = chord_data['root']
                            chord_type = chord_data.get('type', 'major')
                            
                            # Generate basic triads based on chord type
                            if chord_type == 'major':
                                pitches = [root, root + 4, root + 7]  # Major triad
                            elif chord_type == 'minor':
                                pitches = [root, root + 3, root + 7]  # Minor triad
                            elif chord_type == '7':
                                pitches = [root, root + 4, root + 7, root + 10]  # Dominant 7th
                            else:
                                pitches = [root, root + 4, root + 7]  # Default to major
                    else:
                        # Maybe it's a string like "Cmaj" or a list of pitches
                        pitches = [60, 64, 67]  # Default C major
                        duration = 2.0
                    
                    # Add each pitch as a note
                    for pitch in pitches:
                        if 36 <= pitch <= 84:  # Reasonable chord range
                            notes.append({
                                'pitch': pitch,
                                'start': current_time,
                                'end': current_time + duration,
                                'velocity': 75  # Medium velocity for chords
                            })
                    
                    current_time += duration
                
                # Adjust to fit the requested duration
                if notes:
                    target_duration = duration * 60  # in seconds
                    current_duration = notes[-1]['end']
                    
                    # Scale if needed
                    if current_duration > target_duration:
                        # Trim excess
                        notes = [n for n in notes if n['start'] < target_duration]
                    elif current_duration < target_duration:
                        # Scale to fit
                        scale_factor = target_duration / current_duration
                        for note in notes:
                            note['start'] *= scale_factor
                            note['end'] *= scale_factor
                
                if len(notes) > 10:  # If we have enough chord notes
                    used_rag_directly = True
                    logging.info(f"[ChordAgent] Successfully created {len(notes)} chord notes using RAG data directly")
            
            except Exception as e:
                logging.error(f"[ChordAgent] Error using RAG chord data directly: {e}")
                used_rag_directly = False
        
        # Fallback to LLM generation if needed
        if not used_rag_directly:
            logging.info(f"[ChordAgent] Falling back to LLM generation with RAG guidance")
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
        
        # Use safe state update instead of direct modification
        chord_data = {'chord_track': chord_track}
        logging.info(f"[ChordAgent] Sophisticated chord progression generated with {len(notes)} notes")
        return safe_state_update(state, {'chords': chord_data}, "ChordAgent")
        
    except Exception as e:
        logging.error(f"[ChordAgent] Error: {e}")
        # Provide fallback using safe state update
        fallback_chords = {'chord_track': {'name': 'Chords', 'program': 24, 'is_drum': False, 'notes': []}}
        return safe_state_update(state, {'chords': fallback_chords}, "ChordAgent")