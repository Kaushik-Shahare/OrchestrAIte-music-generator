import logging
from typing import Any
from utils.gemini_llm import gemini_generate
import ast
import math
import re

def quantize_time(time, step=0.25):
    return round(time / step) * step

def clean_llm_output(text):
    # Remove code block markers and leading/trailing whitespace
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
            logging.error(f"[MelodyAgent] Failed to parse LLM output after auto-fix: {e}")
            return []

def notes_array_to_dicts(notes_array):
    return [
        {'pitch': n[0], 'start': n[1], 'end': n[2], 'velocity': n[3]}
        for n in notes_array if isinstance(n, (list, tuple)) and len(n) == 4
    ]

def melody_agent(state: Any) -> Any:
    """
    Generate sophisticated, context-aware melodies with dynamic section variations.
    """
    logging.info("[MelodyAgent] Generating sophisticated melody with comprehensive context.")
    
    try:
        # Extract comprehensive context
        artist_profile = state.get('artist_profile', {})
        musical_context = state.get('musical_context', {})
        musical_vision = state.get('musical_vision', {})
        arrangement_guidelines = state.get('arrangement_guidelines', {})
        section_arrangement_plan = state.get('section_arrangement_plan', {})
        structured_sections = state.get('structured_sections', [])
        director_summary = state.get('director_summary', '')
        
        # Basic parameters
        genre = musical_context.get('full_genre', state.get('genre', 'pop'))
        mood = musical_context.get('full_mood', state.get('mood', 'happy'))
        tempo = state.get('tempo', 120)
        duration = state.get('duration', 2)
        key_signature = state.get('key_signature', 'C major')
        time_signature = state.get('time_signature', '4/4')
        
        # Complexity and style parameters
        harmonic_complexity = state.get('harmonic_complexity', 'medium')
        rhythmic_complexity = state.get('rhythmic_complexity', 'medium')
        emotional_arc = state.get('emotional_arc', '')
        special_techniques = state.get('special_techniques', '')
        instrument_description = state.get('instrument_description', '')
        
        # Artist-specific characteristics
        melody_characteristics = artist_profile.get('melody_characteristics', 'expressive melodic lines with natural phrasing')
        signature_sounds = artist_profile.get('signature_sounds', '')
        style_summary = artist_profile.get('style_summary', '')
        
        # Musical director's vision
        melodic_character = musical_vision.get('melodic_character', 'memorable, expressive themes')
        emotional_narrative = musical_vision.get('emotional_narrative', 'engaging emotional development')
        dynamic_architecture = musical_vision.get('dynamic_architecture', 'thoughtful dynamic progression')
        
        # Build comprehensive prompt
        context_description = f"""
        Create a {genre} melody with {mood} emotional character following this artistic vision:
        
        MUSICAL DIRECTOR'S GUIDANCE:
        {director_summary}
        
        MELODIC CHARACTER: {melodic_character}
        EMOTIONAL NARRATIVE: {emotional_narrative}
        DYNAMIC ARCHITECTURE: {dynamic_architecture}
        
        MUSICAL PARAMETERS:
        - Key: {key_signature}
        - Time Signature: {time_signature}  
        - Tempo: {tempo} BPM
        - Duration: {duration} minutes
        - Harmonic Complexity: {harmonic_complexity}
        - Rhythmic Complexity: {rhythmic_complexity}
        
        STYLE CONTEXT:
        - Melody Style: {melody_characteristics}
        - Signature Elements: {signature_sounds}
        {f'- Artist Context: {style_summary}' if style_summary else ''}
        {f'- Emotional Journey: {emotional_arc}' if emotional_arc else ''}
        {f'- Special Techniques: {special_techniques}' if special_techniques else ''}
        {f'- Performance Instructions: {instrument_description}' if instrument_description else ''}
        """
        
        # Add section-specific instructions with director's plan
        section_instructions = ""
        if structured_sections and len(structured_sections) > 2:
            section_instructions = "\nCREATE DISTINCT MELODIC CONTENT FOR EACH SECTION FOLLOWING THE DIRECTOR'S PLAN:\n"
            for i, section in enumerate(structured_sections):
                section_start = section['start_time']
                section_end = section['end_time']
                section_name = section['name'].upper()
                
                # Get director's plan for this section if available
                section_plan = section_arrangement_plan.get(section['name'], {})
                energy_level = section_plan.get('energy_level', 5)
                complexity_level = section_plan.get('complexity_level', 'medium')
                emotional_character = section_plan.get('emotional_character', mood)
                
                # Define section characteristics enhanced by director's vision
                if 'verse' in section['name'].lower():
                    characteristics = f"narrative melodic line (energy: {energy_level}/10), {emotional_character}, {complexity_level} complexity"
                elif 'chorus' in section['name'].lower():
                    characteristics = f"memorable hook (energy: {energy_level}/10), {emotional_character}, increased melodic prominence"
                elif 'bridge' in section['name'].lower():
                    characteristics = f"contrasting melodic material (energy: {energy_level}/10), {emotional_character}, unique approach"
                elif 'intro' in section['name'].lower():
                    characteristics = f"establishing theme (energy: {energy_level}/10), building anticipation, {complexity_level} complexity"
                elif 'outro' in section['name'].lower():
                    characteristics = f"concluding material (energy: {energy_level}/10), resolution or fade approach"
                else:
                    characteristics = f"section-appropriate development (energy: {energy_level}/10), {emotional_character}"
                
                section_instructions += f"- {section_name} ({section_start:.1f}s-{section_end:.1f}s): {characteristics}\n"
        
        # Dynamic complexity mapping
        complexity_guidance = {
            'simple': 'Use mostly stepwise motion, simple rhythms, clear phrases',
            'medium': 'Mix stepwise and leap motion, varied rhythms, interesting phrases',
            'complex': 'Include large intervals, syncopation, extended phrases, melodic ornamentation'
        }
        
        rhythm_guidance = {
            'simple': 'Mostly quarter and eighth notes, minimal syncopation',
            'medium': 'Mix of note values, moderate syncopation, some rests',
            'complex': 'Complex rhythmic patterns, polyrhythms, advanced syncopation'
        }
        
        final_prompt = f"""
        {context_description}
        
        {section_instructions}
        
        COMPLEXITY GUIDELINES:
        - Melodic: {complexity_guidance.get(harmonic_complexity, complexity_guidance['medium'])}
        - Rhythmic: {rhythm_guidance.get(rhythmic_complexity, rhythm_guidance['medium'])}
        
        GENRE-SPECIFIC MELODIC REQUIREMENTS FOR {genre.upper()}:
        """
        
        # Add genre-specific melodic guidance
        genre_lower = genre.lower()
        if 'metal' in genre_lower or 'rock' in genre_lower:
            final_prompt += """
        - Use aggressive melodic intervals and power-chord friendly notes
        - Include fast scalar runs and chromatic passing tones
        - Emphasize strong downbeats and syncopated rhythms
        - Use palm-muted sections and sustained power notes
        - Include dramatic octave leaps and aggressive bends
        - Pitch range: 40-80 (low E to high E on guitar)
        """
        elif 'jazz' in genre_lower:
            final_prompt += """
        - Use complex chord tones, extensions (9ths, 11ths, 13ths)
        - Include blue notes, chromatic approach tones
        - Swing rhythm patterns with triplet subdivisions
        - Sophisticated harmonic movement and voice leading
        - Pitch range: 50-85 (jazz piano/horn range)
        """
        elif 'classical' in genre_lower:
            final_prompt += """
        - Use stepwise motion with tasteful leaps
        - Classical phrase structures with clear cadences
        - Ornamentations like trills, turns, grace notes
        - Balanced melodic contour with arch-like phrases
        - Pitch range: 48-84 (classical instrumental range)
        """
        elif 'electronic' in genre_lower or 'edm' in genre_lower:
            final_prompt += """
        - Use synthesizer-friendly intervals and patterns
        - Include filter sweeps, arpeggiated sequences
        - Rhythmic patterns aligned to electronic beats
        - Repetitive but evolving melodic phrases
        - Pitch range: 36-96 (wide electronic range)
        """
        elif 'blues' in genre_lower:
            final_prompt += """
        - Heavily use blue notes (♭3, ♭5, ♭7)
        - Call-and-response melodic phrases
        - Bending and sliding between pitches
        - 12-bar blues structure awareness
        - Pitch range: 40-75 (blues guitar/vocal range)
        """
        elif 'folk' in genre_lower or 'country' in genre_lower:
            final_prompt += """
        - Simple, singable melodic lines
        - Pentatonic and modal scales
        - Clear phrase structure with repetition
        - Natural speech rhythm patterns
        - Pitch range: 50-77 (vocal-friendly range)
        """
        else:  # Pop/other
            final_prompt += """
        - Catchy, memorable melodic hooks
        - Mix of stepwise motion and moderate leaps
        - Clear phrase structure with repetition and variation
        - Accessible melodic content for broad appeal
        - Pitch range: 48-81 (pop vocal/instrumental range)
        """
        
        final_prompt += f"""
        
        TEMPO-SPECIFIC RHYTHMIC REQUIREMENTS FOR {tempo} BPM:
        """
        
        # Add tempo-specific rhythmic guidance
        if tempo < 80:
            final_prompt += """
        - Use longer note values (half notes, whole notes)
        - Include expressive rubato and ritardando
        - Fewer but more meaningful melodic events
        - Allow for breath and space between phrases
        """
        elif tempo < 120:
            final_prompt += """
        - Balanced mix of quarter and eighth notes
        - Moderate rhythmic activity with clear pulse
        - Natural breathing points in phrases
        - Syncopation for interest but not overwhelming
        """
        elif tempo < 160:
            final_prompt += """
        - Emphasis on eighth and sixteenth note patterns
        - Active rhythmic movement with driving pulse
        - Quick melodic passages and energetic phrases
        - Strategic use of rests for rhythmic punch
        """
        else:  # Very fast tempo
            final_prompt += """
        - Rapid sixteenth note passages and scalar runs
        - Aggressive rhythmic patterns with strong accents
        - Short, punchy melodic cells repeated and developed
        - Minimal rests - continuous melodic energy
        """
        
        final_prompt += f"""
        
        REQUIREMENTS:
        1. Create melodic content that spans the FULL {duration} minutes ({duration * 60} seconds)
        2. Include rests and phrase breaks for musical breathing
        3. Vary velocity (40-120) to create dynamic expression that matches {mood} mood
        4. Use appropriate pitch range for the style and instruments
        5. Include section transitions and connecting phrases
        6. Add fills, runs, or ornaments where stylistically appropriate
        7. ENSURE MELODIC CONTENT IS DISTINCTLY {genre.upper()} in character
        8. CREATE DIFFERENT MELODIC MATERIAL FOR EACH SECTION
        
        Output ONLY a valid Python list of lists: [[pitch, start_time, end_time, velocity], ...]
        
        PITCH VALUES: MIDI numbers (C4=60, C5=72, etc.) - USE THE SPECIFIED RANGE ABOVE
        TIME VALUES: Precise seconds (e.g., 0.0, 0.5, 1.25, 2.0)
        VELOCITY VALUES: 40-120 for dynamic expression appropriate to {mood} mood
        
        NO explanations, NO text, ONLY the list.
        """
        
        logging.info(f"[MelodyAgent] Generating melody with prompt length: {len(final_prompt)} characters")
        melody_text = gemini_generate(final_prompt)
        logging.info(f"[MelodyAgent] Raw Gemini output preview: {melody_text[:200]}...")
        
        cleaned = clean_llm_output(melody_text)
        notes_array = safe_literal_eval(cleaned)
        notes = notes_array_to_dicts(notes_array)
        
        # Post-process and validate notes
        if notes:
            # Sort by start time
            notes.sort(key=lambda n: n['start'])
            
            # Quantize note timings
            for note in notes:
                note['start'] = quantize_time(note['start'])
                note['end'] = quantize_time(note['end'])
                
                # Ensure minimum note duration
                if note['end'] <= note['start']:
                    note['end'] = note['start'] + 0.25
                    
                # Validate velocity range
                note['velocity'] = max(40, min(120, note['velocity']))
            
            # Check coverage
            max_end = max(n['end'] for n in notes)
            target_duration = duration * 60
            coverage_ratio = max_end / target_duration
            
            logging.info(f"[MelodyAgent] Generated {len(notes)} notes covering {max_end:.1f}s "
                        f"({coverage_ratio:.1%} of target {target_duration}s)")
            
            if coverage_ratio < 0.8:
                logging.warning(f"[MelodyAgent] Melody coverage is low ({coverage_ratio:.1%})")
        else:
            logging.error("[MelodyAgent] No valid notes generated!")
            notes = []
        
        # Calculate structural information
        tempo = state.get('tempo', 120)
        beats_per_bar = 4 if time_signature.startswith('4') else 3
        seconds_per_beat = 60.0 / tempo
        total_beats = int((duration * 60) / seconds_per_beat)
        bar_times = [quantize_time(i * beats_per_bar * seconds_per_beat) 
                    for i in range(math.ceil(total_beats / beats_per_bar))]
        
        note_onsets = sorted(set([quantize_time(n['start']) for n in notes])) if notes else []
        
        melody_track = {
            'name': 'Melody',
            'program': 0,  # Piano by default
            'is_drum': False,
            'notes': notes
        }
        
        state['melody'] = {'melody_track': melody_track}
        state['structure'] = {
            'melody_onsets': note_onsets,
            'bar_times': bar_times,
            'tempo': tempo,
            'duration': duration,
            'beats_per_bar': beats_per_bar,
            'time_signature': time_signature,
            'key_signature': key_signature
        }
        
        logging.info(f"[MelodyAgent] Sophisticated melody generated with {len(notes)} notes, "
                    f"{len(note_onsets)} unique onsets, {len(bar_times)} bars")
        
    except Exception as e:
        logging.error(f"[MelodyAgent] Error: {e}")
        # Provide fallback empty state
        state['melody'] = {'melody_track': {'name': 'Melody', 'program': 0, 'is_drum': False, 'notes': []}}
        state['structure'] = {'melody_onsets': [], 'bar_times': [], 'tempo': 120, 'duration': 2}
    
    return state 