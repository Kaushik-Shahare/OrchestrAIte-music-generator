#!/usr/bin/env python3
"""
Pattern Application Utility
Applies extracted MIDI patterns directly to music generation.
"""

import logging
import random
from typing import Dict, List, Any, Tuple
import pretty_midi

class PatternApplicator:
    """
    Applies real MIDI patterns to new music generation.
    """
    
    def __init__(self):
        self.base_tempo = 120
        self.base_time_signature = (4, 4)
    
    def apply_melody_pattern(self, pattern: Dict, start_time: float = 0.0, 
                           duration: float = 8.0, key_offset: int = 0) -> List[pretty_midi.Note]:
        """
        Apply a melodic pattern to generate actual MIDI notes.
        """
        notes = []
        
        try:
            intervals = pattern.get('intervals', [])
            start_pitch = pattern.get('start_pitch', 60) + key_offset
            
            if not intervals:
                return notes
            
            current_time = start_time
            current_pitch = start_pitch
            note_duration = 0.5  # Default half beat
            
            # Generate notes following the interval pattern
            for i, interval in enumerate(intervals):
                if current_time >= start_time + duration:
                    break
                
                # Apply interval
                current_pitch += interval
                
                # Keep pitch in reasonable range
                current_pitch = max(24, min(108, current_pitch))
                
                # Create note
                note = pretty_midi.Note(
                    velocity=random.randint(70, 100),
                    pitch=current_pitch,
                    start=current_time,
                    end=current_time + note_duration
                )
                notes.append(note)
                
                # Advance time
                current_time += note_duration
                
                # Vary note duration slightly for musicality
                note_duration = random.choice([0.25, 0.5, 0.75, 1.0])
            
            logging.info(f"[PatternApplicator] Generated {len(notes)} melody notes from pattern")
            return notes
            
        except Exception as e:
            logging.error(f"[PatternApplicator] Error applying melody pattern: {e}")
            return []
    
    def apply_chord_progression_pattern(self, pattern: Dict, start_time: float = 0.0,
                                      duration: float = 16.0, key_offset: int = 0) -> List[pretty_midi.Note]:
        """
        Apply a chord progression pattern to generate chord notes.
        """
        notes = []
        
        try:
            chords = pattern.get('chords', [])
            
            if not chords:
                return notes
            
            current_time = start_time
            
            # Repeat chord progression to fill duration
            while current_time < start_time + duration:
                for chord_info in chords:
                    if current_time >= start_time + duration:
                        break
                    
                    chord_duration = chord_info.get('duration', 2.0)
                    pitches = chord_info.get('pitches', [0, 4, 7])
                    base_octave = 48  # C3
                    
                    # Create chord notes
                    for i, pitch_class in enumerate(pitches):
                        pitch = base_octave + pitch_class + key_offset
                        if i > 0:  # Stack higher notes in higher octaves
                            pitch += 12 * (i // 3)
                        
                        # Keep in reasonable range
                        pitch = max(24, min(84, pitch))
                        
                        note = pretty_midi.Note(
                            velocity=random.randint(60, 85),
                            pitch=pitch,
                            start=current_time,
                            end=current_time + chord_duration
                        )
                        notes.append(note)
                    
                    current_time += chord_duration
            
            logging.info(f"[PatternApplicator] Generated {len(notes)} chord notes from pattern")
            return notes
            
        except Exception as e:
            logging.error(f"[PatternApplicator] Error applying chord pattern: {e}")
            return []
    
    def apply_rhythm_pattern(self, pattern: Dict, start_time: float = 0.0,
                           duration: float = 8.0, base_pitch: int = 36) -> List[pretty_midi.Note]:
        """
        Apply a rhythm pattern to generate rhythmic notes (bass/drums).
        """
        notes = []
        
        try:
            intervals = pattern.get('intervals', [])
            density = pattern.get('density', 2.0)
            
            if not intervals:
                return notes
            
            current_time = start_time
            
            # Apply rhythm pattern
            while current_time < start_time + duration:
                for interval in intervals:
                    if current_time >= start_time + duration:
                        break
                    
                    # Create rhythmic note
                    note_duration = min(interval, 0.5)  # Short punchy notes
                    velocity = random.randint(80, 110)
                    
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=base_pitch + random.randint(-2, 2),  # Slight pitch variation
                        start=current_time,
                        end=current_time + note_duration
                    )
                    notes.append(note)
                    
                    current_time += interval
            
            logging.info(f"[PatternApplicator] Generated {len(notes)} rhythm notes from pattern")
            return notes
            
        except Exception as e:
            logging.error(f"[PatternApplicator] Error applying rhythm pattern: {e}")
            return []
    
    def apply_musical_segment(self, segment: Dict, start_time: float = 0.0,
                            key_offset: int = 0) -> Dict[str, List[pretty_midi.Note]]:
        """
        Apply a complete musical segment (most valuable for authentic generation).
        Returns notes organized by instrument.
        """
        segment_notes = {}
        
        try:
            instruments = segment.get('instruments', [])
            segment_duration = segment.get('duration', 16.0)
            
            for inst_info in instruments:
                inst_name = inst_info.get('name', 'Piano')
                note_sequence = inst_info.get('notes', [])
                
                if not note_sequence:
                    continue
                
                inst_notes = []
                
                # Apply the actual note sequence from the segment
                for note_data in note_sequence:
                    if len(note_data) >= 4:  # pitch, start, duration, velocity
                        pitch, rel_start, note_duration, velocity = note_data[:4]
                        
                        # Apply key offset
                        adjusted_pitch = pitch + key_offset
                        adjusted_pitch = max(24, min(108, adjusted_pitch))
                        
                        # Adjust timing
                        note_start = start_time + rel_start
                        note_end = note_start + note_duration
                        
                        note = pretty_midi.Note(
                            velocity=int(velocity),
                            pitch=int(adjusted_pitch),
                            start=note_start,
                            end=note_end
                        )
                        inst_notes.append(note)
                
                if inst_notes:
                    segment_notes[inst_name] = inst_notes
                    logging.info(f"[PatternApplicator] Applied {len(inst_notes)} notes for {inst_name}")
            
            return segment_notes
            
        except Exception as e:
            logging.error(f"[PatternApplicator] Error applying musical segment: {e}")
            return {}
    
    def blend_patterns_with_generated(self, generated_notes: List[pretty_midi.Note],
                                    pattern_notes: List[pretty_midi.Note],
                                    blend_ratio: float = 0.3) -> List[pretty_midi.Note]:
        """
        Blend pattern-derived notes with LLM-generated notes.
        """
        try:
            # Sort both by start time
            generated_notes.sort(key=lambda n: n.start)
            pattern_notes.sort(key=lambda n: n.start)
            
            blended_notes = []
            
            # Take pattern notes for first part
            pattern_duration = len(pattern_notes) * blend_ratio
            for note in pattern_notes:
                if note.start <= pattern_duration:
                    blended_notes.append(note)
            
            # Take generated notes for remaining part, but offset timing
            time_offset = pattern_duration if pattern_notes else 0
            for note in generated_notes:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start + time_offset,
                    end=note.end + time_offset
                )
                blended_notes.append(new_note)
            
            # Occasionally interleave pattern elements throughout
            for i, note in enumerate(pattern_notes):
                if i % 4 == 0 and random.random() < 0.2:  # 20% chance every 4th note
                    scattered_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start + time_offset + random.uniform(0, 8),
                        end=note.end + time_offset + random.uniform(0, 8)
                    )
                    blended_notes.append(scattered_note)
            
            return blended_notes
            
        except Exception as e:
            logging.error(f"[PatternApplicator] Error blending patterns: {e}")
            return generated_notes

# Global instance
pattern_applicator = PatternApplicator()

def apply_patterns_to_generation(patterns: Dict, generated_midi: pretty_midi.PrettyMIDI,
                                blend_ratio: float = 0.4) -> pretty_midi.PrettyMIDI:
    """
    Apply extracted patterns to enhance generated MIDI.
    """
    try:
        if not patterns or not patterns.get('patterns'):
            return generated_midi
        
        pattern_data = patterns['patterns']
        
        # Apply segments first (highest priority)
        if pattern_data.get('segments'):
            segment = pattern_data['segments'][0]  # Use first segment
            segment_notes = pattern_applicator.apply_musical_segment(segment, 0.0, 0)
            
            # Add segment notes to appropriate instruments
            for inst_name, notes in segment_notes.items():
                if notes:
                    # Find or create appropriate instrument
                    target_inst = None
                    for inst in generated_midi.instruments:
                        if inst_name.lower() in pretty_midi.program_to_instrument_name(inst.program).lower():
                            target_inst = inst
                            break
                    
                    if not target_inst and generated_midi.instruments:
                        target_inst = generated_midi.instruments[0]  # Use first available
                    
                    if target_inst:
                        # Blend with existing notes
                        target_inst.notes = pattern_applicator.blend_patterns_with_generated(
                            target_inst.notes, notes, blend_ratio
                        )
        
        # Apply melodies
        if pattern_data.get('melodies'):
            melody_pattern = pattern_data['melodies'][0]
            melody_notes = pattern_applicator.apply_melody_pattern(melody_pattern, 16.0, 8.0, 0)
            
            if melody_notes and generated_midi.instruments:
                # Add to lead instrument
                lead_inst = generated_midi.instruments[0]
                lead_inst.notes.extend(melody_notes)
        
        # Apply chord progressions
        if pattern_data.get('chord_progressions'):
            chord_pattern = pattern_data['chord_progressions'][0]
            chord_notes = pattern_applicator.apply_chord_progression_pattern(chord_pattern, 24.0, 16.0, 0)
            
            if chord_notes and len(generated_midi.instruments) > 1:
                # Add to harmonic instrument
                harm_inst = generated_midi.instruments[1]
                harm_inst.notes.extend(chord_notes)
        
        logging.info(f"[PatternApplicator] Applied patterns from {patterns.get('source', 'unknown')} source")
        return generated_midi
        
    except Exception as e:
        logging.error(f"[PatternApplicator] Error applying patterns to generation: {e}")
        return generated_midi
