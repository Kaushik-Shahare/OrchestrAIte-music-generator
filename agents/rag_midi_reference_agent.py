#!/usr/bin/env python3
"""
RAG-Enhanced MIDI Reference Agent
Uses vector database to retrieve similar musical patterns and provide them as concrete examples
"""

import logging
from typing import Any, Dict, List
from utils.gemini_llm import gemini_generate
from utils.midi_rag_system import midi_rag, initialize_midi_rag_database
import json
import random

def generate_fallback_patterns(genre: str, instruments: List[str]) -> Dict:
    """Generate fallback patterns when no MIDI files are found in the dataset
    
    This ensures we always have some patterns to work with even if the RAG system fails
    """
    logging.info(f"[RAGMidiReferenceAgent] Generating fallback patterns for genre: {genre}")
    
    # Basic patterns by genre
    genre_patterns = {
        'classical': {
            'piano': {
                'notes': [[60, 0.0, 0.5, 80], [64, 0.5, 1.0, 75], [67, 1.0, 1.5, 85], [72, 1.5, 2.0, 80]],
                'intervals': [4, 3, 5, -5, -3, -4],
                'rhythm': [0.5, 0.5, 0.5, 0.5, 0.75, 0.25]
            },
            'violin': {
                'notes': [[76, 0.0, 1.0, 75], [79, 1.0, 2.0, 80], [77, 2.0, 3.0, 75]],
                'intervals': [3, -2, 4, -3, 2, -4],
                'rhythm': [1.0, 1.0, 1.0, 0.5, 0.5, 1.0]
            },
            'chords': [
                {'root': 60, 'pitches': [60, 64, 67], 'duration': 2.0},
                {'root': 67, 'pitches': [67, 71, 74], 'duration': 2.0},
                {'root': 65, 'pitches': [65, 69, 72], 'duration': 2.0},
                {'root': 62, 'pitches': [62, 65, 69], 'duration': 2.0}
            ]
        },
        'jazz': {
            'piano': {
                'notes': [[60, 0.0, 0.5, 70], [63, 0.5, 1.0, 75], [67, 1.0, 1.5, 70], [70, 1.5, 2.0, 80]],
                'intervals': [3, 4, 3, -3, -4, -3],
                'rhythm': [0.5, 0.5, 0.5, 0.5, 0.75, 0.25]
            },
            'bass': {
                'notes': [[43, 0.0, 0.5, 90], [45, 1.0, 1.5, 85], [48, 2.0, 2.5, 90], [50, 3.0, 3.5, 85]],
                'intervals': [2, 3, 2, -2, -3, -2],
                'rhythm': [0.5, 0.5, 0.5, 0.5, 1.0, 0.5]
            },
            'chords': [
                {'root': 60, 'pitches': [60, 64, 67, 71], 'duration': 2.0},
                {'root': 67, 'pitches': [67, 71, 74, 77], 'duration': 2.0},
                {'root': 65, 'pitches': [65, 69, 72, 76], 'duration': 2.0},
                {'root': 62, 'pitches': [62, 65, 69, 72], 'duration': 2.0}
            ]
        },
        'rock': {
            'guitar': {
                'notes': [[40, 0.0, 0.5, 100], [40, 0.5, 1.0, 95], [47, 1.0, 1.5, 100], [47, 1.5, 2.0, 95]],
                'intervals': [0, 7, 0, -7, 5, -5],
                'rhythm': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5]
            },
            'bass': {
                'notes': [[28, 0.0, 1.0, 95], [28, 1.0, 2.0, 90], [35, 2.0, 3.0, 95], [35, 3.0, 4.0, 90]],
                'intervals': [0, 7, 0, -7, 5, -5],
                'rhythm': [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
            },
            'chords': [
                {'root': 40, 'pitches': [40, 47, 52], 'duration': 2.0},
                {'root': 45, 'pitches': [45, 52, 57], 'duration': 2.0},
                {'root': 47, 'pitches': [47, 54, 59], 'duration': 2.0},
                {'root': 38, 'pitches': [38, 45, 50], 'duration': 2.0}
            ]
        },
        'metal': {
            'guitar': {
                'notes': [[40, 0.0, 0.2, 120], [40, 0.2, 0.4, 115], [47, 0.4, 0.6, 120], [47, 0.6, 0.8, 115]],
                'intervals': [0, 7, 0, -7, 5, -5],
                'rhythm': [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
            },
            'bass': {
                'notes': [[28, 0.0, 0.4, 110], [28, 0.4, 0.8, 105], [35, 0.8, 1.2, 110], [35, 1.2, 1.6, 105]],
                'intervals': [0, 7, 0, -7, 5, -5],
                'rhythm': [0.4, 0.4, 0.4, 0.4, 0.2, 0.2]
            },
            'chords': [
                {'root': 40, 'pitches': [40, 47], 'duration': 0.8},
                {'root': 45, 'pitches': [45, 52], 'duration': 0.8},
                {'root': 47, 'pitches': [47, 54], 'duration': 0.8},
                {'root': 38, 'pitches': [38, 45], 'duration': 0.8}
            ]
        },
        'pop': {
            'piano': {
                'notes': [[60, 0.0, 0.5, 80], [64, 0.5, 1.0, 85], [67, 1.0, 1.5, 80], [64, 1.5, 2.0, 85]],
                'intervals': [4, 3, -3, 3, -3, -4],
                'rhythm': [0.5, 0.5, 0.5, 0.5, 0.25, 0.25]
            },
            'guitar': {
                'notes': [[52, 0.0, 0.5, 85], [52, 0.5, 1.0, 80], [59, 1.0, 1.5, 85], [59, 1.5, 2.0, 80]],
                'intervals': [0, 7, 0, -7, 4, -4],
                'rhythm': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5]
            },
            'chords': [
                {'root': 60, 'pitches': [60, 64, 67], 'duration': 1.0},
                {'root': 65, 'pitches': [65, 69, 72], 'duration': 1.0},
                {'root': 67, 'pitches': [67, 71, 74], 'duration': 1.0},
                {'root': 62, 'pitches': [62, 65, 69], 'duration': 1.0}
            ]
        }
    }
    
    # Default to pop if genre not found in our fallback patterns
    if genre.lower() not in genre_patterns:
        fallback_genre = 'pop'
        logging.info(f"[RAGMidiReferenceAgent] No fallback patterns for genre '{genre}', using '{fallback_genre}'")
        genre_lower = fallback_genre
    else:
        genre_lower = genre.lower()
    
    # Create a segments entry with pattern data
    instruments_data = []
    for instrument in instruments:
        # Map instrument to closest matching one in our patterns
        inst_key = 'piano'  # default
        if 'guitar' in instrument.lower():
            inst_key = 'guitar'
        elif 'bass' in instrument.lower():
            inst_key = 'bass'
        elif 'violin' in instrument.lower() or 'string' in instrument.lower():
            inst_key = 'violin' if 'violin' in genre_patterns[genre_lower] else 'piano'
        
        # Use fallback if instrument not found
        if inst_key not in genre_patterns[genre_lower]:
            inst_key = list(genre_patterns[genre_lower].keys())[0]
        
        # Create instrument pattern
        pattern = genre_patterns[genre_lower][inst_key]
        instruments_data.append({
            'name': instrument,
            'note_sequence': pattern['notes'],
            'melodic_intervals': pattern['intervals'],
            'rhythm_pattern': pattern['rhythm']
        })
    
    # Create the basic fallback patterns
    fallback_patterns = {
        'segments': [{
            'description': f"Fallback {genre} pattern",
            'similarity_score': 0.8,
            'genre': genre,
            'pattern_data': {
                'instruments': instruments_data,
                'duration': 4.0
            }
        }],
        'progressions': [{
            'description': f"Fallback {genre} chord progression",
            'similarity_score': 0.8,
            'genre': genre,
            'pattern_data': {
                'chords': genre_patterns[genre_lower]['chords']
            }
        }],
        'melodies': [{
            'description': f"Fallback {genre} melody",
            'similarity_score': 0.8,
            'genre': genre,
            'pattern_data': {
                'intervals': genre_patterns[genre_lower][list(genre_patterns[genre_lower].keys())[0]]['intervals'],
                'instrument': instruments[0] if instruments else 'piano',
                'start_pitch': 60
            }
        }],
        'metadata': {
            'query_genre': genre,
            'query_artist': '',
            'total_retrieved': 3,  # 1 segment + 1 progression + 1 melody
            'is_fallback': True
        }
    }
    
    logging.info(f"[RAGMidiReferenceAgent] Generated fallback patterns for {genre} with {len(instruments)} instruments")
    return fallback_patterns

def rag_midi_reference_agent(state: Dict) -> Dict:
    """
    RAG-enhanced MIDI reference agent that retrieves similar patterns from vector database
    """
    logging.info("[RAGMidiReferenceAgent] Finding similar musical patterns using vector search...")
    
    try:
        # Extract context from state
        artist = state.get('artist', '')
        genre = state.get('genre', 'pop')
        subgenre = state.get('subgenre', '')
        instruments = state.get('instruments', ['piano'])
        mood = state.get('mood', 'neutral')
        tempo = state.get('tempo', 120)
        
        # Full genre context
        full_genre = f"{genre}"
        if subgenre:
            full_genre += f" {subgenre}"
        
        logging.info(f"[RAGMidiReferenceAgent] Searching for patterns: genre='{full_genre}', artist='{artist}'")
        
        # Add debug log
        logging.info(f"[RAGMidiReferenceAgent] DEBUG: Request state: genre={genre}, subgenre={subgenre}, full_genre={full_genre}")
        
        # Initialize RAG database if needed
        try:
            stats = initialize_midi_rag_database()
            logging.info(f"[RAGMidiReferenceAgent] Database stats: {stats}")
        except Exception as e:
            logging.warning(f"[RAGMidiReferenceAgent] Could not initialize RAG database: {e}")
        
        # Retrieve similar patterns from vector database
        retrieved_patterns = {
            'segments': [],
            'progressions': [],
            'melodies': [],
            'metadata': {
                'query_genre': full_genre,
                'query_artist': artist,
                'total_retrieved': 0
            }
        }
        
        # Retrieve musical segments (most important for similarity)
        try:
            segments = midi_rag.retrieve_similar_patterns(full_genre, artist, "segment", top_k=5)
            retrieved_patterns['segments'] = segments
            logging.info(f"[RAGMidiReferenceAgent] Retrieved {len(segments)} musical segments")
        except Exception as e:
            logging.error(f"[RAGMidiReferenceAgent] Error retrieving segments: {e}")
        
        # Retrieve chord progressions
        try:
            progressions = midi_rag.retrieve_similar_patterns(full_genre, artist, "progression", top_k=3)
            retrieved_patterns['progressions'] = progressions
            logging.info(f"[RAGMidiReferenceAgent] Retrieved {len(progressions)} chord progressions")
        except Exception as e:
            logging.error(f"[RAGMidiReferenceAgent] Error retrieving progressions: {e}")
        
        # Retrieve melodies
        try:
            melodies = midi_rag.retrieve_similar_patterns(full_genre, artist, "melody", top_k=3)
            retrieved_patterns['melodies'] = melodies
            logging.info(f"[RAGMidiReferenceAgent] Retrieved {len(melodies)} melodies")
        except Exception as e:
            logging.error(f"[RAGMidiReferenceAgent] Error retrieving melodies: {e}")
        
        # Calculate total retrieved
        total_retrieved = len(retrieved_patterns['segments']) + len(retrieved_patterns['progressions']) + len(retrieved_patterns['melodies'])
        retrieved_patterns['metadata']['total_retrieved'] = total_retrieved
        
        # If no patterns were found, use the fallback patterns
        if total_retrieved == 0:
            logging.warning(f"[RAGMidiReferenceAgent] No patterns found in RAG system, using fallback patterns for {full_genre}")
            retrieved_patterns = generate_fallback_patterns(genre, instruments)
            total_retrieved = retrieved_patterns['metadata']['total_retrieved']
        
        # Create instruction set for generation agents
        rag_instructions = create_rag_instructions(retrieved_patterns, full_genre, artist, instruments)
        
        # Add debug logs for pattern counts
        logging.info(f"[RAGMidiReferenceAgent] DEBUG: Retrieved patterns - segments: {len(retrieved_patterns['segments'])}, "
                    f"progressions: {len(retrieved_patterns['progressions'])}, melodies: {len(retrieved_patterns['melodies'])}")
        logging.info(f"[RAGMidiReferenceAgent] DEBUG: Total patterns: {total_retrieved}")
        
        # Store in state
        state['rag_patterns'] = retrieved_patterns
        state['rag_instructions'] = rag_instructions
        state['pattern_source'] = 'vector_database_rag'
        
        # Create summary for logging
        summary = f"""
        RAG PATTERN RETRIEVAL SUMMARY:
        - Genre Query: {full_genre}
        - Artist Query: {artist}
        - Musical Segments Found: {len(retrieved_patterns['segments'])}
        - Chord Progressions Found: {len(retrieved_patterns['progressions'])}
        - Melody Sequences Found: {len(retrieved_patterns['melodies'])}
        - Total Patterns Retrieved: {total_retrieved}
        
        PATTERN SIMILARITY SCORES:
        """
        
        # Add top similarity scores
        if retrieved_patterns['segments']:
            top_segment = retrieved_patterns['segments'][0]
            summary += f"- Best Segment Match: {top_segment['similarity_score']:.3f}\n"
        
        if retrieved_patterns['progressions']:
            top_progression = retrieved_patterns['progressions'][0]
            summary += f"- Best Progression Match: {top_progression['similarity_score']:.3f}\n"
        
        if retrieved_patterns['melodies']:
            top_melody = retrieved_patterns['melodies'][0]
            summary += f"- Best Melody Match: {top_melody['similarity_score']:.3f}\n"
        
        state['rag_summary'] = summary
        
        logging.info(f"[RAGMidiReferenceAgent] Successfully retrieved {total_retrieved} similar patterns")
        
    except Exception as e:
        logging.error(f"[RAGMidiReferenceAgent] Error in RAG retrieval: {e}")
        # Fallback to empty patterns
        state['rag_patterns'] = {'segments': [], 'progressions': [], 'melodies': [], 'metadata': {}}
        state['rag_instructions'] = "No similar patterns found. Generate using genre conventions."
        state['pattern_source'] = 'fallback'
        state['rag_summary'] = f"RAG retrieval failed: {e}"
    
    # Add debug log for state modification verification
    logging.info(f"[RAGMidiReferenceAgent] DEBUG: State after RAG processing - has 'rag_patterns': {'rag_patterns' in state}")
    if 'rag_patterns' in state:
        logging.info(f"[RAGMidiReferenceAgent] DEBUG: rag_patterns keys: {list(state['rag_patterns'].keys())}")
        logging.info(f"[RAGMidiReferenceAgent] DEBUG: Total patterns in state: {state['rag_patterns']['metadata'].get('total_retrieved', 0)}")
    
    return state

def create_rag_instructions(patterns: Dict, genre: str, artist: str, instruments: List[str]) -> str:
    """
    Create specific instructions for generation agents based on retrieved patterns
    """
    instructions = []
    
    instructions.append(f"=== RAG-RETRIEVED PATTERNS FOR {genre.upper()} ===")
    instructions.append(f"Artist Style Reference: {artist}")
    instructions.append("")
    
    # Process musical segments (highest priority)
    segments = patterns.get('segments', [])
    if segments:
        instructions.append("🎼 SIMILAR MUSICAL SEGMENTS TO FOLLOW:")
        for i, segment in enumerate(segments[:3]):  # Top 3 segments
            similarity = segment.get('similarity_score', 0.0)
            pattern_data = segment.get('pattern_data', {})
            
            instructions.append(f"\nSEGMENT {i+1} (Similarity: {similarity:.3f}):")
            instructions.append(f"Description: {segment.get('description', '')}")
            
            # Extract concrete musical data
            if pattern_data:
                segment_instruments = pattern_data.get('instruments', [])
                duration = pattern_data.get('duration', 16.0)
                
                instructions.append(f"Duration: {duration:.1f} seconds")
                instructions.append("APPLY THESE EXACT PATTERNS:")
                
                for inst in segment_instruments[:2]:  # Top 2 instruments
                    inst_name = inst.get('name', 'Unknown')
                    note_sequence = inst.get('note_sequence', [])
                    melodic_intervals = inst.get('melodic_intervals', [])
                    rhythm_pattern = inst.get('rhythm_pattern', [])
                    
                    if note_sequence:
                        instructions.append(f"  {inst_name} Notes: {note_sequence[:10]}")  # First 10 notes
                    if melodic_intervals:
                        instructions.append(f"  {inst_name} Intervals: {melodic_intervals[:8]}")  # First 8 intervals  
                    if rhythm_pattern:
                        instructions.append(f"  {inst_name} Rhythm: {rhythm_pattern[:8]}")  # First 8 durations
    
    # Process chord progressions
    progressions = patterns.get('progressions', [])
    if progressions:
        instructions.append("\n🎹 SIMILAR CHORD PROGRESSIONS TO USE:")
        for i, progression in enumerate(progressions[:2]):  # Top 2 progressions
            similarity = progression.get('similarity_score', 0.0)
            pattern_data = progression.get('pattern_data', {})
            
            instructions.append(f"\nPROGRESSION {i+1} (Similarity: {similarity:.3f}):")
            
            if pattern_data:
                chords = pattern_data.get('chords', [])
                if chords:
                    instructions.append("APPLY THIS EXACT CHORD SEQUENCE:")
                    chord_desc = []
                    for chord in chords[:6]:  # First 6 chords
                        root = chord.get('root', 0)
                        pitches = chord.get('pitches', [])
                        duration = chord.get('duration', 2.0)
                        chord_desc.append(f"Root:{root} Pitches:{pitches} Duration:{duration:.1f}s")
                    instructions.append("  " + " | ".join(chord_desc))
    
    # Process melodies
    melodies = patterns.get('melodies', [])
    if melodies:
        instructions.append("\n🎵 SIMILAR MELODY PATTERNS TO FOLLOW:")
        for i, melody in enumerate(melodies[:2]):  # Top 2 melodies
            similarity = melody.get('similarity_score', 0.0)
            pattern_data = melody.get('pattern_data', {})
            
            instructions.append(f"\nMELODY {i+1} (Similarity: {similarity:.3f}):")
            
            if pattern_data:
                intervals = pattern_data.get('intervals', [])
                start_pitch = pattern_data.get('start_pitch', 60)
                instrument = pattern_data.get('instrument', 'Piano')
                
                if intervals:
                    instructions.append(f"Instrument: {instrument}")
                    instructions.append(f"Start Pitch: {start_pitch}")
                    instructions.append(f"APPLY THESE EXACT INTERVALS: {intervals[:12]}")  # First 12 intervals
    
    # Add generation directives
    instructions.append("\n" + "="*50)
    instructions.append("🎯 GENERATION DIRECTIVES:")
    instructions.append("1. Use the retrieved patterns as EXACT templates")
    instructions.append("2. Apply the note sequences, intervals, and rhythms directly")
    instructions.append("3. Transpose to appropriate keys but maintain the patterns")
    instructions.append("4. Combine patterns creatively but stay close to the examples")
    instructions.append("5. If patterns don't fit your instrument, adapt but keep the essence")
    instructions.append("6. These patterns come from REAL music in this genre - they work!")
    
    return "\n".join(instructions)

def get_rag_patterns_for_instrument(state: Dict, instrument: str, role: str = "lead") -> str:
    """
    Get RAG-specific instructions for a particular instrument
    """
    try:
        rag_patterns = state.get('rag_patterns', {})
        rag_instructions = state.get('rag_instructions', '')
        
        if not rag_patterns or not rag_instructions:
            return "No RAG patterns available. Use genre conventions."
        
        # Filter patterns relevant to this instrument
        instrument_instructions = []
        instrument_instructions.append(f"=== RAG PATTERNS FOR {instrument.upper()} ({role.upper()}) ===")
        
        # Look for patterns with similar instruments
        segments = rag_patterns.get('segments', [])
        relevant_segments = []
        
        for segment in segments:
            pattern_data = segment.get('pattern_data', {})
            segment_instruments = pattern_data.get('instruments', [])
            
            for inst in segment_instruments:
                inst_name = inst.get('name', '').lower()
                if instrument.lower() in inst_name or any(word in inst_name for word in instrument.lower().split('_')):
                    relevant_segments.append((segment, inst))
                    break
        
        if relevant_segments:
            instrument_instructions.append(f"\nFOUND {len(relevant_segments)} SIMILAR {instrument.upper()} PATTERNS:")
            
            for i, (segment, inst_data) in enumerate(relevant_segments[:2]):  # Top 2 matches
                similarity = segment.get('similarity_score', 0.0)
                note_sequence = inst_data.get('note_sequence', [])
                melodic_intervals = inst_data.get('melodic_intervals', [])
                rhythm_pattern = inst_data.get('rhythm_pattern', [])
                
                instrument_instructions.append(f"\nPATTERN {i+1} (Similarity: {similarity:.3f}):")
                
                if note_sequence:
                    instrument_instructions.append("EXACT NOTES TO USE:")
                    # Group notes for easier reading
                    notes_grouped = [note_sequence[j:j+5] for j in range(0, min(len(note_sequence), 20), 5)]
                    for group in notes_grouped:
                        instrument_instructions.append(f"  {group}")
                
                if melodic_intervals:
                    instrument_instructions.append(f"MELODIC INTERVALS: {melodic_intervals[:10]}")
                
                if rhythm_pattern:
                    instrument_instructions.append(f"RHYTHM PATTERN: {rhythm_pattern[:8]}")
                
                instrument_instructions.append("^ APPLY THESE PATTERNS DIRECTLY ^")
        else:
            instrument_instructions.append(f"\nNo specific {instrument} patterns found. Use general patterns:")
            instrument_instructions.append(rag_instructions[-500:])  # Last 500 chars of general instructions
        
        return "\n".join(instrument_instructions)
        
    except Exception as e:
        logging.error(f"[RAGMidiReferenceAgent] Error getting patterns for {instrument}: {e}")
        return f"Error retrieving RAG patterns for {instrument}. Use genre conventions."

# Main agent function for the graph
def midi_reference_agent(state: Dict) -> Dict:
    """
    Main MIDI reference agent function (called from composer graph)
    """
    return rag_midi_reference_agent(state)
