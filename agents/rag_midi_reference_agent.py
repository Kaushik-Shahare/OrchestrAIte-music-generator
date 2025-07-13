#!/usr/bin/env python3
"""
RAG-Enhanced MIDI Reference Agent
Uses vector database to retrieve similar musical patterns and provide them as concrete examples
"""

import logging
from typing import Any, Dict, List, Optional, Union
from utils.gemini_llm import gemini_generate
from utils.midi_rag_system import midi_rag, initialize_midi_rag_database
import json
import random
import copy

def generate_fallback_patterns(genre: str, instruments: List[str]) -> Dict:
    """Generate fallback patterns when no MIDI files are found in the dataset
    
    This ensures we always have some patterns to work with even if the RAG system fails.
    Returns a consistent data structure to avoid LangGraph errors.
    """
    logging.info(f"[RAGMidiReferenceAgent] Generating fallback patterns for genre: {genre}")
    
    # Ensure we have valid inputs
    if not isinstance(genre, str):
        genre = "pop"
        logging.warning(f"[RAGMidiReferenceAgent] Invalid genre type, defaulting to 'pop'")
        
    if not isinstance(instruments, list):
        instruments = ["piano"]
        logging.warning(f"[RAGMidiReferenceAgent] Invalid instruments type, defaulting to ['piano']")
    
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
    
    # Create a segments entry with pattern data - with deep copying to avoid reference issues
    instruments_data = []
    try:
        for instrument in instruments:
            # Ensure instrument is a string
            instrument_str = str(instrument)
            
            # Map instrument to closest matching one in our patterns
            inst_key = 'piano'  # default
            if 'guitar' in instrument_str.lower():
                inst_key = 'guitar'
            elif 'bass' in instrument_str.lower():
                inst_key = 'bass'
            elif 'violin' in instrument_str.lower() or 'string' in instrument_str.lower():
                inst_key = 'violin' if 'violin' in genre_patterns[genre_lower] else 'piano'
            
            # Use fallback if instrument not found
            if inst_key not in genre_patterns[genre_lower]:
                available_keys = list(genre_patterns[genre_lower].keys())
                inst_key = available_keys[0] if available_keys else 'piano'
                if not available_keys:
                    logging.warning(f"[RAGMidiReferenceAgent] No instruments found for genre: {genre_lower}, using empty pattern")
                    continue
            
            # Create instrument pattern with deep copying
            pattern = genre_patterns[genre_lower][inst_key]
            
            # Deep copy all the data to ensure no shared references
            notes_copy = []
            if 'notes' in pattern:
                for note in pattern['notes']:
                    if isinstance(note, list):
                        notes_copy.append(list(note))  # Copy each note array
                    else:
                        notes_copy.append(note)
                        
            intervals_copy = []
            if 'intervals' in pattern:
                intervals_copy = list(pattern['intervals'])
                
            rhythm_copy = []
            if 'rhythm' in pattern:
                rhythm_copy = list(pattern['rhythm'])
            
            instruments_data.append({
                'name': instrument_str,
                'note_sequence': notes_copy,
                'melodic_intervals': intervals_copy,
                'rhythm_pattern': rhythm_copy
            })
    except Exception as e:
        logging.error(f"[RAGMidiReferenceAgent] Error creating instrument patterns: {e}")
        # Add at least one default instrument if we hit an error
        if not instruments_data:
            instruments_data.append({
                'name': 'piano',
                'note_sequence': [[60, 0.0, 0.5, 80], [64, 0.5, 1.0, 75]],
                'melodic_intervals': [4, -4],
                'rhythm_pattern': [0.5, 0.5]
            })
    
    # Create deep copies of chord data to avoid reference issues
    chords_copy = []
    try:
        for chord in genre_patterns[genre_lower]['chords']:
            if isinstance(chord, dict):
                # Create a fresh dict for each chord
                new_chord = {
                    'root': chord.get('root', 60),
                    'duration': float(chord.get('duration', 1.0))
                }
                
                # Deep copy the pitches array
                if 'pitches' in chord and isinstance(chord['pitches'], list):
                    new_chord['pitches'] = list(chord['pitches'])
                else:
                    new_chord['pitches'] = [60, 64, 67]  # Default C major
                    
                chords_copy.append(new_chord)
    except Exception as e:
        logging.error(f"[RAGMidiReferenceAgent] Error copying chord data: {e}")
        # Add default chord if we hit an error
        chords_copy = [{'root': 60, 'pitches': [60, 64, 67], 'duration': 1.0}]
    
    # Deep copy intervals data
    intervals_copy = []
    try:
        first_instrument_key = list(genre_patterns[genre_lower].keys())[0]
        source_intervals = genre_patterns[genre_lower][first_instrument_key].get('intervals', [])
        intervals_copy = list(source_intervals)
    except Exception as e:
        logging.error(f"[RAGMidiReferenceAgent] Error copying intervals data: {e}")
        intervals_copy = [0, 2, 2, 1, 2, 2, 2, -1]  # Default scale pattern
    
    # Create the basic fallback patterns with all serializable data
    fallback_patterns = {
        'segments': [{
            'description': f"Fallback {genre} pattern",
            'similarity_score': 0.8,
            'genre': str(genre),
            'pattern_data': {
                'instruments': instruments_data,
                'duration': 4.0
            }
        }],
        'progressions': [{
            'description': f"Fallback {genre} chord progression",
            'similarity_score': 0.8,
            'genre': str(genre),
            'pattern_data': {
                'chords': chords_copy
            }
        }],
        'melodies': [{
            'description': f"Fallback {genre} melody",
            'similarity_score': 0.8,
            'genre': str(genre),
            'pattern_data': {
                'intervals': intervals_copy,
                'instrument': str(instruments[0]) if instruments else 'piano',
                'start_pitch': 60
            }
        }],
        'metadata': {
            'query_genre': str(genre),
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
    
    This function ensures we return a properly validated state to avoid LangGraph concurrent update errors.
    It uses deep copy and validation to ensure all data is properly structured and serializable.
    """
    import copy
    
    logging.info("[RAGMidiReferenceAgent] Finding similar musical patterns using vector search...")
    
    # Create a new clean state to avoid modifying the original and break references
    try:
        # Use clean state instead of just copying, to ensure data integrity
        new_state = ensure_valid_state(state)
        logging.info(f"[RAGMidiReferenceAgent] Created clean state copy with keys: {list(new_state.keys())}")
    except Exception as e:
        logging.error(f"[RAGMidiReferenceAgent] Error creating state copy: {e}, falling back to empty state")
        new_state = {
            'genre': 'pop',
            'instruments': ['piano'],
            'artist': '',
            'tempo': 120,
            'key_signature': 'C Major',
            'mood': 'neutral'
        }
    
    try:
        # Extract context from validated state
        artist = new_state.get('artist', '')
        genre = new_state.get('genre', 'pop')
        subgenre = new_state.get('subgenre', '')
        instruments = new_state.get('instruments', ['piano'])
        mood = new_state.get('mood', 'neutral')
        tempo = new_state.get('tempo', 120)
        
        # Validate critical inputs
        if not isinstance(genre, str):
            logging.warning(f"[RAGMidiReferenceAgent] Genre is not a string: {type(genre)}")
            genre = 'pop'
            
        if not isinstance(instruments, list):
            logging.warning(f"[RAGMidiReferenceAgent] Instruments is not a list: {type(instruments)}")
            instruments = ['piano']
        
        # Full genre context
        full_genre = f"{genre}"
        if subgenre and isinstance(subgenre, str):
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
        
        # Create retrieved patterns dictionary with clean structure
        retrieved_patterns = {
            'segments': [],
            'progressions': [],
            'melodies': [],
            'metadata': {
                'query_genre': full_genre,
                'query_artist': artist,
                'total_retrieved': 0,
                'is_fallback': False
            }
        }
        
        # Use try/except for each retrieval step to make the process more robust
        segments, progressions, melodies = [], [], []
        
        # Retrieve musical segments (most important for similarity)
        try:
            segments = midi_rag.retrieve_similar_patterns(full_genre, artist, "segment", top_k=5)
            # Validate and clean each segment to ensure they're serializable
            clean_segments = []
            for segment in segments:
                if isinstance(segment, dict):
                    # Test serialization
                    try:
                        json.dumps(segment)
                        clean_segments.append(segment)
                    except (TypeError, OverflowError):
                        # Create a cleaned version with string values for non-serializable items
                        clean_segment = {}
                        for key, value in segment.items():
                            if isinstance(value, (str, int, float, bool, type(None))):
                                clean_segment[key] = value
                            else:
                                clean_segment[key] = str(value)
                        clean_segments.append(clean_segment)
            
            retrieved_patterns['segments'] = clean_segments
            logging.info(f"[RAGMidiReferenceAgent] Retrieved {len(clean_segments)} musical segments")
        except Exception as e:
            logging.error(f"[RAGMidiReferenceAgent] Error retrieving segments: {e}")
        
        # Retrieve chord progressions
        try:
            progressions = midi_rag.retrieve_similar_patterns(full_genre, artist, "progression", top_k=3)
            # Clean progressions
            clean_progressions = []
            for progression in progressions:
                if isinstance(progression, dict):
                    try:
                        json.dumps(progression)
                        clean_progressions.append(progression)
                    except (TypeError, OverflowError):
                        # Create a cleaned version
                        clean_progression = {}
                        for key, value in progression.items():
                            if isinstance(value, (str, int, float, bool, type(None))):
                                clean_progression[key] = value
                            else:
                                clean_progression[key] = str(value)
                        clean_progressions.append(clean_progression)
                        
            retrieved_patterns['progressions'] = clean_progressions
            logging.info(f"[RAGMidiReferenceAgent] Retrieved {len(clean_progressions)} chord progressions")
        except Exception as e:
            logging.error(f"[RAGMidiReferenceAgent] Error retrieving progressions: {e}")
        
        # Retrieve melodies
        try:
            melodies = midi_rag.retrieve_similar_patterns(full_genre, artist, "melody", top_k=3)
            # Clean melodies
            clean_melodies = []
            for melody in melodies:
                if isinstance(melody, dict):
                    try:
                        json.dumps(melody)
                        clean_melodies.append(melody)
                    except (TypeError, OverflowError):
                        # Create a cleaned version
                        clean_melody = {}
                        for key, value in melody.items():
                            if isinstance(value, (str, int, float, bool, type(None))):
                                clean_melody[key] = value
                            else:
                                clean_melody[key] = str(value)
                        clean_melodies.append(clean_melody)
                        
            retrieved_patterns['melodies'] = clean_melodies
            logging.info(f"[RAGMidiReferenceAgent] Retrieved {len(clean_melodies)} melodies")
        except Exception as e:
            logging.error(f"[RAGMidiReferenceAgent] Error retrieving melodies: {e}")
        
        # Calculate total retrieved
        total_retrieved = len(retrieved_patterns['segments']) + len(retrieved_patterns['progressions']) + len(retrieved_patterns['melodies'])
        retrieved_patterns['metadata']['total_retrieved'] = total_retrieved
        
        # If no patterns were found, use the fallback patterns
        if total_retrieved == 0:
            logging.warning(f"[RAGMidiReferenceAgent] No patterns found in RAG system, using fallback patterns for {full_genre}")
            
            try:
                fallback_patterns = generate_fallback_patterns(genre, instruments)
                # Test if we can serialize the fallback patterns
                json.dumps(fallback_patterns)
                retrieved_patterns = fallback_patterns
                total_retrieved = retrieved_patterns['metadata']['total_retrieved']
            except Exception as e:
                logging.error(f"[RAGMidiReferenceAgent] Error with fallback patterns: {e}")
                # If fallback fails, use an empty but valid structure
                retrieved_patterns = {
                    'segments': [],
                    'progressions': [],
                    'melodies': [],
                    'metadata': {
                        'query_genre': full_genre,
                        'query_artist': artist,
                        'total_retrieved': 0,
                        'is_fallback': True
                    }
                }
                total_retrieved = 0
        
        # Create instruction set for generation agents
        try:
            rag_instructions = create_rag_instructions(retrieved_patterns, full_genre, artist, instruments)
        except Exception as e:
            logging.error(f"[RAGMidiReferenceAgent] Error creating instructions: {e}")
            rag_instructions = f"Use {genre} conventions to create music. No reference patterns were found."
        
        # Add debug logs for pattern counts
        logging.info(f"[RAGMidiReferenceAgent] DEBUG: Retrieved patterns - segments: {len(retrieved_patterns['segments'])}, "
                     f"progressions: {len(retrieved_patterns['progressions'])}, melodies: {len(retrieved_patterns['melodies'])}")
        logging.info(f"[RAGMidiReferenceAgent] DEBUG: Total patterns: {total_retrieved}")
        
        # Store in new state with careful validation
        new_state['rag_patterns'] = retrieved_patterns
        new_state['rag_instructions'] = rag_instructions
        new_state['pattern_source'] = 'vector_database_rag' if total_retrieved > 0 else 'fallback'
        
        # Create summary for logging
        summary = f"""
        RAG PATTERN RETRIEVAL SUMMARY:
        - Genre Query: {full_genre}
        - Artist Query: {artist}
        - Musical Segments Found: {len(retrieved_patterns['segments'])}
        - Chord Progressions Found: {len(retrieved_patterns['progressions'])}
        - Melody Sequences Found: {len(retrieved_patterns['melodies'])}
        - Total Patterns Retrieved: {total_retrieved}
        - Source: {'Database' if total_retrieved > 0 else 'Fallback'}
        
        PATTERN SIMILARITY SCORES:
        """
        
        # Add top similarity scores
        if retrieved_patterns['segments'] and len(retrieved_patterns['segments']) > 0:
            top_segment = retrieved_patterns['segments'][0]
            summary += f"- Best Segment Match: {top_segment.get('similarity_score', 0):.3f}\n"
        
        if retrieved_patterns['progressions'] and len(retrieved_patterns['progressions']) > 0:
            top_progression = retrieved_patterns['progressions'][0]
            summary += f"- Best Progression Match: {top_progression.get('similarity_score', 0):.3f}\n"
        
        if retrieved_patterns['melodies'] and len(retrieved_patterns['melodies']) > 0:
            top_melody = retrieved_patterns['melodies'][0]
            summary += f"- Best Melody Match: {top_melody.get('similarity_score', 0):.3f}\n"
        
        new_state['rag_summary'] = summary
        
        logging.info(f"[RAGMidiReferenceAgent] Successfully retrieved {total_retrieved} similar patterns")
        
    except Exception as e:
        logging.error(f"[RAGMidiReferenceAgent] Error in RAG retrieval: {e}")
        # Fallback to empty patterns with a valid structure
        new_state['rag_patterns'] = {
            'segments': [],
            'progressions': [],
            'melodies': [], 
            'metadata': {
                'total_retrieved': 0,
                'is_fallback': True,
                'error': str(e)
            }
        }
        new_state['rag_instructions'] = "No similar patterns found. Generate using genre conventions."
        new_state['pattern_source'] = 'fallback'
        new_state['rag_summary'] = f"RAG retrieval failed: {e}"
    
    # Add debug log for state verification
    logging.info(f"[RAGMidiReferenceAgent] Final state has 'rag_patterns': {'rag_patterns' in new_state}")
    if 'rag_patterns' in new_state:
        logging.info(f"[RAGMidiReferenceAgent] Final rag_patterns keys: {list(new_state['rag_patterns'].keys())}")
        logging.info(f"[RAGMidiReferenceAgent] Total patterns in state: {new_state['rag_patterns'].get('metadata', {}).get('total_retrieved', 0)}")
    
    # Final validation to ensure the state is clean and serializable
    try:
        # Test if the entire state can be serialized
        json.dumps(new_state)
    except (TypeError, OverflowError) as e:
        logging.error(f"[RAGMidiReferenceAgent] Final state is still not serializable: {e}")
        # If not serializable, run through the full validation again
        new_state = ensure_valid_state(new_state)
    
    # Return the validated state
    return new_state

def create_rag_instructions(patterns: Dict, genre: str, artist: str, instruments: List[str]) -> str:
    """
    Create specific instructions for generation agents based on retrieved patterns
    
    With added safeguards against potential type errors that could cause LangGraph issues
    """
    instructions = []
    
    # Ensure we're working with strings
    genre_str = str(genre).upper()
    artist_str = str(artist)
    
    instructions.append(f"=== RAG-RETRIEVED PATTERNS FOR {genre_str} ===")
    instructions.append(f"Artist Style Reference: {artist_str}")
    instructions.append("")
    
    # Process musical segments (highest priority) - with type checking
    if not isinstance(patterns, dict):
        logging.warning(f"[RAGMidiReferenceAgent] Patterns is not a dictionary: {type(patterns)}")
        patterns = {}
    
    segments = []
    if isinstance(patterns.get('segments'), list):
        segments = patterns.get('segments', [])
        
    if segments:
        instructions.append("🎼 SIMILAR MUSICAL SEGMENTS TO FOLLOW:")
        for i, segment in enumerate(segments[:3]):  # Top 3 segments
            if not isinstance(segment, dict):
                continue
                
            try:
                similarity = float(segment.get('similarity_score', 0.0))
            except (ValueError, TypeError):
                similarity = 0.0
                
            pattern_data = {}
            if isinstance(segment.get('pattern_data'), dict):
                pattern_data = segment.get('pattern_data', {})
            
            instructions.append(f"\nSEGMENT {i+1} (Similarity: {similarity:.3f}):")
            instructions.append(f"Description: {str(segment.get('description', ''))}")
            
            # Extract concrete musical data
            if pattern_data:
                segment_instruments = []
                if isinstance(pattern_data.get('instruments'), list):
                    segment_instruments = pattern_data.get('instruments', [])
                    
                try:
                    duration = float(pattern_data.get('duration', 16.0))
                except (ValueError, TypeError):
                    duration = 16.0
                
                instructions.append(f"Duration: {duration:.1f} seconds")
                instructions.append("APPLY THESE EXACT PATTERNS:")
                
                for inst in segment_instruments[:2]:  # Top 2 instruments
                    if not isinstance(inst, dict):
                        continue
                        
                    inst_name = str(inst.get('name', 'Unknown'))
                    
                    note_sequence = []
                    if isinstance(inst.get('note_sequence'), list):
                        note_sequence = inst.get('note_sequence', [])
                        
                    melodic_intervals = []
                    if isinstance(inst.get('melodic_intervals'), list):
                        melodic_intervals = inst.get('melodic_intervals', [])
                        
                    rhythm_pattern = []
                    if isinstance(inst.get('rhythm_pattern'), list):
                        rhythm_pattern = inst.get('rhythm_pattern', [])
                    
                    if note_sequence:
                        instructions.append(f"  {inst_name} Notes: {note_sequence[:10]}")  # First 10 notes
                    if melodic_intervals:
                        instructions.append(f"  {inst_name} Intervals: {melodic_intervals[:8]}")  # First 8 intervals  
                    if rhythm_pattern:
                        instructions.append(f"  {inst_name} Rhythm: {rhythm_pattern[:8]}")  # First 8 durations
    
    # Process chord progressions - with type checking
    progressions = []
    if isinstance(patterns.get('progressions'), list):
        progressions = patterns.get('progressions', [])
        
    if progressions:
        instructions.append("\n🎹 SIMILAR CHORD PROGRESSIONS TO USE:")
        for i, progression in enumerate(progressions[:2]):  # Top 2 progressions
            if not isinstance(progression, dict):
                continue
                
            try:
                similarity = float(progression.get('similarity_score', 0.0))
            except (ValueError, TypeError):
                similarity = 0.0
                
            pattern_data = {}
            if isinstance(progression.get('pattern_data'), dict):
                pattern_data = progression.get('pattern_data', {})
            
            instructions.append(f"\nPROGRESSION {i+1} (Similarity: {similarity:.3f}):")
            
            if pattern_data:
                chords = []
                if isinstance(pattern_data.get('chords'), list):
                    chords = pattern_data.get('chords', [])
                    
                if chords:
                    instructions.append("APPLY THIS EXACT CHORD SEQUENCE:")
                    chord_desc = []
                    for chord in chords[:6]:  # First 6 chords
                        if not isinstance(chord, dict):
                            continue
                            
                        root = chord.get('root', 0)
                        
                        pitches = []
                        if isinstance(chord.get('pitches'), list):
                            pitches = chord.get('pitches', [])
                            
                        try:
                            duration = float(chord.get('duration', 2.0))
                        except (ValueError, TypeError):
                            duration = 2.0
                            
                        chord_desc.append(f"Root:{root} Pitches:{pitches} Duration:{duration:.1f}s")
                    
                    if chord_desc:
                        instructions.append("  " + " | ".join(chord_desc))
    
    # Process melodies - with type checking
    melodies = []
    if isinstance(patterns.get('melodies'), list):
        melodies = patterns.get('melodies', [])
        
    if melodies:
        instructions.append("\n🎵 SIMILAR MELODY PATTERNS TO FOLLOW:")
        for i, melody in enumerate(melodies[:2]):  # Top 2 melodies
            if not isinstance(melody, dict):
                continue
                
            try:
                similarity = float(melody.get('similarity_score', 0.0))
            except (ValueError, TypeError):
                similarity = 0.0
                
            pattern_data = {}
            if isinstance(melody.get('pattern_data'), dict):
                pattern_data = melody.get('pattern_data', {})
            
            instructions.append(f"\nMELODY {i+1} (Similarity: {similarity:.3f}):")
            
            if pattern_data:
                intervals = []
                if isinstance(pattern_data.get('intervals'), list):
                    intervals = pattern_data.get('intervals', [])
                    
                start_pitch = pattern_data.get('start_pitch', 60)
                instrument = str(pattern_data.get('instrument', 'Piano'))
                
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
    
    This function safely extracts pattern data without modifying the state
    """
    try:
        # Make sure we're accessing read-only data from the state
        rag_patterns = state.get('rag_patterns', {})
        if not isinstance(rag_patterns, dict):
            logging.warning(f"[RAGMidiReferenceAgent] rag_patterns is not a dictionary: {type(rag_patterns)}")
            rag_patterns = {}
            
        rag_instructions = state.get('rag_instructions', '')
        
        if not rag_patterns or not rag_instructions:
            return "No RAG patterns available. Use genre conventions."
        
        # Filter patterns relevant to this instrument
        instrument_instructions = []
        instrument_instructions.append(f"=== RAG PATTERNS FOR {instrument.upper()} ({role.upper()}) ===")
        
        # Look for patterns with similar instruments - safely access segments
        segments = []
        if isinstance(rag_patterns.get('segments'), list):
            segments = rag_patterns.get('segments', [])
        
        relevant_segments = []
        
        for segment in segments:
            if not isinstance(segment, dict):
                continue
                
            pattern_data = segment.get('pattern_data', {})
            if not isinstance(pattern_data, dict):
                continue
                
            segment_instruments = []
            if isinstance(pattern_data.get('instruments'), list):
                segment_instruments = pattern_data.get('instruments', [])
            
            for inst in segment_instruments:
                if not isinstance(inst, dict):
                    continue
                    
                inst_name = str(inst.get('name', '')).lower()
                if instrument.lower() in inst_name or any(word in inst_name for word in instrument.lower().split('_')):
                    relevant_segments.append((segment, inst))
                    break
        
        if relevant_segments:
            instrument_instructions.append(f"\nFOUND {len(relevant_segments)} SIMILAR {instrument.upper()} PATTERNS:")
            
            for i, (segment, inst_data) in enumerate(relevant_segments[:2]):  # Top 2 matches
                similarity = segment.get('similarity_score', 0.0)
                
                # Safely access nested data
                note_sequence = []
                if isinstance(inst_data.get('note_sequence'), list):
                    note_sequence = inst_data.get('note_sequence', [])
                    
                melodic_intervals = []
                if isinstance(inst_data.get('melodic_intervals'), list):
                    melodic_intervals = inst_data.get('melodic_intervals', [])
                    
                rhythm_pattern = []
                if isinstance(inst_data.get('rhythm_pattern'), list):
                    rhythm_pattern = inst_data.get('rhythm_pattern', [])
                
                instrument_instructions.append(f"\nPATTERN {i+1} (Similarity: {float(similarity):.3f}):")
                
                if note_sequence:
                    instrument_instructions.append("EXACT NOTES TO USE:")
                    # Group notes for easier reading - with safeguards for list access
                    max_notes = min(len(note_sequence), 20)
                    notes_grouped = []
                    for j in range(0, max_notes, 5):
                        end_idx = min(j+5, max_notes)
                        notes_grouped.append(note_sequence[j:end_idx])
                        
                    for group in notes_grouped:
                        instrument_instructions.append(f"  {group}")
                
                if melodic_intervals:
                    instrument_instructions.append(f"MELODIC INTERVALS: {melodic_intervals[:10]}")
                
                if rhythm_pattern:
                    instrument_instructions.append(f"RHYTHM PATTERN: {rhythm_pattern[:8]}")
                
                instrument_instructions.append("^ APPLY THESE PATTERNS DIRECTLY ^")
        else:
            instrument_instructions.append(f"\nNo specific {instrument} patterns found. Use general patterns:")
            # Safely take a substring with boundary checking
            if len(rag_instructions) > 500:
                instrument_instructions.append(rag_instructions[-500:])
            else:
                instrument_instructions.append(rag_instructions)
        
        return "\n".join(instrument_instructions)
        
    except Exception as e:
        logging.error(f"[RAGMidiReferenceAgent] Error getting patterns for {instrument}: {e}")
        return f"Error retrieving RAG patterns for {instrument}. Use genre conventions."

def ensure_valid_state(state: Dict) -> Dict:
    """
    Ensure the state is valid and compatible with LangGraph expectations
    
    This helps prevent the INVALID_CONCURRENT_GRAPH_UPDATE error by ensuring
    consistent state structure and all data is serializable
    """
    import copy
    
    # If state isn't a dictionary, create a new empty one
    if not isinstance(state, dict):
        logging.error(f"[MidiReferenceAgent] State is not a dictionary: {type(state)}")
        return {}
    
    # Create a brand new state dictionary instead of modifying the input
    # This helps prevent LangGraph concurrent update issues
    clean_state = {}
    
    # LangGraph requires consistent state - these are expected fields
    required_keys = ['genre', 'instruments', 'artist', 'tempo', 'key_signature', 'mood']
    
    # Process each key in the state, ensuring all data is serializable
    for key, value in state.items():
        try:
            # For critical keys, ensure they exist with appropriate defaults
            if key in required_keys and value is None:
                if key == 'genre':
                    clean_state[key] = 'pop'
                elif key == 'instruments':
                    clean_state[key] = ['piano']
                elif key == 'artist':
                    clean_state[key] = ''
                elif key == 'tempo':
                    clean_state[key] = 120
                elif key == 'key_signature':
                    clean_state[key] = 'C Major'
                elif key == 'mood':
                    clean_state[key] = 'neutral'
                else:
                    clean_state[key] = ''
                continue
                
            # Test if the value is directly JSON serializable
            json.dumps(value)
            
            # Handle special cases for key types that might cause issues
            if key == 'rag_patterns':
                # Do a deep copy and validation of the RAG patterns
                if isinstance(value, dict):
                    clean_patterns = {}
                    
                    # Process each pattern type (segments, progressions, melodies)
                    for pattern_type in ['segments', 'progressions', 'melodies', 'metadata']:
                        if pattern_type in value:
                            if pattern_type == 'metadata' and isinstance(value[pattern_type], dict):
                                # Copy metadata dictionary
                                clean_patterns[pattern_type] = dict(value[pattern_type])
                            elif pattern_type != 'metadata' and isinstance(value[pattern_type], list):
                                # For pattern lists, sanitize each item
                                clean_list = []
                                for item in value[pattern_type]:
                                    if isinstance(item, dict):
                                        # Try to serialize each item to ensure it's clean
                                        try:
                                            # Test serialization
                                            json.dumps(item)
                                            clean_list.append(item)
                                        except (TypeError, OverflowError):
                                            # If serialization fails, create a sanitized version
                                            clean_item = {}
                                            for k, v in item.items():
                                                if isinstance(v, (str, int, float, bool, type(None))):
                                                    clean_item[k] = v
                                                elif isinstance(v, dict):
                                                    # Try to create a simplified dict
                                                    clean_item[k] = {sk: str(sv) for sk, sv in v.items()}
                                                elif isinstance(v, list):
                                                    # Try to create a simplified list
                                                    clean_item[k] = [str(x) for x in v]
                                                else:
                                                    clean_item[k] = str(v)
                                            clean_list.append(clean_item)
                                clean_patterns[pattern_type] = clean_list
                            else:
                                # Use empty defaults for invalid types
                                if pattern_type == 'metadata':
                                    clean_patterns[pattern_type] = {}
                                else:
                                    clean_patterns[pattern_type] = []
                    
                    # Add the cleaned patterns to the state
                    clean_state[key] = clean_patterns
                else:
                    # If rag_patterns isn't a dict, initialize with empty structure
                    clean_state[key] = {
                        'segments': [],
                        'progressions': [],
                        'melodies': [],
                        'metadata': {
                            'total_retrieved': 0,
                            'is_fallback': True
                        }
                    }
            else:
                # For regular keys, use the value directly if it's serializable
                clean_state[key] = value
                
        except (TypeError, OverflowError, ValueError):
            logging.warning(f"[MidiReferenceAgent] Key '{key}' contains non-serializable data, sanitizing")
            
            # Handle different data types appropriately
            if isinstance(value, (list, tuple)):
                # For lists/tuples, create a clean list of strings
                try:
                    clean_state[key] = [str(item) for item in value]
                except:
                    clean_state[key] = ["Unserializable list item"]
            elif isinstance(value, dict):
                # For dictionaries, create a clean dict with string values
                try:
                    clean_state[key] = {k: str(v) for k, v in value.items()}
                except:
                    clean_state[key] = {"error": "Unserializable dictionary"}
            else:
                # For other types, convert to string or use placeholder
                try:
                    clean_state[key] = str(value)
                except:
                    clean_state[key] = "Unserializable value"
    
    # Verify essential fields are present
    for key in required_keys:
        if key not in clean_state:
            if key == 'genre':
                clean_state[key] = 'pop'
            elif key == 'instruments':
                clean_state[key] = ['piano']
            elif key == 'artist':
                clean_state[key] = ''
            elif key == 'tempo':
                clean_state[key] = 120
            elif key == 'key_signature':
                clean_state[key] = 'C Major'
            elif key == 'mood':
                clean_state[key] = 'neutral'
            else:
                clean_state[key] = ''
    
    # Make one final verification that everything is serializable
    try:
        json.dumps(clean_state)
    except Exception as e:
        logging.error(f"[MidiReferenceAgent] Final state still not serializable: {e}")
        # If we still have issues, return a minimal valid state
        return {
            'genre': 'pop',
            'instruments': ['piano'],
            'artist': '',
            'tempo': 120,
            'key_signature': 'C Major',
            'mood': 'neutral',
            'rag_patterns': {
                'segments': [],
                'progressions': [],
                'melodies': [],
                'metadata': {'total_retrieved': 0}
            },
            'rag_instructions': "No valid patterns found. Use genre conventions.",
            'pattern_source': 'fallback',
            'error': str(e)
        }
            
    return clean_state

# Main agent function for the graph
def midi_reference_agent(state: Dict) -> Dict:
    """
    Main MIDI reference agent function (called from composer graph)
    
    This ensures we're only returning a single state value for LangGraph compatibility
    with proper validation to prevent concurrent update errors
    """
    # Create a deep copy of the state to avoid modifying the original
    # This is critical for preventing LangGraph's INVALID_CONCURRENT_GRAPH_UPDATE error
    import copy
    try:
        safe_state = copy.deepcopy(state)
        logging.info(f"[MidiReferenceAgent] Created deep copy of state with keys: {list(safe_state.keys())}")
    except Exception as e:
        logging.error(f"[MidiReferenceAgent] Error making deep copy of state: {e}")
        # If deepcopy fails, use ensure_valid_state to create a clean state
        safe_state = ensure_valid_state(state)

    try:
        # Get the result of the RAG agent on our safe copy
        new_state = rag_midi_reference_agent(safe_state)
    
        # Log the state transitions for debugging
        logging.info(f"[MidiReferenceAgent] Input state keys: {list(safe_state.keys())}")
        logging.info(f"[MidiReferenceAgent] Output state keys before cleaning: {list(new_state.keys())}")
    
        # Ensure we have a valid, serializable state
        clean_state = ensure_valid_state(new_state)
    
        # Log the final state
        logging.info(f"[MidiReferenceAgent] Final output state keys: {list(clean_state.keys())}")
    except Exception as e:
        logging.error(f"[MidiReferenceAgent] Critical error in state processing: {e}")
        # Return a minimal valid state that won't break the graph
        clean_state = {
            'genre': safe_state.get('genre', 'pop'),
            'instruments': safe_state.get('instruments', ['piano']),
            'artist': safe_state.get('artist', ''),
            'tempo': safe_state.get('tempo', 120),
            'key_signature': safe_state.get('key_signature', 'C Major'),
            'mood': safe_state.get('mood', 'neutral'),
            'rag_patterns': {
                'segments': [],
                'progressions': [],
                'melodies': [],
                'metadata': {'total_retrieved': 0, 'is_fallback': True}
            },
            'rag_instructions': "An error occurred. Use genre conventions.",
            'pattern_source': 'fallback_error',
            'error': str(e)
        }
    
    # Test the final state for serialization
    try:
        # Verify we can serialize the state
        json.dumps(clean_state)
    except Exception as e:
        logging.error(f"[MidiReferenceAgent] Final state is not serializable: {e}")
        # Last resort fallback - create minimal state from scratch
        clean_state = {
            'genre': 'pop',
            'instruments': ['piano'],
            'artist': '',
            'tempo': 120,
            'key_signature': 'C Major',
            'mood': 'neutral',
            'rag_patterns': {
                'segments': [],
                'progressions': [],
                'melodies': [],
                'metadata': {'total_retrieved': 0}
            },
            'rag_instructions': "Error occurred. Use genre conventions.",
            'pattern_source': 'fallback'
        }
    
    # Return exactly one clean state for LangGraph compatibility
    return clean_state
