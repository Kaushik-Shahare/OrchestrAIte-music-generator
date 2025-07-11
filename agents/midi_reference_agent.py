import logging
from typing import Any, Dict, List
from utils.midi_rag_system import midi_rag, initialize_midi_rag_database
import json

def get_rag_patterns(genre: str, artist: str = "") -> Dict:
    """
    Retrieve patterns from the RAG system based on genre
    """
    logging.info(f"[MidiReferenceAgent] Retrieving RAG patterns for genre: '{genre}'")
    
    try:
        # Initialize RAG database if needed
        try:
            stats = initialize_midi_rag_database()
            logging.info(f"[MidiReferenceAgent] RAG database stats: {stats}")
        except Exception as e:
            logging.warning(f"[MidiReferenceAgent] Could not initialize RAG database: {e}")
            return {'segments': [], 'progressions': [], 'melodies': []}
        
        # Create retrieved patterns dictionary
        retrieved_patterns = {
            'segments': [],
            'progressions': [],
            'melodies': [],
            'metadata': {
                'query_genre': genre,
                'query_artist': artist,
                'total_retrieved': 0
            }
        }
        
        # Retrieve musical segments
        try:
            segments = midi_rag.retrieve_similar_patterns(genre, artist, "segment", top_k=5)
            
            # Process segments - ensure they have the expected structure
            processed_segments = []
            for segment in segments:
                # Check if pattern_data is a string (JSON) and parse it if needed
                if isinstance(segment.get('pattern_data'), str):
                    try:
                        segment['pattern_data'] = json.loads(segment['pattern_data'])
                        logging.info("[MidiReferenceAgent] Successfully parsed pattern_data JSON")
                    except Exception as e:
                        logging.warning(f"[MidiReferenceAgent] Failed to parse segment pattern_data JSON: {e}")
                
                processed_segments.append(segment)
            
            retrieved_patterns['segments'] = processed_segments
            logging.info(f"[MidiReferenceAgent] Retrieved {len(processed_segments)} musical segments for '{genre}'")
            
            # Log first segment details for debugging
            if processed_segments:
                first_segment = processed_segments[0]
                logging.info(f"[MidiReferenceAgent] First segment genre: {first_segment.get('genre', 'unknown')}")
                pattern_data = first_segment.get('pattern_data', {})
                if isinstance(pattern_data, dict):
                    instruments = pattern_data.get('instruments', [])
                    logging.info(f"[MidiReferenceAgent] First segment has {len(instruments)} instruments")
        except Exception as e:
            logging.error(f"[MidiReferenceAgent] Error retrieving segments: {e}")
        
        # Retrieve chord progressions
        try:
            progressions = midi_rag.retrieve_similar_patterns(genre, artist, "progression", top_k=3)
            
            # Process progressions - ensure they have the expected structure
            processed_progressions = []
            for prog in progressions:
                # Check if pattern_data is a string (JSON) and parse it if needed
                if isinstance(prog.get('pattern_data'), str):
                    try:
                        prog['pattern_data'] = json.loads(prog['pattern_data'])
                        logging.info("[MidiReferenceAgent] Successfully parsed progression pattern_data JSON")
                    except Exception as e:
                        logging.warning(f"[MidiReferenceAgent] Failed to parse progression pattern_data JSON: {e}")
                
                processed_progressions.append(prog)
            
            retrieved_patterns['progressions'] = processed_progressions
            logging.info(f"[MidiReferenceAgent] Retrieved {len(processed_progressions)} chord progressions for '{genre}'")
        except Exception as e:
            logging.error(f"[MidiReferenceAgent] Error retrieving progressions: {e}")
        
        # Retrieve melodies
        try:
            melodies = midi_rag.retrieve_similar_patterns(genre, artist, "melody", top_k=3)
            
            # Process melodies - ensure they have the expected structure
            processed_melodies = []
            for melody in melodies:
                # Check if pattern_data is a string (JSON) and parse it if needed
                if isinstance(melody.get('pattern_data'), str):
                    try:
                        melody['pattern_data'] = json.loads(melody['pattern_data'])
                        logging.info("[MidiReferenceAgent] Successfully parsed melody pattern_data JSON")
                    except Exception as e:
                        logging.warning(f"[MidiReferenceAgent] Failed to parse melody pattern_data JSON: {e}")
                
                processed_melodies.append(melody)
            
            retrieved_patterns['melodies'] = processed_melodies
            logging.info(f"[MidiReferenceAgent] Retrieved {len(processed_melodies)} melodies for '{genre}'")
        except Exception as e:
            logging.error(f"[MidiReferenceAgent] Error retrieving melodies: {e}")
        
        # Calculate total retrieved
        total_retrieved = len(retrieved_patterns['segments']) + len(retrieved_patterns['progressions']) + len(retrieved_patterns['melodies'])
        retrieved_patterns['metadata']['total_retrieved'] = total_retrieved
        
        logging.info(f"[MidiReferenceAgent] Successfully retrieved {total_retrieved} RAG patterns for {genre}")
        
        return retrieved_patterns
        
    except Exception as e:
        logging.error(f"[MidiReferenceAgent] Error in RAG retrieval: {e}")
        return {'segments': [], 'progressions': [], 'melodies': []}

def create_rag_instructions(patterns: Dict, genre: str) -> str:
    """
    Create specific instructions for generation agents based on retrieved patterns
    """
    instructions = []
    
    instructions.append(f"=== RAG-RETRIEVED PATTERNS FOR {genre.upper()} ===")
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
    instructions.append("5. These patterns come from REAL music in this genre - they work!")
    
    return "\n".join(instructions)

def midi_reference_agent(state: Dict) -> Dict:
    """
    Use RAG-based retrieval to provide musical reference patterns.
    This version focuses exclusively on the RAG system.
    """
    logging.info("[MidiReferenceAgent] Starting RAG-only musical reference retrieval")
    
    try:
        # Extract key parameters
        genre = state.get('genre', 'classical')
        artist = state.get('artist', '')
        
        # Log the exact genre being used
        logging.info(f"[MidiReferenceAgent] Detected genre: '{genre}' for RAG retrieval")
        
        # Ensure proper genre value - if none specified, use classical as default
        if not genre:
            genre = 'classical'
            logging.info(f"[MidiReferenceAgent] No genre specified, defaulting to: '{genre}'")
        
        # Get RAG-based patterns based on genre
        logging.info(f"[MidiReferenceAgent] Retrieving RAG patterns for genre: '{genre}'")
        retrieved_patterns = get_rag_patterns(genre, artist)
        
        # Create RAG instructions
        rag_instructions = create_rag_instructions(retrieved_patterns, genre)
        
        # Store RAG patterns in state (critical for the export agent to find them)
        state['rag_patterns'] = retrieved_patterns
        state['rag_instructions'] = rag_instructions
        state['pattern_source'] = 'vector_database_rag'
        
        total_patterns = retrieved_patterns['metadata']['total_retrieved'] 
        logging.info(f"[MidiReferenceAgent] Added {total_patterns} RAG patterns to state")
        
        # Create pattern summary for other agents
        pattern_summary = f"""
        RAG MUSICAL REFERENCE:
        - Genre: {genre}
        - Segments: {len(retrieved_patterns['segments'])}
        - Chord Progressions: {len(retrieved_patterns['progressions'])}
        - Melodies: {len(retrieved_patterns['melodies'])}
        - Total Patterns: {total_patterns}
        
        Use these patterns as foundation for authentic {genre} sound!
        """
        
        state['pattern_summary'] = pattern_summary
        
        # Additional debug info
        logging.info(f"[MidiReferenceAgent] State keys after processing: {list(state.keys())}")
        if 'rag_patterns' in state:
            logging.info(f"[MidiReferenceAgent] RAG patterns in state: {total_patterns} total")
            logging.info(f"[MidiReferenceAgent] RAG patterns structure: segments={len(retrieved_patterns['segments'])}, "
                         f"progressions={len(retrieved_patterns['progressions'])}, melodies={len(retrieved_patterns['melodies'])}")
        
    except Exception as e:
        logging.error(f"[MidiReferenceAgent] Error: {e}")
        state['pattern_summary'] = "Error retrieving reference patterns."
        state['rag_patterns'] = {'segments': [], 'progressions': [], 'melodies': [], 'metadata': {'total_retrieved': 0}}
    
    return state
