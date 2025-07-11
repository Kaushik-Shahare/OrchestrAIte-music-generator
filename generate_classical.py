#!/usr/bin/env python3
"""
Classical Music Generator
------------------------
Simple script to generate classical music using the composer graph.
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import composer graph
from graph.composer_graph import composer_app
from utils.midi_rag_system import midi_rag, initialize_midi_rag_database
import json
import logging

def generate_classical_music(duration=60, 
                             instruments=None,
                             tempo=90,
                             mood="expressive",
                             output_dir="output"):
    """
    Generate classical music with the composer graph
    """
    if instruments is None:
        instruments = ["piano", "strings"]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create initial state
    print(f"Generating {duration} seconds of classical music with {', '.join(instruments)}...")
    initial_state = {
        "user_request": f"Create a classical piece with {', '.join(instruments)}",
        "genre": "classical",
        "subgenre": "",
        "artist": "",
        "instruments": instruments,
        "tempo": tempo,
        "mood": mood,
        "duration": duration,
        "vocals": False
    }
    
    # Start time measurement
    start_time = time.time()
    
    # Check RAG collections before execution
    try:
        stats = midi_rag.get_collection_stats()
        print(f"\nRAG Database status before generation:")
        print(f"Collections: {stats}")
    except Exception as e:
        print(f"Error getting RAG stats: {e}")
    
    # Execute the composer graph
    result = composer_app.invoke(initial_state)
    
    # Log all keys in the result to diagnose
    print("\nDEBUG: Result keys returned from composer_app.invoke():")
    print(json.dumps(list(result.keys()), indent=2))
    
    # Check for RAG patterns specifically
    if "rag_patterns" in result:
        print("\nDEBUG: RAG patterns structure:")
        rag_patterns = result["rag_patterns"]
        print(f"- Type: {type(rag_patterns)}")
        print(f"- Keys: {list(rag_patterns.keys()) if isinstance(rag_patterns, dict) else 'Not a dictionary'}")
        
        # Check each expected collection
        for collection_name in ["segments", "progressions", "melodies", "metadata"]:
            if collection_name in rag_patterns:
                if collection_name == "metadata":
                    print(f"- {collection_name}: {rag_patterns[collection_name]}")
                else:
                    print(f"- {collection_name}: {len(rag_patterns[collection_name])} items")
    else:
        print("\nDEBUG: No 'rag_patterns' found in result")
    
    # Calculate generation time
    gen_time = time.time() - start_time
    
    # Check if output file was created
    if "output_file" in result and result["output_file"]:
        output_path = result["output_file"]
        if os.path.exists(output_path):
            print(f"\n✓ Generated classical music in {gen_time:.1f} seconds")
            print(f"✓ Output file: {output_path}")
            
            # Check if RAG patterns were used
            if "rag_patterns" in result:
                patterns = result["rag_patterns"]
                pattern_count = sum(len(patterns.get(pt, [])) for pt in ["segments", "progressions", "melodies"])
                print(f"\n✓ Used {pattern_count} RAG patterns from the database")
                
                # Print detailed information about retrieved patterns
                print("\n=== RAG PATTERNS RETRIEVED AND USED ===")
                
                # Print segments
                segments = patterns.get("segments", [])
                if segments:
                    print(f"\n📋 MUSICAL SEGMENTS ({len(segments)}):")
                    for i, segment in enumerate(segments):
                        print(f"  Segment {i+1}:")
                        print(f"    - Genre: {segment.get('genre', 'unknown')}")
                        print(f"    - Source: {segment.get('source_file', 'unknown')}")
                        
                        # Get pattern_data and ensure it's not a string
                        pattern_data = segment.get('pattern_data', {})
                        if isinstance(pattern_data, str):
                            try:
                                pattern_data = json.loads(pattern_data)
                            except:
                                pattern_data = {}
                        
                        # Extract instruments from pattern_data
                        instruments = []
                        segment_instruments = pattern_data.get('instruments', [])
                        for inst in segment_instruments:
                            if isinstance(inst, dict):
                                instruments.append(inst.get('name', 'Unknown'))
                        print(f"    - Instruments: {', '.join(instruments) if instruments else 'unknown'}")
                        
                        # Get duration and notes
                        duration = pattern_data.get('duration', 0)
                        print(f"    - Duration: {duration:.1f}s")
                        
                        # Calculate total notes
                        total_notes = 0
                        for inst in segment_instruments:
                            if isinstance(inst, dict):
                                notes = inst.get('notes', [])
                                total_notes += len(notes)
                        print(f"    - Total notes: {total_notes}")
                
                # Print chord progressions
                chord_progressions = patterns.get("progressions", [])
                if chord_progressions:
                    print(f"\n🎹 CHORD PROGRESSIONS ({len(chord_progressions)}):")
                    for i, progression in enumerate(chord_progressions):
                        print(f"  Progression {i+1}:")
                        print(f"    - Genre: {progression.get('genre', 'unknown')}")
                        print(f"    - Source: {progression.get('source_file', 'unknown')}")
                        
                        # Get pattern_data and ensure it's not a string
                        pattern_data = progression.get('pattern_data', {})
                        if isinstance(pattern_data, str):
                            try:
                                pattern_data = json.loads(pattern_data)
                            except:
                                pattern_data = {}
                        
                        # Extract chords
                        chords = pattern_data.get('chords', [])
                        if chords and len(chords) > 0:
                            sample_chords = [str(c.get('pitches', [])) for c in chords[:3]]
                            print(f"    - Example chords: {' → '.join(sample_chords)}")
                        print(f"    - Chord count: {len(chords)}")
                
                # Print melodies
                melodies = patterns.get("melodies", [])
                if melodies:
                    print(f"\n🎵 MELODIES ({len(melodies)}):")
                    for i, melody in enumerate(melodies):
                        print(f"  Melody {i+1}:")
                        print(f"    - Genre: {melody.get('genre', 'unknown')}")
                        print(f"    - Source: {melody.get('source_file', 'unknown')}")
                        
                        # Get pattern_data and ensure it's not a string
                        pattern_data = melody.get('pattern_data', {})
                        if isinstance(pattern_data, str):
                            try:
                                pattern_data = json.loads(pattern_data)
                            except:
                                pattern_data = {}
                                
                        # Extract melody information
                        instrument = pattern_data.get('instrument', 'Unknown')
                        intervals = pattern_data.get('intervals', [])
                        start_pitch = pattern_data.get('start_pitch', 'unknown')
                        
                        print(f"    - Instrument: {instrument}")
                        print(f"    - Intervals: {len(intervals)} notes")
                        print(f"    - Start pitch: {start_pitch}")
                
                # Print metadata
                if "metadata" in patterns:
                    print("\n📊 RAG METADATA:")
                    metadata = patterns.get("metadata", {})
                    for key, value in metadata.items():
                        print(f"  - {key}: {value}")
                
                print("\n=== END OF RAG PATTERNS REPORT ===\n")
            else:
                print("\n⚠️ No RAG patterns found in result. The RAG system may not be working properly.")
            
            return output_path
    
    print(f"\n✗ Failed to generate music: {result.get('error', 'Unknown error')}")
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate classical music")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--instruments", type=str, default="piano,strings", 
                       help="Comma-separated list of instruments")
    parser.add_argument("--tempo", type=int, default=90, help="Tempo in BPM")
    parser.add_argument("--mood", type=str, default="expressive", 
                       help="Mood of the music (e.g., expressive, calm, energetic)")
    
    args = parser.parse_args()
    
    # Process instruments
    instrument_list = [i.strip() for i in args.instruments.split(',')]
    
    # Generate music
    generate_classical_music(
        duration=args.duration,
        instruments=instrument_list,
        tempo=args.tempo,
        mood=args.mood
    )
