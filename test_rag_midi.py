#!/usr/bin/env python3
"""
Unified RAG MIDI Test Script
----------------------------
Tests the RAG-enhanced music generation using real MIDI data from ChromaDB

Features:
- Tests ChromaDB collection connection
- Verifies pattern retrieval by genre
- Tests complete RAG-enhanced music generation pipeline
- Includes visualization and analysis of results
"""

import sys
import os
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import random

import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from dotenv import load_dotenv
import chromadb

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import from project
from graph.composer_graph import composer_app
from utils.midi_rag_system import midi_rag
from utils.pattern_application import apply_patterns_to_generation

# Setup console
console = Console()

# Load environment variables
load_dotenv()

def setup_logging():
    """Configure logging for the test script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_test.log')
        ]
    )
    return logging.getLogger('rag_test')

def verify_chromadb_collections() -> Dict:
    """Verify ChromaDB collections exist and contain data"""
    results = {
        "status": "success",
        "collections": {},
        "total_patterns": 0
    }
    
    try:
        # Get collection stats from midi_rag system
        stats = midi_rag.get_collection_stats()
        if not stats:
            results["status"] = "error"
            results["error"] = "Failed to get collection stats"
            return results
        
        results["collections"] = {k: v for k, v in stats.items() if k != "total"}
        results["total_patterns"] = stats.get("total", 0)
        
        if results["total_patterns"] == 0:
            results["status"] = "empty"
            results["error"] = "No patterns found in ChromaDB"
        
        return results
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        return results

def test_pattern_retrieval(genre: str, n_results: int = 3) -> Dict:
    """Test retrieving patterns by genre from ChromaDB"""
    results = {
        "status": "success",
        "genre": genre,
        "patterns": {
            "segments": [],
            "chord_progressions": [],
            "melodies": []
        },
        "total_retrieved": 0
    }
    
    try:
        # Query for patterns by genre
        patterns = midi_rag.retrieve_patterns_by_genre(
            genre=genre,
            n_results=n_results
        )
        
        if not patterns:
            results["status"] = "no_results"
            return results
        
        # Extract pattern data
        for pattern_type in ["segments", "chord_progressions", "melodies"]:
            type_patterns = patterns.get(pattern_type, [])
            results["patterns"][pattern_type] = type_patterns
            results["total_retrieved"] += len(type_patterns)
        
        # Add metadata
        results["metadata"] = patterns.get("metadata", {})
        
        return results
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        return results

def test_rag_generation(genre: str, instruments: List[str], 
                       duration: int = 30, tempo: int = 120) -> Dict:
    """Test complete RAG-enhanced music generation"""
    start_time = time.time()
    results = {
        "status": "success",
        "genre": genre,
        "instruments": instruments,
        "duration": duration,
        "generation_time": 0,
        "output_file": None
    }
    
    try:
        # Create initial state
        initial_state = {
            "user_request": f"Create a {genre} piece with {', '.join(instruments)}",
            "genre": genre,
            "subgenre": "",
            "artist": "",
            "instruments": instruments,
            "tempo": tempo,
            "mood": "expressive",
            "duration": duration,
            "vocals": False
        }
        
        # Execute the composer graph
        result = composer_app.invoke(initial_state)
        
        # Calculate generation time
        results["generation_time"] = time.time() - start_time
        
        # Check if output file was created
        if "output_file" in result and result["output_file"]:
            output_path = result["output_file"]
            if os.path.exists(output_path):
                results["output_file"] = output_path
                results["file_size"] = os.path.getsize(output_path)
            else:
                results["status"] = "file_missing"
        
        # Check if RAG patterns were retrieved and used
        if "rag_patterns" in result:
            results["rag_patterns_used"] = True
            results["rag_metadata"] = result.get("rag_patterns", {}).get("metadata", {})
            
            # Count patterns by type
            pattern_counts = {}
            for pattern_type in ["segments", "chord_progressions", "melodies"]:
                pattern_counts[pattern_type] = len(result.get("rag_patterns", {}).get(pattern_type, []))
            results["pattern_counts"] = pattern_counts
            
        else:
            results["rag_patterns_used"] = False
        
        # Additional results analysis
        if "melody_notes" in result:
            results["melody_note_count"] = len(result["melody_notes"])
        
        if "chord_progression" in result:
            results["chord_count"] = len(result["chord_progression"])
        
        if "instrument_tracks" in result:
            results["track_count"] = len(result["instrument_tracks"])
        
        return results
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        return results

def analyze_midi_file(midi_path: str) -> Dict:
    """Analyze a MIDI file for melodic and harmonic properties"""
    results = {
        "status": "success",
        "file_path": midi_path,
        "analysis": {}
    }
    
    try:
        # Load MIDI file
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # Basic properties
        results["analysis"]["duration"] = pm.get_end_time()
        results["analysis"]["tempo"] = pm.get_tempo_changes()[1][0] if pm.get_tempo_changes()[1].size > 0 else 120
        results["analysis"]["instrument_count"] = len(pm.instruments)
        
        # Count notes by instrument
        instrument_notes = {}
        total_notes = 0
        
        for i, instrument in enumerate(pm.instruments):
            inst_name = pretty_midi.program_to_instrument_name(instrument.program)
            note_count = len(instrument.notes)
            total_notes += note_count
            
            instrument_notes[f"Instrument {i+1} ({inst_name})"] = note_count
        
        results["analysis"]["total_notes"] = total_notes
        results["analysis"]["instrument_notes"] = instrument_notes
        
        # Extract pitch range
        all_pitches = []
        for instrument in pm.instruments:
            for note in instrument.notes:
                all_pitches.append(note.pitch)
        
        if all_pitches:
            results["analysis"]["min_pitch"] = min(all_pitches)
            results["analysis"]["max_pitch"] = max(all_pitches)
            results["analysis"]["pitch_range"] = max(all_pitches) - min(all_pitches)
        
        # Estimate chord complexity
        chords = []
        step = 0.5  # Sample every half second
        for time in np.arange(0, pm.get_end_time(), step):
            notes_at_time = []
            for instrument in pm.instruments:
                for note in instrument.notes:
                    if note.start <= time < note.end:
                        notes_at_time.append(note.pitch % 12)  # Get pitch class
            
            if len(notes_at_time) >= 3:  # Consider it a chord if 3+ notes
                chords.append(len(set(notes_at_time)))
        
        if chords:
            results["analysis"]["avg_chord_complexity"] = sum(chords) / len(chords)
            results["analysis"]["max_chord_complexity"] = max(chords)
        
        return results
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        return results

def plot_midi_visualization(midi_path: str, output_path: str) -> bool:
    """Generate a piano roll visualization of a MIDI file"""
    try:
        # Load MIDI file
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # Create figure and axes
        plt.figure(figsize=(12, 8))
        
        # Plot piano roll for each instrument
        for i, instrument in enumerate(pm.instruments):
            # Get instrument name
            program_name = pretty_midi.program_to_instrument_name(instrument.program)
            
            # Create piano roll
            piano_roll = instrument.get_piano_roll(fs=100)
            
            # Plot as an image
            plt.subplot(len(pm.instruments), 1, i+1)
            plt.imshow(piano_roll, aspect='auto', origin='lower', 
                       extent=[0, pm.get_end_time(), 0, 127])
            plt.colorbar(label='Velocity')
            plt.ylabel('Pitch')
            
            if i == len(pm.instruments) - 1:
                plt.xlabel('Time (s)')
            
            plt.title(f"Instrument {i+1}: {program_name}")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to visualize MIDI: {e}")
        return False

def run_test_suite():
    """Run the complete test suite for the RAG MIDI system"""
    logger = setup_logging()
    
    # Print header
    console.print(Panel.fit(
        "RAG MIDI System Test Suite", 
        subtitle="Testing real MIDI data retrieval and generation"
    ))
    
    # Step 1: Verify database connection and content
    console.print("\n[bold cyan]Step 1: Verifying ChromaDB Collections[/bold cyan]")
    db_check = verify_chromadb_collections()
    
    if db_check["status"] != "success":
        console.print(f"[bold red]Database check failed: {db_check.get('error', 'Unknown error')}[/bold red]")
        console.print("[yellow]Please run load_tegridy_midi_to_chromadb.py first.[/yellow]")
        return False
    
    # Print collection statistics
    stats_table = Table(title="ChromaDB Pattern Statistics")
    stats_table.add_column("Collection", style="cyan")
    stats_table.add_column("Pattern Count", style="green")
    
    for collection, count in db_check["collections"].items():
        stats_table.add_row(collection, str(count))
    stats_table.add_row("Total Patterns", str(db_check["total_patterns"]))
    
    console.print(stats_table)
    
    # Step 2: Test pattern retrieval by genre
    console.print("\n[bold cyan]Step 2: Testing Pattern Retrieval by Genre[/bold cyan]")
    
    test_genres = ["classical", "jazz", "pop", "rock", "ambient"]
    genre_results = {}
    
    for genre in track(test_genres, description="Testing genres..."):
        genre_results[genre] = test_pattern_retrieval(genre)
    
    # Print retrieval results
    retrieval_table = Table(title="Pattern Retrieval Results by Genre")
    retrieval_table.add_column("Genre", style="cyan")
    retrieval_table.add_column("Status", style="green")
    retrieval_table.add_column("Total Retrieved", style="green")
    retrieval_table.add_column("Segments", style="green")
    retrieval_table.add_column("Chords", style="green")
    retrieval_table.add_column("Melodies", style="green")
    
    for genre, result in genre_results.items():
        status = result["status"]
        status_color = "green" if status == "success" else "red"
        
        retrieval_table.add_row(
            genre,
            f"[{status_color}]{status}[/{status_color}]",
            str(result["total_retrieved"]),
            str(len(result["patterns"]["segments"])),
            str(len(result["patterns"]["chord_progressions"])),
            str(len(result["patterns"]["melodies"]))
        )
    
    console.print(retrieval_table)
    
    # Step 3: Test music generation with RAG
    console.print("\n[bold cyan]Step 3: Testing Complete RAG Music Generation[/bold cyan]")
    
    test_cases = [
        {
            "name": "Classical Piano",
            "genre": "classical",
            "instruments": ["piano"],
            "tempo": 90,
            "duration": 30
        },
        {
            "name": "Jazz Trio",
            "genre": "jazz",
            "instruments": ["piano", "bass", "drums"],
            "tempo": 120,
            "duration": 30
        },
        {
            "name": "Pop Song",
            "genre": "pop",
            "instruments": ["piano", "guitar", "bass", "drums"],
            "tempo": 120,
            "duration": 30
        },
        {
            "name": "Ambient Soundscape",
            "genre": "ambient",
            "instruments": ["synth", "strings"],
            "tempo": 80,
            "duration": 30
        }
    ]
    
    generation_results = {}
    successful_outputs = []
    
    for i, test_case in enumerate(track(test_cases, description="Generating music...")):
        console.print(f"\nTest case {i+1}: {test_case['name']}")
        
        # Generate music
        result = test_rag_generation(
            genre=test_case["genre"],
            instruments=test_case["instruments"],
            tempo=test_case["tempo"],
            duration=test_case["duration"]
        )
        
        generation_results[test_case["name"]] = result
        
        if result["status"] == "success" and result.get("output_file"):
            successful_outputs.append(result["output_file"])
            
            # Analyze the output
            analysis = analyze_midi_file(result["output_file"])
            
            if analysis["status"] == "success":
                # Generate visualization
                vis_path = f"output/visualization_{test_case['genre']}_{i+1}.png"
                plot_midi_visualization(result["output_file"], vis_path)
                
                # Print analysis
                console.print(f"[green]✓[/green] Generated {test_case['genre']} piece")
                console.print(f"  - Duration: {analysis['analysis'].get('duration', 0):.1f}s")
                console.print(f"  - Notes: {analysis['analysis'].get('total_notes', 0)}")
                console.print(f"  - Instruments: {analysis['analysis'].get('instrument_count', 0)}")
                
                if "avg_chord_complexity" in analysis["analysis"]:
                    console.print(f"  - Avg chord complexity: {analysis['analysis']['avg_chord_complexity']:.2f} notes")
        else:
            console.print(f"[red]✗[/red] Failed to generate {test_case['genre']} piece: {result.get('error', 'Unknown error')}")
    
    # Step 4: Final summary
    console.print("\n[bold cyan]Step 4: Test Suite Summary[/bold cyan]")
    
    # Print generation results
    generation_table = Table(title="Music Generation Results")
    generation_table.add_column("Test Case", style="cyan")
    generation_table.add_column("Status", style="green")
    generation_table.add_column("Generation Time", style="green")
    generation_table.add_column("RAG Patterns Used", style="green")
    generation_table.add_column("Output File", style="green")
    
    for name, result in generation_results.items():
        status = result["status"]
        status_color = "green" if status == "success" else "red"
        
        rag_used = "Yes" if result.get("rag_patterns_used", False) else "No"
        gen_time = f"{result.get('generation_time', 0):.1f}s"
        output_file = result.get("output_file", "None")
        
        generation_table.add_row(
            name,
            f"[{status_color}]{status}[/{status_color}]",
            gen_time,
            rag_used,
            os.path.basename(output_file) if output_file else "None"
        )
    
    console.print(generation_table)
    
    # Success rate
    success_rate = sum(1 for r in generation_results.values() 
                     if r["status"] == "success") / len(generation_results) * 100
    
    rag_usage_rate = sum(1 for r in generation_results.values() 
                      if r.get("rag_patterns_used", False)) / len(generation_results) * 100
    
    console.print(f"\n[bold green]Test Suite Complete![/bold green]")
    console.print(f"Success rate: {success_rate:.1f}%")
    console.print(f"RAG pattern usage rate: {rag_usage_rate:.1f}%")
    
    if successful_outputs:
        console.print(f"\nSuccessful outputs:")
        for output in successful_outputs:
            console.print(f"  - {output}")
    
    return True

if __name__ == "__main__":
    try:
        run_test_suite()
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Unhandled exception: {e}[/bold red]")
        logging.exception("Unhandled exception in test suite")
        sys.exit(1)
