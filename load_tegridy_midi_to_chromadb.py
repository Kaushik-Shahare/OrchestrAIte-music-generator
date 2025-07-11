#!/usr/bin/env python3
"""
Unified MIDI Dataset Loader
---------------------------
Loads MIDI files from the Tegridy-MIDI-Dataset into ChromaDB
Extracts instrument-specific patterns and indexes by genre.

Features:
- Extracts and processes ZIP/GZ archives automatically
- Analyzes MIDI files for instruments and patterns
- Stores patterns in ChromaDB by genre
- Terminal dashboard with real-time metrics
"""

import os
import sys
import time
import glob
import zipfile
import gzip
import tarfile
import shutil
import json
import argparse
import random
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
import pretty_midi
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.text import Text
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# Ensure the script runs from project root directory
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup rich console for terminal dashboard
console = Console()

# Load environment variables (Gemini API key)
load_dotenv()

# Validate API key is present
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    console.print("[bold red]ERROR: GEMINI_API_KEY not found in environment or .env file[/bold red]")
    console.print("Please create a .env file with your Gemini API key: GEMINI_API_KEY=your_key_here")
    sys.exit(1)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Global stats for dashboard
class Stats:
    def __init__(self):
        self.total_files_processed = 0
        self.total_files_failed = 0
        self.total_patterns_extracted = 0
        self.patterns_by_genre = {}
        self.patterns_by_type = {"segments": 0, "melodies": 0, "chord_progressions": 0, "rhythms": 0}
        self.current_file = ""
        self.current_genre = ""
        self.extraction_rate = 0
        self.start_time = time.time()
        self.processed_size_mb = 0
        self.lock = threading.Lock()
    
    def update(self, file_processed=False, file_failed=False, patterns=0, genre=None, 
               pattern_types=None, current_file="", file_size_mb=0):
        with self.lock:
            if file_processed:
                self.total_files_processed += 1
                self.processed_size_mb += file_size_mb
            if file_failed:
                self.total_files_failed += 1
            
            self.total_patterns_extracted += patterns
            
            if genre and patterns > 0:
                if genre not in self.patterns_by_genre:
                    self.patterns_by_genre[genre] = 0
                self.patterns_by_genre[genre] += patterns
            
            if pattern_types:
                for p_type, count in pattern_types.items():
                    if p_type in self.patterns_by_type:
                        self.patterns_by_type[p_type] += count
            
            if current_file:
                self.current_file = current_file
            
            if genre:
                self.current_genre = genre
            
            # Calculate extraction rate (patterns per minute)
            elapsed_mins = (time.time() - self.start_time) / 60.0
            if elapsed_mins > 0:
                self.extraction_rate = self.total_patterns_extracted / elapsed_mins

# Initialize global stats
stats = Stats()

# Gemini embedding function for ChromaDB
class GeminiEmbeddingFunction(EmbeddingFunction):
    """Embedding function that uses Gemini for vectorization"""
    
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the input documents"""
        result = []
        
        # Handle input as a batch
        for doc in input:
            try:
                # Get the embedding from Gemini
                response = genai.embed_content(
                    model="models/embedding-001",
                    content=doc,
                    task_type="retrieval_document",
                    title="MIDI Pattern"
                )
                
                embedding = response["embedding"]
                
                # If it's a NumPy array, convert it to a plain Python list
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                
                result.append(embedding)
            except Exception as e:
                console.print(f"[red]Embedding error: {e}[/red]")
                # Return a zero vector as fallback
                result.append([0.0] * 768)  # Gemini embeddings are 768 dimensions
        
        return result

def extract_archive(archive_path: Path, extract_dir: Path) -> Optional[Path]:
    """Extract ZIP, GZ, or TAR archives to the specified directory"""
    try:
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
        
        archive_path_str = str(archive_path)
        
        if archive_path_str.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            return extract_dir
            
        elif archive_path_str.endswith('.gz') and not archive_path_str.endswith('.tar.gz'):
            # Single file gzip
            output_path = extract_dir / archive_path.stem
            with gzip.open(archive_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return output_path
            
        elif archive_path_str.endswith('.tar.gz') or archive_path_str.endswith('.tgz'):
            # Tar gzip
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
            return extract_dir
            
        else:
            console.print(f"[yellow]Unsupported archive format: {archive_path}[/yellow]")
            return None
            
    except Exception as e:
        console.print(f"[red]Failed to extract {archive_path}: {e}[/red]")
        return None

def detect_genre_from_path(file_path: str) -> str:
    """Detect music genre from file path"""
    path_lower = file_path.lower()
    file_name = os.path.basename(path_lower)
    parent_dir = os.path.basename(os.path.dirname(path_lower))
    grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(path_lower)))
    
    # First check for Tegridy dataset specific directories
    tegridy_specific_genres = {
        'classical': ['classical-piano-strings', 'piano', 'tegridy-piano'],
        'jazz': ['tegridy-jazz', 'jazz-collection'],
        'rock': ['rock-collection', 'tegridy-rock'],
        'pop': ['tegridy-pop', 'pop-collection'],
        'ambient': ['ambient', 'relax-in-tegridy'],
        'children': ['tegridy-children', 'children-songs'],
        'anime': ['beautiful-anime', 'anime-masterpieces'],
        'film': ['gothic-horror-movies', 'soundtrack']
    }
    
    # Check each dataset-specific folder naming pattern
    for genre, folder_patterns in tegridy_specific_genres.items():
        for pattern in folder_patterns:
            if (pattern in parent_dir or pattern in grandparent_dir or 
                pattern in file_name):
                return genre
    
    # If no specific match, use generic keyword matching
    genre_keywords = {
        'classical': ['classical', 'baroque', 'romantic', 'piano', 'mozart', 'beethoven', 'bach', 'chopin', 'strings'],
        'jazz': ['jazz', 'blues', 'swing', 'bebop', 'fusion', 'dixieland'],
        'rock': ['rock', 'metal', 'punk', 'indie', 'alternative', 'guitar'],
        'pop': ['pop', 'dance', 'disco', 'edm', 'electronic', 'synth'],
        'folk': ['folk', 'country', 'bluegrass', 'acoustic', 'americana'],
        'hip-hop': ['hip', 'hop', 'rap', 'trap', 'beats', 'rhyme'],
        'r&b': ['r&b', 'rnb', 'soul', 'funk', 'motown'],
        'latin': ['latin', 'salsa', 'bossa', 'tango', 'samba', 'flamenco'],
        'world': ['world', 'ethnic', 'traditional', 'asian', 'african', 'indian'],
        'ambient': ['ambient', 'chillout', 'lounge', 'relaxation', 'meditation', 'relax'],
        'film': ['soundtrack', 'score', 'film', 'movie', 'cinematic', 'horror'],
        'video-game': ['game', 'gaming', 'video game', '8bit', 'chiptune', 'arcade']
    }
    
    # Check for known Tegridy datasets
    if 'tegridy' in path_lower:
        if 'piano' in path_lower or 'classical' in path_lower:
            return 'classical'
        if 'melodies' in path_lower:
            return 'misc'
        if 'children' in path_lower:
            return 'children'
    
    # Check for various versions of 'beautiful' datasets
    if 'beautiful' in path_lower:
        if 'anime' in path_lower:
            return 'anime'
        if 'music' in path_lower:
            return 'classical'
    
    # Check each genre's keywords in file/folder names
    full_path = f"{file_name} {parent_dir} {grandparent_dir}"
    for genre, keywords in genre_keywords.items():
        for keyword in keywords:
            if keyword in full_path:
                return genre
    
    # Try to extract genre from the file path
    # e.g., if the file is in a genre-named folder
    path_parts = path_lower.split(os.sep)
    for part in path_parts:
        for genre, keywords in genre_keywords.items():
            if part in keywords:
                return genre
    
    # Default to "misc" if no genre is detected
    return "misc"

def normalize_instrument_name(pretty_midi_instrument) -> str:
    """Convert a PrettyMIDI instrument into a normalized name category"""
    
    # Get the program number and check if it's a drum kit
    program = pretty_midi_instrument.program
    is_drum = pretty_midi_instrument.is_drum
    
    if is_drum:
        return "drums"
    
    # Use PrettyMIDI's built-in mapping
    inst_name = pretty_midi.program_to_instrument_name(program).lower()
    
    # Categorize into common groups
    if any(x in inst_name for x in ["piano", "grand", "bright", "honky"]):
        return "piano"
    elif any(x in inst_name for x in ["guitar", "acoustic"]):
        return "acoustic_guitar"
    elif any(x in inst_name for x in ["electric guitar"]):
        return "electric_guitar"
    elif any(x in inst_name for x in ["bass"]):
        return "bass"
    elif any(x in inst_name for x in ["violin", "viola", "cello", "contrabass", "string"]):
        return "strings"
    elif any(x in inst_name for x in ["trumpet", "trombone", "tuba", "brass", "horn"]):
        return "brass"
    elif any(x in inst_name for x in ["sax", "clarinet", "flute", "piccolo", "oboe", "wind"]):
        return "woodwinds"
    elif any(x in inst_name for x in ["synth"]):
        return "synth"
    elif any(x in inst_name for x in ["organ"]):
        return "organ"
    else:
        return "other"

def extract_notes_from_instrument(instrument, max_notes=100) -> List:
    """Extract note data from a PrettyMIDI instrument"""
    notes = []
    
    for note in instrument.notes[:max_notes]:  # Limit to prevent excessive data
        # Store (pitch, start_time, duration, velocity)
        notes.append([
            int(note.pitch),
            float(note.start),
            float(note.end - note.start),
            int(note.velocity)
        ])
    
    return notes

def extract_chord_progression(pm_obj: pretty_midi.PrettyMIDI, start_time=0, end_time=None, min_notes=3) -> List:
    """Extract chord progression from a PrettyMIDI object"""
    if end_time is None:
        end_time = pm_obj.get_end_time()
    
    # Find instruments likely to contain chords (piano, guitar, etc.)
    chord_instruments = []
    for instrument in pm_obj.instruments:
        name = normalize_instrument_name(instrument)
        if name in ["piano", "acoustic_guitar", "electric_guitar", "organ", "synth"]:
            chord_instruments.append(instrument)
    
    if not chord_instruments:
        return []
    
    # Analyze chords at regular intervals
    chords = []
    step = 2.0  # seconds between chord samples
    current_time = start_time
    
    while current_time < end_time:
        # Find all notes sounding at this time point across relevant instruments
        active_pitches = set()
        for instrument in chord_instruments:
            for note in instrument.notes:
                if note.start <= current_time < note.end:
                    # Only use pitch class (0-11) for chord detection
                    pitch_class = note.pitch % 12
                    active_pitches.add(pitch_class)
        
        # If we have enough notes for a chord
        if len(active_pitches) >= min_notes:
            chords.append({
                "pitches": sorted(list(active_pitches)),
                "duration": step,
                "time": current_time
            })
        
        current_time += step
    
    return chords

def extract_melody_pattern(pm_obj: pretty_midi.PrettyMIDI) -> Dict:
    """Extract melody pattern (usually highest-pitched prominent line)"""
    if not pm_obj.instruments:
        return {}
    
    # Find instruments likely to carry melody
    melody_instruments = []
    for instrument in pm_obj.instruments:
        name = normalize_instrument_name(instrument)
        if name in ["piano", "woodwinds", "brass", "strings", "synth"]:
            if len(instrument.notes) >= 10:  # Ensure enough notes to be a melody
                melody_instruments.append(instrument)
    
    if not melody_instruments:
        # Fall back to any instrument with notes
        melody_instruments = [i for i in pm_obj.instruments if len(i.notes) >= 10]
    
    if not melody_instruments:
        return {}
    
    # Select instrument with most notes in higher register
    lead_instrument = max(
        melody_instruments,
        key=lambda x: sum(1 for note in x.notes if note.pitch > 60)
    )
    
    # Extract notes and calculate intervals
    if not lead_instrument.notes:
        return {}
    
    notes = sorted(lead_instrument.notes, key=lambda x: x.start)[:50]  # First 50 notes max
    intervals = []
    start_pitch = notes[0].pitch
    
    for i in range(1, len(notes)):
        intervals.append(notes[i].pitch - notes[i-1].pitch)
    
    return {
        "intervals": intervals,
        "start_pitch": start_pitch,
        "instrument_name": normalize_instrument_name(lead_instrument)
    }

def extract_patterns_from_midi(midi_path: str) -> Dict:
    """Extract various musical patterns from a MIDI file"""
    try:
        # Load MIDI file
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # Skip if no instruments or extremely short file
        if not pm.instruments or pm.get_end_time() < 5.0:
            return {}
        
        # Extract musical segment (complete section with instruments)
        segment = {"instruments": [], "duration": pm.get_end_time()}
        for instrument in pm.instruments:
            if len(instrument.notes) < 5:  # Skip instruments with very few notes
                continue
                
            inst_name = normalize_instrument_name(instrument)
            notes = extract_notes_from_instrument(instrument)
            
            if notes:
                segment["instruments"].append({
                    "name": inst_name,
                    "program": int(instrument.program),
                    "is_drum": bool(instrument.is_drum),
                    "notes": notes
                })
        
        # Get chord progression
        chord_progression = extract_chord_progression(pm)
        
        # Get melody pattern
        melody_pattern = extract_melody_pattern(pm)
        
        # Determine genre from filename
        genre = detect_genre_from_path(midi_path)
        
        # Prepare the return dictionary with all extracted patterns
        patterns = {
            "segments": [segment] if segment["instruments"] else [],
            "chord_progressions": [{"chords": chord_progression}] if chord_progression else [],
            "melodies": [melody_pattern] if melody_pattern else [],
            "filename": os.path.basename(midi_path),
            "genre": genre
        }
        
        # Count extracted patterns
        pattern_counts = {
            "segments": 1 if segment["instruments"] else 0,
            "chord_progressions": 1 if chord_progression else 0,
            "melodies": 1 if melody_pattern else 0,
            "rhythms": 0  # Not implemented yet
        }
        
        # Update stats with pattern counts by type
        return patterns, pattern_counts
        
    except Exception as e:
        #console.print(f"[red]Error processing {os.path.basename(midi_path)}: {e}[/red]")
        return {}, {}

def format_pattern_for_chromadb(pattern_data: Dict, pattern_type: str) -> Tuple[str, Dict]:
    """Format a pattern for storage in ChromaDB with metadata"""
    
    # Create description based on pattern type
    if pattern_type == "segments":
        instrument_count = len(pattern_data.get("instruments", []))
        instrument_names = [i.get("name", "unknown") for i in pattern_data.get("instruments", [])][:3]
        description = f"Musical segment with {instrument_count} instruments: {', '.join(instrument_names)}"
        
        # Count total notes
        total_notes = sum(len(i.get("notes", [])) for i in pattern_data.get("instruments", []))
        description += f", containing {total_notes} notes over {pattern_data.get('duration', 0):.1f} seconds"
    
    elif pattern_type == "chord_progressions":
        chord_count = len(pattern_data.get("chords", []))
        description = f"Chord progression with {chord_count} chords"
        
        # Add sample of chord pitches
        if chord_count > 0:
            sample_chords = [str(c.get("pitches", [])) for c in pattern_data.get("chords", [])[:3]]
            description += f": {' → '.join(sample_chords)}"
    
    elif pattern_type == "melodies":
        interval_count = len(pattern_data.get("intervals", []))
        start_pitch = pattern_data.get("start_pitch", 0)
        instrument = pattern_data.get("instrument_name", "unknown")
        description = f"Melody pattern with {interval_count} intervals starting at pitch {start_pitch} on {instrument}"
    
    else:
        description = f"Musical pattern of type {pattern_type}"
    
    # Prepare metadata
    metadata = {
        "type": pattern_type,
        "genre": pattern_data.get("genre", "unknown"),
        "filename": pattern_data.get("filename", "unknown"),
    }
    
    # Add pattern-specific metadata
    if pattern_type == "segments":
        metadata["instrument_count"] = len(pattern_data.get("instruments", []))
        metadata["duration"] = pattern_data.get("duration", 0)
    elif pattern_type == "chord_progressions":
        metadata["chord_count"] = len(pattern_data.get("chords", []))
    elif pattern_type == "melodies":
        metadata["interval_count"] = len(pattern_data.get("intervals", []))
    
    # Convert pattern_data to a serializable format (handle numpy values)
    serialized_data = json.dumps(pattern_data, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    
    # Include the serialized pattern data in the metadata
    metadata["pattern_data"] = serialized_data
    
    return description, metadata

def create_vector_db(db_path: str):
    """Create or get ChromaDB collections for different pattern types"""
    
    # Ensure directory exists
    os.makedirs(db_path, exist_ok=True)
    
    # Create ChromaDB client
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Create embedding function
    embedding_func = GeminiEmbeddingFunction()
    
    # Create collections for each pattern type
    collections = {}
    for pattern_type in ["segments", "chord_progressions", "melodies", "rhythms"]:
        collections[pattern_type] = chroma_client.get_or_create_collection(
            name=f"midi_{pattern_type}",
            embedding_function=embedding_func,
        )
    
    return collections

def add_patterns_to_db(collections: Dict, patterns: Dict) -> int:
    """Add extracted patterns to ChromaDB collections"""
    total_added = 0
    
    # Process each pattern type
    for pattern_type in ["segments", "chord_progressions", "melodies"]:
        pattern_list = patterns.get(pattern_type, [])
        
        if not pattern_list:
            continue
        
        # Get the collection
        collection = collections.get(pattern_type)
        if not collection:
            continue
        
        # Process each pattern
        for i, pattern in enumerate(pattern_list):
            # Skip empty patterns
            if not pattern:
                continue
            
            # Set genre and filename if available at top level
            if "genre" not in pattern and "genre" in patterns:
                pattern["genre"] = patterns["genre"]
            if "filename" not in pattern and "filename" in patterns:
                pattern["filename"] = patterns["filename"]
            
            # Format for ChromaDB
            description, metadata = format_pattern_for_chromadb(pattern, pattern_type)
            
            # Create unique ID
            pattern_id = f"{patterns.get('filename', 'unknown')}_{pattern_type}_{i}"
            
            try:
                # Add to collection
                collection.add(
                    ids=[pattern_id],
                    documents=[description],
                    metadatas=[metadata]
                )
                total_added += 1
            except Exception as e:
                console.print(f"[red]Error adding pattern {pattern_id}: {e}[/red]")
                # Continue with next pattern
    
    return total_added

def process_midi_file(midi_path: str, collections: Dict) -> Dict:
    """Process a single MIDI file and add patterns to ChromaDB"""
    result = {
        "file_processed": True,
        "file_failed": False,
        "patterns_added": 0,
        "pattern_types": {},
        "genre": "unknown",
        "file_size_mb": os.path.getsize(midi_path) / (1024 * 1024)
    }
    
    try:
        # Extract patterns
        patterns, pattern_counts = extract_patterns_from_midi(midi_path)
        
        if not patterns:
            result["file_processed"] = False
            result["file_failed"] = True
            return result
        
        # Set genre from patterns
        result["genre"] = patterns.get("genre", "unknown")
        result["pattern_types"] = pattern_counts
        
        # Add to database
        patterns_added = add_patterns_to_db(collections, patterns)
        result["patterns_added"] = patterns_added
        
        return result
        
    except Exception as e:
        result["file_processed"] = False
        result["file_failed"] = True
        return result

def render_dashboard(stats: Stats) -> Layout:
    """Create a rich dashboard layout with statistics"""
    layout = Layout()
    
    # Create the header
    header = Panel(
        Text("TEGRIDY MIDI DATASET LOADER", style="bold white on blue", justify="center"),
        style="bold white on blue"
    )
    
    # Create stats tables
    processing_stats = Table(title="Processing Statistics", box=None)
    processing_stats.add_column("Metric", style="cyan")
    processing_stats.add_column("Value", style="green")
    
    # Add processing stats rows
    elapsed_time = time.time() - stats.start_time
    elapsed_str = f"{int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}"
    processing_rate = stats.total_files_processed / elapsed_time if elapsed_time > 0 else 0
    
    processing_stats.add_row("Files Processed", f"{stats.total_files_processed}")
    processing_stats.add_row("Files Failed", f"{stats.total_files_failed}")
    processing_stats.add_row("Processing Rate", f"{processing_rate:.2f} files/sec")
    processing_stats.add_row("Data Processed", f"{stats.processed_size_mb:.2f} MB")
    processing_stats.add_row("Elapsed Time", elapsed_str)
    processing_stats.add_row("Current File", stats.current_file[-40:] if len(stats.current_file) > 40 else stats.current_file)
    
    # Create pattern stats table
    pattern_stats = Table(title="Pattern Statistics", box=None)
    pattern_stats.add_column("Metric", style="cyan")
    pattern_stats.add_column("Value", style="green")
    
    # Add pattern stats rows
    pattern_stats.add_row("Total Patterns Extracted", f"{stats.total_patterns_extracted}")
    pattern_stats.add_row("Extraction Rate", f"{stats.extraction_rate:.2f} patterns/min")
    pattern_stats.add_row("Musical Segments", f"{stats.patterns_by_type['segments']}")
    pattern_stats.add_row("Chord Progressions", f"{stats.patterns_by_type['chord_progressions']}")
    pattern_stats.add_row("Melody Patterns", f"{stats.patterns_by_type['melodies']}")
    
    # Create genre stats table
    genre_stats = Table(title="Patterns by Genre", box=None)
    genre_stats.add_column("Genre", style="cyan")
    genre_stats.add_column("Count", style="green")
    
    # Add genre stats rows
    for genre, count in sorted(stats.patterns_by_genre.items(), key=lambda x: x[1], reverse=True):
        genre_stats.add_row(genre, str(count))
    
    # Create footer with info
    footer = Panel(
        Text("Press Ctrl+C to stop processing", style="italic", justify="center"),
        style="dim"
    )
    
    # Arrange layout
    layout.split(
        Layout(header, size=3),
        Layout(name="main", ratio=1),
        Layout(footer, size=3)
    )
    
    layout["main"].split_row(
        Layout(processing_stats, name="left"),
        Layout(name="right"),
    )
    
    layout["right"].split(
        Layout(pattern_stats),
        Layout(genre_stats)
    )
    
    return layout

def process_directory(midi_dir: str, collections: Dict, temp_dir: str, max_files: int = None):
    """Process all MIDI files in a directory and its subdirectories"""
    # Get list of all potential MIDI files
    midi_paths = []
    archive_paths = []
    
    # Find MIDI files
    for extension in ["mid", "midi", "MID", "MIDI"]:
        midi_paths.extend(glob.glob(os.path.join(midi_dir, f"**/*.{extension}"), recursive=True))
    
    # Find archives that may contain MIDI files
    for extension in ["zip", "gz", "tgz", "tar.gz"]:
        archive_paths.extend(glob.glob(os.path.join(midi_dir, f"**/*.{extension}"), recursive=True))
    
    # Limit number of files if specified
    if max_files:
        midi_paths = midi_paths[:max_files]
        archive_remaining = max(0, max_files - len(midi_paths))
        archive_paths = archive_paths[:archive_remaining]
    
    console.print(f"[green]Found {len(midi_paths)} MIDI files and {len(archive_paths)} archives to process[/green]")
    
    # Create temp directory for archive extraction
    temp_extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    # Process archives first to extract additional MIDI files
    with Live(render_dashboard(stats), refresh_per_second=1) as live:
        for archive_path in archive_paths:
            try:
                stats.update(current_file=f"Extracting {os.path.basename(archive_path)}")
                live.update(render_dashboard(stats))
                
                extracted_path = extract_archive(Path(archive_path), Path(temp_extract_dir) / Path(archive_path).stem)
                
                if extracted_path:
                    # Find MIDI files in the extracted directory
                    if os.path.isdir(extracted_path):
                        for extension in ["mid", "midi", "MID", "MIDI"]:
                            extracted_midis = glob.glob(os.path.join(extracted_path, f"**/*.{extension}"), recursive=True)
                            midi_paths.extend(extracted_midis)
            except Exception as e:
                console.print(f"[red]Failed to process archive {archive_path}: {e}[/red]")
    
    # Update total count after archive extraction
    total_midi_files = len(midi_paths)
    console.print(f"[green]Total MIDI files to process (including extracted): {total_midi_files}[/green]")
    
    # Limit if needed after extraction
    if max_files and len(midi_paths) > max_files:
        midi_paths = random.sample(midi_paths, max_files)
        console.print(f"[yellow]Limiting to {max_files} random files[/yellow]")
    
    # Process all MIDI files with a live dashboard
    with Live(render_dashboard(stats), refresh_per_second=1) as live:
        # Set up thread pool
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit all files for processing
            futures = [executor.submit(process_midi_file, path, collections) for path in midi_paths]
            
            # Process results as they complete
            for future in as_completed(futures):
                result = future.result()
                
                # Update stats
                if result:
                    current_file = midi_paths[futures.index(future)]
                    stats.update(
                        file_processed=result["file_processed"],
                        file_failed=result["file_failed"],
                        patterns=result["patterns_added"],
                        genre=result["genre"],
                        pattern_types=result["pattern_types"],
                        current_file=os.path.basename(current_file),
                        file_size_mb=result["file_size_mb"]
                    )
                    
                    # Update the dashboard
                    live.update(render_dashboard(stats))

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Load MIDI files from Tegridy dataset into ChromaDB")
    parser.add_argument("--midi_dir", type=str, default="./Tegridy-MIDI-Dataset-master", 
                        help="Directory containing MIDI files or archives")
    parser.add_argument("--db_path", type=str, default="./chroma_midi_db", 
                        help="Path to store ChromaDB database")
    parser.add_argument("--temp_dir", type=str, default="./temp", 
                        help="Temporary directory for extraction")
    parser.add_argument("--max_files", type=int, default=None, 
                        help="Maximum number of files to process")
    
    args = parser.parse_args()
    
    # Print banner
    console.print(Panel.fit(
        Text("TEGRIDY MIDI DATASET LOADER", justify="center"),
        title="Music Generator RAG System",
        subtitle="Loading real MIDI data into ChromaDB"
    ))
    
    # Check if input directory exists
    if not os.path.exists(args.midi_dir):
        console.print(f"[red]Error: MIDI directory {args.midi_dir} not found[/red]")
        console.print("[yellow]Please download the Tegridy-MIDI-Dataset or provide a valid path[/yellow]")
        return 1
    
    # Create ChromaDB collections
    console.print("[green]Creating ChromaDB collections...[/green]")
    collections = create_vector_db(args.db_path)
    
    # Process all files
    console.print(f"[green]Starting processing of MIDI files from {args.midi_dir}[/green]")
    process_directory(args.midi_dir, collections, args.temp_dir, args.max_files)
    
    # Final stats
    console.print("\n[bold green]Processing complete![/bold green]")
    console.print(f"Total files processed: {stats.total_files_processed}")
    console.print(f"Total files failed: {stats.total_files_failed}")
    console.print(f"Total patterns extracted: {stats.total_patterns_extracted}")
    
    console.print("\n[bold cyan]Patterns by type:[/bold cyan]")
    for pattern_type, count in stats.patterns_by_type.items():
        console.print(f"  - {pattern_type}: {count}")
    
    console.print("\n[bold cyan]Patterns by genre:[/bold cyan]")
    for genre, count in sorted(stats.patterns_by_genre.items(), key=lambda x: x[1], reverse=True):
        console.print(f"  - {genre}: {count}")
    
    # Display ChronaDB collection info
    console.print("\n[bold cyan]ChromaDB Collections:[/bold cyan]")
    for pattern_type, collection in collections.items():
        try:
            count = collection.count()
            console.print(f"  - midi_{pattern_type}: {count} patterns")
        except Exception:
            console.print(f"  - midi_{pattern_type}: Unknown count")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        console.print(f"Partial results saved to database - {stats.total_patterns_extracted} patterns extracted")
        sys.exit(130)
