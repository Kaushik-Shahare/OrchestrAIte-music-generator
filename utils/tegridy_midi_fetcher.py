#!/usr/bin/env python3
"""
Tegridy MIDI Fetcher
-------------------
Fetches and processes MIDI patterns from the Tegridy-MIDI-Dataset
Compatible with the existing ChromaDB loader functionality.
"""

import os
import sys
import glob
import random
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import pretty_midi
import numpy as np

# Import pattern extraction functions from the existing loader
sys.path.append(str(Path(__file__).parent.parent))
from load_tegridy_midi_to_chromadb import (
    extract_patterns_from_midi,
    detect_genre_from_path,
    normalize_instrument_name,
    extract_notes_from_instrument,
    extract_chord_progression,
    extract_melody_pattern
)

class TegridyMIDIFetcher:
    """
    Fetches and processes MIDI patterns from the Tegridy-MIDI-Dataset
    """
    
    def __init__(self, tegridy_path: str = "./Tegridy-MIDI-Dataset-master"):
        """
        Initialize the Tegridy MIDI Fetcher
        
        Args:
            tegridy_path: Path to the Tegridy MIDI dataset directory
        """
        self.tegridy_path = Path(tegridy_path)
        self.logger = logging.getLogger(__name__)
        
        # Cache for MIDI file paths by genre
        self._genre_file_cache = {}
        self._initialize_file_cache()
    
    def _initialize_file_cache(self):
        """Initialize cache of MIDI files organized by genre"""
        if not self.tegridy_path.exists():
            self.logger.warning(f"Tegridy dataset path not found: {self.tegridy_path}")
            return
        
        # Find all MIDI files
        midi_patterns = ["**/*.mid", "**/*.midi", "**/*.MID", "**/*.MIDI"]
        all_midi_files = []
        
        for pattern in midi_patterns:
            all_midi_files.extend(self.tegridy_path.glob(pattern))
        
        # Organize by genre
        for midi_file in all_midi_files:
            genre = detect_genre_from_path(str(midi_file))
            if genre not in self._genre_file_cache:
                self._genre_file_cache[genre] = []
            self._genre_file_cache[genre].append(str(midi_file))
        
        self.logger.info(f"Cached {len(all_midi_files)} MIDI files across {len(self._genre_file_cache)} genres")
    
    def get_patterns_for_artist_genre(self, artist: str, genre: str, max_files: int = 10) -> Dict[str, Any]:
        """
        Get musical patterns for a specific artist/genre combination
        
        Args:
            artist: Artist name (currently ignored, can be empty string)
            genre: Music genre to filter by
            max_files: Maximum number of files to process
            
        Returns:
            Dictionary containing extracted musical patterns
        """
        if genre not in self._genre_file_cache:
            self.logger.warning(f"No files found for genre: {genre}")
            return {}
        
        # Get random sample of files for this genre
        genre_files = self._genre_file_cache[genre]
        sample_files = random.sample(genre_files, min(max_files, len(genre_files)))
        
        # Extract patterns from sampled files
        all_segments = []
        all_chord_progressions = []
        all_melodies = []
        
        for midi_file in sample_files:
            try:
                patterns, pattern_counts = extract_patterns_from_midi(midi_file)
                
                if not patterns:
                    continue
                
                # Collect patterns by type
                segments = patterns.get("segments", [])
                chord_progressions = patterns.get("chord_progressions", [])
                melodies = patterns.get("melodies", [])
                
                # Add metadata to each pattern
                for segment in segments:
                    segment["source_file"] = os.path.basename(midi_file)
                    segment["genre"] = genre
                    all_segments.append(segment)
                
                for chord_prog in chord_progressions:
                    chord_prog["source_file"] = os.path.basename(midi_file)
                    chord_prog["genre"] = genre
                    all_chord_progressions.append(chord_prog)
                
                for melody in melodies:
                    melody["source_file"] = os.path.basename(midi_file)
                    melody["genre"] = genre
                    all_melodies.append(melody)
                    
            except Exception as e:
                self.logger.warning(f"Failed to process {midi_file}: {e}")
                continue
        
        result = {
            "musical_segments": all_segments,
            "chord_progressions": all_chord_progressions,
            "melody_sequences": all_melodies,
            "metadata": {
                "genre": genre,
                "artist": artist,
                "files_processed": len(sample_files),
                "total_patterns": len(all_segments) + len(all_chord_progressions) + len(all_melodies)
            }
        }
        
        self.logger.info(f"Extracted {result['metadata']['total_patterns']} patterns from {len(sample_files)} {genre} files")
        
        return result
    
    def get_available_genres(self) -> List[str]:
        """Get list of available genres in the dataset"""
        return list(self._genre_file_cache.keys())
    
    def get_genre_file_count(self, genre: str) -> int:
        """Get number of files available for a specific genre"""
        return len(self._genre_file_cache.get(genre, []))
    
    def get_random_patterns_by_genre(self, genre: str, pattern_type: str = "segments", n_patterns: int = 5) -> List[Dict]:
        """
        Get random patterns of a specific type for a genre
        
        Args:
            genre: Music genre
            pattern_type: Type of pattern ("segments", "chord_progressions", "melodies")
            n_patterns: Number of patterns to return
            
        Returns:
            List of pattern dictionaries
        """
        if genre not in self._genre_file_cache:
            return []
        
        # Get a random file for this genre
        genre_files = self._genre_file_cache[genre]
        sample_file = random.choice(genre_files)
        
        try:
            patterns, _ = extract_patterns_from_midi(sample_file)
            
            if not patterns:
                return []
            
            # Get patterns of requested type
            if pattern_type == "segments":
                available_patterns = patterns.get("segments", [])
            elif pattern_type == "chord_progressions":
                available_patterns = patterns.get("chord_progressions", [])
            elif pattern_type == "melodies":
                available_patterns = patterns.get("melodies", [])
            else:
                return []
            
            # Add metadata and return random sample
            for pattern in available_patterns:
                pattern["source_file"] = os.path.basename(sample_file)
                pattern["genre"] = genre
            
            return random.sample(available_patterns, min(n_patterns, len(available_patterns)))
            
        except Exception as e:
            self.logger.warning(f"Failed to extract patterns from {sample_file}: {e}")
            return []
    
    def search_patterns_by_instrument(self, instrument_name: str, genre: str = None, max_results: int = 10) -> List[Dict]:
        """
        Search for patterns containing a specific instrument
        
        Args:
            instrument_name: Name of instrument to search for
            genre: Optional genre filter
            max_results: Maximum number of results to return
            
        Returns:
            List of matching pattern dictionaries
        """
        matching_patterns = []
        
        # Filter files by genre if specified
        if genre and genre in self._genre_file_cache:
            search_files = self._genre_file_cache[genre]
        else:
            # Search all files
            search_files = []
            for genre_files in self._genre_file_cache.values():
                search_files.extend(genre_files)
        
        # Randomly sample files to search
        sample_files = random.sample(search_files, min(20, len(search_files)))
        
        for midi_file in sample_files:
            if len(matching_patterns) >= max_results:
                break
                
            try:
                patterns, _ = extract_patterns_from_midi(midi_file)
                
                if not patterns:
                    continue
                
                # Check segments for matching instruments
                for segment in patterns.get("segments", []):
                    for instrument in segment.get("instruments", []):
                        if instrument.get("name", "").lower() == instrument_name.lower():
                            segment["source_file"] = os.path.basename(midi_file)
                            segment["genre"] = detect_genre_from_path(midi_file)
                            matching_patterns.append(segment)
                            break
                    
                    if len(matching_patterns) >= max_results:
                        break
                        
            except Exception as e:
                self.logger.warning(f"Failed to search {midi_file}: {e}")
                continue
        
        return matching_patterns[:max_results]
