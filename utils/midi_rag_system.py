#!/usr/bin/env python3
"""
MIDI RAG System - Vector Database for Musical Pattern Retrieval
Uses ChromaDB with Gemini embeddings to store and retrieve similar MIDI patterns
"""

import os
import json
import logging
import copy
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
import pretty_midi

from utils.tegridy_midi_fetcher import TegridyMIDIFetcher

# Load environment variables
load_dotenv()

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

class MIDIEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for MIDI patterns using Gemini
    """
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=self.api_key)
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Convert MIDI pattern descriptions to embeddings
        """
        try:
            embeddings = []
            
            for text in input:
                # Get embedding from Gemini
                response = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document",
                    title="MIDI Musical Pattern"
                )
                
                embedding = response["embedding"]
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logging.error(f"[MIDIEmbeddingFunction] Error creating embeddings: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 768 for _ in input]

class MIDIPatternVectorizer:
    """
    Converts MIDI musical patterns into text descriptions for vectorization
    """
    
    @staticmethod
    def vectorize_musical_segment(segment: Dict, genre: str, artist: str = "") -> str:
        """
        Convert a musical segment into a descriptive text for embedding
        """
        try:
            description_parts = []
            
            # Add genre and artist context
            description_parts.append(f"Genre: {genre}")
            if artist:
                description_parts.append(f"Artist style: {artist}")
            
            # Segment duration and structure
            duration = segment.get('duration', 16.0)
            description_parts.append(f"Duration: {duration:.1f} seconds")
            
            # Analyze instruments in segment
            instruments = segment.get('instruments', [])
            if instruments:
                inst_names = [inst.get('name', 'Unknown') for inst in instruments]
                description_parts.append(f"Instruments: {', '.join(inst_names)}")
                
                # Analyze each instrument's pattern
                for inst in instruments:
                    inst_name = inst.get('name', 'Unknown')
                    note_sequence = inst.get('note_sequence', [])
                    melodic_intervals = inst.get('melodic_intervals', [])
                    rhythm_pattern = inst.get('rhythm_pattern', [])
                    
                    if note_sequence:
                        # Analyze pitch range
                        pitches = [note[0] for note in note_sequence if len(note) >= 1]
                        if pitches:
                            pitch_min, pitch_max = min(pitches), max(pitches)
                            description_parts.append(
                                f"{inst_name} pitch range: {pitch_min}-{pitch_max}"
                            )
                    
                    if melodic_intervals:
                        # Describe melodic movement
                        intervals_list = [float(x) for x in melodic_intervals[:10]]  # Convert to float
                        avg_interval = np.mean([abs(x) for x in intervals_list])
                        direction = "ascending" if np.mean(intervals_list) > 0 else "descending"
                        description_parts.append(
                            f"{inst_name} melodic movement: {direction}, average interval {avg_interval:.1f} semitones"
                        )
                    
                    if rhythm_pattern:
                        # Describe rhythm characteristics
                        rhythm_list = [float(x) for x in rhythm_pattern[:10]]  # Convert to float
                        avg_duration = np.mean(rhythm_list)
                        rhythm_type = "fast" if avg_duration < 0.25 else "moderate" if avg_duration < 1.0 else "slow"
                        description_parts.append(
                            f"{inst_name} rhythm: {rhythm_type} notes, average duration {avg_duration:.2f}s"
                        )
            
            description_text = " | ".join(description_parts)
            
            # Add the serialized pattern data to the end for complete retrieval
            try:
                pattern_data_json = json.dumps(segment)
                description_text += f"\n\n==PATTERN_DATA=={pattern_data_json}"
            except Exception as e:
                logging.warning(f"[MIDIPatternVectorizer] Could not serialize pattern data: {e}")
            
            return description_text
            
        except Exception as e:
            logging.error(f"[MIDIPatternVectorizer] Error vectorizing segment: {e}")
            return f"Genre: {genre} | Artist: {artist} | Musical segment"
    
    @staticmethod
    def vectorize_chord_progression(progression: Dict, genre: str, artist: str = "") -> str:
        """
        Convert chord progression into descriptive text
        """
        try:
            description_parts = []
            description_parts.append(f"Genre: {genre}")
            if artist:
                description_parts.append(f"Artist style: {artist}")
            
            chords = progression.get('chords', [])
            if chords:
                # Analyze harmonic movement
                roots = [chord.get('root', 0) for chord in chords]
                chord_names = []
                
                for i, chord in enumerate(chords):
                    root = chord.get('root', 0)
                    pitches = chord.get('pitches', [])
                    
                    # Simple chord recognition
                    if len(pitches) >= 3:
                        intervals = sorted([p % 12 for p in pitches])
                        if intervals == [0, 4, 7]:
                            chord_names.append(f"Major_{root}")
                        elif intervals == [0, 3, 7]:
                            chord_names.append(f"Minor_{root}")
                        else:
                            chord_names.append(f"Complex_{root}")
                    else:
                        chord_names.append(f"Root_{root}")
                
                description_parts.append(f"Chord progression: {' -> '.join(chord_names[:8])}")
                
                # Analyze harmonic rhythm
                durations = [float(chord.get('duration', 2.0)) for chord in chords]  # Convert to float
                avg_duration = np.mean(durations)
                description_parts.append(f"Harmonic rhythm: {avg_duration:.1f}s per chord")
            
            description_text = " | ".join(description_parts)
            
            # Add the serialized pattern data to the end for complete retrieval
            try:
                pattern_data_json = json.dumps(progression)
                description_text += f"\n\n==PATTERN_DATA=={pattern_data_json}"
            except Exception as e:
                logging.warning(f"[MIDIPatternVectorizer] Could not serialize progression data: {e}")
            
            return description_text
            
        except Exception as e:
            logging.error(f"[MIDIPatternVectorizer] Error vectorizing progression: {e}")
            return f"Genre: {genre} | Artist: {artist} | Chord progression"
    
    @staticmethod
    def vectorize_melody_sequence(melody: Dict, genre: str, artist: str = "") -> str:
        """
        Convert melody sequence into descriptive text
        """
        try:
            description_parts = []
            description_parts.append(f"Genre: {genre}")
            if artist:
                description_parts.append(f"Artist style: {artist}")
            
            intervals = melody.get('intervals', [])
            instrument = melody.get('instrument', 'Unknown')
            start_pitch = melody.get('start_pitch', 60)
            
            description_parts.append(f"Instrument: {instrument}")
            description_parts.append(f"Starting pitch: {start_pitch}")
            
            if intervals:
                # Analyze melodic characteristics - convert to float to avoid numpy issues
                intervals_float = [float(x) for x in intervals]
                avg_interval = np.mean([abs(x) for x in intervals_float])
                max_interval = max([abs(x) for x in intervals_float])
                direction_changes = sum(1 for i in range(1, len(intervals_float)) 
                                      if intervals_float[i] * intervals_float[i-1] < 0)
                
                description_parts.append(f"Average interval: {avg_interval:.1f} semitones")
                description_parts.append(f"Maximum leap: {max_interval} semitones")
                description_parts.append(f"Direction changes: {direction_changes}")
                
                # Characterize melody type
                if avg_interval < 1.5:
                    melody_type = "stepwise"
                elif avg_interval < 3.0:
                    melody_type = "moderate leaps"
                else:
                    melody_type = "large leaps"
                
                description_parts.append(f"Melodic style: {melody_type}")
            
            description_text = " | ".join(description_parts)
            
            # Add the serialized pattern data to the end for complete retrieval
            try:
                pattern_data_json = json.dumps(melody)
                description_text += f"\n\n==PATTERN_DATA=={pattern_data_json}"
            except Exception as e:
                logging.warning(f"[MIDIPatternVectorizer] Could not serialize melody data: {e}")
            
            return description_text
            
        except Exception as e:
            logging.error(f"[MIDIPatternVectorizer] Error vectorizing melody: {e}")
            return f"Genre: {genre} | Artist: {artist} | Melody sequence"

class MIDIRAGSystem:
    """
    Main RAG system for MIDI pattern storage and retrieval
    """
    
    def __init__(self, db_path: str = "./chroma_midi_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        self.embedding_function = MIDIEmbeddingFunction()
        self.vectorizer = MIDIPatternVectorizer()
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Collections for different pattern types
        self.segments_collection = self._get_or_create_collection("midi_segments")
        self.progressions_collection = self._get_or_create_collection("midi_chord_progressions")
        self.melodies_collection = self._get_or_create_collection("midi_melodies")
        
        self.tegridy_fetcher = TegridyMIDIFetcher()
        
        # Initialize synthetic pattern templates
        self._init_synthetic_patterns()
        
        logging.info(f"[MIDIRAGSystem] Initialized with database at {self.db_path}")
        
    def _init_synthetic_patterns(self):
        """
        Initialize templates for synthetic pattern generation when no real data is available
        """
        # Basic templates for different genres
        self.synthetic_templates = {
            # Pop music patterns
            "pop": {
                "segments": [
                    {
                        "instruments": [
                            {
                                "name": "piano",
                                "note_sequence": [[60, 0.0, 0.5, 80], [64, 0.5, 1.0, 75], [67, 1.0, 1.5, 70], [64, 1.5, 2.0, 75]],
                                "melodic_intervals": [4, 3, -3, 4, -4, -3],
                                "rhythm_pattern": [0.5, 0.5, 0.5, 0.5, 0.25, 0.25]
                            },
                            {
                                "name": "bass",
                                "note_sequence": [[48, 0.0, 1.0, 90], [48, 1.0, 2.0, 85], [55, 2.0, 3.0, 90], [55, 3.0, 4.0, 85]],
                                "melodic_intervals": [0, 7, 0, -7],
                                "rhythm_pattern": [1.0, 1.0, 1.0, 1.0]
                            }
                        ],
                        "duration": 4.0
                    }
                ],
                "chord_progressions": [
                    {
                        "chords": [
                            {"root": 60, "pitches": [60, 64, 67], "duration": 1.0},
                            {"root": 67, "pitches": [67, 71, 74], "duration": 1.0},
                            {"root": 65, "pitches": [65, 69, 72], "duration": 1.0},
                            {"root": 62, "pitches": [62, 65, 69], "duration": 1.0}
                        ],
                        "instrument": "piano"
                    }
                ],
                "melody_sequences": [
                    {
                        "intervals": [0, 2, 2, 1, 2, 2, 2, 1, -1, -2, -2, -2, -1, -2],
                        "start_pitch": 60,
                        "instrument": "piano"
                    }
                ]
            },
            # Rock music patterns
            "rock": {
                "segments": [
                    {
                        "instruments": [
                            {
                                "name": "electric_guitar",
                                "note_sequence": [[40, 0.0, 0.5, 100], [40, 0.5, 1.0, 95], [47, 1.0, 1.5, 100], [47, 1.5, 2.0, 95]],
                                "melodic_intervals": [0, 7, 0, -7, 5, -5],
                                "rhythm_pattern": [0.25, 0.25, 0.25, 0.25, 0.5, 0.5]
                            },
                            {
                                "name": "bass",
                                "note_sequence": [[28, 0.0, 1.0, 95], [28, 1.0, 2.0, 90], [35, 2.0, 3.0, 95], [35, 3.0, 4.0, 90]],
                                "melodic_intervals": [0, 7, 0, -7, 5, -5],
                                "rhythm_pattern": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
                            }
                        ],
                        "duration": 4.0
                    }
                ],
                "chord_progressions": [
                    {
                        "chords": [
                            {"root": 40, "pitches": [40, 47, 52], "duration": 2.0},
                            {"root": 45, "pitches": [45, 52, 57], "duration": 2.0},
                            {"root": 47, "pitches": [47, 54, 59], "duration": 2.0},
                            {"root": 38, "pitches": [38, 45, 50], "duration": 2.0}
                        ],
                        "instrument": "guitar"
                    }
                ],
                "melody_sequences": [
                    {
                        "intervals": [0, 3, 2, 0, 0, 2, 3, 0, -3, -2, 0, 0, -2, -3],
                        "start_pitch": 64,
                        "instrument": "guitar"
                    }
                ]
            },
            # Classical music patterns
            "classical": {
                "segments": [
                    {
                        "instruments": [
                            {
                                "name": "piano",
                                "note_sequence": [[60, 0.0, 0.5, 80], [64, 0.5, 1.0, 75], [67, 1.0, 1.5, 85], [72, 1.5, 2.0, 80]],
                                "melodic_intervals": [4, 3, 5, -5, -3, -4],
                                "rhythm_pattern": [0.5, 0.5, 0.5, 0.5, 0.75, 0.25]
                            },
                            {
                                "name": "violin",
                                "note_sequence": [[76, 0.0, 1.0, 75], [79, 1.0, 2.0, 80], [77, 2.0, 3.0, 75]],
                                "melodic_intervals": [3, -2, 4, -3, 2, -4],
                                "rhythm_pattern": [1.0, 1.0, 1.0, 0.5, 0.5, 1.0]
                            }
                        ],
                        "duration": 4.0
                    }
                ],
                "chord_progressions": [
                    {
                        "chords": [
                            {"root": 60, "pitches": [60, 64, 67], "duration": 2.0},
                            {"root": 67, "pitches": [67, 71, 74], "duration": 2.0},
                            {"root": 65, "pitches": [65, 69, 72], "duration": 2.0},
                            {"root": 62, "pitches": [62, 65, 69], "duration": 2.0}
                        ],
                        "instrument": "piano"
                    }
                ],
                "melody_sequences": [
                    {
                        "intervals": [0, 2, 2, 1, 0, -1, -2, -2, 2, 2, 1, 2, 2, 2],
                        "start_pitch": 67,
                        "instrument": "violin"
                    }
                ]
            },
            # Jazz music patterns
            "jazz": {
                "segments": [
                    {
                        "instruments": [
                            {
                                "name": "piano",
                                "note_sequence": [[60, 0.0, 0.5, 70], [63, 0.5, 1.0, 75], [67, 1.0, 1.5, 70], [70, 1.5, 2.0, 80]],
                                "melodic_intervals": [3, 4, 3, -3, -4, -3],
                                "rhythm_pattern": [0.5, 0.5, 0.5, 0.5, 0.75, 0.25]
                            },
                            {
                                "name": "bass",
                                "note_sequence": [[43, 0.0, 0.5, 90], [45, 1.0, 1.5, 85], [48, 2.0, 2.5, 90], [50, 3.0, 3.5, 85]],
                                "melodic_intervals": [2, 3, 2, -2, -3, -2],
                                "rhythm_pattern": [0.5, 0.5, 0.5, 0.5, 1.0, 0.5]
                            }
                        ],
                        "duration": 4.0
                    }
                ],
                "chord_progressions": [
                    {
                        "chords": [
                            {"root": 60, "pitches": [60, 64, 67, 71], "duration": 2.0},
                            {"root": 55, "pitches": [55, 59, 62, 65], "duration": 2.0},
                            {"root": 50, "pitches": [50, 54, 57, 60], "duration": 2.0},
                            {"root": 57, "pitches": [57, 61, 64, 67], "duration": 2.0}
                        ],
                        "instrument": "piano"
                    }
                ],
                "melody_sequences": [
                    {
                        "intervals": [0, 3, 1, 1, -1, -1, 4, -3, -1, -1, -2, 3, 2, -5],
                        "start_pitch": 65,
                        "instrument": "saxophone"
                    }
                ]
            },
            # Default for any other genre
            "default": {
                "segments": [
                    {
                        "instruments": [
                            {
                                "name": "piano",
                                "note_sequence": [[60, 0.0, 0.5, 80], [64, 0.5, 1.0, 75], [67, 1.0, 1.5, 70], [64, 1.5, 2.0, 75]],
                                "melodic_intervals": [4, 3, -3, 4, -4, -3],
                                "rhythm_pattern": [0.5, 0.5, 0.5, 0.5, 0.25, 0.25]
                            }
                        ],
                        "duration": 4.0
                    }
                ],
                "chord_progressions": [
                    {
                        "chords": [
                            {"root": 60, "pitches": [60, 64, 67], "duration": 1.0},
                            {"root": 67, "pitches": [67, 71, 74], "duration": 1.0},
                            {"root": 65, "pitches": [65, 69, 72], "duration": 1.0},
                            {"root": 62, "pitches": [62, 65, 69], "duration": 1.0}
                        ],
                        "instrument": "piano"
                    }
                ],
                "melody_sequences": [
                    {
                        "intervals": [0, 2, 2, 1, 2, 2, 2, -1, -2, -2, -2, -1, -2],
                        "start_pitch": 60,
                        "instrument": "piano"
                    }
                ]
            }
        }
    
    def _get_random_patterns_for_adaptation(self, target_genre: str) -> Dict:
        """
        Get random patterns from existing collections to adapt for a genre that has no specific patterns.
        This ensures we always use real patterns from the database rather than synthetic ones.
        """
        try:
            logging.info(f"[MIDIRAGSystem] Getting random patterns to adapt for genre: {target_genre}")
            
            patterns = {
                "musical_segments": [],
                "chord_progressions": [],
                "melody_sequences": []
            }
            
            # Helper function to get random patterns from a collection
            def get_random_patterns_from_collection(collection, pattern_key, count=2):
                if not collection:
                    logging.warning(f"[MIDIRAGSystem] Collection not available for {pattern_key}")
                    return []
                
                try:
                    # Get collection size
                    collection_size = collection.count()
                    if collection_size == 0:
                        logging.warning(f"[MIDIRAGSystem] No patterns in collection for {pattern_key}")
                        return []
                    
                    # Use a generic query to get all patterns
                    generic_query = "Musical pattern"
                    results = collection.query(
                        query_texts=[generic_query],
                        n_results=min(collection_size, count*3)  # Get more to ensure we have enough valid patterns
                    )
                    
                    if not results['documents'] or len(results['documents']) == 0 or len(results['documents'][0]) == 0:
                        logging.warning(f"[MIDIRAGSystem] No results for generic query in {pattern_key} collection")
                        return []
                    
                    # Get documents and metadata
                    documents = results['documents'][0]
                    metadatas = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0 else []
                    
                    # Randomize selection
                    import random
                    if len(documents) > count:
                        indices = random.sample(range(len(documents)), count)
                    else:
                        indices = list(range(len(documents)))
                    
                    # Extract patterns
                    extracted_patterns = []
                    for idx in indices:
                        doc = documents[idx] if idx < len(documents) else ""
                        metadata = metadatas[idx] if idx < len(metadatas) else {}
                        
                        pattern_data = {}
                        
                        # First try to extract from metadata
                        if metadata and 'pattern_data' in metadata:
                            try:
                                if isinstance(metadata['pattern_data'], str):
                                    pattern_data = json.loads(metadata['pattern_data'])
                                else:
                                    pattern_data = metadata['pattern_data']
                            except json.JSONDecodeError:
                                logging.warning(f"[MIDIRAGSystem] Failed to parse pattern data from metadata in {pattern_key}")
                        
                        # If no data from metadata, try document
                        if not pattern_data and '==PATTERN_DATA==' in doc:
                            try:
                                pattern_data_str = doc.split('==PATTERN_DATA==')[1].strip()
                                pattern_data = json.loads(pattern_data_str)
                            except (IndexError, json.JSONDecodeError):
                                logging.warning(f"[MIDIRAGSystem] Failed to extract pattern data from document in {pattern_key}")
                        
                        # If we have valid data, adapt it for the target genre
                        if pattern_data:
                            # Get original genre for reference
                            original_genre = metadata.get('genre', 'unknown')
                            
                            # Adapt the pattern for the target genre
                            pattern_data['genre'] = target_genre
                            if 'description' in pattern_data:
                                pattern_data['description'] = f"{target_genre} style based on {original_genre} pattern"
                            
                            extracted_patterns.append(pattern_data)
                    
                    return extracted_patterns
                    
                except Exception as e:
                    logging.error(f"[MIDIRAGSystem] Error getting random patterns from {pattern_key} collection: {str(e)}")
                    return []
            
            # Get random patterns from each collection
            patterns['musical_segments'] = get_random_patterns_from_collection(
                self.segments_collection, "musical_segments")
            patterns['chord_progressions'] = get_random_patterns_from_collection(
                self.progressions_collection, "chord_progressions")
            patterns['melody_sequences'] = get_random_patterns_from_collection(
                self.melodies_collection, "melody_sequences")
            
            # Check if we found any patterns
            total_patterns = sum(len(patterns[k]) for k in patterns.keys())
            
            if total_patterns > 0:
                logging.info(f"[MIDIRAGSystem] Successfully adapted {total_patterns} random patterns for genre: {target_genre}")
                return patterns
            else:
                logging.warning(f"[MIDIRAGSystem] No patterns available to adapt for genre: {target_genre}")
                return None
                
        except Exception as e:
            logging.error(f"[MIDIRAGSystem] Error adapting random patterns: {str(e)}")
            return None
    
    def _generate_synthetic_patterns(self, genre: str) -> Dict:
        """
        Generate synthetic patterns for a genre when no real patterns are available
        """
        logging.info(f"[MIDIRAGSystem] Generating synthetic patterns for genre: {genre}")
        
        # Get templates for this genre or default
        genre_lower = genre.lower()
        if genre_lower not in self.synthetic_templates:
            genre_lower = "default"
            logging.info(f"[MIDIRAGSystem] No synthetic templates for {genre}, using default")
        
        templates = self.synthetic_templates[genre_lower]
        
        # Create patterns dictionary
        patterns = {
            "musical_segments": [],
            "chord_progressions": [],
            "melody_sequences": []
        }
        
        # Add segments with slight variations
        for template in templates["segments"]:
            # Create variations by tweaking velocity and timing
            for i in range(3):  # Generate 3 variations
                segment = copy.deepcopy(template)
                
                # Add slight variations to each instrument
                for inst in segment["instruments"]:
                    # Vary note velocities
                    for note in inst["note_sequence"]:
                        if len(note) > 3:  # Make sure note has velocity component
                            note[3] = max(30, min(127, note[3] + random.randint(-10, 10)))
                
                patterns["musical_segments"].append(segment)
        
        # Add chord progressions
        for template in templates["chord_progressions"]:
            patterns["chord_progressions"].append(copy.deepcopy(template))
            
        # Add melody sequences
        for template in templates["melody_sequences"]:
            patterns["melody_sequences"].append(copy.deepcopy(template))
            
        return patterns
    
    def _get_or_create_collection(self, name: str):
        """
        Get or create a ChromaDB collection
        """
        try:
            return self.chroma_client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            logging.error(f"[MIDIRAGSystem] Error creating collection {name}: {e}")
            return None
    
    def index_midi_patterns_from_tegridy(self, genres: List[str] = None, max_files_per_genre: int = 10):
        """
        Download and index MIDI patterns from Tegridy dataset
        """
        if genres is None:
            genres = ['classical', 'jazz', 'pop', 'rock', 'metal']
        
        logging.info(f"[MIDIRAGSystem] Starting indexing for genres: {genres}")
        
        total_patterns = 0
        successful_genres = []
        failed_genres = []
        
        for genre in genres:
            try:
                logging.info(f"[MIDIRAGSystem] Processing genre: {genre}")
                
                # Get patterns from Tegridy dataset
                patterns = self.tegridy_fetcher.get_patterns_for_artist_genre("", genre, max_files_per_genre)
                
                if not patterns:
                    logging.warning(f"[MIDIRAGSystem] No patterns found for genre: {genre} (genre may not exist in dataset)")
                    
                    # Try to get existing patterns from other genres to adapt
                    adapted_patterns = self._get_random_patterns_for_adaptation(genre)
                    
                    if adapted_patterns:
                        patterns = adapted_patterns
                        logging.info(f"[MIDIRAGSystem] Using adapted patterns from other genres for: {genre}")
                    else:
                        # If no patterns can be adapted, fall back to synthetic ones
                        patterns = self._generate_synthetic_patterns(genre)
                        if not patterns:
                            failed_genres.append(genre)
                            continue
                        else:
                            logging.info(f"[MIDIRAGSystem] Using synthetic patterns for genre: {genre}")
                else:
                    logging.info(f"[MIDIRAGSystem] Found {sum(len(patterns.get(k, [])) for k in ['musical_segments', 'chord_progressions', 'melody_sequences'])} patterns from Tegridy dataset for genre: {genre}")
                
                genre_pattern_count = 0
                
                # Index musical segments
                segments = patterns.get('musical_segments', [])
                for i, segment in enumerate(segments):
                    try:
                        text_repr = self.vectorizer.vectorize_musical_segment(segment, genre)
                        segment_id = f"{genre}_segment_{i}_{total_patterns}"
                        
                        # Convert numpy types before JSON serialization
                        clean_segment = convert_numpy_types(segment)
                        
                        self.segments_collection.add(
                            ids=[segment_id],
                            documents=[text_repr],
                            metadatas=[{
                                'genre': genre,
                                'type': 'segment',
                                'duration': float(segment.get('duration', 16.0)),
                                'instruments': len(segment.get('instruments', [])),
                                'pattern_data': json.dumps(clean_segment)
                            }]
                        )
                        total_patterns += 1
                        
                    except Exception as e:
                        logging.error(f"[MIDIRAGSystem] Error indexing segment {i} for {genre}: {e}")
                
                # Index chord progressions
                progressions = patterns.get('chord_progressions', [])
                for i, progression in enumerate(progressions):
                    try:
                        text_repr = self.vectorizer.vectorize_chord_progression(progression, genre)
                        prog_id = f"{genre}_progression_{i}_{total_patterns}"
                        
                        # Convert numpy types before JSON serialization
                        clean_progression = convert_numpy_types(progression)
                        
                        self.progressions_collection.add(
                            ids=[prog_id],
                            documents=[text_repr],
                            metadatas=[{
                                'genre': genre,
                                'type': 'progression',
                                'instrument': progression.get('instrument', 'Unknown'),
                                'pattern_data': json.dumps(clean_progression)
                            }]
                        )
                        total_patterns += 1
                        
                    except Exception as e:
                        logging.error(f"[MIDIRAGSystem] Error indexing progression {i} for {genre}: {e}")
                
                # Index melodies from melody sequences
                melodies = patterns.get('melody_sequences', [])
                for i, melody in enumerate(melodies):
                    try:
                        text_repr = self.vectorizer.vectorize_melody_sequence(melody, genre)
                        melody_id = f"{genre}_melody_{i}_{total_patterns}"
                        
                        # Convert numpy types before JSON serialization
                        clean_melody = convert_numpy_types(melody)
                        
                        self.melodies_collection.add(
                            ids=[melody_id],
                            documents=[text_repr],
                            metadatas=[{
                                'genre': genre,
                                'type': 'melody',
                                'instrument': melody.get('instrument', 'Unknown'),
                                'pattern_data': json.dumps(clean_melody)
                            }]
                        )
                        total_patterns += 1
                        genre_pattern_count += 1
                        
                    except Exception as e:
                        logging.error(f"[MIDIRAGSystem] Error indexing melody {i} for {genre}: {e}")
                
                if genre_pattern_count > 0:
                    successful_genres.append(genre)
                    logging.info(f"[MIDIRAGSystem] Successfully indexed {genre_pattern_count} patterns for {genre}")
                else:
                    failed_genres.append(genre)
                    logging.info(f"[MIDIRAGSystem] No valid patterns indexed for {genre}")
                
            except Exception as e:
                logging.error(f"[MIDIRAGSystem] Error processing genre {genre}: {e}")
                failed_genres.append(genre)
        
        logging.info(f"[MIDIRAGSystem] Indexing complete. Total patterns indexed: {total_patterns}")
        logging.info(f"[MIDIRAGSystem] Successful genres: {successful_genres}")
        if failed_genres:
            logging.info(f"[MIDIRAGSystem] Genres with no patterns: {failed_genres}")
        return total_patterns
    
    def retrieve_similar_patterns(self, query_genre: str, query_artist: str = "", 
                                 pattern_type: str = "segment", top_k: int = 5) -> List[Dict]:
        """
        Retrieve similar musical patterns based on genre and artist
        """
        try:
            logging.info(f"[MIDIRAGSystem] DEBUG: retrieve_similar_patterns called with genre='{query_genre}', artist='{query_artist}', type='{pattern_type}'")
            
            # Construct query to match the format used in vectorization functions
            query_parts = []
            query_parts.append(f"Genre: {query_genre}")
            if query_artist:
                query_parts.append(f"Artist style: {query_artist}")
            
            # Add pattern-specific context with musical characteristics
            if pattern_type == "segment":
                query_parts.append("Musical segment with instruments and rhythm patterns")
                # Add some common musical segment descriptors that might appear in embeddings
                if query_genre.lower() in ['rock', 'metal']:
                    query_parts.append("Strong rhythm, guitar, drums, medium to high intensity")
                elif query_genre.lower() in ['jazz']:
                    query_parts.append("Complex harmonies, swing rhythm, improvisation")
                elif query_genre.lower() in ['classical']:
                    query_parts.append("Orchestra, piano, violin, structured composition")
                elif query_genre.lower() in ['electronic', 'edm']:
                    query_parts.append("Synthesizers, repetitive beats, electronic sounds")
                collection = self.segments_collection
            elif pattern_type == "progression":
                query_parts.append("Chord progression with harmonic movement")
                # Add common chord progression characteristics
                if query_genre.lower() in ['pop', 'rock']:
                    query_parts.append("Major chords, standard cadences, 4-chord patterns")
                elif query_genre.lower() in ['jazz']:
                    query_parts.append("Complex chords, sevenths, ninths, jazz harmonies")
                collection = self.progressions_collection
            elif pattern_type == "melody":
                query_parts.append("Melody sequence with melodic intervals")
                # Add common melodic characteristics
                if query_genre.lower() in ['pop']:
                    query_parts.append("Catchy, stepwise movement, moderate range")
                elif query_genre.lower() in ['jazz']:
                    query_parts.append("Large leaps, complex patterns, chromatic elements")
                collection = self.melodies_collection
            else:
                raise ValueError(f"Unknown pattern type: {pattern_type}")
                
            # Join all parts with the same separator as in the vectorization
            query_text = " | ".join(query_parts)
            
            logging.info(f"[MIDIRAGSystem] DEBUG: Enhanced query text: '{query_text}'")
            
            if not collection:
                logging.error(f"[MIDIRAGSystem] Collection for {pattern_type} not available")
                return []
            
            # Retrieve similar patterns with more results to ensure we get valid patterns
            results = collection.query(
                query_texts=[query_text],
                n_results=top_k * 2  # Get more results in case some are malformed
            )
            
            # Parse results
            retrieved_patterns = []
            
            valid_patterns = 0
            # Parse and process results
            documents = []
            metadatas = []
            distances = []
            
            if results['documents'] and len(results['documents']) > 0 and len(results['documents'][0]) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0 else []
                distances = results['distances'][0] if 'distances' in results and results['distances'] and len(results['distances']) > 0 else []
                
                logging.info(f"[MIDIRAGSystem] Retrieved {len(documents)} documents for query: '{query_text[:50]}...'")
                if documents and len(documents) > 0:
                    logging.info(f"[MIDIRAGSystem] First document excerpt: '{documents[0][:100]}...'")
                if metadatas and len(metadatas) > 0:
                    logging.info(f"[MIDIRAGSystem] First metadata: {metadatas[0]}")
            else:
                logging.warning(f"[MIDIRAGSystem] No results found for query: '{query_text[:50]}...'")
            
            # Process the results to extract pattern data
            valid_patterns = 0
            for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                distance = distances[i] if i < len(distances) else 1.0
                
                pattern_data = {}
                pattern_data_source = "none"
                
                # First try to get pattern data from metadata
                if 'pattern_data' in metadata:
                    try:
                        if isinstance(metadata['pattern_data'], str):
                            pattern_data = json.loads(metadata['pattern_data'])
                            pattern_data_source = "metadata"
                        else:
                            pattern_data = metadata['pattern_data']
                            pattern_data_source = "metadata"
                    except json.JSONDecodeError:
                        logging.warning(f"[MIDIRAGSystem] Could not parse pattern data from metadata for result {i}")
                
                # If metadata parsing failed, try to extract from document text
                if not pattern_data and '==PATTERN_DATA==' in doc:
                    try:
                        pattern_data_str = doc.split('==PATTERN_DATA==')[1].strip()
                        pattern_data = json.loads(pattern_data_str)
                        pattern_data_source = "document"
                        logging.info(f"[MIDIRAGSystem] Successfully extracted pattern data from document text")
                    except (IndexError, json.JSONDecodeError) as e:
                        logging.warning(f"[MIDIRAGSystem] Failed to extract pattern data from document: {e}")
                
                # Skip patterns with no data
                if not pattern_data:
                    logging.warning(f"[MIDIRAGSystem] No pattern data found for result {i}, skipping")
                    continue
                    
                # Validate pattern data structure based on pattern type
                is_valid = False
                if pattern_type == "segment" and isinstance(pattern_data, dict) and "instruments" in pattern_data:
                    is_valid = True
                elif pattern_type == "progression" and isinstance(pattern_data, dict) and "chords" in pattern_data:
                    is_valid = True
                elif pattern_type == "melody" and isinstance(pattern_data, dict) and "intervals" in pattern_data:
                    is_valid = True
                    
                if not is_valid:
                    logging.warning(f"[MIDIRAGSystem] Invalid pattern data structure for {pattern_type} in result {i}, skipping")
                    continue
                    
                valid_patterns += 1
                
                # Add the pattern if it's valid (which it is if we got here)
                retrieved_patterns.append({
                    'description': doc.split('==PATTERN_DATA==')[0] if '==PATTERN_DATA==' in doc else doc,
                    'metadata': metadata,
                    'pattern_data': pattern_data,
                    'similarity_score': 1.0 - distance,  # Convert distance to similarity
                    'genre': metadata.get('genre', query_genre),  # Fallback to query genre if not in metadata
                    'type': metadata.get('type', pattern_type),
                    'data_source': pattern_data_source  # Add source for debugging
                })
            
            logging.info(f"[MIDIRAGSystem] Retrieved {len(retrieved_patterns)} valid {pattern_type} patterns for {query_genre} (from {valid_patterns} total)")
            
            # Trim to the requested number if we have more valid patterns than requested
            if len(retrieved_patterns) > top_k:
                retrieved_patterns = retrieved_patterns[:top_k]
                logging.info(f"[MIDIRAGSystem] Trimmed results to requested {top_k} patterns")
            
            # Add debug info about data sources
            from_metadata = sum(1 for p in retrieved_patterns if p.get('data_source') == 'metadata')
            from_document = sum(1 for p in retrieved_patterns if p.get('data_source') == 'document')
            logging.info(f"[MIDIRAGSystem] Pattern data sources: {from_metadata} from metadata, {from_document} from document text")
            
            # If we still have no patterns, use random samples from the database as a fallback
            if not retrieved_patterns:
                logging.warning(f"[MIDIRAGSystem] No valid patterns found for {query_genre}, fetching random samples from database")
                
                random_patterns = []
                try:
                    # Try to get random patterns from each collection
                    if pattern_type == "segment" and self.segments_collection:
                        # Get collection size
                        collection_size = self.segments_collection.count()
                        if collection_size > 0:
                            # Get random samples (up to top_k)
                            sample_size = min(top_k, collection_size)
                            # Use a generic query that will match most items
                            generic_results = self.segments_collection.query(
                                query_texts=["Musical segment"],
                                n_results=collection_size  # Get all available
                            )
                            
                            if generic_results['documents'] and len(generic_results['documents']) > 0 and len(generic_results['documents'][0]) > 0:
                                # Get the documents and metadatas
                                all_docs = generic_results['documents'][0]
                                all_metadatas = generic_results['metadatas'][0] if 'metadatas' in generic_results and generic_results['metadatas'] and len(generic_results['metadatas']) > 0 else []
                                
                                # Select random indices
                                if len(all_docs) > sample_size:
                                    indices = random.sample(range(len(all_docs)), sample_size)
                                else:
                                    indices = range(len(all_docs))
                                
                                # Process the random samples
                                for idx in indices:
                                    doc = all_docs[idx]
                                    metadata = all_metadatas[idx] if idx < len(all_metadatas) else {}
                                    
                                    pattern_data = {}
                                    pattern_data_source = "none"
                                    
                                    # Try to get pattern data from metadata
                                    if 'pattern_data' in metadata:
                                        try:
                                            if isinstance(metadata['pattern_data'], str):
                                                pattern_data = json.loads(metadata['pattern_data'])
                                                pattern_data_source = "metadata"
                                            else:
                                                pattern_data = metadata['pattern_data']
                                                pattern_data_source = "metadata"
                                        except json.JSONDecodeError:
                                            pass
                                    
                                    # If metadata parsing failed, try to extract from document text
                                    if not pattern_data and '==PATTERN_DATA==' in doc:
                                        try:
                                            pattern_data_str = doc.split('==PATTERN_DATA==')[1].strip()
                                            pattern_data = json.loads(pattern_data_str)
                                            pattern_data_source = "document"
                                        except (IndexError, json.JSONDecodeError):
                                            pass
                                    
                                    # Add the pattern if we got data
                                    if pattern_data:
                                        source_genre = metadata.get('genre', 'unknown')
                                        random_patterns.append({
                                            'description': f"Adapted {source_genre} segment for {query_genre}",
                                            'metadata': metadata,
                                            'pattern_data': pattern_data,
                                            'similarity_score': 0.5,  # Lower score to indicate it's not a perfect match
                                            'genre': query_genre,  # Override with requested genre
                                            'type': pattern_type,
                                            'data_source': pattern_data_source,
                                            'original_genre': source_genre
                                        })
                    
                    # Similar approach for progressions
                    elif pattern_type == "progression" and self.progressions_collection:
                        collection_size = self.progressions_collection.count()
                        if collection_size > 0:
                            sample_size = min(top_k, collection_size)
                            generic_results = self.progressions_collection.query(
                                query_texts=["Chord progression"],
                                n_results=collection_size
                            )
                            
                            if generic_results['documents'] and len(generic_results['documents']) > 0 and len(generic_results['documents'][0]) > 0:
                                all_docs = generic_results['documents'][0]
                                all_metadatas = generic_results['metadatas'][0] if 'metadatas' in generic_results and generic_results['metadatas'] and len(generic_results['metadatas']) > 0 else []
                                
                                if len(all_docs) > sample_size:
                                    indices = random.sample(range(len(all_docs)), sample_size)
                                else:
                                    indices = range(len(all_docs))
                                
                                for idx in indices:
                                    doc = all_docs[idx]
                                    metadata = all_metadatas[idx] if idx < len(all_metadatas) else {}
                                    
                                    pattern_data = {}
                                    pattern_data_source = "none"
                                    
                                    if 'pattern_data' in metadata:
                                        try:
                                            if isinstance(metadata['pattern_data'], str):
                                                pattern_data = json.loads(metadata['pattern_data'])
                                                pattern_data_source = "metadata"
                                            else:
                                                pattern_data = metadata['pattern_data']
                                                pattern_data_source = "metadata"
                                        except json.JSONDecodeError:
                                            pass
                                    
                                    if not pattern_data and '==PATTERN_DATA==' in doc:
                                        try:
                                            pattern_data_str = doc.split('==PATTERN_DATA==')[1].strip()
                                            pattern_data = json.loads(pattern_data_str)
                                            pattern_data_source = "document"
                                        except (IndexError, json.JSONDecodeError):
                                            pass
                                    
                                    if pattern_data:
                                        source_genre = metadata.get('genre', 'unknown')
                                        random_patterns.append({
                                            'description': f"Adapted {source_genre} progression for {query_genre}",
                                            'metadata': metadata,
                                            'pattern_data': pattern_data,
                                            'similarity_score': 0.5,
                                            'genre': query_genre,
                                            'type': pattern_type,
                                            'data_source': pattern_data_source,
                                            'original_genre': source_genre
                                        })
                    
                    # And for melodies
                    elif pattern_type == "melody" and self.melodies_collection:
                        collection_size = self.melodies_collection.count()
                        if collection_size > 0:
                            sample_size = min(top_k, collection_size)
                            generic_results = self.melodies_collection.query(
                                query_texts=["Melody sequence"],
                                n_results=collection_size
                            )
                            
                            if generic_results['documents'] and len(generic_results['documents']) > 0 and len(generic_results['documents'][0]) > 0:
                                all_docs = generic_results['documents'][0]
                                all_metadatas = generic_results['metadatas'][0] if 'metadatas' in generic_results and generic_results['metadatas'] and len(generic_results['metadatas']) > 0 else []
                                
                                if len(all_docs) > sample_size:
                                    indices = random.sample(range(len(all_docs)), sample_size)
                                else:
                                    indices = range(len(all_docs))
                                
                                for idx in indices:
                                    doc = all_docs[idx]
                                    metadata = all_metadatas[idx] if idx < len(all_metadatas) else {}
                                    
                                    pattern_data = {}
                                    pattern_data_source = "none"
                                    
                                    if 'pattern_data' in metadata:
                                        try:
                                            if isinstance(metadata['pattern_data'], str):
                                                pattern_data = json.loads(metadata['pattern_data'])
                                                pattern_data_source = "metadata"
                                            else:
                                                pattern_data = metadata['pattern_data']
                                                pattern_data_source = "metadata"
                                        except json.JSONDecodeError:
                                            pass
                                    
                                    if not pattern_data and '==PATTERN_DATA==' in doc:
                                        try:
                                            pattern_data_str = doc.split('==PATTERN_DATA==')[1].strip()
                                            pattern_data = json.loads(pattern_data_str)
                                            pattern_data_source = "document"
                                        except (IndexError, json.JSONDecodeError):
                                            pass
                                    
                                    if pattern_data:
                                        source_genre = metadata.get('genre', 'unknown')
                                        random_patterns.append({
                                            'description': f"Adapted {source_genre} melody for {query_genre}",
                                            'metadata': metadata,
                                            'pattern_data': pattern_data,
                                            'similarity_score': 0.5,
                                            'genre': query_genre,
                                            'type': pattern_type,
                                            'data_source': pattern_data_source,
                                            'original_genre': source_genre
                                        })
                
                except Exception as e:
                    logging.error(f"[MIDIRAGSystem] Error retrieving random patterns: {e}")
                
                # Add the random patterns to the retrieved patterns
                if random_patterns:
                    retrieved_patterns = random_patterns
                    logging.info(f"[MIDIRAGSystem] Added {len(random_patterns)} random patterns from database adapted for {query_genre}")
                else:
                    # Only use synthetic patterns if we couldn't get any random patterns from the database
                    logging.warning(f"[MIDIRAGSystem] No random patterns available, falling back to synthetic patterns")
                    
                    # Generate synthetic patterns
                    synth_patterns = self._generate_synthetic_patterns(query_genre)
                    
                    if pattern_type == "segment" and synth_patterns.get("musical_segments"):
                        for i, segment in enumerate(synth_patterns["musical_segments"]):
                            if i >= top_k:
                                break
                            retrieved_patterns.append({
                                'description': f"Synthetic {query_genre} segment",
                                'metadata': {'genre': query_genre, 'type': 'segment', 'synthetic': True},
                                'pattern_data': segment,
                                'similarity_score': 0.3,  # Even lower score for synthetic
                                'genre': query_genre,
                                'type': pattern_type,
                                'data_source': 'synthetic'
                            })
                    elif pattern_type == "progression" and synth_patterns.get("chord_progressions"):
                        for i, prog in enumerate(synth_patterns["chord_progressions"]):
                            if i >= top_k:
                                break
                            retrieved_patterns.append({
                                'description': f"Synthetic {query_genre} chord progression",
                                'metadata': {'genre': query_genre, 'type': 'progression', 'synthetic': True},
                                'pattern_data': prog,
                                'similarity_score': 0.3,
                                'genre': query_genre,
                                'type': pattern_type,
                                'data_source': 'synthetic'
                            })
                    elif pattern_type == "melody" and synth_patterns.get("melody_sequences"):
                        for i, melody in enumerate(synth_patterns["melody_sequences"]):
                            if i >= top_k:
                                break
                            retrieved_patterns.append({
                                'description': f"Synthetic {query_genre} melody",
                                'metadata': {'genre': query_genre, 'type': 'melody', 'synthetic': True},
                                'pattern_data': melody,
                                'similarity_score': 0.3,
                                'genre': query_genre,
                                'type': pattern_type,
                                'data_source': 'synthetic'
                            })
                    
                    if retrieved_patterns:
                        logging.info(f"[MIDIRAGSystem] Added {len(retrieved_patterns)} synthetic patterns for {query_genre}")
                    else:
                        logging.error(f"[MIDIRAGSystem] Failed to generate any patterns for {query_genre}")
            
            return retrieved_patterns
            
        except Exception as e:
            logging.error(f"[MIDIRAGSystem] Error retrieving patterns: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get statistics about indexed patterns
        """
        stats = {}
        
        try:
            if self.segments_collection:
                stats['segments'] = self.segments_collection.count()
            if self.progressions_collection:
                stats['progressions'] = self.progressions_collection.count()
            if self.melodies_collection:
                stats['melodies'] = self.melodies_collection.count()
            
            stats['total'] = sum(stats.values())
            
        except Exception as e:
            logging.error(f"[MIDIRAGSystem] Error getting stats: {e}")
            stats = {'error': str(e)}
        
        return stats

# Global instance
midi_rag = MIDIRAGSystem()

def initialize_midi_rag_database(force_reindex: bool = False):
    """
    Initialize the MIDI RAG database with patterns from Tegridy dataset
    """
    try:
        # Check if database already has patterns
        stats = midi_rag.get_collection_stats()
        total_patterns = stats.get('total', 0)
        
        if total_patterns > 0 and not force_reindex:
            logging.info(f"[MIDIRAGSystem] Database already contains {total_patterns} patterns. Use force_reindex=True to rebuild.")
            return stats
        
        logging.info("[MIDIRAGSystem] Initializing MIDI pattern database...")
        
        # Index patterns from multiple genres - prioritize common genres first
        genres = ['classical', 'jazz', 'pop', 'rock', 'metal', 'folk', 'electronic', 'country', 'blues']
        # Try to index more files to ensure we have data
        logging.info("[MIDIRAGSystem] Indexing MIDI patterns from genres: " + str(genres))
        total_indexed = midi_rag.index_midi_patterns_from_tegridy(genres, max_files_per_genre=10)
        
        final_stats = midi_rag.get_collection_stats()
        logging.info(f"[MIDIRAGSystem] Database initialization complete: {final_stats}")
        
        return final_stats
        
    except Exception as e:
        logging.error(f"[MIDIRAGSystem] Error initializing database: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Test the RAG system
    logging.basicConfig(level=logging.INFO)
    
    # Initialize database
    print("Initializing MIDI RAG database...")
    stats = initialize_midi_rag_database()
    print(f"Database stats: {stats}")
    
    # Test retrieval
    print("\nTesting pattern retrieval...")
    
    # Test segments for metal
    metal_segments = midi_rag.retrieve_similar_patterns("heavy metal", "Metallica", "segment", top_k=3)
    print(f"Found {len(metal_segments)} metal segments")
    for i, pattern in enumerate(metal_segments):
        print(f"  {i+1}. {pattern['description'][:100]}...")
        print(f"     Similarity: {pattern['similarity_score']:.3f}")
    
    # Test progressions for jazz
    jazz_progressions = midi_rag.retrieve_similar_patterns("jazz", "Miles Davis", "progression", top_k=3)
    print(f"\nFound {len(jazz_progressions)} jazz progressions")
    for i, pattern in enumerate(jazz_progressions):
        print(f"  {i+1}. {pattern['description'][:100]}...")
        print(f"     Similarity: {pattern['similarity_score']:.3f}")
