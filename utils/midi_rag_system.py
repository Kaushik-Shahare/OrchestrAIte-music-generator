#!/usr/bin/env python3
"""
MIDI RAG System - Vector Database for Musical Pattern Retrieval
Uses ChromaDB with Gemini embeddings to store and retrieve similar MIDI patterns
"""

import os
import json
import logging
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
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
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
            
            return " | ".join(description_parts)
            
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
            
            return " | ".join(description_parts)
            
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
            
            return " | ".join(description_parts)
            
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
        
        logging.info(f"[MIDIRAGSystem] Initialized with database at {self.db_path}")
    
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
            genres = ['rock', 'metal', 'jazz', 'pop', 'classical']
        
        logging.info(f"[MIDIRAGSystem] Starting indexing for genres: {genres}")
        
        total_patterns = 0
        
        for genre in genres:
            try:
                logging.info(f"[MIDIRAGSystem] Processing genre: {genre}")
                
                # Get patterns from Tegridy dataset
                patterns = self.tegridy_fetcher.get_patterns_for_artist_genre("", genre, max_files_per_genre)
                
                if not patterns:
                    logging.warning(f"[MIDIRAGSystem] No patterns found for genre: {genre}")
                    continue
                
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
                        
                    except Exception as e:
                        logging.error(f"[MIDIRAGSystem] Error indexing melody {i} for {genre}: {e}")
                
                logging.info(f"[MIDIRAGSystem] Completed indexing for {genre}: {len(segments)} segments, {len(progressions)} progressions, {len(melodies)} melodies")
                
            except Exception as e:
                logging.error(f"[MIDIRAGSystem] Error processing genre {genre}: {e}")
        
        logging.info(f"[MIDIRAGSystem] Indexing complete. Total patterns indexed: {total_patterns}")
        return total_patterns
    
    def retrieve_similar_patterns(self, query_genre: str, query_artist: str = "", 
                                 pattern_type: str = "segment", top_k: int = 5) -> List[Dict]:
        """
        Retrieve similar musical patterns based on genre and artist
        """
        try:
            # Construct query
            query_text = f"Genre: {query_genre}"
            if query_artist:
                query_text += f" | Artist style: {query_artist}"
            
            # Add pattern-specific context
            if pattern_type == "segment":
                query_text += " | Musical segment with instruments and rhythm patterns"
                collection = self.segments_collection
            elif pattern_type == "progression":
                query_text += " | Chord progression with harmonic movement"
                collection = self.progressions_collection
            elif pattern_type == "melody":
                query_text += " | Melody sequence with melodic intervals"
                collection = self.melodies_collection
            else:
                raise ValueError(f"Unknown pattern type: {pattern_type}")
            
            if not collection:
                logging.error(f"[MIDIRAGSystem] Collection for {pattern_type} not available")
                return []
            
            # Retrieve similar patterns
            results = collection.query(
                query_texts=[query_text],
                n_results=top_k
            )
            
            # Parse results
            retrieved_patterns = []
            
            if results['documents'] and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else []
                distances = results['distances'][0] if results['distances'] else []
                
                for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                    distance = distances[i] if i < len(distances) else 1.0
                    
                    pattern_data = {}
                    if 'pattern_data' in metadata:
                        try:
                            pattern_data = json.loads(metadata['pattern_data'])
                        except json.JSONDecodeError:
                            logging.warning(f"[MIDIRAGSystem] Could not parse pattern data for result {i}")
                    
                    retrieved_patterns.append({
                        'description': doc,
                        'metadata': metadata,
                        'pattern_data': pattern_data,
                        'similarity_score': 1.0 - distance,  # Convert distance to similarity
                        'genre': metadata.get('genre', ''),
                        'type': metadata.get('type', pattern_type)
                    })
            
            logging.info(f"[MIDIRAGSystem] Retrieved {len(retrieved_patterns)} {pattern_type} patterns for {query_genre}")
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
        
        # Index patterns from multiple genres
        genres = ['metal', 'rock', 'jazz', 'pop', 'classical']
        total_indexed = midi_rag.index_midi_patterns_from_tegridy(genres, max_files_per_genre=5)
        
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
