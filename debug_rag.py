#!/usr/bin/env python3
"""
Debug script for MIDI RAG System random pattern adaptation
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import the MIDI RAG system
from utils.midi_rag_system import midi_rag

# Test retrieving patterns for a non-existent genre
genre = "funk"
print(f"\nTesting pattern retrieval for genre: {genre}\n")

# Get segments
segments = midi_rag.retrieve_similar_patterns(genre, "", "segment", top_k=1)
print(f"Segments found: {len(segments)}")

# Get progressions
progressions = midi_rag.retrieve_similar_patterns(genre, "", "progression", top_k=1)
print(f"Progressions found: {len(progressions)}")

# Get melodies
melodies = midi_rag.retrieve_similar_patterns(genre, "", "melody", top_k=1)
print(f"Melodies found: {len(melodies)}")

# Print database stats
stats = midi_rag.get_collection_stats()
print(f"\nDatabase stats: {stats}")
