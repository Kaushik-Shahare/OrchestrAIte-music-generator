# OrchestrAIte: AI-Powered Multi-Agent Music Composer

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-brightgreen)](https://github.com/langchain-ai/langgraph)
[![RAG](https://img.shields.io/badge/RAG-Vector%20Database-orange)](https://github.com/chroma-core/chroma)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## OrchestrAIte

**OrchestrAIte** is an advanced, production-ready, multi-agent music composition system built with [LangGraph](https://github.com/langchain-ai/langgraph) and powered by RAG (Retrieval-Augmented Generation). It generates original, high-quality, multi-instrumental MIDI and MP3 music using a sophisticated pipeline of specialized AI agents that learn from real musical patterns.

---

## Key Features

### Multi-Agent Architecture
- **Specialized Agents**: Dedicated agents for melody, chords, instruments, drums, vocals, and audio rendering
- **LangGraph Orchestration**: Flexible, scalable workflow with conditional branching
- **Musical Intelligence**: Enhanced note correction and musical flow validation
- **Real-time Collaboration**: Agents share musical context and adapt to each other

### RAG-Powered Pattern Learning
- **Vector Database**: ChromaDB-powered storage of musical patterns from real MIDI files
- **Pattern Matching**: Intelligent retrieval of similar musical segments, chord progressions, and melodies
- **Authentic Generation**: Uses real musical data as reference for authentic genre-specific composition
- **Fallback System**: Multi-tier fallback ensures robust pattern availability

### Advanced Music Generation
- **Natural Language Input**: Generate music from text descriptions (e.g., "upbeat jazz with piano and drums")
- **Melody Coverage**: Automatic extension ensures melodies span full song duration
- **Musical Quality**: Preserves harmonic relationships and melodic contours
- **Genre Intelligence**: Deep understanding of musical styles and conventions

### Comprehensive Control
- **Genres**: Jazz, rock, pop, classical, blues, electronic, and more
- **Moods**: Happy, sad, energetic, mellow, dramatic, mysterious, etc.
- **Instruments**: Piano, guitar, bass, drums, violin, saxophone, and many more
- **AI Vocals**: Optional vocal generation with lyrics
- **Flexible Parameters**: Tempo, duration, key signature, complexity levels

---

## System Architecture

### Multi-Agent Workflow
```
User Input → Description Parser → MIDI Reference (RAG) → Artist Context → Musical Director
     ↓
Chord Agent → Melody Agent → Instrument Agent → Drum Agent → [Vocal Agent] → MIDI Synth → Audio Renderer → Export
```

### Agent Responsibilities
- **Description Parser**: Converts natural language to musical parameters
- **MIDI Reference (RAG)**: Retrieves similar patterns from vector database  
- **Artist Context**: Applies artist-specific styling and characteristics
- **Musical Director**: Creates overall vision and section arrangement
- **Chord Agent**: Generates harmonic progressions using RAG patterns
- **Melody Agent**: Creates melodic lines with automatic coverage extension
- **Instrument Agent**: Adds instrumental parts with musical intelligence
- **Drum Agent**: Builds rhythmic foundations
- **Vocal Agent**: Generates AI vocals with lyrics (optional)
- **Audio Pipeline**: MIDI synthesis and MP3 conversion

### RAG System
- **ChromaDB Vector Store**: Indexes musical patterns by genre, style, and characteristics
- **Pattern Types**: Segments, chord progressions, melodic patterns
- **Similarity Search**: Finds most relevant musical examples
- **Fallback Layers**: Multiple backup strategies ensure robust operation

---

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Kaushik-Shahare/OrchestrAIte-music-generator.git
cd OrchestrAIte-music-generator/music_generator

# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies
# macOS:
brew install fluidsynth ffmpeg

# Ubuntu/Debian:
sudo apt-get install fluidsynth ffmpeg

# Set up environment
cp .env.example .env
# Add your GEMINI_API_KEY to .env file
```

### 2. Generate Music

#### Basic Generation
```bash
python main.py --genre jazz --mood happy --tempo 120 --duration 2 --instruments piano,bass
```

#### Natural Language Generation
```bash
python main.py --description "Create an upbeat rock song with electric guitar and drums, energetic and driving"
```

#### Advanced Generation
```bash
python main.py \
  --genre blues \
  --mood soulful \
  --tempo 80 \
  --duration 3 \
  --instruments guitar,piano,bass,drums \
  --vocals true \
  --lyrics "My heart sings the blues tonight" \
  --artist "B.B. King" \
  --harmonic_complexity high \
  --rhythmic_complexity medium
```

### 3. Play Generated Music
```bash
# Play MIDI file
python play_midi.py output/song_YYYYMMDD_HHMMSS.mid

# Or play MP3 file
open output/song_YYYYMMDD_HHMMSS.mp3  # macOS
# or any audio player
```

---

## Example Output

### Generation Log
```
RAG SYSTEM DATA USAGE REPORT FOR GENRE: JAZZ
================================================================================
REQUEST DETAILS:
   - Requested Genre: jazz
   - Total Patterns Retrieved: 8
   - Is Fallback: False

RESPONSE BREAKDOWN:
   - Musical Segments: 3
   - Chord Progressions: 3  
   - Melody Patterns: 2

MUSICAL SEGMENTS BEING USED:
   Segment 1:
     - Similarity Score: 0.892
     - Source File: jazz_standards_collection.mid
     - Instruments: Piano, Bass

HOW THIS DATA WILL BE USED:
   - Agents will receive 8 reference patterns
   - Chord agent will use progression patterns for harmonic structure
   - Melody agent will use interval patterns and note sequences
================================================================================

MIDI generated: output/song_20250715_143022.mid
MP3 generated: output/song_20250715_143022.mp3

Music Generation Successful!
Generation Summary:
   Genre: jazz
   Mood: mellow
   Tempo: 100 BPM
   Duration: 2 minutes
   Instruments: piano, bass
   RAG Patterns Used: 8
```

---

## Advanced Features

### Natural Language Description
Generate music from free-form text descriptions:
```bash
python main.py --description "A haunting classical piece with violin and piano, mysterious and atmospheric, slow tempo"
```

### Artist Style Emulation
```bash
python main.py --genre rock --artist "Led Zeppelin" --mood energetic
```

### Complex Arrangements
```bash
python main.py \
  --song_structure "intro,verse,chorus,verse,chorus,bridge,chorus,outro" \
  --dynamic_range wide \
  --harmonic_complexity high
```

### RAG Pattern Analysis
View detailed information about which musical patterns are being used:
- Real-time logging shows RAG queries and responses
- Pattern similarity scores and source files
- How patterns influence each agent's generation

---

## Project Structure
```
music_generator/
├── agents/                 # AI Agents
│   ├── description_parser_agent.py     # Natural language processing
│   ├── midi_reference_agent.py         # RAG pattern retrieval  
│   ├── artist_context_agent.py         # Artist style application
│   ├── chord_agent.py                  # Harmonic generation
│   ├── melody_agent.py                 # Melodic composition
│   ├── instrument_agent.py             # Instrumental parts
│   ├── drum_agent.py                   # Rhythmic foundations
│   ├── vocal_agent.py                  # AI vocal generation
│   └── audio_renderer_agent.py         # Audio synthesis
├── graph/                  # LangGraph Workflow
│   └── composer_graph.py               # Multi-agent orchestration
├── models/                 # Music Models
│   ├── chord_model.py                  # Chord progression logic
│   ├── melody_model.py                 # Melodic structure
│   └── instrument_model.py             # Instrument-specific models
├── utils/                  # Utilities
│   ├── midi_rag_system.py              # Vector database system
│   ├── gemini_llm.py                   # LLM integration
│   ├── midi_utils.py                   # MIDI processing
│   ├── audio_utils.py                  # Audio conversion
│   └── lyrics_utils.py                 # Lyric generation
├── output/                 # Generated Files
├── chroma_midi_db/         # RAG Vector Database
├── main.py                 # CLI Interface
├── play_midi.py            # MIDI Playback
├── examples_description_generation.py  # Usage examples
└── requirements.txt        # Dependencies
```

---

## Technical Requirements

### System Dependencies
- **Python**: 3.9+ with pip
- **FluidSynth**: For MIDI synthesis (`brew install fluidsynth` or `apt-get install fluidsynth`)
- **FFmpeg**: For audio conversion (`brew install ffmpeg` or `apt-get install ffmpeg`)

### Python Dependencies  
- **LangGraph**: Multi-agent workflow orchestration
- **ChromaDB**: Vector database for RAG system
- **Google Generative AI**: Gemini LLM integration
- **pretty_midi**: MIDI file manipulation
- **pyfluidsynth**: Audio synthesis
- **soundfile**: Audio file I/O
- **numpy/scipy**: Numerical computing
- **pygame**: MIDI playback

### API Keys
- **GEMINI_API_KEY**: Required for AI generation (get from Google AI Studio)

---

## Use Cases

### Musicians & Composers
- **Inspiration**: Generate ideas and starting points for compositions
- **Collaboration**: Use AI as a creative partner in the composition process
- **Learning**: Analyze how different musical styles and patterns work
- **Rapid Prototyping**: Quickly test musical ideas and arrangements

### Developers & Researchers  
- **AI Research**: Explore multi-agent systems and RAG applications
- **Music Technology**: Build upon the modular architecture
- **Educational**: Learn about music theory through code
- **Integration**: Embed music generation into larger applications

### Content Creators
- **Background Music**: Generate royalty-free music for videos, podcasts, games
- **Mood Setting**: Create atmosphere-appropriate soundtracks
- **Customization**: Tailor music to specific scenes or emotions
- **Rapid Production**: Generate multiple variations quickly

---

## Customization & Extension

### Adding New Genres
1. Update genre patterns in `midi_reference_agent.py`
2. Add genre-specific logic in individual agents
3. Expand RAG database with new musical examples

### Creating Custom Agents
1. Inherit from base agent structure
2. Implement agent-specific logic
3. Add to LangGraph workflow in `composer_graph.py`

### Extending RAG System
1. Add new pattern types to `midi_rag_system.py`
2. Expand vector database with additional musical data
3. Enhance similarity search algorithms

---

## Troubleshooting

### Common Issues

**No audio output / Silent MP3**
- Verify FluidSynth installation
- Check audio system permissions
- Ensure SoundFont files are available

**RAG system not working**
- Set GEMINI_API_KEY in environment
- Check ChromaDB database exists in `chroma_midi_db/`
- Verify network connectivity for API calls

**Low melody coverage warnings**
- System automatically extends melodies
- Check tempo and duration parameters
- Review LLM generation logs

**Import errors**
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.9+)

---

## Contributing

We welcome contributions! Areas of interest:
- **New musical genres and styles**
- **Enhanced RAG patterns and algorithms**  
- **Additional agent capabilities**
- **Performance optimizations**
- **Documentation improvements**
- **Bug fixes and testing**

### Development Setup
```bash
git clone https://github.com/Kaushik-Shahare/OrchestrAIte-music-generator.git
cd OrchestrAIte-music-generator
pip install -r requirements.txt
# Set up pre-commit hooks for code quality
```

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

### Technologies
- **[LangGraph](https://github.com/langchain-ai/langgraph)**: Multi-agent workflow framework
- **[ChromaDB](https://github.com/chroma-core/chroma)**: Vector database for RAG
- **[Google Gemini](https://ai.google.dev/)**: Large language model
- **[pretty_midi](https://github.com/craffel/pretty-midi)**: MIDI manipulation
- **[FluidSynth](https://www.fluidsynth.org/)**: Software synthesizer

### Inspiration
- Modern AI music generation research
- Multi-agent system architectures  
- Retrieval-augmented generation applications
- Open-source music technology community

---

## Get Started Today!

Create your first AI-generated song in minutes:

```bash
git clone https://github.com/Kaushik-Shahare/OrchestrAIte-music-generator.git
cd OrchestrAIte-music-generator/music_generator
pip install -r requirements.txt
python main.py --description "Create a cheerful pop song with piano and light percussion"
```

**OrchestrAIte**: Where artificial intelligence meets musical creativity! 