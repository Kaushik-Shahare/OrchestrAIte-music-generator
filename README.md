# OrchestrAIte: LangGraph Multi-Agent Music Composer

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-brightgreen)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## OrchestrAIte

**OrchestrAIte** is a modular, production-ready, multi-agent music composition system built with [LangGraph](https://github.com/langchain-ai/langgraph). It generates original, multi-instrumental MIDI and MP3 music using a pipeline of specialized agents for melody, chords, instrumentation, drums, vocals, and audio rendering.

---

## Features
- Multi-agent architecture: Each music aspect handled by a dedicated agent (melody, chords, drums, etc.)
- LangGraph orchestration: Flexible, scalable, and easy to extend
- Genre, mood, tempo, and instrument control
- Optional AI-generated vocals
- MIDI and MP3 output
- CLI for easy use
- Modern, clean, and modular Python codebase

---

## Architecture

```
User Input → Melody → Chord → Instrument → Drum → [Vocal if enabled] → MIDI Synth → Audio Renderer → Export
```
- Each step is a LangGraph node/agent
- State is passed as a Python dict
- Conditional branching for vocals

---

## Quickstart

### 1. Install requirements
```bash
pip install -r music_generator/requirements.txt
# Also install system dependencies:
# macOS: brew install fluidsynth ffmpeg
# Ubuntu: sudo apt-get install fluidsynth ffmpeg
```

### 2. Generate music
From the project root:
```bash
python music_generator/main.py --genre jazz --mood happy --tempo 120 --duration 2 --instruments piano,guitar --vocals false
```

### 3. Play the MIDI file
```bash
python music_generator/play_midi.py output/song_YYYYMMDD_HHMMSS.mid
```
Or open the MP3 in any audio player.

---

## Example Output
```
MIDI generated: output/song_YYYYMMDD_HHMMSS.mid
MP3 generated: output/song_YYYYMMDD_HHMMSS.mp3
```

---

## Project Structure
```
music_generator/
├── agents/           # All agent logic (melody, chord, drum, etc.)
├── graph/            # LangGraph workflow definition
├── models/           # Music model logic
├── utils/            # MIDI/audio utilities
├── output/           # Generated music files
├── main.py           # CLI entrypoint
├── play_midi.py      # Script to play MIDI files
└── requirements.txt  # Python dependencies
```

---

## Requirements
- Python 3.9+
- [LangGraph](https://github.com/langchain-ai/langgraph)
- pretty_midi, mido, pyfluidsynth, soundfile, ffmpeg-python, huggingface_hub, pygame
- System: fluidsynth, ffmpeg

---

## Credits
- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- MIDI/audio: [pretty_midi](https://github.com/craffel/pretty-midi), [pyfluidsynth](https://github.com/nwhitehead/pyfluidsynth), [ffmpeg](https://ffmpeg.org/)
- Inspired by open-source music AI and agent systems

---

## License
MIT License

---

## Inspiration
OrchestrAIte is designed for musicians, AI researchers, and developers who want to explore the intersection of generative AI, music, and multi-agent systems. Compose, experiment, and extend! 