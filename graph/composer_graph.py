import os
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

from agents.user_input_agent import user_input_agent
from agents.artist_context_agent import artist_context_agent
# Removed artist style agent as requested - focusing only on vector database
from agents.midi_reference_agent import midi_reference_agent
from agents.musical_director_agent import musical_director_agent
from agents.melody_agent import melody_agent
from agents.chord_agent import chord_agent
from agents.instrument_agent import instrument_agent
from agents.drum_agent import drum_agent
from agents.vocal_agent import vocal_agent
from agents.midi_synth_agent import midi_synth_agent
from agents.audio_renderer_agent import audio_renderer_agent
from agents.export_agent import export_agent

import logging

logging.basicConfig(level=logging.INFO)

def user_input_node(state: dict) -> dict:
    return user_input_agent(state)

def artist_context_node(state: dict) -> dict:
    return artist_context_agent(state)

def midi_reference_node(state: dict) -> dict:
    return midi_reference_agent(state)

def musical_director_node(state: dict) -> dict:
    return musical_director_agent(state)

def melody_node(state: dict) -> dict:
    return melody_agent(state)

def chord_node(state: dict) -> dict:
    return chord_agent(state)

def instrument_node(state: dict) -> dict:
    return instrument_agent(state)

def drum_node(state: dict) -> dict:
    return drum_agent(state)

def vocal_node(state: dict) -> dict:
    return vocal_agent(state)

def midi_synth_node(state: dict) -> dict:
    return midi_synth_agent(state)

def audio_renderer_node(state: dict) -> dict:
    return audio_renderer_agent(state)

def export_node(state: dict) -> dict:
    return export_agent(state)

def build_composer_app():
    graph = StateGraph(dict)
    graph.add_node("user_input", user_input_node)
    graph.add_node("artist_context", artist_context_node)
    graph.add_node("midi_reference", midi_reference_node)
    graph.add_node("musical_director", musical_director_node)
    graph.add_node("melody", melody_node)
    graph.add_node("chord", chord_node)
    graph.add_node("instrument", instrument_node)
    graph.add_node("drum", drum_node)
    graph.add_node("vocal", vocal_node)
    graph.add_node("midi_synth", midi_synth_node)
    graph.add_node("audio_renderer", audio_renderer_node)
    graph.add_node("export", export_node)

    graph.add_edge(START, "user_input")
    graph.add_edge("user_input", "artist_context")
    # Skip artist_style since we want to use vector database only
    graph.add_edge("artist_context", "midi_reference")
    graph.add_edge("midi_reference", "musical_director")
    graph.add_edge("musical_director", "melody")
    graph.add_edge("melody", "chord")
    graph.add_edge("chord", "instrument")
    
    # Check if drums are explicitly requested
    def instrument_branch(state):
        instruments = state.get("instruments", [])
        instruments_str = ','.join(instruments).lower() if isinstance(instruments, list) else str(instruments).lower()
        if "drum" in instruments_str or "percussion" in instruments_str:
            return "drum"
        else:
            if state.get("vocals", False):
                return "vocal"
            else:
                return "midi_synth"
                
    graph.add_conditional_edges("instrument", instrument_branch)

    # Add edges from drums when they're included
    graph.add_edge("drum", "vocal")
    graph.add_edge("drum", "midi_synth")
    graph.add_edge("vocal", "midi_synth")
    graph.add_edge("midi_synth", "audio_renderer")
    graph.add_edge("audio_renderer", "export")
    graph.add_edge("export", END)

    return graph.compile()

# For direct import
composer_app = build_composer_app()

composer_app.get_graph().print_ascii()
