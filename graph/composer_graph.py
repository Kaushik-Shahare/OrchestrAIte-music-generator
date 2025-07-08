import os
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

# Placeholder imports for agents (to be implemented in agents/)
from agents.user_input_agent import user_input_agent
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
    graph.add_node("melody", melody_node)
    graph.add_node("chord", chord_node)
    graph.add_node("instrument", instrument_node)
    graph.add_node("drum", drum_node)
    graph.add_node("vocal", vocal_node)
    graph.add_node("midi_synth", midi_synth_node)
    graph.add_node("audio_renderer", audio_renderer_node)
    graph.add_node("export", export_node)

    graph.add_edge(START, "user_input")
    graph.add_edge("user_input", "melody")
    graph.add_edge("melody", "chord")
    graph.add_edge("chord", "instrument")
    graph.add_edge("instrument", "drum")

    # Conditional: if vocals enabled, go to vocal, else midi_synth
    def drum_branch(state):
        return "vocal" if state.get("vocals", False) else "midi_synth"

    graph.add_conditional_edges("drum", drum_branch)
    graph.add_edge("vocal", "midi_synth")
    graph.add_edge("midi_synth", "audio_renderer")
    graph.add_edge("audio_renderer", "export")
    graph.add_edge("export", END)

    return graph.compile()

# For direct import
composer_app = build_composer_app() 

composer_app.get_graph().print_ascii()
