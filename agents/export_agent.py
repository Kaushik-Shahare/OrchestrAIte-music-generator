import logging
from typing import Any
import os
import json
from utils.state_utils import validate_agent_return, safe_state_update

def export_agent(state: Any) -> Any:
    """
    Export agent that finalizes the generation process and ensures all
    important state data is preserved in the result.
    """
    logging.info("[ExportAgent] Exporting generated files and preserving RAG patterns")
    
    try:
        # Log all keys in the state at the export stage
        logging.info(f"[ExportAgent] Final state keys: {list(state.keys())}")
        
        # Get output paths
        midi_path = state.get('midi_path')
        mp3_path = state.get('mp3_path')
        
        # Log output file paths
        if midi_path and os.path.exists(midi_path):
            logging.info(f"[ExportAgent] MIDI generated: {midi_path}")
            print(f"\n🎵 MIDI generated: {midi_path}")
        else:
            logging.warning("[ExportAgent] MIDI file missing.")
            
        if mp3_path and os.path.exists(mp3_path):
            logging.info(f"[ExportAgent] MP3 generated: {mp3_path}")
            print(f"🎶 MP3 generated: {mp3_path}\n")
        else:
            logging.warning("[ExportAgent] MP3 file missing.")
        
        # Prepare result dictionary with all important data
        result = {
            'midi': midi_path,
            'mp3': mp3_path,
            'output_file': midi_path or mp3_path
        }
        
        # Preserve RAG patterns in the result
        if 'rag_patterns' in state:
            rag_patterns = state['rag_patterns']
            logging.info(f"[ExportAgent] Preserving RAG patterns in result: {rag_patterns['metadata'] if isinstance(rag_patterns, dict) and 'metadata' in rag_patterns else 'unknown structure'}")
            result['rag_patterns'] = rag_patterns
            
            # Count patterns by type for logging
            if isinstance(rag_patterns, dict):
                segments_count = len(rag_patterns.get('segments', []))
                progressions_count = len(rag_patterns.get('progressions', []))
                melodies_count = len(rag_patterns.get('melodies', []))
                total_patterns = segments_count + progressions_count + melodies_count
                
                logging.info(f"[ExportAgent] RAG pattern counts: segments={segments_count}, "
                           f"progressions={progressions_count}, melodies={melodies_count}, total={total_patterns}")
        else:
            logging.warning("[ExportAgent] No RAG patterns found in state!")
            
        # Preserve other important generation data
        for key in ['genre', 'instruments', 'tempo', 'mood', 'duration', 'artist_patterns', 'musical_vision']:
            if key in state:
                result[key] = state[key]
        
        # Prepare export data
        export_data = {'result': result}
        
        # Also store output_file at top level for compatibility
        if midi_path:
            export_data['output_file'] = midi_path
            
        return safe_state_update(state, export_data, "ExportAgent")
            
    except Exception as e:
        logging.error(f"[ExportAgent] Error: {e}")
        return safe_state_update(state, {'result': {'error': str(e)}}, "ExportAgent")