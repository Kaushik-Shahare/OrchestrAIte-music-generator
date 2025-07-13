#!/usr/bin/env python3
"""
State Management Utilities for LangGraph Compatibility

This module provides utilities to ensure consistent state handling across all agents
to prevent the INVALID_CONCURRENT_GRAPH_UPDATE error.
"""

import logging
import json
import copy
from typing import Dict, Any

def ensure_consistent_state(state: Dict, agent_name: str = "Unknown") -> Dict:
    """
    Ensure the state is consistent and serializable for LangGraph compatibility.
    
    This function creates a new state object instead of modifying the original,
    preventing concurrent update errors.
    
    Args:
        state: The input state dictionary
        agent_name: Name of the agent for logging purposes
        
    Returns:
        A new, clean state dictionary
    """
    if not isinstance(state, dict):
        logging.error(f"[{agent_name}] State is not a dictionary: {type(state)}")
        return {}
    
    # Create a brand new state dictionary
    try:
        # Use deep copy to ensure complete separation from original state
        new_state = copy.deepcopy(state)
        logging.debug(f"[{agent_name}] Created deep copy of state with keys: {list(new_state.keys())}")
    except Exception as e:
        logging.error(f"[{agent_name}] Error creating deep copy: {e}")
        # Fallback to manual copying
        new_state = {}
        for key, value in state.items():
            try:
                new_state[key] = copy.deepcopy(value)
            except Exception:
                # If deep copy fails, convert to string representation
                try:
                    new_state[key] = json.loads(json.dumps(value, default=str))
                except Exception:
                    new_state[key] = str(value)
    
    # Ensure essential fields exist
    essential_fields = {
        'genre': 'pop',
        'instruments': ['piano'],
        'artist': '',
        'tempo': 120,
        'key_signature': 'C Major',
        'mood': 'neutral',
        'duration': 2
    }
    
    for field, default_value in essential_fields.items():
        if field not in new_state or new_state[field] is None:
            new_state[field] = default_value
            logging.debug(f"[{agent_name}] Added missing field '{field}' with default value")
    
    # Validate serialization
    try:
        json.dumps(new_state, default=str)
    except Exception as e:
        logging.error(f"[{agent_name}] State not serializable: {e}")
        # Clean up non-serializable data
        new_state = _clean_non_serializable_data(new_state, agent_name)
    
    return new_state

def _clean_non_serializable_data(state: Dict, agent_name: str) -> Dict:
    """
    Clean non-serializable data from the state.
    
    Args:
        state: The state dictionary to clean
        agent_name: Name of the agent for logging
        
    Returns:
        A cleaned state dictionary
    """
    clean_state = {}
    
    for key, value in state.items():
        try:
            # Test if the value is serializable
            json.dumps(value)
            clean_state[key] = value
        except (TypeError, OverflowError):
            logging.warning(f"[{agent_name}] Cleaning non-serializable data for key '{key}'")
            
            if isinstance(value, (list, tuple)):
                # Clean lists/tuples
                clean_list = []
                for item in value:
                    try:
                        json.dumps(item)
                        clean_list.append(item)
                    except Exception:
                        clean_list.append(str(item))
                clean_state[key] = clean_list
            elif isinstance(value, dict):
                # Clean dictionaries recursively
                clean_dict = {}
                for k, v in value.items():
                    try:
                        json.dumps(v)
                        clean_dict[k] = v
                    except Exception:
                        clean_dict[k] = str(v)
                clean_state[key] = clean_dict
            else:
                # Convert other types to strings
                clean_state[key] = str(value)
    
    return clean_state

def validate_agent_return(state: Dict, agent_name: str) -> Dict:
    """
    Validate that an agent is returning a proper state for LangGraph.
    
    This should be called by every agent before returning state.
    
    Args:
        state: The state to validate
        agent_name: Name of the agent for logging
        
    Returns:
        A validated state dictionary
    """
    if not isinstance(state, dict):
        logging.error(f"[{agent_name}] Agent returning non-dict state: {type(state)}")
        return {
            'genre': 'pop',
            'instruments': ['piano'],
            'artist': '',
            'tempo': 120,
            'key_signature': 'C Major',
            'mood': 'neutral',
            'error': f'{agent_name} returned invalid state type'
        }
    
    # Ensure we have a clean, serializable state
    validated_state = ensure_consistent_state(state, agent_name)
    
    # Final validation
    try:
        json.dumps(validated_state, default=str)
        logging.debug(f"[{agent_name}] State validation passed")
        return validated_state
    except Exception as e:
        logging.error(f"[{agent_name}] Final state validation failed: {e}")
        # Return minimal valid state
        return {
            'genre': state.get('genre', 'pop'),
            'instruments': state.get('instruments', ['piano']),
            'artist': state.get('artist', ''),
            'tempo': state.get('tempo', 120),
            'key_signature': state.get('key_signature', 'C Major'),
            'mood': state.get('mood', 'neutral'),
            'error': f'{agent_name} state validation failed: {str(e)}'
        }

def safe_state_update(original_state: Dict, updates: Dict, agent_name: str) -> Dict:
    """
    Safely update state with new values without modifying the original.
    
    Args:
        original_state: The original state dictionary
        updates: Dictionary of updates to apply
        agent_name: Name of the agent for logging
        
    Returns:
        A new state dictionary with updates applied
    """
    # Start with a clean copy of the original state
    new_state = ensure_consistent_state(original_state, agent_name)
    
    # Apply updates
    for key, value in updates.items():
        try:
            # Test if the value is serializable
            json.dumps(value, default=str)
            new_state[key] = value
        except Exception as e:
            logging.warning(f"[{agent_name}] Non-serializable update for key '{key}': {e}")
            new_state[key] = str(value)
    
    return validate_agent_return(new_state, agent_name)
