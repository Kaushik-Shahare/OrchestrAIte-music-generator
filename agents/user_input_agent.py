import logging
from typing import Any, Dict

def user_input_agent(state: Any) -> Any:
    """
    Parse user input parameters from the state and update state accordingly.
    """
    logging.info("[UserInputAgent] Parsing user input parameters.")
    # Extract and validate user input
    params = state.get('user_input', {})
    genre = params.get('genre', 'pop')
    mood = params.get('mood', 'happy')
    tempo = int(params.get('tempo', 120))
    duration = int(params.get('duration', 2))
    instruments = params.get('instruments', ['piano'])
    if isinstance(instruments, str):
        instruments = [i.strip() for i in instruments.split(',')]
    vocals = bool(params.get('vocals', False))
    logging.info(f"[UserInputAgent] genre={genre}, mood={mood}, tempo={tempo}, duration={duration}, instruments={instruments}, vocals={vocals}")
    state['genre'] = genre
    state['mood'] = mood
    state['tempo'] = tempo
    state['duration'] = duration
    state['instruments'] = instruments
    state['vocals'] = vocals
    return state 