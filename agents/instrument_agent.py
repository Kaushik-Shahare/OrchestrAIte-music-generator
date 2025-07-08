import logging
from typing import Any
from models.instrument_model import InstrumentModel

def instrument_agent(state: Any) -> Any:
    logging.info("[InstrumentAgent] Layering instruments.")
    try:
        model = InstrumentModel()
        instrument_data = model.add_instruments(
            melody=state.get('melody'),
            chords=state.get('chords'),
            instruments=state.get('instruments')
        )
        # instrument_data should be a list of track dicts
        tracks = instrument_data.get('tracks', [])
        state['instrument_tracks'] = {'instrument_tracks': tracks}
        logging.info("[InstrumentAgent] Instrument tracks added.")
    except Exception as e:
        logging.error(f"[InstrumentAgent] Error: {e}")
    return state 