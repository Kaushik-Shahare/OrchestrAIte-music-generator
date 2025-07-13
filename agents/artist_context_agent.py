import logging
import requests
import os
from utils.state_utils import validate_agent_return, safe_state_update

def fetch_genius_lyrics(artist, max_songs=3):
    """
    Fetch top song lyrics for the artist from Genius API.
    Returns a list of dicts: {title, lyrics}
    """
    api_key = os.environ.get('GENIUS_API_KEY')
    if not api_key:
        logging.warning('[ArtistContextAgent] GENIUS_API_KEY not set.')
        return []
    base_url = 'https://api.genius.com'
    headers = {'Authorization': f'Bearer {api_key}'}
    search_url = f'{base_url}/search'
    params = {'q': artist}
    try:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
        hits = response.json()['response']['hits']
        songs = []
        for hit in hits[:max_songs]:
            song_title = hit['result']['title']
            song_url = hit['result']['url']
            # Scrape lyrics from song_url (Genius API does not provide lyrics directly)
            lyrics = fetch_lyrics_from_url(song_url)
            songs.append({'title': song_title, 'lyrics': lyrics})
        return songs
    except Exception as e:
        logging.error(f'[ArtistContextAgent] Genius API error: {e}')
        return []

def fetch_lyrics_from_url(url):
    # Placeholder: In production, use BeautifulSoup to scrape lyrics from the Genius page
    # For now, just return the URL as a placeholder
    return f'[See lyrics at {url}]'

def artist_context_agent(state: dict) -> dict:
    artist = state.get('user_input', {}).get('artist', '')
    if not artist:
        logging.info('[ArtistContextAgent] No artist specified')
        return safe_state_update(state, {'artist_context': {}}, "ArtistContextAgent")
    
    logging.info(f'[ArtistContextAgent] Fetching context for artist: {artist}')
    lyrics_data = fetch_genius_lyrics(artist)
    # Optionally, add chord progressions, arrangement info from other sources
    artist_context_data = {
        'lyrics': lyrics_data,
        # 'chords': ...
        # 'arrangement': ...
    }
    logging.info(f'[ArtistContextAgent] Artist context fetched: {artist_context_data}')
    return safe_state_update(state, {'artist_context': artist_context_data}, "ArtistContextAgent") 