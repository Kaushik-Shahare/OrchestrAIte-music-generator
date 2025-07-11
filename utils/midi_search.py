import logging
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, quote
import os
from typing import List, Dict, Optional
import pretty_midi
import numpy as np
from utils.tegridy_midi_fetcher import fetch_artist_midi_patterns

def search_and_analyze_artist_midi(artist_name: str, song_name: str = "") -> Dict:
    """
    Primary function to search for and analyze MIDI patterns from an artist.
    First tries Tegridy dataset, then falls back to web search.
    """
    logging.info(f"[MidiSearch] Searching for {artist_name} patterns...")
    
    try:
        # First, try to get patterns from Tegridy dataset
        genre = determine_genre_from_artist(artist_name)
        tegridy_patterns = fetch_artist_midi_patterns(artist_name, genre)
        
        if tegridy_patterns and tegridy_patterns.get('files_analyzed', 0) > 0:
            logging.info(f"[MidiSearch] Found Tegridy patterns for {artist_name}")
            return tegridy_patterns
        
        # Fallback to web search
        logging.info(f"[MidiSearch] No Tegridy patterns found, trying web search...")
        web_results = search_midi_files(artist_name, song_name)
        
        if web_results:
            # Try to download and analyze first result
            first_result = web_results[0]
            if 'url' in first_result:
                analysis = download_and_analyze_midi(first_result['url'])
                if analysis:
                    return analysis
        
        return {}
        
    except Exception as e:
        logging.error(f"[MidiSearch] Error: {e}")
        return {}

def determine_genre_from_artist(artist_name: str) -> str:
    """Determine genre from artist name for Tegridy dataset selection."""
    artist_lower = artist_name.lower()
    
    metal_rock = ['metallica', 'iron maiden', 'sabbath', 'megadeth', 'slayer', 'pantera',
                  'in flames', 'linkin park', 'korn', 'disturbed', 'tool', 'acdc']
    jazz = ['miles davis', 'coltrane', 'evans', 'parker', 'ellington', 'basie', 'monk']
    classical = ['bach', 'mozart', 'beethoven', 'chopin', 'liszt', 'debussy']
    
    for artist in metal_rock:
        if artist in artist_lower:
            return 'rock'
    for artist in jazz:
        if artist in artist_lower:
            return 'jazz'  
    for artist in classical:
        if artist in artist_lower:
            return 'classical'
    
    return 'pop'

def download_and_analyze_midi(url: str) -> Dict:
    """Download and analyze a MIDI file from URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Save to temp file and analyze
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            f.write(response.content)
            temp_path = f.name
        
        try:
            midi_data = pretty_midi.PrettyMIDI(temp_path)
            analysis = analyze_midi_structure(midi_data)
            return analysis
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        logging.error(f"[MidiSearch] Failed to download/analyze {url}: {e}")
        return {}

def analyze_midi_structure(midi_data) -> Dict:
    """Analyze MIDI structure and return patterns."""
    try:
        tempo_times, tempos = midi_data.get_tempo_changes()
        
        patterns = {
            'source': 'web_search',
            'chord_progressions': [],
            'rhythm_patterns': [],
            'typical_tempos': [float(np.mean(tempos))] if len(tempos) > 0 else [120],
            'pitch_ranges': {},
            'files_analyzed': 1
        }
        
        # Analyze each instrument
        for instrument in midi_data.instruments:
            if instrument.is_drum or not instrument.notes:
                continue
                
            inst_name = pretty_midi.program_to_instrument_name(instrument.program)
            pitches = [note.pitch for note in instrument.notes]
            
            patterns['pitch_ranges'][inst_name] = {
                'min': int(min(pitches)),
                'max': int(max(pitches)),
                'common': int(np.median(pitches))
            }
            
            # Basic rhythm analysis
            if len(instrument.notes) > 1:
                intervals = [instrument.notes[i+1].start - instrument.notes[i].start 
                           for i in range(min(10, len(instrument.notes)-1))]
                if intervals:
                    patterns['rhythm_patterns'].append({
                        'instrument': inst_name,
                        'common_interval': float(np.median(intervals)),
                        'note_density': len(instrument.notes) / midi_data.get_end_time()
                    })
        
        return patterns
        
    except Exception as e:
        logging.error(f"[MidiSearch] MIDI analysis failed: {e}")
        return {}

def search_midi_files(artist_name: str, song_name: str = "") -> List[Dict]:
    """
    Search for MIDI files from various online sources.
    Returns a list of potential MIDI file URLs and metadata.
    """
    logging.info(f"[MidiSearcher] Searching for {artist_name} MIDI files...")
    
    midi_results = []
    
    # Search query
    query = f"{artist_name} {song_name} MIDI".strip()
    
    try:
        # Search in MIDI World
        midi_world_results = search_midi_world(query)
        midi_results.extend(midi_world_results)
        
        # Search in FreeMidi.org
        freemidi_results = search_freemidi(query)
        midi_results.extend(freemidi_results)
        
        # Search general web for MIDI files
        web_results = search_web_midi(query)
        midi_results.extend(web_results)
        
        logging.info(f"[MidiSearcher] Found {len(midi_results)} potential MIDI files")
        return midi_results
        
    except Exception as e:
        logging.error(f"[MidiSearcher] Error searching: {e}")
        return []

def search_midi_world(query: str) -> List[Dict]:
    """Search MidiWorld.com for MIDI files."""
    results = []
    try:
        search_url = f"https://www.midiworld.com/search/?q={quote(query)}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for MIDI file links
            midi_links = soup.find_all('a', href=re.compile(r'\.mid$|\.midi$'))
            
            for link in midi_links:
                midi_url = urljoin(search_url, link.get('href'))
                title = link.get_text(strip=True) or "Unknown"
                
                results.append({
                    'url': midi_url,
                    'title': title,
                    'source': 'midiworld',
                    'type': 'midi'
                })
                
                if len(results) >= 5:  # Limit results
                    break
                    
    except Exception as e:
        logging.error(f"[MidiSearcher] Error searching MidiWorld: {e}")
    
    return results

def search_freemidi(query: str) -> List[Dict]:
    """Search FreeMidi.org for MIDI files."""
    results = []
    try:
        search_url = f"https://freemidi.org/search?q={quote(query)}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for download links
            download_links = soup.find_all('a', href=re.compile(r'download.*\.mid|\.midi'))
            
            for link in download_links:
                midi_url = urljoin(search_url, link.get('href'))
                title = link.get_text(strip=True) or "Unknown"
                
                results.append({
                    'url': midi_url,
                    'title': title,
                    'source': 'freemidi',
                    'type': 'midi'
                })
                
                if len(results) >= 5:
                    break
                    
    except Exception as e:
        logging.error(f"[MidiSearcher] Error searching FreeMidi: {e}")
    
    return results

def search_web_midi(query: str) -> List[Dict]:
    """Search general web for MIDI files using common patterns."""
    results = []
    
    # Common MIDI hosting sites
    sites = [
        "midiworld.com",
        "freemidi.org", 
        "8notes.com",
        "piano-midi.de",
        "classicalarchives.com"
    ]
    
    for site in sites:
        try:
            search_query = f"site:{site} {query} filetype:mid OR filetype:midi"
            
            # This would normally use a search API, but for demo we'll use direct URLs
            # In production, you'd use Google Custom Search API or similar
            
            # For now, create some example URLs based on common patterns
            potential_urls = [
                f"https://{site}/download/{query.replace(' ', '_').lower()}.mid",
                f"https://{site}/midi/{query.replace(' ', '-').lower()}.mid",
                f"https://{site}/files/{query.replace(' ', '_').lower()}.midi"
            ]
            
            for url in potential_urls:
                results.append({
                    'url': url,
                    'title': f"{query} - {site}",
                    'source': site,
                    'type': 'midi'
                })
                
        except Exception as e:
            logging.error(f"[MidiSearcher] Error searching {site}: {e}")
            continue
    
    return results[:10]  # Limit results

def download_midi_file(midi_info: Dict, download_dir: str = "downloaded_midi") -> Optional[str]:
    """
    Download a MIDI file from the given URL.
    Returns the local file path if successful, None otherwise.
    """
    try:
        os.makedirs(download_dir, exist_ok=True)
        
        url = midi_info['url']
        title = midi_info['title']
        
        # Create safe filename
        safe_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', title)
        if not safe_filename.endswith(('.mid', '.midi')):
            safe_filename += '.mid'
        
        file_path = os.path.join(download_dir, safe_filename)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Verify it's actually a MIDI file
            if response.content.startswith(b'MThd'):
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                logging.info(f"[MidiSearcher] Downloaded: {file_path}")
                return file_path
            else:
                logging.warning(f"[MidiSearcher] File doesn't appear to be MIDI: {url}")
        else:
            logging.warning(f"[MidiSearcher] Failed to download {url}: {response.status_code}")
            
    except Exception as e:
        logging.error(f"[MidiSearcher] Error downloading {midi_info['url']}: {e}")
    
    return None

def analyze_midi_file(midi_path: str) -> Dict:
    """
    Analyze a MIDI file to extract musical patterns.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        analysis = {
            'tempo': 120,
            'time_signatures': ['4/4'],
            'key_signatures': ['C major'],
            'chord_progressions': [],
            'rhythm_patterns': [],
            'melodic_phrases': [],
            'instruments': [],
            'note_density': 0,
            'pitch_ranges': {},
            'velocity_patterns': [],
            'duration_patterns': []
        }
        
        # Extract tempo
        tempo_changes = midi_data.get_tempo_changes()
        if len(tempo_changes[1]) > 0:
            analysis['tempo'] = int(tempo_changes[1][0])
        
        # Analyze each instrument
        for i, instrument in enumerate(midi_data.instruments):
            if instrument.is_drum:
                continue
                
            inst_name = pretty_midi.program_to_instrument_name(instrument.program)
            analysis['instruments'].append({
                'name': inst_name,
                'program': instrument.program,
                'note_count': len(instrument.notes)
            })
            
            if instrument.notes:
                # Extract pitch patterns
                pitches = [note.pitch for note in instrument.notes]
                analysis['pitch_ranges'][inst_name] = {
                    'min': min(pitches),
                    'max': max(pitches),
                    'common': max(set(pitches), key=pitches.count)
                }
                
                # Extract rhythm patterns
                note_times = [note.start for note in instrument.notes]
                if len(note_times) > 1:
                    intervals = [note_times[i+1] - note_times[i] for i in range(len(note_times)-1)]
                    common_interval = max(set(intervals), key=intervals.count)
                    analysis['rhythm_patterns'].append({
                        'instrument': inst_name,
                        'common_interval': common_interval,
                        'note_density': len(instrument.notes) / midi_data.get_end_time()
                    })
                
                # Extract chord progressions (simplified)
                if len(instrument.notes) >= 3:
                    # Group notes by time to find chords
                    time_groups = {}
                    for note in instrument.notes:
                        time_key = round(note.start * 4) / 4  # Quarter note resolution
                        if time_key not in time_groups:
                            time_groups[time_key] = []
                        time_groups[time_key].append(note.pitch)
                    
                    # Extract chord progressions
                    chords = []
                    for time_key in sorted(time_groups.keys()):
                        notes = time_groups[time_key]
                        if len(notes) >= 2:  # At least 2 notes for harmony
                            chords.append(sorted(notes))
                    
                    if chords:
                        analysis['chord_progressions'].append({
                            'instrument': inst_name,
                            'progression': chords[:8]  # First 8 chords
                        })
        
        # Calculate overall note density
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments if not inst.is_drum)
        if midi_data.get_end_time() > 0:
            analysis['note_density'] = total_notes / midi_data.get_end_time()
        
        logging.info(f"[MidiAnalyzer] Analyzed {midi_path}: {len(analysis['instruments'])} instruments")
        return analysis
        
    except Exception as e:
        logging.error(f"[MidiAnalyzer] Error analyzing {midi_path}: {e}")
        return {}

def search_and_analyze_artist_midi(artist_name: str, song_name: str = "") -> Dict:
    """
    Search for, download, and analyze MIDI files for an artist.
    Returns combined analysis of found patterns.
    """
    logging.info(f"[MidiReference] Searching and analyzing MIDI for {artist_name}")
    
    # Search for MIDI files
    midi_results = search_midi_files(artist_name, song_name)
    
    if not midi_results:
        logging.warning(f"[MidiReference] No MIDI files found for {artist_name}")
        return {}
    
    # Try to download and analyze the first few results
    combined_analysis = {
        'tempo': [],
        'chord_progressions': [],
        'rhythm_patterns': [],
        'pitch_ranges': {},
        'instruments': [],
        'note_density': []
    }
    
    successful_downloads = 0
    
    for midi_info in midi_results[:3]:  # Try first 3 results
        file_path = download_midi_file(midi_info)
        
        if file_path:
            analysis = analyze_midi_file(file_path)
            
            if analysis:
                # Combine analysis results
                if analysis.get('tempo'):
                    combined_analysis['tempo'].append(analysis['tempo'])
                
                combined_analysis['chord_progressions'].extend(analysis.get('chord_progressions', []))
                combined_analysis['rhythm_patterns'].extend(analysis.get('rhythm_patterns', []))
                combined_analysis['instruments'].extend(analysis.get('instruments', []))
                
                if analysis.get('note_density'):
                    combined_analysis['note_density'].append(analysis['note_density'])
                
                # Merge pitch ranges
                for inst, range_info in analysis.get('pitch_ranges', {}).items():
                    if inst not in combined_analysis['pitch_ranges']:
                        combined_analysis['pitch_ranges'][inst] = range_info
                
                successful_downloads += 1
            
            # Clean up downloaded file
            try:
                os.remove(file_path)
            except:
                pass
    
    if successful_downloads > 0:
        logging.info(f"[MidiReference] Successfully analyzed {successful_downloads} MIDI files for {artist_name}")
        return combined_analysis
    else:
        logging.warning(f"[MidiReference] Could not analyze any MIDI files for {artist_name}")
        return {}
