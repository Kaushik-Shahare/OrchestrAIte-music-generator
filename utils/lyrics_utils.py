import re
from typing import List, Dict

def parse_lyrics_sections(lyrics: str) -> List[Dict]:
    """
    Parse lyrics for section markers (e.g., [Verse], [Chorus], [Bridge], etc.)
    and return a list of sections with their names and line indices.
    """
    lines = lyrics.splitlines()
    sections = []
    current_section = None
    for idx, line in enumerate(lines):
        match = re.match(r"\[(.+?)\]", line.strip(), re.IGNORECASE)
        if match:
            if current_section:
                current_section['end'] = idx
                sections.append(current_section)
            current_section = {'name': match.group(1).strip().lower(), 'start': idx + 1, 'end': None}
    if current_section:
        current_section['end'] = len(lines)
        sections.append(current_section)
    return sections 