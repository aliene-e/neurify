import os
import numpy as np
import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

# 1. Spotify Authentication
load_dotenv()

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")

scope = "user-modify-playback-state,playlist-read-private"

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scope,
    )
)

# 2. Spotify Track Lookup Function
def get_track_uri(track_name, artist_name):
    query = f"track:{track_name} artist:{artist_name}"
    results = sp.search(q=query, type="track", limit=1)

    if results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        return track["uri"]
    else:
        return None


# 3. Curated Song Database (Valence / Arousal Map)
SONG_DATABASE = [
    {"title": "Blinding Lights", "artist": "The Weeknd", "v": 0.334, "a": 0.73},
    {"title": "In Paradisum", "artist": "Gabriel Fauré", "v": -0.743, "a": -0.372},
    {"title": "Roommates", "artist": "Malcolm Todd", "v": 0.292, "a": -0.437},
    {"title": "I Don't Like", "artist": "Chief Keef", "v": -0.416, "a": 0.844},
]

# 4. Vector Distance Matcher
def find_nearest_song(valence: float, arousal: float, database: list) -> dict:
    best_match = None
    min_distance = float("inf")

    for song in database:
        dist = np.sqrt((song["v"] - valence) ** 2 + (song["a"] - arousal) ** 2)

        if dist < min_distance:
            min_distance = dist
            best_match = song
    
    return best_match, min_distance

# 5. Main Integrated Execution
if __name__ == "__main__":
    from cleaning import compute_valence_arousal

    # Load cleaned EEG coordinates from Pt 1
    raw_csv = "Shriya_Elaine_raw_muse_session.csv"
    df = pd.read_csv(raw_csv)
    live_valence, live_arousal = compute_valence_arousal(df)

    print(
        f"Neural Input Coordinates: Valence = {live_valence:.4f}, Arousal = {live_arousal:.4f}"
    )

    # Match coordinates to nearest song entry in database
    matched_entry, distance = find_nearest_song(
        live_valence, live_arousal, SONG_DATABASE
    )
    song_title = matched_entry["title"]
    artist_name = matched_entry["artist"]

    print(
        f"Match Identified: '{song_title}' by {artist_name} (Distance: {distance:.4f})"
    )
    sp.add_to_queue(get_track_uri(song_title, artist_name))