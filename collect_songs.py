# import os
# import numpy as np
import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials
from pathlib import Path



# 1. Spotify Authentication
load_dotenv()

auth_manager = SpotifyClientCredentials() 
spotify = spotipy.Spotify(auth_manager=auth_manager)

song_vector = [
    ("ZONE", "NCT WISH"),
    ("Life Goes On", "BTS"),
    ("Run BTS", "BTS"),
    ("SOS", "NCT DREAM"),
    ("RUDE!", "Hearts2Hearts"),
    ("needy", "Ariana Grande"),
    ("Melt Inside My Pocket", "NCT WISH"),
    ("All Of Me", "John Legend"),
    ("Answer: Love Myself", "BTS"),
    ("The Girl In The Bubble", "Ariana Grande")
]

def get_track_info(track_name, artist_name):
    query = f"track:{track_name} artist:{artist_name}"
    results = spotify.search(q=query, type="track", limit=1)

    if results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        return track["uri"]
    else:
        return None


