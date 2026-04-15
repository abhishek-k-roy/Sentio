import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
import os
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ==============================
# API KEYS
# ==============================
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# ==============================
# Emotion → Search Mapping
# ==============================
EMOTION_KEYWORDS = {
    "joy": "happy upbeat feel good songs",
    "sadness": "motivational uplifting healing songs",
    "anger": "calm relaxing stress relief songs",
    "fear": "peaceful meditation anxiety relief songs",
    "love": "romantic love songs"
}

# OMDb searches TITLES, not moods.
# So we map each emotion to actual movie titles, then use OMDb properly.
EMOTION_MOVIE_TITLES = {
    "joy": [
        "The Pursuit of Happyness",
        "Zindagi Na Milegi Dobara",
        "Inside Out",
        "The Secret Life of Walter Mitty",
        "School of Rock"
    ],
    "sadness": [
        "Good Will Hunting",
        "The Shawshank Redemption",
        "A Beautiful Mind",
        "Taare Zameen Par",
        "Soul"
    ],
    "anger": [
        "Rocky",
        "Creed",
        "The Dark Knight",
        "Lakshya",
        "Whiplash"
    ],
    "fear": [
        "Life of Pi",
        "Gravity",
        "Cast Away",
        "The Martian",
        "127 Hours"
    ],
    "love": [
        "The Notebook",
        "Before Sunrise",
        "La La Land",
        "Titanic",
        "About Time"
    ]
}

ACTIVITIES = {
    "sadness": [
        "Take a 15-minute walk 🌿",
        "Call a close friend ☎️",
        "Do light stretching 🧘"
    ],
    "anger": [
        "Do push-ups 💪",
        "Practice deep breathing 🌬️"
    ],
    "fear": [
        "Meditate for 10 minutes 🧘‍♂️",
        "Write your thoughts in a journal 📖"
    ],
    "joy": [
        "Dance to your favorite song 💃",
        "Share your happiness 😊"
    ],
    "love": [
        "Plan a small surprise 🎁",
        "Write a gratitude note ❤️"
    ]
}

# ==============================
# Helpers
# ==============================
def _safe_get_json(url, *, params=None, headers=None, timeout=15):
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()

def _spotify_client():
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        raise ValueError("Spotify credentials are missing. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.")

    auth_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    return spotipy.Spotify(auth_manager=auth_manager, requests_timeout=15)

# ==============================
# Spotify
# ==============================
def get_spotify_songs(emotion, limit=5):
    try:
        sp = _spotify_client()
        query = EMOTION_KEYWORDS.get(emotion.lower(), "feel good songs")

        # market helps avoid some unavailable-result issues
        results = sp.search(
            q=query,
            type="track",
            limit=limit,
            market="IN"
        )

        items = results.get("tracks", {}).get("items", [])
        songs = []

        for item in items:
            name = item.get("name", "Unknown Title")
            artists = item.get("artists", [])
            artist_names = ", ".join(a.get("name", "Unknown Artist") for a in artists) or "Unknown Artist"
            url = item.get("external_urls", {}).get("spotify", "")
            if url:
                songs.append(f"{name} - {artist_names}\n{url}")
            else:
                songs.append(f"{name} - {artist_names}")

        if songs:
            return songs

        return ["No Spotify songs found for this emotion."]

    except spotipy.exceptions.SpotifyException as e:
        # Cleaner message for UI
        if getattr(e, "http_status", None) == 403:
            return ["Spotify access denied (403). Check your Spotify app settings, credentials, and account/app access."]
        if getattr(e, "http_status", None) == 401:
            return ["Spotify authentication failed (401). Check client ID and client secret."]
        return [f"Spotify Error: {str(e)}"]

    except Exception as e:
        return [f"Spotify Error: {str(e)}"]

# ==============================
# YouTube
# ==============================
def get_youtube_videos(emotion, limit=5):
    try:
        if not YOUTUBE_API_KEY:
            return ["YouTube API key missing."]

        query = EMOTION_KEYWORDS.get(emotion.lower(), "motivational songs")
        url = "https://www.googleapis.com/youtube/v3/search"

        params = {
            "part": "snippet",
            "q": query,
            "key": YOUTUBE_API_KEY,
            "maxResults": limit,
            "type": "video"
        }

        response = _safe_get_json(url, params=params)

        videos = []
        for item in response.get("items", []):
            title = item.get("snippet", {}).get("title", "Untitled Video")
            video_id = item.get("id", {}).get("videoId", "")
            if video_id:
                link = f"https://www.youtube.com/watch?v={video_id}"
                videos.append(f"{title}\n{link}")

        return videos if videos else ["No YouTube videos found."]

    except Exception as e:
        return [f"YouTube Error: {str(e)}"]

# ==============================
# OMDb Movies
# ==============================
def _fetch_movie_by_title(title):
    url = "https://www.omdbapi.com/"
    params = {
        "apikey": OMDB_API_KEY,
        "t": title,
        "type": "movie"
    }

    data = _safe_get_json(url, params=params)

    if data.get("Response") == "True":
        imdb_id = data.get("imdbID", "")
        imdb_link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else ""
        display = f"{data.get('Title', title)} ({data.get('Year', 'N/A')})"
        return f"{display} - {imdb_link}" if imdb_link else display

    return None

def get_movies(emotion, limit=5):
    try:
        if not OMDB_API_KEY:
            return ["OMDb API key missing."]

        titles = EMOTION_MOVIE_TITLES.get(emotion.lower(), [])
        if not titles:
            return ["No suitable movies found 🎬"]

        movies = []
        for title in titles:
            movie = _fetch_movie_by_title(title)
            if movie:
                movies.append(movie)
            if len(movies) >= limit:
                break

        return movies if movies else ["No suitable movies found 🎬"]

    except requests.HTTPError as e:
        return [f"Movie API Error: HTTP {e.response.status_code}"]
    except Exception as e:
        return [f"Movie API Error: {str(e)}"]

# ==============================
# Activities
# ==============================
def get_activities(emotion):
    return ACTIVITIES.get(emotion.lower(), ["Take care of yourself 💙"])

# ==============================
# Test Block
# ==============================
if __name__ == "__main__":
    test_emotion = "sadness"

    print("\n🎵 Spotify Songs:")
    for song in get_spotify_songs(test_emotion):
        print("-", song)

    print("\n📺 YouTube Videos:")
    for video in get_youtube_videos(test_emotion):
        print("-", video)

    print("\n🎬 Movies:")
    for movie in get_movies(test_emotion):
        print("-", movie)

    print("\n🧘 Activities:")
    for act in get_activities(test_emotion):
        print("-", act)