import streamlit as st
import requests
import json
from pathlib import Path
from datetime import datetime
from inference import EmotionDetector
from recomm import (
    get_spotify_songs,
    get_youtube_videos,
    get_movies,
    get_activities
)

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Sentio - Emotion Detection",
    page_icon="🎭",
    layout="wide"
)

# ==============================
# Custom CSS
# ==============================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .emotion-display {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .tracker-card {
        padding: 1rem;
        border-radius: 10px;
        background: #111827;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# Configuration
# ==============================
AUDIO_API_URL = "http://localhost:5000"
TRACKER_FILE = Path("emotion_history.json")

# Emotion mapping: Wav2Vec2 emotions → app emotions
EMOTION_MAPPING = {
    'neutral': 'joy',
    'happy': 'joy',
    'sad': 'sadness',
    'angry': 'anger',
    'fearful': 'fear',
    'disgusted': 'anger',
    'surprised': 'joy'
}

EMOTION_EMOJIS = {
    'joy': '😊',
    'sadness': '😢',
    'anger': '😠',
    'fear': '😨',
    'love': '❤️'
}

EMOTION_SCORE_MAP = {
    'fear': 1,
    'sadness': 2,
    'anger': 3,
    'love': 4,
    'joy': 5
}

SCORE_TO_EMOTION = {
    1: 'fear',
    2: 'sadness',
    3: 'anger',
    4: 'love',
    5: 'joy'
}

# ==============================
# Session State Init
# ==============================
if "text_result" not in st.session_state:
    st.session_state["text_result"] = None

if "audio_result" not in st.session_state:
    st.session_state["audio_result"] = None

# ==============================
# Load Text Model (Cached)
# ==============================
@st.cache_resource
def load_text_model():
    """Load the text emotion detection model"""
    try:
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "model" / "trained_emotion_model"

        if not model_path.exists():
            st.error(f"Text model folder not found: {model_path}")
            return None

        return EmotionDetector(model_path=str(model_path))
    except Exception as e:
        st.error(f"Error loading text model: {e}")
        return None

# ==============================
# Audio API Functions
# ==============================
def check_audio_api():
    """Check if audio API is available"""
    try:
        response = requests.get(f"{AUDIO_API_URL}/health", timeout=3)
        return response.status_code == 200
    except Exception:
        return False

def predict_audio_emotion(audio_file):
    """Predict emotion from audio file"""
    try:
        audio_file.seek(0)
        files = {'file': audio_file}
        data = {'return_all_scores': 'true'}

        response = requests.post(
            f"{AUDIO_API_URL}/predict",
            files=files,
            data=data,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            wav2vec_emotion = result['emotion']
            mapped_emotion = EMOTION_MAPPING.get(wav2vec_emotion, 'joy')

            return {
                'emotion': mapped_emotion,
                'original_emotion': wav2vec_emotion,
                'confidence': result['confidence'],
                'all_scores': result.get('all_scores', {})
            }
        else:
            try:
                err = response.json().get('error', 'Unknown error')
            except Exception:
                err = response.text
            st.error(f"API Error: {err}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# ==============================
# Negation Handler
# ==============================
def handle_negation(text):
    """Handle negation in text"""
    negation_words = ["not", "never", "no", "n't"]
    text_lower = text.lower()

    for word in negation_words:
        if word in text_lower:
            return True
    return False

# ==============================
# Emotion Tracker Functions
# ==============================
def load_emotion_history():
    if TRACKER_FILE.exists():
        try:
            with open(TRACKER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_emotion_history(history):
    try:
        with open(TRACKER_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save emotion history: {e}")

def add_emotion_to_tracker(source, emotion, confidence):
    history = load_emotion_history()
    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "emotion": emotion,
        "confidence": float(confidence),
        "score": EMOTION_SCORE_MAP.get(emotion, 3)
    })
    save_emotion_history(history)

def build_tracker_chart_data(history):
    rows = []
    for item in history:
        rows.append({
            "Time": item["timestamp"],
            "Emotion Score": item["score"],
            "Emotion": item["emotion"],
            "Source": item["source"],
            "Confidence": item["confidence"]
        })
    return rows

# ==============================
# Display Recommendations
# ==============================
def display_recommendations(emotion):
    """Display all recommendations for the detected emotion"""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎵 Spotify Recommendations")
        with st.spinner("Loading songs..."):
            songs = get_spotify_songs(emotion)
            if songs:
                for song in songs:
                    st.markdown(f"- {song}")
            else:
                st.write("No songs found.")

        st.markdown("### 🎬 Movie Recommendations")
        with st.spinner("Loading movies..."):
            movies = get_movies(emotion)
            if movies:
                for movie in movies:
                    st.markdown(f"- {movie}")
            else:
                st.write("No movies found.")

    with col2:
        st.markdown("### 📺 YouTube Recommendations")
        with st.spinner("Loading videos..."):
            videos = get_youtube_videos(emotion)
            if videos:
                for video in videos:
                    st.markdown(f"- {video}")
            else:
                st.write("No videos found.")

        st.markdown("### 🧘 Suggested Activities")
        activities = get_activities(emotion)
        if activities:
            for act in activities:
                st.markdown(f"- {act}")
        else:
            st.write("Take care of yourself 💙")

# ==============================
# Result UI Helpers
# ==============================
def render_emotion_card(emotion, confidence, subtitle=None):
    emoji = EMOTION_EMOJIS.get(emotion, '🎭')
    subtitle_html = f"<p style='font-size: 0.9rem; opacity: 0.85;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div class="emotion-display">
        <h1>{emoji}</h1>
        <h2>{emotion.upper()}</h2>
        <p>Confidence: {confidence:.1%}</p>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)

def render_text_result(result):
    col1, col2 = st.columns([1, 2])

    with col1:
        render_emotion_card(result["emotion"], result["confidence"])

    with col2:
        st.subheader("All Emotions")
        for emo, score in sorted(result["all_emotions"].items(), key=lambda x: x[1], reverse=True):
            st.progress(score)
            st.caption(f"{emo}: {score:.1%}")

def render_audio_result(result):
    col1, col2 = st.columns([1, 2])

    with col1:
        render_emotion_card(
            result["emotion"],
            result["confidence"],
            subtitle=f"Detected audio emotion: {result['original_emotion']}"
        )

    with col2:
        st.subheader("All Emotions")
        for emo, score in sorted(result["all_scores"].items(), key=lambda x: x[1], reverse=True):
            st.progress(score)
            st.caption(f"{emo}: {score:.1%}")

# ==============================
# Main App
# ==============================
def main():
    st.markdown('<p class="main-header">🎭 Sentio</p>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem;'>Technology that truly listens</p>",
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        app_mode = st.radio(
            "Choose a section:",
            ["📝 Text", "🎤 Audio", "📈 Emotion Tracker"],
            index=0
        )

        st.divider()

        if app_mode == "🎤 Audio":
            api_status = check_audio_api()
            if api_status:
                st.success("✅ Audio API: Connected")
            else:
                st.error("❌ Audio API: Disconnected")
                st.info("Start API: `python audio_api.py`")

        st.divider()

        with st.expander("ℹ️ About"):
            st.write("""
            **Sentio** supports:
            - 📝 Text emotion detection
            - 🎤 Voice emotion detection
            - 📈 Emotion tracking over time

            It also gives personalized recommendations based on detected emotions.
            """)

        with st.expander("🎯 Supported Emotions"):
            st.write("😊 Joy")
            st.write("😢 Sadness")
            st.write("😠 Anger")
            st.write("😨 Fear")
            st.write("❤️ Love")

    st.markdown("---")

    # ==============================
    # TEXT TAB
    # ==============================
    if app_mode == "📝 Text":
        st.header("📝 Text Emotion Detection")

        user_text = st.text_area(
            "Enter your text here:",
            placeholder="Type how you feel...",
            height=150
        )

        if st.button("🔍 Analyze Text", type="primary", key="text_button"):
            if user_text.strip() == "":
                st.warning("⚠️ Please enter some text.")
            else:
                detector = load_text_model()

                if detector:
                    with st.spinner("Analyzing text..."):
                        result = detector.predict(user_text)
                        emotion = result["primary_emotion"].lower()
                        confidence = result["confidence"]

                        if handle_negation(user_text) and emotion == "joy":
                            emotion = "sadness"
                            st.info("ℹ️ Negation detected - adjusted emotion")

                        text_result = {
                            "emotion": emotion,
                            "confidence": confidence,
                            "all_emotions": result.get("all_emotions", {})
                        }

                        st.session_state["text_result"] = text_result
                        add_emotion_to_tracker("text", emotion, confidence)

        if st.session_state["text_result"]:
            render_text_result(st.session_state["text_result"])

            st.markdown("---")
            st.header(f"✨ Recommendations for {st.session_state['text_result']['emotion'].upper()}")
            display_recommendations(st.session_state["text_result"]["emotion"])
        else:
            st.info("👆 Analyze your text above to get personalized recommendations.")

    # ==============================
    # AUDIO TAB
    # ==============================
    elif app_mode == "🎤 Audio":
        st.header("🎤 Audio Emotion Detection")

        if not check_audio_api():
            st.error("❌ Audio API is not running!")
            st.info("""
            **To enable audio detection:**
            1. Open a new terminal
            2. Run: `python audio_api.py`
            3. Keep it running
            4. Refresh this page
            """)
        else:
            audio_source = st.radio(
                "Choose audio input method:",
                ["🎙️ Record with Mic", "📁 Upload Audio File"],
                horizontal=True
            )

            audio_file = None

            if audio_source == "🎙️ Record with Mic":
                st.write("Speak directly to the app.")
                audio_file = st.audio_input("Tap below to record your voice")

                if audio_file is not None:
                    st.audio(audio_file)

            else:
                audio_file = st.file_uploader(
                    "Upload audio file:",
                    type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
                    help="Upload a 2-10 second audio clip"
                )

                if audio_file is not None:
                    st.audio(audio_file)

            if audio_file is not None:
                if st.button("🔍 Analyze Audio", type="primary", key="audio_button"):
                    with st.spinner("Analyzing audio... Please wait"):
                        audio_result = predict_audio_emotion(audio_file)

                        if audio_result:
                            st.session_state["audio_result"] = audio_result
                            add_emotion_to_tracker(
                                "audio",
                                audio_result["emotion"],
                                audio_result["confidence"]
                            )

        if st.session_state["audio_result"]:
            render_audio_result(st.session_state["audio_result"])

            st.markdown("---")
            st.header(f"✨ Recommendations for {st.session_state['audio_result']['emotion'].upper()}")
            display_recommendations(st.session_state["audio_result"]["emotion"])
        else:
            st.info("👆 Record or upload audio above to get personalized recommendations.")

    # ==============================
    # EMOTION TRACKER TAB
    # ==============================
    elif app_mode == "📈 Emotion Tracker":
        st.header("📈 Emotion Tracker")
        st.write("Track how your emotions change over time.")

        history = load_emotion_history()

        if not history:
            st.info("No emotion history yet. Analyze text or audio first to start tracking.")
        else:
            chart_rows = build_tracker_chart_data(history)

            st.subheader("Emotion Trend Over Time")
            st.line_chart(
                data=[row["Emotion Score"] for row in chart_rows],
                x_label="Entries",
                y_label="Emotion Score"
            )

            st.caption("Emotion scale: 1 = Fear, 2 = Sadness, 3 = Anger, 4 = Love, 5 = Joy")

            with st.expander("View Emotion History"):
                for item in reversed(history[-20:]):
                    emoji = EMOTION_EMOJIS.get(item["emotion"], "🎭")
                    st.markdown(
                        f"**{emoji} {item['emotion'].upper()}** | "
                        f"{item['source'].title()} | "
                        f"{item['confidence']:.1%} confidence | "
                        f"{item['timestamp']}"
                    )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Entries", len(history))
            with col2:
                latest = history[-1]
                st.metric("Latest Emotion", latest["emotion"].upper())

            if st.button("🗑️ Clear Emotion History"):
                save_emotion_history([])
                st.success("Emotion history cleared.")
                st.rerun()

if __name__ == "__main__":
    main()