# 🎧 Sentio - Multimodal Emotion Detection & Recommendation System

## 📌 Project Overview

**Sentio** is an advanced AI-powered system that detects human emotions using both **text and audio inputs**, tracks emotional patterns over time, and provides **personalized recommendations** to improve user well-being.

Unlike traditional emotion detection systems, Sentio integrates **real-time emotion tracking** and an intelligent **recommendation engine**, making it a complete emotional intelligence platform.

---

## 🚀 Key Features

### 🎤 Multimodal Emotion Detection

* Text-based emotion analysis using NLP
* Audio-based emotion recognition using speech models
* Combines both for higher accuracy

### 📈 Real-Time Emotion Tracker

* Tracks user emotions continuously over time
* Visualizes emotional trends using a **line graph**
* Helps identify mood patterns and behavioral insights

### 🎯 Smart Recommendation System

Based on detected emotions, Sentio suggests:

* 🎵 Songs (via Spotify)
* 🎬 Movies (via OMDb)
* 📺 Videos (via YouTube)
* 🧘 Physical activities (exercise, relaxation techniques)

---

## 🧠 Technologies Used

### 🔹 AI / Machine Learning

* DistilBERT (Text Emotion Detection)
* Wav2Vec2 (Audio Emotion Recognition)
* Logistic Regression (Baseline Model)

### 🔹 Data Visualization

* Matplotlib / Plotly (Emotion tracking graphs)

### 🔹 APIs & Integrations

* Spotify API
* YouTube API
* OMDb API

### 🔹 Tools & Libraries

* Python
* Jupyter Notebook / Google Colab
* Scikit-learn
* Pandas, NumPy
* Librosa
* HuggingFace Transformers

---

## ⚙️ System Architecture

1. **User Input**

   * Text or audio input

2. **Preprocessing**

   * Text cleaning & tokenization
   * Audio feature extraction

3. **Model Processing**

   * DistilBERT → Text emotion
   * Wav2Vec2 → Audio emotion

4. **Emotion Tracking Module**

   * Stores emotion history
   * Generates time-based graph

5. **Recommendation Engine**

   * Maps emotions → suitable content
   * Fetches data from external APIs

6. **Output**

   * Emotion label + Graph + Recommendations

---

## 📂 Project Structure

```
Sentio/
│── data/
│── models/
│── notebooks/
│── src/
│── app.py
│── requirements.txt
│── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/your-username/sentio-project.git
cd sentio-project
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
python app.py
```

---

## 📊 Output Example

* Input: "I feel stressed and tired"
* Detected Emotion: **Sad 😔**

📈 Emotion Graph:

* Shows fluctuation of mood over time

🎯 Recommendations:

* 🎵 Calm music playlist
* 🎬 Motivational movies
* 📺 Relaxing videos
* 🧘 Meditation / breathing exercises

---
## 🔐 Environment Variables

Create a `.env` file in the root directory and add the following:

SPOTIFY_CLIENT_ID=your_key  
SPOTIFY_CLIENT_SECRET=your_key  
YOUTUBE_API_KEY=your_key  
OMDB_API_KEY=your_key  

## 🎯 Use Cases

* Mental health monitoring
* Personalized entertainment recommendations
* Stress management systems
* Human-computer interaction
* Smart wellness applications

---

## 📈 Future Improvements

* Add facial emotion recognition
* Deploy as a web/mobile app
* Improve recommendation accuracy using user feedback
* Real-time streaming emotion detection

---

## 👨‍💻 Author

**Abhishek Roy**
Final Year BCA Student

---

## ⭐ Acknowledgements

* HuggingFace Transformers
* Spotify, YouTube, and OMDb APIs
* Open-source emotion datasets

---

## 📌 Keywords

Multimodal AI, Emotion Detection, Real-Time Tracking, Recommendation System, NLP, Speech Recognition, DistilBERT, Wav2Vec2, Data Visualization, Machine Learning

