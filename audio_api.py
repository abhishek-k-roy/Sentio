"""
Audio Emotion Detection API Server
Flask REST API for Wav2Vec2 emotion detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', './wav2vec2_emotion_model/final_model')
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

# Global predictor
predictor = None


class AudioEmotionPredictor:
    """Audio emotion predictor using Wav2Vec2"""
    
    def __init__(self, model_path):
        """Initialize the predictor"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading model from {model_path}...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mappings
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Emotions: {list(self.label2id.keys())}")
    
    def preprocess_audio(self, audio_path, target_sr=16000):
        """Load and preprocess audio file"""
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio
    
    def predict(self, audio_path, return_all_scores=False):
        """Predict emotion from audio file"""
        # Preprocess audio
        audio = self.preprocess_audio(audio_path)
        
        # Extract features
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get predictions
        probs = probs.cpu().numpy()[0]
        predicted_id = np.argmax(probs)
        predicted_emotion = self.id2label[predicted_id]
        confidence = probs[predicted_id]
        
        if return_all_scores:
            return {self.id2label[i]: float(probs[i]) for i in range(len(probs))}
        else:
            return predicted_emotion, float(confidence)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_predictor():
    """Initialize the emotion predictor"""
    global predictor
    if predictor is None:
        logger.info(f"Loading model from {MODEL_PATH}...")
        try:
            predictor = AudioEmotionPredictor(MODEL_PATH)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if predictor is None:
            init_predictor()
        return jsonify({
            'status': 'healthy',
            'model_loaded': predictor is not None,
            'emotions': list(predictor.label2id.keys()) if predictor else []
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict_emotion():
    """Predict emotion from uploaded audio file"""
    try:
        # Initialize predictor if needed
        if predictor is None:
            init_predictor()
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {ALLOWED_EXTENSIONS}'
            }), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'error': f'File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024} MB'
            }), 400
        
        # Get parameters
        return_all_scores = request.form.get('return_all_scores', 'false').lower() == 'true'
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Predict emotion
            if return_all_scores:
                all_scores = predictor.predict(temp_path, return_all_scores=True)
                emotion = max(all_scores, key=all_scores.get)
                confidence = all_scores[emotion]
                
                response = {
                    'emotion': emotion,
                    'confidence': confidence,
                    'all_scores': all_scores
                }
            else:
                emotion, confidence = predictor.predict(temp_path)
                response = {
                    'emotion': emotion,
                    'confidence': confidence
                }
            
            logger.info(f"Prediction: {emotion} ({confidence:.2%})")
            return jsonify(response), 200
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/emotions', methods=['GET'])
def get_emotions():
    """Get list of supported emotions"""
    try:
        if predictor is None:
            init_predictor()
        
        return jsonify({
            'emotions': list(predictor.label2id.keys())
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    # Initialize predictor on startup
    try:
        init_predictor()
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        logger.info("Server will start but predictions will fail until model is loaded")
    
    # Run server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)