from transformers import pipeline
import os

class EmotionDetector:
    def __init__(self, model_path="model/trained_emotion_model"):
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}. Train the model first!")

        print(f"Loading model from: {model_path}")
        self.classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            top_k=None,
            device=-1
        )
        print("Model loaded successfully!")

    def predict(self, text):
        if not text or not text.strip():
            return {
                "primary_emotion": "None",
                "confidence": 0.0,
                "all_emotions": {}
            }

        results = self.classifier(text)[0]
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        return {
            "primary_emotion": results[0]['label'],
            "confidence": float(results[0]['score']),
            "all_emotions": {r['label']: float(r['score']) for r in results}
        }

if __name__ == "__main__":
    detector = EmotionDetector()
    print(detector.predict("I am very happy today"))
