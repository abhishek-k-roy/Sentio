"""
Wav2Vec2 Fine-tuning for Audio Emotion Detection
This script fine-tunes Meta's Wav2Vec2 model for emotion classification from audio
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import librosa
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for training"""
    
    # Model settings
    MODEL_NAME = "facebook/wav2vec2-base"  # Or "facebook/wav2vec2-large-960h"
    
    # Emotion labels - customize based on your dataset
    EMOTION_LABELS = [
        "neutral",
        "happy", 
        "sad",
        "angry",
        "fearful",
        "disgusted",
        "surprised"
    ]
    
    # Audio processing
    SAMPLING_RATE = 16000  # Wav2Vec2 expects 16kHz
    MAX_DURATION = 10  # seconds
    
    # Training hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 20
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Paths
    DATA_DIR = "./data"  # Your audio files directory
    OUTPUT_DIR = "./wav2vec2_emotion_model"
    CACHE_DIR = "./cache"

    # Run modes
    EVAL_ONLY = True  # True = only evaluate using a saved checkpoint, False = train + evaluate
    CKPT_PATH = "./wav2vec2_emotion_model/checkpoint-2268"  # change to your latest checkpoint
    FINAL_DIR = "./wav2vec2_emotion_model/final_model"
    
    # Training settings
    EVAL_STRATEGY = "epoch"
    SAVE_STRATEGY = "epoch"
    SAVE_TOTAL_LIMIT = 3
    EARLY_STOPPING_PATIENCE = 3
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_dataset(audio_paths, labels, label2id, sampling_rate=16000):
    """
    Prepare dataset from audio files and labels
    
    Args:
        audio_paths: List of paths to audio files
        labels: List of emotion labels
        label2id: Dictionary mapping labels to IDs
        sampling_rate: Target sampling rate
    
    Returns:
        Dataset object
    """
    data = {
        "audio": audio_paths,
        "label": [label2id[label] for label in labels]
    }
    
    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate, decode=True))
    
    return dataset


def load_data_from_directory(data_dir, emotion_labels):
    """
    Load audio files from directory structure: data_dir/emotion_name/*.wav
    
    Args:
        data_dir: Root directory containing emotion subdirectories
        emotion_labels: List of emotion labels
    
    Returns:
        audio_paths, labels
    """
    audio_paths = []
    labels = []
    
    for emotion in emotion_labels:
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: Directory {emotion_dir} not found, skipping...")
            continue
            
        for audio_file in os.listdir(emotion_dir):
            if audio_file.endswith(('.wav', '.mp3', '.flac')):
                audio_paths.append(os.path.join(emotion_dir, audio_file))
                labels.append(emotion)
    
    print(f"\nLoaded {len(audio_paths)} audio files")
    print(f"Label distribution:")
    for emotion in emotion_labels:
        count = labels.count(emotion)
        print(f"  {emotion}: {count}")
    
    return audio_paths, labels


def load_data_from_csv(csv_path):
    """
    Load audio files from CSV with columns: 'path', 'emotion'
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        audio_paths, labels
    """
    df = pd.read_csv(csv_path)
    audio_paths = df['path'].tolist()
    labels = df['emotion'].tolist()
    
    print(f"\nLoaded {len(audio_paths)} audio files from CSV")
    print(f"Label distribution:")
    print(df['emotion'].value_counts())
    
    return audio_paths, labels


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_function(examples, feature_extractor, max_length):
    """
    Preprocess audio data for Wav2Vec2
    
    Args:
        examples: Batch of examples
        feature_extractor: Wav2Vec2FeatureExtractor
        max_length: Maximum audio length in samples
    
    Returns:
        Preprocessed features
    """
    audio_arrays = [x["array"] for x in examples["audio"]]
    
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    inputs["labels"] = examples["label"]
    
    return inputs


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics
    
    Args:
        eval_pred: Tuple of (predictions, labels)
    
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


# ============================================================================
# TRAINING
# ============================================================================

@dataclass
class DataCollatorSpeechWithPadding:
    feature_extractor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate labels and inputs
        labels = [f["labels"] for f in features]
        inputs = [{"input_values": f["input_values"]} for f in features]

        # Pad audio sequences
        batch = self.feature_extractor.pad(
            inputs,
            padding=self.padding,
            return_tensors="pt"
        )

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

def train_model(train_dataset, eval_dataset, config):
    """
    Train the Wav2Vec2 model
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Configuration object
    
    Returns:
        Trained model and trainer
    """
    # Create label mappings
    label2id = {label: i for i, label in enumerate(config.EMOTION_LABELS)}
    id2label = {i: label for i, label in enumerate(config.EMOTION_LABELS)}
    
    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        config.MODEL_NAME,
        cache_dir=config.CACHE_DIR
    )
    
    # Load model for sequence classification
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.EMOTION_LABELS),
        label2id=label2id,
        id2label=id2label,
        cache_dir=config.CACHE_DIR
    )
    
    # Freeze feature encoder layers (optional - for faster training)
    # Uncomment if you want to freeze the feature extractor
    # model.freeze_feature_encoder()
    
    # Preprocess datasets
    max_length = config.MAX_DURATION * config.SAMPLING_RATE
    
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, feature_extractor, max_length),
        remove_columns=train_dataset.column_names,
        batched=True,
        batch_size=config.BATCH_SIZE
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, feature_extractor, max_length),
        remove_columns=eval_dataset.column_names,
        batched=True,
        batch_size=config.BATCH_SIZE
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        evaluation_strategy=config.EVAL_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=os.path.join(config.OUTPUT_DIR, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        dataloader_num_workers=0,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # Initialize trainer
    data_collator = DataCollatorSpeechWithPadding(feature_extractor=feature_extractor)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
        data_collator=data_collator
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting Training...")
    print("="*50 + "\n")
    
    trainer.train()
    
    return model, trainer, feature_extractor


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(trainer, test_dataset, id2label, feature_extractor, config):
    """
    Evaluate model on test set
    """
    print("\n" + "="*50)
    print("Evaluating on Test Set...")
    print("="*50 + "\n")

    # Preprocess test dataset (IMPORTANT)
    max_length = config.MAX_DURATION * config.SAMPLING_RATE

    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, feature_extractor, max_length),
        remove_columns=test_dataset.column_names,
        batched=True,
        batch_size=config.BATCH_SIZE
    )

    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    print("\nClassification Report:")
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=[id2label[i] for i in range(len(id2label))]
    ))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)

    return pred_labels, true_labels

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline"""

    config = Config()

    print("=" * 50)
    print("Wav2Vec2 Emotion Detection Training")
    print("=" * 50)
    print(f"\nDevice: {config.DEVICE}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Emotions: {config.EMOTION_LABELS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.NUM_EPOCHS}")

    # Create directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.FINAL_DIR, exist_ok=True)

    # Load data
    audio_paths, labels = load_data_from_directory(config.DATA_DIR, config.EMOTION_LABELS)

    if len(audio_paths) == 0:
        print("\n⚠️ No audio files found!")
        print(f"Please place your audio files in: {config.DATA_DIR}")
        print("Directory structure should be:")
        print("  data/")
        print("    neutral/")
        print("      audio1.wav")
        print("    happy/")
        print("      audio2.wav")
        print("    ...")
        return

    # Create label mappings
    label2id = {label: i for i, label in enumerate(config.EMOTION_LABELS)}
    id2label = {i: label for i, label in enumerate(config.EMOTION_LABELS)}

    # Split data: 70% train, 15% validation, 15% test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        audio_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_paths)}")
    print(f"  Validation: {len(val_paths)}")
    print(f"  Test: {len(test_paths)}")

    # Prepare datasets
    train_dataset = prepare_dataset(train_paths, train_labels, label2id, config.SAMPLING_RATE)
    val_dataset = prepare_dataset(val_paths, val_labels, label2id, config.SAMPLING_RATE)
    test_dataset = prepare_dataset(test_paths, test_labels, label2id, config.SAMPLING_RATE)

    # ---------------------------
    # TRAIN or EVAL ONLY
    # ---------------------------
    if not config.EVAL_ONLY:
        model, trainer, feature_extractor = train_model(train_dataset, val_dataset, config)

    else:
        print("\n✅ EVAL_ONLY is ON — Loading model from checkpoint...\n")

        # Load feature extractor from base model
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.MODEL_NAME)

        # Load trained model weights from checkpoint
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            config.CKPT_PATH,
            num_labels=len(config.EMOTION_LABELS),
            label2id=label2id,
            id2label=id2label,
        )

        training_args = TrainingArguments(
            output_dir=config.OUTPUT_DIR,
            per_device_eval_batch_size=config.BATCH_SIZE,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        data_collator = DataCollatorSpeechWithPadding(feature_extractor=feature_extractor)

        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

    # ---------------------------
    # Evaluate on test set
    # ---------------------------
    pred_labels, true_labels = evaluate_model(
        trainer,
        test_dataset,
        id2label,
        feature_extractor,
        config
    )

    # ---------------------------
    # Save final model
    # ---------------------------
    model.save_pretrained(config.FINAL_DIR)
    feature_extractor.save_pretrained(config.FINAL_DIR)

    print(f"\n✅ Final model saved to: {config.FINAL_DIR}")
    print("\n✅ Process completed successfully!")
    print("To use the model for inference, run: python inference.py")

if __name__ == "__main__":
    main()