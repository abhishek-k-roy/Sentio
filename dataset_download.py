"""
Download and Prepare RAVDESS Dataset for Emotion Detection
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Free dataset with 7356 files covering 8 emotions

This script will download and prepare the dataset automatically.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
import shutil
from tqdm import tqdm
import pandas as pd

# Dataset information
DATASET_NAME = "RAVDESS"
DATASET_URL = "https://zenodo.org/record/1188976/files/"

# Emotion mapping for RAVDESS
# Format: identifier-emotion
EMOTION_MAPPING = {
    '01': 'neutral',
    '02': 'neutral',  # calm -> neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgusted',
    '08': 'surprised'
}

# Actor IDs (24 actors total)
ACTORS = list(range(1, 25))


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_ravdess_dataset(output_dir="./data_raw"):
    """
    Download RAVDESS dataset
    
    Args:
        output_dir: Directory to save downloaded files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Downloading RAVDESS Dataset")
    print("="*60)
    print("\nThis dataset contains:")
    print("  - 1440 speech files (60 trials per actor x 24 actors)")
    print("  - 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised")
    print("  - 24 professional actors (12 female, 12 male)")
    print()
    
    # Download each actor's audio files
    for actor_id in ACTORS:
        actor_folder = f"Actor_{actor_id:02d}"
        zip_filename = f"Audio_Speech_Actors_01-24.zip"
        
        print(f"\nDownloading Actor {actor_id:02d}...")
        
        # RAVDESS dataset structure
        actor_url = f"https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
        zip_path = os.path.join(output_dir, zip_filename)
        
        # Download only once
        if actor_id == 1:
            if not os.path.exists(zip_path):
                try:
                    download_url(actor_url, zip_path)
                except Exception as e:
                    print(f"Error downloading: {e}")
                    print("\n⚠️ Automatic download failed.")
                    print("\nPlease download manually:")
                    print("1. Go to: https://zenodo.org/record/1188976")
                    print("2. Download 'Audio_Speech_Actors_01-24.zip'")
                    print(f"3. Place it in: {output_dir}/")
                    print("4. Run this script again")
                    return False
            
            # Extract
            print("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            print("✓ Dataset downloaded and extracted!")
            break
    
    return True


def prepare_ravdess_dataset(raw_dir="./data_raw", output_dir="./data"):
    """
    Organize RAVDESS dataset into emotion folders
    
    Args:
        raw_dir: Directory with extracted RAVDESS files
        output_dir: Output directory with organized structure
    """
    print("\n" + "="*60)
    print("Organizing Dataset")
    print("="*60)
    
    # Create emotion directories
    for emotion in set(EMOTION_MAPPING.values()):
        os.makedirs(os.path.join(output_dir, emotion), exist_ok=True)
    
    # Find all audio files
    audio_files = []
    actors_dir = raw_dir
    
    if not os.path.exists(actors_dir):
        print(f"❌ Directory not found: {actors_dir}")
        print("Please run the download step first.")
        return False
    
    for actor_folder in os.listdir(actors_dir):
        actor_path = os.path.join(actors_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        
        for audio_file in os.listdir(actor_path):
            if audio_file.endswith('.wav'):
                audio_files.append(os.path.join(actor_path, audio_file))
    
    print(f"\nFound {len(audio_files)} audio files")
    
    # Organize files by emotion
    stats = {emotion: 0 for emotion in set(EMOTION_MAPPING.values())}
    
    for audio_path in tqdm(audio_files, desc="Organizing files"):
        # RAVDESS filename format: 03-01-06-01-02-01-12.wav
        # Position 3 is emotion identifier
        filename = os.path.basename(audio_path)
        parts = filename.split('-')
        
        if len(parts) >= 3:
            emotion_id = parts[2]
            emotion = EMOTION_MAPPING.get(emotion_id, 'unknown')
            
            if emotion != 'unknown':
                # Copy file to emotion folder
                dest_path = os.path.join(output_dir, emotion, filename)
                shutil.copy2(audio_path, dest_path)
                stats[emotion] += 1
    
    # Print statistics
    print("\n✓ Dataset organized successfully!")
    print("\nDataset statistics:")
    total = 0
    for emotion, count in sorted(stats.items()):
        print(f"  {emotion:12s}: {count:4d} files")
        total += count
    print(f"  {'TOTAL':12s}: {total:4d} files")
    
    return True


def create_metadata_csv(data_dir="./data", output_csv="./dataset_metadata.csv"):
    """
    Create metadata CSV for the dataset
    
    Args:
        data_dir: Directory with organized emotion folders
        output_csv: Output CSV path
    """
    print("\n" + "="*60)
    print("Creating Metadata CSV")
    print("="*60)
    
    data = []
    
    for emotion in os.listdir(data_dir):
        emotion_dir = os.path.join(data_dir, emotion)
        
        if not os.path.isdir(emotion_dir):
            continue
        
        for audio_file in os.listdir(emotion_dir):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(emotion_dir, audio_file)
                
                # Parse RAVDESS filename
                # Format: Modality-VocalChannel-Emotion-EmotionIntensity-Statement-Repetition-Actor
                parts = audio_file.replace('.wav', '').split('-')
                
                if len(parts) == 7:
                    data.append({
                        'path': audio_path,
                        'emotion': emotion,
                        'filename': audio_file,
                        'actor': int(parts[6]),
                        'gender': 'female' if int(parts[6]) % 2 == 0 else 'male',
                        'intensity': 'normal' if parts[3] == '01' else 'strong'
                    })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    print(f"✓ Metadata saved to: {output_csv}")
    print(f"\nDataset summary:")
    print(f"  Total files: {len(df)}")
    print(f"  Emotions: {df['emotion'].nunique()}")
    print(f"  Actors: {df['actor'].nunique()}")
    print(f"\nGender distribution:")
    print(df['gender'].value_counts())
    print(f"\nEmotion distribution:")
    print(df['emotion'].value_counts())
    
    return df


def main():
    """Main function to download and prepare dataset"""
    
    print("="*60)
    print("RAVDESS Dataset Setup")
    print("="*60)
    print("\nThis script will:")
    print("1. Download RAVDESS dataset (~2.5 GB)")
    print("2. Extract and organize files by emotion")
    print("3. Create metadata CSV")
    print()
    
    # Ask for confirmation
    response = input("Continue? (y/n): ").lower().strip()
    if response != 'y':
        print("Setup cancelled.")
        return
    
    # Step 1: Download
    success = download_ravdess_dataset("./data_raw")
    if not success:
        return
    
    # Step 2: Organize
    success = prepare_ravdess_dataset("./data_raw", "./data")
    if not success:
        return
    
    # Step 3: Create metadata
    create_metadata_csv("./data", "./dataset_metadata.csv")
    
    print("\n" + "="*60)
    print("✓ Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Validate the dataset:")
    print("   python data_preparation.py validate --input ./data")
    print()
    print("2. Train the model:")
    print("   python train_wav2vec2_emotion.py")
    print()
    print("3. The training will take approximately:")
    print("   - With GPU: 1-2 hours")
    print("   - With CPU: 6-8 hours")
    print()


if __name__ == "__main__":
    main()