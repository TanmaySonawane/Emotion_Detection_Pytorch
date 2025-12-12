# ================================
# mel.py - Part 1
# Imports and Path Setup
# ================================

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Root dataset path (your audio files)
BASE_AUDIO_PATH = r"C:\Users\Owner\OneDrive - University of Massachusetts Dartmouth\Documents\Capstone\Data\RAVDESS"

# Destination path for Mel Spectrogram images
BASE_MEL_PATH = r"C:\Users\Owner\OneDrive - University of Massachusetts Dartmouth\Documents\Capstone\Data\RAVDESSMEL"


def check_and_create_folders():
    """
    Checks if the output folder exists.
    If not, create it and its 24 actor subfolders.
    """

    print("\n[INFO] Checking output folder structure...")

    # Create main mel folder if it does not exist
    if not os.path.exists(BASE_MEL_PATH):
        os.makedirs(BASE_MEL_PATH)
        print(f"[INFO] Created main folder: {BASE_MEL_PATH}")
    else:
        print(f"[INFO] Main folder already exists: {BASE_MEL_PATH}")

    # Create Actor subfolders
    for i in range(1, 25):
        folder_name = f"Actor_{str(i).zfill(2)}"
        folder_path = os.path.join(BASE_MEL_PATH, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"[INFO] Created subfolder: {folder_path}")
        else:
            print(f"[INFO] Subfolder exists: {folder_path}")

def convert_wav_to_mel_png(wav_path, png_path):
    """
    Convert a single WAV file into a Mel Spectrogram PNG.

    Args:
        wav_path (str): Full path to the .wav file
        png_path (str): Full path to save the .png file
    """

    try:
        print(f"\n[INFO] Processing file: {wav_path}")

        # Load audio
        signal, sr = librosa.load(wav_path, sr=22050)
        print(f"[DEBUG] Loaded audio. Shape: {signal.shape}, Sample rate: {sr}")

        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

        print("[DEBUG] Mel spectrogram created.")

        # Convert to decibel scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Plot and save
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512)
        plt.axis('off')

        # Save spectrogram
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"[SUCCESS] Saved PNG: {png_path}")

    except Exception as e:
        print(f"[ERROR] Failed to process {wav_path}")
        print(f"[ERROR MESSAGE] {str(e)}")

def generate_all_mels():
    """
    Loop through all Actor folders,
    convert each WAV into a Mel spectrogram PNG.
    """

    print("\n[INFO] Starting FULL mel spectrogram generation process...")
    total_files = 0
    converted_files = 0

    for i in range(1, 25):
        actor_folder = f"Actor_{str(i).zfill(2)}"
        audio_folder_path = os.path.join(BASE_AUDIO_PATH, actor_folder)
        mel_folder_path = os.path.join(BASE_MEL_PATH, actor_folder)

        print(f"\n[INFO] Scanning folder: {audio_folder_path}")

        if not os.path.exists(audio_folder_path):
            print(f"[WARNING] Folder not found: {audio_folder_path}")
            continue

        files = os.listdir(audio_folder_path)

        for file in tqdm(files):
            if file.endswith(".wav"):
                total_files += 1

                # Create corresponding PNG file name
                png_filename = file.replace(".wav", ".png")
                wav_path = os.path.join(audio_folder_path, file)
                png_path = os.path.join(mel_folder_path, png_filename)

                # Skip if PNG already exists (no repeats)
                if os.path.exists(png_path):
                    print(f"[SKIP] Already exists: {png_path}")
                    continue

                # Convert file
                convert_wav_to_mel_png(wav_path, png_path)
                converted_files += 1

    print("\n======= SUMMARY =======")
    print(f"Total WAV files found: {total_files}")
    print(f"Total PNG files created: {converted_files}")
    print("[INFO] Mel generation complete!")

if __name__ == "__main__":
    print("\n===== MEL SPECTROGRAM SCRIPT STARTED =====")

    check_and_create_folders()
    generate_all_mels()

    print("\n===== MEL SPECTROGRAM SCRIPT FINISHED =====")
