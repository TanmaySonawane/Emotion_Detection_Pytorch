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

def convert_wav_to_mel_npy(wav_path, npy_path, save_png=False, png_path=None):
    """
    Convert a single WAV file into a Mel spectrogram and save as a normalized .npy file.

    Args:
        wav_path (str): Full path to the .wav file
        npy_path (str): Full path to save the .npy (numpy float32) file
        save_png (bool): If True, also save PNG image at png_path (optional)
        png_path (str): Path to save PNG if save_png=True
    Behavior:
        - Loads audio at sr=22050
        - Computes log-mel spectrogram (dB via power_to_db)
        - Normalizes the spectrogram (per-spectrogram mean/std -> mean=0,std=1)
        - Saves normalized spectrogram as np.float32 in .npy
        - Optionally creates a PNG for visualization (not used for training)
    """
    try:
        print(f"\n[INFO] Processing file (to .npy): {wav_path}")

        # 1) Load audio
        signal, sr = librosa.load(wav_path, sr=22050)
        if signal.size == 0:
            raise ValueError("Loaded audio empty")

        # 2) Compute Mel spectrogram (power)
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            power=2.0
        )

        # 3) Convert to decibel (log) scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 4) Per-spectrogram normalization (standard score)
        # compute mean/std over all time-frequency cells and normalize
        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db)
        if std < 1e-6:
            std = 1.0
        mel_norm = (mel_spec_db - mean) / std
        mel_norm = mel_norm.astype(np.float32)  # save space and be explicit

        # 5) Save as .npy
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, mel_norm)
        print(f"[SUCCESS] Saved .npy: {npy_path} (shape={mel_norm.shape}, mean={mel_norm.mean():.4f}, std={mel_norm.std():.4f})")

        # 6) Optionally save PNG (for reporting/visualization only) — not used for training
        if save_png and png_path is not None:
            plt.figure(figsize=(6, 3))
            librosa.display.specshow(mel_spec_db, sr=sr, hop_length=512)
            plt.axis('off')
            plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"[INFO] Also saved PNG for visualization: {png_path}")

    except Exception as e:
        print(f"[ERROR] Failed to process {wav_path}")
        print(f"[ERROR MESSAGE] {str(e)}")
        raise

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
                npy_filename = file.replace(".wav", ".npy")
                wav_path = os.path.join(audio_folder_path, file)
                npy_path = os.path.join(mel_folder_path, npy_filename)

                # Skip if NPY already exists (no repeats)
                if os.path.exists(npy_path):
                    print(f"[SKIP] Already exists: {npy_path}")
                    continue

                # Convert file and save .npy (optionally set save_png=True and provide png_path)
                convert_wav_to_mel_npy(wav_path, npy_path, save_png=False, png_path=None)

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
