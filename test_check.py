# Test script to check train_pytorch_spectrogram.py
import sys
import os

# Add Scripts folder to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(script_dir, 'Scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from train_pytorch_spectrogram import build_spectrogram_dataframe, main_train_entry

BASE_AUDIO = r"C:\Users\Owner\OneDrive - University of Massachusetts Dartmouth\Documents\Capstone\Data\RAVDESS"
BASE_MEL   = r"C:\Users\Owner\OneDrive - University of Massachusetts Dartmouth\Documents\Capstone\Data\RAVDESSMEL"

print("[INFO] Building dataframe...")
df = build_spectrogram_dataframe(BASE_AUDIO, BASE_MEL)

print("\n[INFO] Checking dataframe structure...")
print("Columns:", df.columns.tolist())
print("\nRow sample:\n", df.head(3).to_string(index=False))
print("\nCounts per emotion (by label):\n", df['emotion_label'].value_counts())
print("\nAny missing mel files? ", df['mel_path'].isna().sum(), "  Has_png false:", (~df['has_png']).sum())

print("\n[INFO] Starting training...")
metrics = main_train_entry(BASE_AUDIO, BASE_MEL, out_dir="./results_test", batch_size=8, num_epochs=1, lr=1e-3, device_str="cpu")

print("\n[INFO] Completed. Metrics:", metrics)

