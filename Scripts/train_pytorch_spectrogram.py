import os
import pandas as pd
"""
This script is used to train CNN+BiLTSM

It takes coded file names, translates them, finds each audio file and its corresponding spectrogram, and makes a dataframe of the data
"""
# Mapping for emotion id -> readable label (RAVDESS specification)
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def parse_ravdess_filename(path_or_filename):
    """
    Parse a RAVDESS filename and return a dictionary of fields.

    Accepts either a filename (e.g., "03-01-06-01-02-01-12.wav")
    or a full path (will use the basename).

    Returns:
        info (dict) with keys:
            - emotion_id (str)
            - emotion_label (str)
            - intensity (str)
            - statement (str)
            - repetition (str)
            - actor_id (str)
            - actor_num (int)
            - gender (str)   ("male" or "female")
            - filename (str) (basename)
    On error returns None and prints an error message.
    """
    try:
        # Accept path or bare filename, use basename to be robust
        filename = os.path.basename(path_or_filename)
        # Debug print disabled to reduce output (uncomment if needed)
        # print(f"[DEBUG] parse_ravdess_filename received: {filename}")

        # Remove extension if present
        name_no_ext, ext = os.path.splitext(filename)
        parts = name_no_ext.split("-")

        if len(parts) != 7:
            print(f"[ERROR] Unexpected filename format (expected 7 parts): {filename}")
            return None

        modality, vocal_channel, emotion_id, intensity, statement, repetition, actor_id = parts

        # Map emotion id to label; default to 'unknown' if not found
        emotion_label = EMOTION_MAP.get(emotion_id, "unknown")

        # Actor numeric id and gender (odd -> male, even -> female)
        actor_num = int(actor_id)
        gender = "male" if (actor_num % 2 == 1) else "female"

        info = {
            "filename": filename,
            "emotion_id": emotion_id,
            "emotion_label": emotion_label,
            "intensity": intensity,
            "statement": statement,
            "repetition": repetition,
            "actor_id": actor_id,
            "actor_num": actor_num,
            "gender": gender
        }

        # Debug print disabled to reduce output (uncomment if needed)
        # print(f"[DEBUG] Parsed: {info}")
        return info

    except Exception as e:
        print(f"[ERROR] Exception in parse_ravdess_filename for input '{path_or_filename}': {e}")
        return None


def build_spectrogram_dataframe(base_audio_path, base_mel_path, combine_calm_neutral=False):
    """
    Build a pandas DataFrame that pairs audio files (.wav) with their corresponding
    spectrogram PNG paths and parsed labels.

    Args:
        base_audio_path (str): Path to root RAVDESS folder containing Actor_01..Actor_24
        base_mel_path (str): Path to root folder where corresponding spectrogram PNGs are stored
                           (must have same actor subfolder structure)
        combine_calm_neutral (bool): If True, maps calm (02) to neutral (01) to create 7 classes.
                                    If False, keeps 8 separate emotion classes.

    Returns:
        df (pd.DataFrame) with columns:
            - wav_path (str)
            - mel_path (str)
            - filename (str)
            - emotion_id (str)  # original emotion_id from filename
            - emotion (int)     # emotion code for model (calm->neutral if combine_calm_neutral=True)
            - emotion_label (str)  
            - actor_id (str)
            - actor_num (int)
            - gender (str)
            - has_png (bool)  # whether png exists at png_path

    Behavior / prints:
        - Prints counts per emotion and summary counts
        - Prints warnings for missing actor folders or missing PNGs
    """
    records = []
    total_wavs_found = 0
    missing_audio_folders = []
    missing_png_count = 0

    print(f"\n[INFO] Building spectrogram dataframe from:\n  audio: {base_audio_path}\n  mel : {base_mel_path}")

    # Expect actor folders Actor_01 .. Actor_24 (but function will gracefully skip missing)
    for i in range(1, 25):
        actor_folder = f"Actor_{str(i).zfill(2)}"
        audio_actor_path = os.path.join(base_audio_path, actor_folder)
        mel_actor_path = os.path.join(base_mel_path, actor_folder)

        if not os.path.exists(audio_actor_path):
            print(f"[WARNING] Missing audio actor folder: {audio_actor_path}")
            missing_audio_folders.append(audio_actor_path)
            continue

        # List files in actor folder
        try:
            files = sorted(os.listdir(audio_actor_path))
        except Exception as e:
            print(f"[ERROR] Cannot list files in {audio_actor_path}: {e}")
            continue

        for fname in files:
            if not fname.lower().endswith(".wav"):
                # ignore non-wav files
                continue

            total_wavs_found += 1
            wav_path = os.path.join(audio_actor_path, fname)

            # Corresponding png has same filename but .png extension in mel folder
            png_fname = os.path.splitext(fname)[0] + ".png"
            png_path = os.path.join(mel_actor_path, png_fname)

            parsed = parse_ravdess_filename(fname)
            if parsed is None:
                # skip malformed filenames but warn
                print(f"[WARNING] Skipping malformed filename: {fname}")
                continue

            has_png = os.path.exists(png_path)
            if not has_png:
                missing_png_count += 1
                print(f"[WARN] PNG not found for: {wav_path} -> expected: {png_path}")

            # Convert emotion_id string to integer for the model
            emotion_id_str = parsed["emotion_id"]
            emotion_int = int(emotion_id_str)
            
            # Optionally combine calm (02) and neutral (01) into one class for 7-class experiments
            if combine_calm_neutral and emotion_int == 2:  # calm emotion
                emotion_int = 1   # map to neutral
                # Debug print disabled to reduce output (uncomment if needed)
                # print(f"[INFO] Combined calm->neutral for: {fname}")

            record = {
                "wav_path": wav_path,
                "mel_path": png_path,   # same file on disk; name standardized to mel_path
                "filename": parsed["filename"],
                "emotion_id": parsed["emotion_id"],         # string like "06" (original, unchanged)
                "emotion": emotion_int,                      # numeric (calm->neutral if combine_calm_neutral=True)
                "emotion_label": parsed["emotion_label"],   # human readable like 'fearful' (original label)
                "actor_id": parsed["actor_id"],
                "actor_num": parsed["actor_num"],
                "gender": parsed["gender"],
                "has_png": has_png
            }
            records.append(record)

    # Build DataFrame
    df = pd.DataFrame.from_records(records)
    
    # Print emotion configuration
    unique_emotions = sorted(df['emotion'].unique())
    num_classes = len(unique_emotions)
    print(f"\n[INFO] Emotion configuration: {num_classes} classes")
    if combine_calm_neutral:
        print(f"[INFO] Calm and neutral are COMBINED into one class (7 total emotions)")
    else:
        print(f"[INFO] Calm and neutral are SEPARATE (8 total emotions)")
    
    # Debug: Print DataFrame info
    print(f"\n[DEBUG] DataFrame shape: {df.shape}")
    print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"[DEBUG] Sample row:\n{df.iloc[0]}")

    # ----- Remove exact duplicate rows based on wav_path (if any exist) -----
    # This removes identical duplicates and keeps the first occurrence.
    before_dedup = len(df)
    df.drop_duplicates(subset=["wav_path"], inplace=True, keep="first")
    after_dedup = len(df)
    removed = before_dedup - after_dedup
    if removed > 0:
        print(f"[INFO] Removed {removed} duplicate records based on wav_path. New count: {after_dedup}")

    # Reset index after dedup
    df.reset_index(drop=True, inplace=True)

    print("\n[SUMMARY] Spectrogram dataframe build complete.")
    print(f"Total WAV files discovered: {total_wavs_found}")
    print(f"Total records in dataframe: {len(df)}")
    print(f"Missing audio actor folders: {len(missing_audio_folders)}")
    print(f"Missing PNG files: {missing_png_count}")

    if len(df) == 0:
        print("[ERROR] DataFrame is empty. Check your base paths and folder structure.")
        return df

    # Print class distribution (emotion) - show both integer codes and labels
    try:
        print("\n[INFO] DataFrame columns:", df.columns.tolist())
        class_counts = df["emotion"].value_counts(dropna=False).sort_index()
        print("\n[INFO] Emotion distribution (integer codes - counts):")
        for emotion_code, cnt in class_counts.items():
            # Get label for this emotion code
            matching_rows = df[df["emotion"] == emotion_code]
            if len(matching_rows) > 0:
                labels = matching_rows["emotion_label"].unique()
                label_str = ", ".join(labels) if len(labels) > 0 else "unknown"
                print(f"   Emotion {emotion_code} ({label_str}): {cnt}")
    except Exception as e:
        print(f"[ERROR] Could not compute class distribution: {e}")

    # Print sample rows for debugging
    print("\n[DEBUG] Sample dataframe rows (first 5):")
    print(df.head(5).to_string(index=False))

    return df


from PIL import Image
import torchvision.transforms as T

def load_spectrogram_image(img_path: str):
    """
    Load a mel-spectrogram PNG and convert it into a normalized tensor.

    Steps:
    1. Open the PNG using PIL
    2. Convert it to grayscale ('L' mode) since mel images are single-channel
    3. Resize to a standard input size
    4. Convert to tensor (shape: 1 × H × W)
    5. Normalize pixel values

    Args:
        img_path: Full path to the mel-spectrogram PNG.

    Returns:
        torch.Tensor of shape (1, H, W)
    """
    try:
        # Debug print disabled to reduce output (uncomment if needed)
        # print(f"[DEBUG] load_spectrogram_image: Loading from {img_path}")
        
        # Check if file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"[ERROR] PNG file not found: {img_path}")
        
        # Standard image size for CNNs; change later if needed
        target_height = 224
        target_width = 224

        transforms = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((target_height, target_width)),
            T.ToTensor(),                           # converts 0–255 to 0–1
            T.Normalize(mean=[0.5], std=[0.5])      # scale to [-1, +1]
        ])

        img = Image.open(img_path)
        img_tensor = transforms(img)
        # Debug print disabled to reduce output
        # print(f"[DEBUG] load_spectrogram_image: Successfully loaded, shape: {img_tensor.shape}")
        return img_tensor
        
    except Exception as e:
        print(f"[ERROR] load_spectrogram_image failed for {img_path}: {str(e)}")
        raise

from torch.utils.data import Dataset
import torch

class EmotionDataset(Dataset):
    """
    PyTorch Dataset for mel-spectrogram based emotion classification.

    Each item returned is:
        image_tensor: the mel-spectrogram (1 × H × W)
        label: integer class label

    The DataFrame must contain:
        mel_path: path to spectrogram PNG
        emotion: integer emotion code from RAVDESS
    """

    def __init__(self, df, emotion_map=None):
        """
        Args:
            df: Pre-built DataFrame linking audio and mel paths.
            emotion_map: Optional dictionary mapping raw emotion integers 
                         to 0..N class indices.
                         Example: {1:0, 2:1, 3:2, ...}
        """
        print(f"[INFO] EmotionDataset.__init__: Starting with {len(df)} rows")
        print(f"[INFO] EmotionDataset.__init__: DataFrame columns: {df.columns.tolist()}")
        
        # Check required columns exist
        required_cols = ['mel_path', 'emotion']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"[ERROR] DataFrame missing required columns: {missing_cols}")

        # Filter out entries with missing spectrograms
        df = df[df['mel_path'].notna()].reset_index(drop=True)
        print(f"[INFO] EmotionDataset.__init__: After filtering missing mel_path, {len(df)} rows remain")
        self.df = df

        # If no mapping supplied, create sorted mapping automatically
        if emotion_map is None:
            unique_emotions = sorted(df['emotion'].unique())
            self.emotion_map = {e: i for i, e in enumerate(unique_emotions)}
            print(f"[INFO] EmotionDataset.__init__: Created emotion_map automatically: {self.emotion_map}")
        else:
            self.emotion_map = emotion_map
            print(f"[INFO] EmotionDataset.__init__: Using provided emotion_map: {self.emotion_map}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            mel_path = row['mel_path']
            raw_emotion = row['emotion']
            
            # Debug print disabled to reduce output (uncomment if needed)
            # print(f"[DEBUG] EmotionDataset.__getitem__: idx={idx}, file={row.get('filename', 'unknown')}, emotion={raw_emotion}")

            # Load spectrogram array from .npy and convert to tensor shaped (1,H,W)
            image_tensor = load_spectrogram_npy(mel_path, target_size=(224,224))


            # Convert raw emotion code into class index
            if raw_emotion not in self.emotion_map:
                raise ValueError(f"[ERROR] Emotion {raw_emotion} not in emotion_map {self.emotion_map}")
            
            label = self.emotion_map[raw_emotion]
            label = torch.tensor(label, dtype=torch.long)
            
            # Debug print disabled to reduce output
            # print(f"[DEBUG] EmotionDataset.__getitem__: Successfully loaded, label={label.item()}")

            return image_tensor, label
            
        except Exception as e:
            print(f"[ERROR] EmotionDataset.__getitem__ failed for idx={idx}: {str(e)}")
            raise

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_cnn_bilstm_model(
    num_classes: int,
    in_channels: int = 1,
    cnn_channels: list = [16, 32, 64],
    kernel_sizes: list = [3, 3, 3],
    pool_kernel: int = 2,
    lstm_hidden: int = 128,
    lstm_layers: int = 1,
    dropout: float = 0.3,
    bidirectional: bool = True
) -> nn.Module:
    """
    Build a CNN + BiLSTM model for spectrogram-based emotion recognition.

    Architecture overview (high level):
      Input: (B, in_channels=1, H, W) -- a mel spectrogram image
      1) Several Conv2D -> BatchNorm -> ReLU -> MaxPool blocks
      2) Collapse frequency dimension: average over freq bins -> produce sequence of length ~W'
         (B, C, H', W') -> avg over H' -> (B, C, W') -> transpose -> (B, W', C)
      3) Feed (B, W', C) to a BiLSTM to capture temporal dependencies across time frames
      4) Take the final BiLSTM hidden state(s) -> feed dropout -> Linear -> logits (num_classes)

    Returns:
        nn.Module (ready to .to(device))
    """

    class CNN_BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()

            # Validate lengths
            assert len(cnn_channels) == len(kernel_sizes), "cnn_channels and kernel_sizes must match lengths."

            # Build convolutional feature extractor
            conv_layers = []
            current_channels = in_channels
            for i, out_ch in enumerate(cnn_channels):
                k = kernel_sizes[i]
                conv_layers.append(nn.Conv2d(current_channels, out_ch, kernel_size=k, padding=k//2))
                conv_layers.append(nn.BatchNorm2d(out_ch))
                conv_layers.append(nn.ReLU(inplace=True))
                conv_layers.append(nn.MaxPool2d(kernel_size=pool_kernel))
                current_channels = out_ch

            self.cnn = nn.Sequential(*conv_layers)
            print(f"[INFO] CNN feature extractor built with channels: {cnn_channels}")

            # After CNN, we'll collapse freq dimension and feed time frames into LSTM
            self.lstm_input_size = current_channels  # features per time-step after collapsing freq
            self.lstm_hidden_size = lstm_hidden
            self.lstm_num_layers = lstm_layers
            self.lstm_bidirectional = bidirectional

            self.bilstm = nn.LSTM(
                input_size=self.lstm_input_size,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_num_layers,
                batch_first=True,
                bidirectional=self.lstm_bidirectional
            )

            # Classifier head
            lstm_directions = 2 if self.lstm_bidirectional else 1
            fc_input_dim = self.lstm_hidden_size * lstm_directions
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(fc_input_dim, fc_input_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fc_input_dim // 2, num_classes)
            )

            print(f"[INFO] BiLSTM created (hidden={lstm_hidden}, layers={lstm_layers}, bidir={bidirectional}).")
            print(f"[INFO] Classifier head created (num_classes={num_classes}).")

        def forward(self, x):
            """
            x: tensor shape (B, 1, H, W)
            Returns logits shape (B, num_classes)
            """
            # 1) CNN feature extraction
            # conv_out shape -> (B, C, H_c, W_c)
            conv_out = self.cnn(x)
            # Debug prints (sizes)
            # Note: don't print inside high-frequency loops in real training; here it's for clarity
            # print("[DEBUG] conv_out.shape:", conv_out.shape)

            # 2) Collapse frequency dimension (H_c) to create sequence across time dimension (W_c)
            # Average across frequency axis to reduce dimensionality: (B, C, H_c, W_c) -> (B, C, W_c)
            seq = conv_out.mean(dim=2)  # average over H_c
            # Now seq shape is (B, C, W_c). We want (B, W_c, C) for LSTM (time-major is second dim)
            seq = seq.permute(0, 2, 1)   # (B, W_c, C)
            # print("[DEBUG] seq.shape (B, T, F):", seq.shape)

            # 3) BiLSTM expects (B, T, F)
            lstm_out, (h_n, c_n) = self.bilstm(seq)
            # lstm_out shape -> (B, T, hidden * num_directions)
            # h_n shape -> (num_layers * num_directions, B, hidden)

            # 4) Pooling / choose final representation:
            # We'll use last time-step output from LSTM (alternatively mean pooling over T)
            # last_output = lstm_out[:, -1, :]  # (B, hidden * num_directions)
            # Another robust option: use h_n (last layer hidden states)
            # h_n has shape (num_layers * num_dirs, B, hidden). We take the last layer's states
            if self.lstm_bidirectional:
                # For bidirectional, the last layer has two directions; we concatenate their final states
                # indices: [-2, -1] correspond to the last layer's forward and backward states
                last_layer_forward = h_n[-2]  # shape (B, hidden)
                last_layer_backward = h_n[-1] # shape (B, hidden)
                final_feat = torch.cat([last_layer_forward, last_layer_backward], dim=1)  # (B, hidden*2)
            else:
                final_feat = h_n[-1]  # (B, hidden)

            # 5) Classification head
            logits = self.classifier(final_feat)  # (B, num_classes)

            return logits

    return CNN_BiLSTM()

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os
import time

def train_pytorch_spectrogram(
    df,
    emotion_col: str = "emotion",
    actor_col: str = "actor_num",
    train_actors: list = None,
    val_actors: list = None,
    test_actors: list = None,
    emotion_map: dict = None,
    num_classes: int = None,
    batch_size: int = 32,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    save_model_path: str = "./best_cnn_bilstm.pth",
    patience: int = 5
):
    """
    Train the CNN+BiLSTM model on spectrogram images using actor-wise splits.

    Key behavior:
      - Splits df into train/val/test based on actor IDs passed as lists (actor-wise split).
      - Builds EmotionDataset and DataLoader for each split.
      - Trains model and validates each epoch.
      - Saves best model (by validation F1-score).
      - Runs final evaluation on test set and prints confusion matrix & classification report.

    Args:
      df: DataFrame produced by build_spectrogram_dataframe()
      train_actors/val_actors/test_actors: lists of actor numeric IDs (e.g., [1,2,...])
      emotion_map: mapping of raw emotion codes to 0..C-1. If None, it will be inferred.
      num_classes: number of output classes. If None, inferred from emotion_map or data.
      device: "cuda" or "cpu"
    """

    start_time = time.time()
    print("\n[INFO] Starting train_pytorch_spectrogram()")

    # ----------------
    # 1) Validate and infer actors if not provided
    # ----------------
    unique_actors = sorted(df[actor_col].unique().tolist())
    print(f"[INFO] Unique actors found: {unique_actors}")

    if train_actors is None or val_actors is None or test_actors is None:
        # Default split: first 18 actors train, next 3 val, final 3 test (adjustable)
        if len(unique_actors) >= 24:
            train_actors = list(range(1, 19))   # Actors 1..18
            val_actors = list(range(19, 22))    # Actors 19..21
            test_actors = list(range(22, 25))   # Actors 22..24
        else:
            # fallback: 70/15/15 split of actor list
            n = len(unique_actors)
            t = int(n * 0.7)
            v = int(n * 0.15)
            train_actors = unique_actors[:t]
            val_actors = unique_actors[t:t+v]
            test_actors = unique_actors[t+v:]
        print(f"[INFO] Using default actor-wise splits -> train:{train_actors}, val:{val_actors}, test:{test_actors}")
    else:
        print(f"[INFO] Using provided actor splits -> train:{train_actors}, val:{val_actors}, test:{test_actors}")

    # ----------------
    # 2) Make splits from df
    # ----------------
    train_df = df[df[actor_col].isin(train_actors)].reset_index(drop=True)
    val_df = df[df[actor_col].isin(val_actors)].reset_index(drop=True)
    test_df = df[df[actor_col].isin(test_actors)].reset_index(drop=True)

    print(f"[INFO] Train records: {len(train_df)}, Val records: {len(val_df)}, Test records: {len(test_df)}")

    # --------------
    # 3) Emotion mapping & classes
    # --------------
    if emotion_map is None:
        unique_emotions = sorted(train_df[emotion_col].unique().tolist())
        emotion_map = {e: i for i, e in enumerate(unique_emotions)}
        print(f"[INFO] Built emotion_map from training data: {emotion_map}")

    if num_classes is None:
        num_classes = len(set(emotion_map.values()))
        print(f"[INFO] Inferred num_classes = {num_classes}")

    # --------------
    # 4) Datasets and DataLoaders
    # --------------
    train_dataset = EmotionDataset(train_df, emotion_map=emotion_map)
    val_dataset = EmotionDataset(val_df, emotion_map=emotion_map)
    test_dataset = EmotionDataset(test_df, emotion_map=emotion_map)

    # Use num_workers=0 on Windows to avoid multiprocessing issues
    import sys
    num_workers = 0 if sys.platform == 'win32' else 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(num_workers > 0))

    print("[INFO] DataLoaders created. Example batch shapes will be printed on first iteration.")

    # --------------
    # 5) Model, optimizer, criterion, scheduler
    # --------------
    model = build_cnn_bilstm_model(num_classes=num_classes)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[INFO] Model moved to device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # --------------
    # 6) Training loop
    # --------------
    best_val_f1 = -1.0
    epochs_without_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\n[TRAIN] Epoch {epoch}/{num_epochs} -------------------------")
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # logits (B, num_classes)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            # Print occasional batch debug info
            if batch_idx % 20 == 0 or batch_idx == 1:
                print(f"[TRAIN] E{epoch} B{batch_idx} Loss: {loss.item():.4f} Acc (batch): {(preds==labels).float().mean().item():.4f}")

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        print(f"[TRAIN] Epoch {epoch} summary: Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # -------------------
        # Validation
        # -------------------
        model.eval()
        all_preds = []
        all_labels = []
        val_loss_accum = 0.0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_accum += loss.item() * images.size(0)
                val_total += images.size(0)
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_loss = val_loss_accum / val_total if val_total > 0 else 0.0
        # Compute validation F1 (macro)
        val_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0.0
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels)) if len(all_labels) > 0 else 0.0

        print(f"[VAL] Epoch {epoch} Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1(macro): {val_f1:.4f}")

        # Scheduler step (based on val_f1)
        scheduler.step(val_f1)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'emotion_map': emotion_map
            }, save_model_path)
            print(f"[INFO] New best model saved at epoch {epoch} with val_f1 {val_f1:.4f} -> {save_model_path}")
        else:
            epochs_without_improve += 1
            print(f"[INFO] No improvement for {epochs_without_improve} epoch(s).")

        if epochs_without_improve >= patience:
            print(f"[INFO] Early stopping triggered after {epochs_without_improve} epochs without improvement.")
            break

    # --------------
    # 7) Final evaluation on test set using best saved model
    # --------------
    print("\n[INFO] Loading best model for final evaluation...")
    checkpoint = torch.load(save_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy().tolist())
            test_labels.extend(labels.cpu().numpy().tolist())

    print("\n[TEST] Classification report:")
    print(classification_report(test_labels, test_preds, digits=4))
    print("\n[TEST] Confusion matrix:")
    print(confusion_matrix(test_labels, test_preds))

    elapsed = time.time() - start_time
    print(f"\n[INFO] Training and evaluation finished in {elapsed/60:.2f} minutes.")

    return {
        "model": model,
        "emotion_map": emotion_map,
        "train_actors": train_actors,
        "val_actors": val_actors,
        "test_actors": test_actors,
        "best_val_f1": best_val_f1,
        "save_model_path": save_model_path
    }

# ==========================
# Utilities & remaining functions
# ==========================
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import json

# --------------------------
# Logging / utility helpers
# --------------------------
def setup_logging(log_dir: str = "./logs", log_filename: str = "training_log.txt"):
    """
    Configure logging:
      - ERROR+ messages to console (stderr)
      - INFO+ messages to log file (log_dir/log_filename)
    """
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, log_filename)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # capture all levels, handlers will filter

    # Remove existing handlers if re-running
    if logger.handlers:
        for h in logger.handlers:
            logger.removeHandler(h)

    # File handler - INFO and above
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler - only ERROR and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch_formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    logger.info(f"Logging initialized. Log file: {log_path}")
    return logger, log_path


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


# --------------------------
# actor-wise splitter
# --------------------------
def actor_wise_split(df, actor_col="actor_num", train_actors=None, val_actors=None, test_actors=None):
    """
    Assign a 'split' column to df using actor IDs.
    If train/val/test lists are None, default split is used:
      train: actors 1..18, val: 19..21, test: 22..24 (if 24 actors exist)
    Returns df with an added 'split' column with values 'train','val','test'
    """
    logger = logging.getLogger()
    unique_actors = sorted(df[actor_col].unique().tolist())
    logger.info(f"actor_wise_split: unique actors detected = {unique_actors}")

    if train_actors is None or val_actors is None or test_actors is None:
        if len(unique_actors) >= 24:
            train_actors = list(range(1, 19))
            val_actors = list(range(19, 22))
            test_actors = list(range(22, 25))
        else:
            n = len(unique_actors)
            t = int(n * 0.7)
            v = int(n * 0.15)
            train_actors = unique_actors[:t]
            val_actors = unique_actors[t:t+v]
            test_actors = unique_actors[t+v:]

        logger.info(f"actor_wise_split: using default actor splits: train={train_actors}, val={val_actors}, test={test_actors}")
    else:
        logger.info(f"actor_wise_split: using provided actor splits: train={train_actors}, val={val_actors}, test={test_actors}")

    # Assign split column
    def choose_split(a):
        if a in train_actors:
            return "train"
        if a in val_actors:
            return "val"
        if a in test_actors:
            return "test"
        return "unused"

    df = df.copy()
    df['split'] = df[actor_col].apply(choose_split)
    counts = df['split'].value_counts().to_dict()
    logger.info(f"actor_wise_split: split counts = {counts}")
    return df, {"train": train_actors, "val": val_actors, "test": test_actors}


# --------------------------
# dataloader builder
# --------------------------
from torch.utils.data import DataLoader

def get_dataloaders(train_df, val_df, test_df, emotion_map, batch_size=32, num_workers=None):
    """
    Build DataLoaders for train/val/test from DataFrames.
    Returns (train_loader, val_loader, test_loader, datasets_dict)
    """
    import sys
    # Use num_workers=0 on Windows to avoid multiprocessing issues
    if num_workers is None:
        num_workers = 0 if sys.platform == 'win32' else 4
    
    logger = logging.getLogger()
    train_ds = EmotionDataset(train_df, emotion_map)
    val_ds = EmotionDataset(val_df, emotion_map)
    test_ds = EmotionDataset(test_df, emotion_map)

    pin_memory = (num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # Small informative log (not printed to console unless error)
    logger.info(f"get_dataloaders: train size={len(train_ds)} val size={len(val_ds)} test size={len(test_ds)} batch_size={batch_size}")
    # Optionally check a mini-batch for shapes (quiet unless logging to file)
    try:
        batch = next(iter(train_loader))
        logger.info(f"get_dataloaders: sample batch shapes - images: {batch[0].shape}, labels: {batch[1].shape}")
    except Exception as e:
        logger.error(f"get_dataloaders: failed to fetch sample batch: {e}")

    datasets = {"train": train_ds, "val": val_ds, "test": test_ds}
    return train_loader, val_loader, test_loader, datasets


# --------------------------
# training helpers (train_one_epoch / validate_epoch)
# --------------------------
import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, criterion, optimizer, device, log_interval=50):
    """
    Train model for one epoch. Returns (epoch_loss, epoch_acc)
    Uses logging to file, only ERRORs to console.
    """
    logger = logging.getLogger()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if batch_idx % log_interval == 0:
            logger.info(f"train_one_epoch: batch {batch_idx} loss={loss.item():.4f} batch_acc={(preds==labels).float().mean().item():.4f}")

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    logger.info(f"train_one_epoch: epoch_loss={epoch_loss:.4f} epoch_acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """
    Run validation: returns (val_loss, val_acc, all_preds, all_labels)
    """
    logger = logging.getLogger()
    model.eval()
    val_loss_accum = 0.0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss_accum += loss.item() * images.size(0)
            val_total += images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    val_loss = val_loss_accum / val_total if val_total > 0 else 0.0
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels)) if len(all_labels) > 0 else 0.0
    val_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0.0
    logger.info(f"validate_epoch: val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
    return val_loss, val_acc, all_preds, all_labels


# --------------------------
# evaluation / reporting
# --------------------------
def plot_confusion_matrix_and_save(y_true, y_pred, class_names, out_path_png, normalize=True):
    """
    Save confusion matrix as PNG. `class_names` is a list of labels in index order.
    """
    ensure_dir(os.path.dirname(out_path_png))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalize:
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    else:
        cm_norm = cm

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f" if normalize else "d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix (normalized)" if normalize else "Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path_png)
    plt.close()
    logging.getLogger().info(f"plot_confusion_matrix_and_save: saved confusion matrix to {out_path_png}")


def save_metrics_to_file(metrics_dict, out_path_txt):
    """
    Save metrics (dictionary) to a human-readable text file (and also JSON for programmatic use).
    """
    ensure_dir(os.path.dirname(out_path_txt))
    # Save plain text
    with open(out_path_txt, 'w', encoding='utf-8') as f:
        for k, v in metrics_dict.items():
            f.write(f"{k}: {v}\n")
    # Save JSON copy
    json_path = out_path_txt + ".json"
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(metrics_dict, jf, indent=2)
    logging.getLogger().info(f"save_metrics_to_file: saved metrics to {out_path_txt} and {json_path}")


def evaluate_and_report(model, test_loader, emotion_map, out_dir="./results"):
    """
    Run final evaluation on test_loader, save confusion matrix PNG and metrics TXT/JSON.
    Returns metrics_dict.
    """
    logger = logging.getLogger()
    ensure_dir(out_dir)

    # Gather predictions
    y_true = []
    y_pred = []

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    # Invert emotion_map to produce class_names in order 0..C-1
    inv_map = {v: k for k, v in emotion_map.items()}  # maps class_index -> original emotion code (int)
    # Build readable class names using the global EMOTION_MAP which uses zero-padded string keys ("01","02",...)
    class_names = [EMOTION_MAP.get(str(inv_map[i]).zfill(2), str(inv_map[i])) for i in range(len(inv_map))]
    # Example: inv_map[i] == 6 --> "06" --> EMOTION_MAP["06"] -> "fearful"

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    clf_report = classification_report(y_true, y_pred, zero_division=0)

    # Prepare output artifact paths
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_png = os.path.join(out_dir, f"confusion_matrix_{ts}.png")
    metrics_txt = os.path.join(out_dir, f"metrics_{ts}.txt")

    # Save confusion matrix PNG
    plot_confusion_matrix_and_save(y_true, y_pred, class_names, cm_png, normalize=True)

    # Save metrics to text
    metrics_dict = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "classification_report": clf_report,
        "num_test_samples": len(y_true)
    }
    save_metrics_to_file(metrics_dict, metrics_txt)

    logger.info(f"evaluate_and_report: evaluation done. metrics saved in {metrics_txt}")
    return metrics_dict


# --------------------------
# plot training curves
# --------------------------
def plot_training_curves(history: dict, out_path="./results/training_curves.png"):
    """
    history should be a dict containing lists:
      history = {
          'train_loss': [...],
          'val_loss': [...],
          'train_acc': [...],
          'val_acc': [...],
      }
    """
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(10, 4))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Loss")

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.getLogger().info(f"plot_training_curves: saved training curves to {out_path}")


# --------------------------
# main() glue function
# --------------------------
def main_train_entry(
    base_audio_path,
    base_mel_path,
    out_dir="./results",
    batch_size=32,
    num_epochs=30,
    lr=1e-3,
    device_str="cuda",
    combine_calm_neutral=False
):
    """
    High-level entry that ties everything together. This function:
      - Sets up logging
      - Builds df via build_spectrogram_dataframe(...)
      - Performs actor-wise split
      - Builds dataloaders
      - Instantiates model and trains via modular train_one_epoch / validate_epoch
      - Saves best model and evaluation artifacts
    
    Args:
        combine_calm_neutral (bool): If True, combines calm and neutral into one class (7 emotions).
                                    If False, keeps 8 separate emotion classes.
    """
    # Logging
    logger, log_path = setup_logging(log_dir="./logs")
    start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(out_dir)

    logger.info(f"main_train_entry started at {start_ts}")
    logger.info(f"combine_calm_neutral={combine_calm_neutral} (will use {'7' if combine_calm_neutral else '8'} emotion classes)")
    
    # Build dataframe (uses your earlier function)
    df = build_spectrogram_dataframe(base_audio_path, base_mel_path, combine_calm_neutral=combine_calm_neutral)

    if df is None or len(df) == 0:
        logger.error("main_train_entry: DataFrame empty or None. Exiting.")
        return

    # Actor-wise split (adds df['split'])
    df, actor_splits = actor_wise_split(df, actor_col="actor_num")
    # Optionally save df snapshot
    df.to_csv(os.path.join(out_dir, f"dataframe_snapshot_{start_ts}.csv"), index=False)
    logger.info(f"Saved dataframe snapshot to {out_dir}")

    # Build train/val/test dfs
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    # Save the split CSVs for reuse by other scripts (e.g., train_pytorch_multimodal.py)
    splits_dir = os.path.join(out_dir, "splits")
    ensure_dir(splits_dir)
    train_df.to_csv(os.path.join(splits_dir, f"train_split_{start_ts}.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, f"val_split_{start_ts}.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, f"test_split_{start_ts}.csv"), index=False)
    logger.info(f"Saved split CSVs to {splits_dir}")


    # Build emotion_map from train_df
    unique_emotions = sorted(train_df['emotion'].unique().tolist())
    emotion_map = {e: i for i, e in enumerate(unique_emotions)}
    logger.info(f"Emotion map: {emotion_map}")

    # Dataloaders
    train_loader, val_loader, test_loader, datasets = get_dataloaders(train_df, val_df, test_df, emotion_map, batch_size=batch_size)
    # ---------------------------
    # Compute class weights (balanced) from training set -> use in CrossEntropyLoss
    # ---------------------------
    try:
        from sklearn.utils.class_weight import compute_class_weight
        train_labels = train_df['emotion'].values
        class_list = np.array(sorted(train_df['emotion'].unique()))
        class_weights = compute_class_weight(class_weight='balanced', classes=class_list, y=train_labels)
        # Map weights into emotion_map class order (0..C-1)
        weight_list_in_order = []
        for orig_emotion in sorted(emotion_map.keys(), key=lambda k: emotion_map[k]):
            # orig_emotion is the original numeric code (e.g., 1,2,...)
            # find index in class_list
            idx = np.where(class_list == orig_emotion)[0][0]
            weight_list_in_order.append(class_weights[idx])
        weight_tensor = torch.tensor(weight_list_in_order, dtype=torch.float32)
        print(f"[INFO] Using class weights: {weight_list_in_order}")
    except Exception as e:
        print(f"[WARNING] Could not compute class weights automatically: {e}")
        weight_tensor = None


    # Model + training setup
    num_classes = len(emotion_map)
    model = build_cnn_bilstm_model(num_classes=num_classes)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Use weighted loss if weights computed
    if weight_tensor is not None:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Training loops but using modular functions
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_f1 = -1.0
    epochs_no_improve = 0
    patience = 6
    save_model_path = os.path.join(out_dir, f"best_model_{start_ts}.pth")

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs} START")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, log_interval=50)
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)

        # compute f1
        val_f1 = f1_score(val_labels, val_preds, average='macro') if len(val_labels) > 0 else 0.0
        logger.info(f"Epoch {epoch} results: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        # record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_f1)

        # save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'emotion_map': emotion_map}, save_model_path)
            logger.info(f"New best model saved at epoch {epoch} with val_f1 {val_f1:.4f}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement epoch counter: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            logger.info("Early stopping triggered.")
            break

    # Save training curves
    plot_training_curves(history, out_path=os.path.join(out_dir, f"training_curves_{start_ts}.png"))

    # Load best model for final test evaluation
    if os.path.exists(save_model_path):
        ckpt = torch.load(save_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        logger.info(f"Loaded best model from {save_model_path}")
    else:
        logger.error("Best model not found; using current weights for evaluation.")

    metrics = evaluate_and_report(model, test_loader, emotion_map, out_dir=out_dir)
    logger.info("main_train_entry finished.")
    return metrics


# ==========================
# Main execution block
# ==========================
if __name__ == "__main__":
    import sys
    
    # Default paths (update these to match your system)
    BASE_AUDIO_PATH = r"C:\Users\Owner\OneDrive - University of Massachusetts Dartmouth\Documents\Capstone\Data\RAVDESS"
    BASE_MEL_PATH = r"C:\Users\Owner\OneDrive - University of Massachusetts Dartmouth\Documents\Capstone\Data\RAVDESSMEL"
    
    print("=" * 80)
    print("PyTorch Spectrogram Training Script")
    print("=" * 80)
    print(f"\nAudio path: {BASE_AUDIO_PATH}")
    print(f"Mel spectrogram path: {BASE_MEL_PATH}")
    
    # Example: Run with 8 emotions (calm and neutral separate)
    print("\n[INFO] Starting training with 8 emotions (calm and neutral SEPARATE)...")
    print("-" * 80)
    
    try:
        metrics = main_train_entry(
            base_audio_path=BASE_AUDIO_PATH,
            base_mel_path=BASE_MEL_PATH,
            out_dir=r"C:\Users\Owner\OneDrive - University of Massachusetts Dartmouth\Documents\Capstone\Results",
            batch_size=16,
            num_epochs=40,  
            lr=1e-3,
            device_str="cpu",  # Use "cuda" if you have GPU
            combine_calm_neutral=False  # 8 emotions
        )
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        print(f"\nOutput files saved in: ./results_8emotions/")
        print(f"\nMetrics summary:")
        if metrics:
            print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
            print(f"  F1-score (macro): {metrics.get('f1_macro', 'N/A')}")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
