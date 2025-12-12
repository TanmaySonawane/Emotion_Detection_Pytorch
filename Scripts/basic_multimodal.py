# train_pytorch_multimodal_minimal.py
"""
Minimal multimodal training script (spectrogram .npy + raw .wav).
Produces a 5-class mapping (as defined below).

Save this as train_pytorch_multimodal_minimal.py and run:
    python train_pytorch_multimodal_minimal.py

Requirements (same as your environment):
    - torch, torchvision
    - librosa, numpy, pandas, scikit-learn, matplotlib, seaborn
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import librosa
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# ----------------------
# Configuration (edit as needed)
# ----------------------
BASE_AUDIO_PATH = r"C:\Users\Owner\OneDrive - University of Massachusetts Dartmouth\Documents\Capstone\Data\RAVDESS"
BASE_MEL_PATH   = r"C:\Users\Owner\OneDrive - University of Massachusetts Dartmouth\Documents\Capstone\Data\RAVDESSMEL"  # .npy files
OUT_DIR = r"C:\Users\Owner\OneDrive - University of Massachusetts Dartmouth\Documents\Capstone\Results_minimal"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0  # recommended 0 on Windows

# Audio preprocessing
AUDIO_SR = 16000
AUDIO_DURATION = 3.0
AUDIO_SAMPLES = int(AUDIO_SR * AUDIO_DURATION)

# Image preprocessing target
IMG_SIZE = (224, 224)  # H, W

# ----------------------
# 5-class mapping (RAVDESS original codes -> new 0..4 labels)
# RAVDESS codes: 1 neutral, 2 calm, 3 happy, 4 sad, 5 angry, 6 fearful, 7 disgust, 8 surprised
# We map:
#   0: neutral/calm (1 or 2)
#   1: fear + sad  (6 or 4)
#   2: happy + surprised (3 or 8)
#   3: angry (5)
#   4: disgust (7)
# ----------------------
LABEL_MAP_5 = {1: 0, 2: 0, 6: 1, 4: 1, 3: 2, 8: 2, 5: 3, 7: 4}

# ----------------------
# Utility: build dataframe (minimal)
# ----------------------
def build_dataframe(base_audio_path, base_mel_path):
    """
    Walk actor folders Actor_01..Actor_24 and build DataFrame with columns:
      wav_path, mel_path (.npy), filename, emotion (int), actor_num (int)
    """
    records = []
    for i in range(1, 25):
        actor_folder = f"Actor_{str(i).zfill(2)}"
        audio_dir = os.path.join(base_audio_path, actor_folder)
        mel_dir = os.path.join(base_mel_path, actor_folder)
        if not os.path.isdir(audio_dir):
            continue
        files = sorted(os.listdir(audio_dir))
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue
            wav_path = os.path.join(audio_dir, fname)
            base_name = os.path.splitext(fname)[0]
            npy_path = os.path.join(mel_dir, base_name + ".npy")  # user produced .npy earlier
            # parse emotion code from filename (RAVDESS)
            parts = base_name.split("-")
            if len(parts) < 7:
                continue
            emotion_id = int(parts[2])  # '01' -> 1
            actor_num = int(parts[6])
            # remap to 5 classes
            if emotion_id not in LABEL_MAP_5:
                continue
            new_label = LABEL_MAP_5[emotion_id]
            records.append({
                "wav_path": wav_path,
                "mel_path": npy_path,
                "filename": fname,
                "emotion_orig": emotion_id,
                "emotion": new_label,
                "actor_num": actor_num
            })
    df = pd.DataFrame.from_records(records)
    return df

# ----------------------
# Dataset
# ----------------------
def load_spectrogram_npy_to_tensor(npy_path, target_size=IMG_SIZE):
    """
    Loads a 2D numpy array from npy_path and returns a float tensor (1,H,W) normalized to [0,1].
    Resizes to target_size using bilinear interpolation if needed.
    """
    try:
        arr = np.load(npy_path)  # expected shape (n_mels, time)
        if arr.ndim == 3:
            # sometimes saved with channel, reduce
            arr = arr.squeeze()
        tensor = torch.from_numpy(arr).float().unsqueeze(0)  # (1, H, W)
        # Resize if necessary: interpolate expects (N,C,H,W)
        if (tensor.shape[1], tensor.shape[2]) != target_size:
            tensor = F.interpolate(tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        # Normalize to [0,1]
        mn = tensor.min()
        mx = tensor.max()
        if mx - mn > 1e-8:
            tensor = (tensor - mn) / (mx - mn)
        else:
            tensor = tensor * 0.0
        return tensor
    except Exception:
        # If loading fails, return zeros
        return torch.zeros(1, target_size[0], target_size[1], dtype=torch.float32)

def load_wav_to_tensor(wav_path, sr=AUDIO_SR, target_samples=AUDIO_SAMPLES):
    """
    load .wav with librosa -> mono -> pad/truncate -> tensor (1, L)
    """
    try:
        y, _ = librosa.load(wav_path, sr=sr, mono=True)
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)), mode='constant')
        else:
            y = y[:target_samples]
        return torch.from_numpy(y).float().unsqueeze(0)
    except Exception:
        return torch.zeros(1, target_samples, dtype=torch.float32)

class EmotionMultimodalMinimalDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_spectrogram_npy_to_tensor(row["mel_path"])
        wav = load_wav_to_tensor(row["wav_path"])
        label = torch.tensor(int(row["emotion"]), dtype=torch.long)
        return img, wav, label

# ----------------------
# Minimal multimodal model
# ----------------------
def build_resnet18_audio_fusion(num_classes, freeze_backbone=False):
    # Load resnet18; handle differences in torchvision versions
    try:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        resnet = torchvision.models.resnet18(weights=weights)
    except Exception:
        resnet = torchvision.models.resnet18(pretrained=True)
    # adapt first conv for 1 channel
    orig_conv = resnet.conv1
    resnet.conv1 = nn.Conv2d(1, orig_conv.out_channels, kernel_size=orig_conv.kernel_size,
                             stride=orig_conv.stride, padding=orig_conv.padding, bias=False)
    if freeze_backbone:
        for p in resnet.parameters():
            p.requires_grad = False
    # feature extractor up to last conv
    feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
    class ResNetAudioFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = feature_extractor
            # small audio encoder (1D CNN)
            aud_layers = []
            in_ch = 1
            for out_ch in [16, 32, 64]:
                aud_layers += [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                               nn.BatchNorm1d(out_ch),
                               nn.ReLU(inplace=True),
                               nn.MaxPool1d(2)]
                in_ch = out_ch
            self.audio_encoder = nn.Sequential(*aud_layers)
            self.audio_pool = nn.AdaptiveAvgPool1d(1)  # -> (B, C, 1)
            # LSTM for image branch will be created lazily
            self.bilstm = None
            self.lstm_hidden = 128
            self.lstm_layers = 1
            self.bidirectional = True
            # classifier to be created lazily
            self.classifier = None

        def forward(self, image, audio):
            # image branch: image (B,1,H,W) -> conv features (B, C, Hf, Wf)
            f = self.feature_extractor(image)
            # collapse freq axis Hf -> mean -> (B, C, Wf), then (B, Wf, C)
            seq = f.mean(dim=2).permute(0, 2, 1)
            input_size = seq.size(2)
            if (self.bilstm is None) or (self.bilstm.input_size != input_size):
                self.bilstm = nn.LSTM(input_size=input_size, hidden_size=self.lstm_hidden,
                                      num_layers=self.lstm_layers, batch_first=True, bidirectional=self.bidirectional)
            lstm_out, (h_n, c_n) = self.bilstm(seq)
            if self.bidirectional:
                last_forward = h_n[-2]
                last_backward = h_n[-1]
                img_feat = torch.cat([last_forward, last_backward], dim=1)
            else:
                img_feat = h_n[-1]
            # audio branch: (B,1,L) -> encoder expects (B, C, L)
            aud = self.audio_encoder(audio)
            aud = self.audio_pool(aud).squeeze(-1)  # (B, C)
            # build classifier lazily
            if self.classifier is None:
                fused_dim = img_feat.size(1) + aud.size(1)
                self.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(fused_dim, fused_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(fused_dim // 2, num_classes)
                )
            fused = torch.cat([img_feat, aud], dim=1)
            logits = self.classifier(fused)
            return logits
    return ResNetAudioFusion()

# ----------------------
# Training & evaluation helpers
# ----------------------
def save_confusion_and_metrics(y_true, y_pred, emotion_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_png = os.path.join(out_dir, f"confusion_{ts}.png")
    metrics_txt = os.path.join(out_dir, f"metrics_{ts}.txt")
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(emotion_names))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, None] + 1e-9)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=emotion_names, yticklabels=emotion_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (normalized)")
    plt.tight_layout()
    plt.savefig(cm_png)
    plt.close()
    # metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, zero_division=0)
    metrics = {"accuracy": float(acc), "f1_macro": float(f1), "classification_report": report}
    with open(metrics_txt, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    # also save human-readable
    with open(metrics_txt.replace(".txt", ".log"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[INFO] Saved metrics: {metrics_txt} and confusion PNG: {cm_png}")
    return cm_png, metrics_txt

# ----------------------
# Minimal training loop
# ----------------------
def train_minimal_multimodal(df, out_dir=OUT_DIR, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, lr=LR, device=DEVICE):
    os.makedirs(out_dir, exist_ok=True)
    # actor-wise split
    unique_actors = sorted(df['actor_num'].unique().tolist())
    train_actors = list(range(1, 19))
    val_actors = list(range(19, 22))
    test_actors = list(range(22, 25))
    train_df = df[df['actor_num'].isin(train_actors)].reset_index(drop=True)
    val_df = df[df['actor_num'].isin(val_actors)].reset_index(drop=True)
    test_df = df[df['actor_num'].isin(test_actors)].reset_index(drop=True)
    print(f"[INFO] Train/Val/Test sizes: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    # emotion_map -> ensure classes are 0..4
    unique_emotions = sorted(train_df['emotion'].unique().tolist())
    emotion_map = {e: i for i, e in enumerate(unique_emotions)}
    print(f"[INFO] emotion_map used: {emotion_map}")
    num_classes = len(emotion_map)
    # datasets + loaders
    train_ds = EmotionMultimodalMinimalDataset(train_df)
    val_ds = EmotionMultimodalMinimalDataset(val_df)
    test_ds = EmotionMultimodalMinimalDataset(test_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    # class weights
    try:
        class_list = np.array(sorted(train_df['emotion'].unique()))
        cw = compute_class_weight(class_weight='balanced', classes=class_list, y=train_df['emotion'].values)
        # order weights according to emotion_map
        weight_list = []
        for orig in sorted(emotion_map.keys(), key=lambda k: emotion_map[k]):
            idx = np.where(class_list == orig)[0][0]
            weight_list.append(float(cw[idx]))
        weight_tensor = torch.tensor(weight_list, dtype=torch.float32).to(device)
        print(f"[INFO] class weights: {weight_list}")
    except Exception as e:
        print(f"[WARN] compute_class_weight failed: {e}")
        weight_tensor = None
    # model, optimizer, criterion
    model = build_resnet18_audio_fusion(num_classes=num_classes, freeze_backbone=False).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if weight_tensor is not None:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    best_val_f1 = -1.0
    save_path = os.path.join(out_dir, f"best_multimodal_5class.pth")
    patience = 6
    epochs_no_improve = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"[TRAIN] Epoch {epoch}/{num_epochs}")
        for batch_idx, (imgs, audios, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device)
            audios = audios.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            logits = model(imgs, audios)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            if batch_idx == 1 or batch_idx % 20 == 0:
                batch_acc = (preds == labels).float().mean().item()
                print(f"  B{batch_idx} Loss: {loss.item():.4f} BatchAcc: {batch_acc:.4f}")
        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        # validation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss_acc = 0.0
        val_total = 0
        with torch.no_grad():
            for imgs, audios, labels in val_loader:
                imgs = imgs.to(device)
                audios = audios.to(device)
                labels = labels.to(device)
                logits = model(imgs, audios)
                loss = criterion(logits, labels)
                val_loss_acc += loss.item() * imgs.size(0)
                val_total += imgs.size(0)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        val_loss = val_loss_acc / val_total if val_total > 0 else 0.0
        val_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0.0
        val_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        print(f"[EVALUATE] TrainLoss: {epoch_loss:.4f} TrainAcc: {epoch_acc:.4f} | ValLoss: {val_loss:.4f} ValAcc: {val_acc:.4f} ValF1: {val_f1:.4f}")
        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'emotion_map': emotion_map, 'num_classes': num_classes}, save_path)
            print(f"[INFO] New best model saved: ValF1={val_f1:.4f} -> {save_path}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] No improvement {epochs_no_improve}/{patience}")
        if epochs_no_improve >= patience:
            print("[INFO] Early stopping")
            break
    # Load best and evaluate on test
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        try:
            model.load_state_dict(ckpt['model_state'])
            print(f"[INFO] Loaded checkpoint from {save_path}")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint exactly: {e} (continuing with current model)")
    model.eval()
    y_t = []
    y_p = []
    with torch.no_grad():
        for imgs, audios, labels in test_loader:
            imgs = imgs.to(device)
            audios = audios.to(device)
            labels = labels.to(device)
            logits = model(imgs, audios)
            preds = logits.argmax(dim=1)
            y_t.extend(labels.cpu().numpy().tolist())
            y_p.extend(preds.cpu().numpy().tolist())
    # emotion names (simple)
    emotion_names = ["neutral/calm", "fear+sad", "happy+surprise", "angry", "disgust"]
    save_confusion_and_metrics(y_t, y_p, emotion_names, out_dir)
    print("[INFO] Test classification report:")
    print(classification_report(y_t, y_p, target_names=emotion_names, zero_division=0))
    return {"best_val_f1": best_val_f1, "save_model_path": save_path}

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    print("=== Minimal multimodal training (5-class) ===")
    df_all = build_dataframe(BASE_AUDIO_PATH, BASE_MEL_PATH)
    print(f"[INFO] Loaded DataFrame with {len(df_all)} records")
    # quick sanity check
    if df_all.empty:
        raise SystemExit("[ERROR] No records found in DataFrame. Check paths and ensure .npy files exist.")
    # run training
    results = train_minimal_multimodal(df_all)
    print("=== Done ===")
    print(results)
