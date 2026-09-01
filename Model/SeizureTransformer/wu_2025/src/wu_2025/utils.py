import math
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import butter, iirnotch, lfilter, resample

from wu_2025.architecture import SeizureTransformer


def load_models(device):
    model = SeizureTransformer()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    checkpoint = torch.load(os.path.join(dir_path, "model.pth"), map_location=device)
    model.load_state_dict(checkpoint)
    return model.to(device)


class SeizureDataset(nn.Module):
    def __init__(self, data, fs=256, window_size=15360, overlap_ratio=0.0):
        super().__init__()
        self.data = data
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.fs = fs
        self.lowcut = 0.5
        self.highcut = 120
        notch_1_b, notch_1_a = iirnotch(1, Q=30, fs=fs)
        notch_60_b, notch_60_a = iirnotch(60, Q=30, fs=fs)
        self.notch_1_b = notch_1_b
        self.notch_1_a = notch_1_a
        self.notch_60_b = notch_60_b
        self.notch_60_a = notch_60_a

    def __len__(self):
        if self.data.shape[1] < self.window_size:
            return 1
        return 1 + math.ceil(
            (self.data.shape[1] - self.window_size)
            / ((1 - self.overlap_ratio) * self.window_size)
        )

    def butter_bandpass_filter(self, data, order=3):
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return lfilter(b, a, data)

    def preprocess_clip(self, eeg_clip):
        bandpass_filtered_signal = self.butter_bandpass_filter(eeg_clip, order=3)
        filtered_1_signal = lfilter(self.notch_1_b, self.notch_1_a, bandpass_filtered_signal)
        return lfilter(self.notch_60_b, self.notch_60_a, filtered_1_signal)

    def __getitem__(self, idx):
        start_idx = int(idx * self.window_size * (1 - self.overlap_ratio))
        if start_idx + self.window_size + 1 > self.data.shape[1]:
            eeg_clip = self.data[:, start_idx:]
            pad = np.zeros((self.data.shape[0], self.window_size - eeg_clip.shape[1]))
            eeg_clip = np.concatenate((eeg_clip, pad), axis=1)
        else:
            eeg_clip = self.data[:, start_idx : start_idx + self.window_size]
        return torch.FloatTensor(self.preprocess_clip(eeg_clip))


def get_dataloader(data, fs, batch_size=256, window_size=15360):
    mu = np.mean(data, axis=1, keepdims=True)
    sd = np.std(data, axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    data = (data - mu) / sd

    if fs != 256:
        new_n_samples = int(data.shape[1] * float(256) / fs)
        data = resample(data, new_n_samples, axis=1)

    dataset = SeizureDataset(data, fs=256, window_size=window_size)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def forward_windows(model, dataloader, device):
    """Run the model on all windows and return per-time-step probabilities."""
    total_output = None
    model.eval()
    for data in dataloader:
        data = data.float().to(device)
        with torch.no_grad():
            output = model(data)
        total_output = output if total_output is None else torch.cat((total_output, output), 0)
    return total_output


def forward_windows_dropout(model, dataloader, device):
    """Single stochastic forward pass (dropout must already be enabled)."""
    total_output = None
    for data in dataloader:
        data = data.float().to(device)
        with torch.no_grad():
            output = model(data)
        total_output = output if total_output is None else torch.cat((total_output, output), 0)
    return total_output


def compute_agreement_trace(model, dataloader, device, seq_len, n_samples, thresh):
    """Monte Carlo dropout agreement index φ per sample (length seq_len)."""
    from experiments_utils.mcd import compute_agreement_index, enable_dropout_only

    binary_runs = []
    enable_dropout_only(model)
    for _ in range(n_samples):
        probs = forward_windows_dropout(model, dataloader, device)
        y = probs.detach().cpu().numpy().flatten()[:seq_len]
        binary_runs.append((y >= thresh).astype(np.float64))
    model.eval()
    return compute_agreement_index(np.stack(binary_runs, axis=0))


def eventize_probabilities(y_predict, seq_len):
    """Convert probabilities to a binary seizure mask using model post-processing."""
    y_predict = y_predict.detach().cpu().numpy().flatten()[:seq_len]

    thresh = float(os.getenv("THRESH", "0.8"))
    open_k = int(os.getenv("OPEN_K", "5"))
    close_k = int(os.getenv("CLOSE_K", "5"))
    min_len = float(os.getenv("MIN_EVENT_SEC", "2.0"))

    binary_output = (y_predict > thresh).astype(int)
    binary_output = morphological_filter_1d(binary_output, operation="opening", kernel_size=open_k)
    binary_output = morphological_filter_1d(binary_output, operation="closing", kernel_size=close_k)
    binary_output = remove_short_events(binary_output, min_length=min_len, fs=256)

    print(
        f"[predict] mean_prob={y_predict.mean():.4f} "
        f"p>{thresh}={(y_predict > thresh).mean():.4f} "
        f"ones_after_eventization={binary_output.mean():.4f}"
    )
    return binary_output


def predict(model, dataloader, device, seq_len):
    probs = forward_windows(model, dataloader, device)
    return eventize_probabilities(probs, seq_len)


def morphological_filter_1d(binary_signal, operation="closing", kernel_size=5):
    structure = np.ones(kernel_size, dtype=bool)
    if operation == "closing":
        filtered = binary_closing(binary_signal, structure=structure)
    elif operation == "opening":
        filtered = binary_opening(binary_signal, structure=structure)
    else:
        raise ValueError("operation must be 'closing' or 'opening'")
    return filtered.astype(int)


def remove_short_events(binary_output, min_length, fs):
    min_samples = int(min_length * fs)
    out = binary_output.copy()
    is_seizure = False
    start_idx = 0
    for i in range(len(binary_output)):
        if not is_seizure and out[i] == 1:
            is_seizure = True
            start_idx = i
        elif is_seizure and (out[i] == 0 or i == len(binary_output) - 1):
            end_idx = i if out[i] == 0 else i + 1
            if end_idx - start_idx < min_samples:
                out[start_idx:end_idx] = 0
            is_seizure = False
    return out
