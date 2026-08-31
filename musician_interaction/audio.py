from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass
class AudioFeatures:
    times: np.ndarray
    rms: np.ndarray
    spectral_flux: np.ndarray
    onset_times: np.ndarray
    sample_rate: int

    def at_video_times(self, video_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.times):
            missing = np.full(video_times.shape, np.nan)
            return missing.copy(), missing
        rms = np.interp(video_times, self.times, self.rms, left=np.nan, right=np.nan)
        flux = np.interp(video_times, self.times, self.spectral_flux, left=np.nan, right=np.nan)
        return rms, flux


def extract_audio(source: Path, output_wav: Path, sample_rate: int = 48000) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s24le", str(output_wav),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {completed.stderr.strip()}")
    return output_wav


def analyze_audio(path: Path, sample_rate: int = 48000, hop_length: int = 512) -> AudioFeatures:
    samples, sr = librosa.load(path, sr=sample_rate, mono=True)
    rms = librosa.feature.rms(y=samples, hop_length=hop_length)[0]
    stft = np.abs(librosa.stft(samples, hop_length=hop_length))
    normalized = stft / np.maximum(stft.sum(axis=0, keepdims=True), 1e-12)
    flux = np.sqrt(np.maximum(0.0, np.diff(normalized, axis=1, prepend=normalized[:, :1])) ** 2).sum(axis=0)
    envelope = librosa.onset.onset_strength(y=samples, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    length = min(len(times), len(rms), len(flux))
    return AudioFeatures(times[:length], rms[:length], flux[:length], onset_times, sr)
