from __future__ import annotations

import json
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
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:a:0"]
    command.extend(["-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s24le", str(output_wav)])
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {completed.stderr.strip()}")
    return output_wav


def probe_audio_start(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=start_time",
         "-of", "json", str(path)], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe audio probe failed: {completed.stderr.strip()}")
    streams = json.loads(completed.stdout).get("streams", [])
    return float(streams[0].get("start_time", 0.0)) if streams else 0.0


def analyze_audio(path: Path, sample_rate: int = 48000, hop_length: int = 512,
                  time_offset_sec: float = 0.0, duration_sec: float | None = None) -> AudioFeatures:
    if duration_sec is not None and duration_sec <= 0:
        empty = np.array([], dtype=float)
        return AudioFeatures(empty, empty.copy(), empty.copy(), empty.copy(), sample_rate)
    samples, sr = librosa.load(path, sr=sample_rate, mono=True, duration=duration_sec)
    if not len(samples):
        empty = np.array([], dtype=float)
        return AudioFeatures(empty, empty.copy(), empty.copy(), empty.copy(), sr)
    rms = librosa.feature.rms(y=samples, hop_length=hop_length)[0]
    stft = np.abs(librosa.stft(samples, hop_length=hop_length))
    normalized = stft / np.maximum(stft.sum(axis=0, keepdims=True), 1e-12)
    spectral_change = np.maximum(0.0, np.diff(normalized, axis=1, prepend=normalized[:, :1]))
    flux = np.sqrt(np.sum(spectral_change ** 2, axis=0))
    envelope = librosa.onset.onset_strength(y=samples, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) + time_offset_sec
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length) + time_offset_sec
    length = min(len(times), len(rms), len(flux))
    return AudioFeatures(times[:length], rms[:length], flux[:length], onset_times, sr)
