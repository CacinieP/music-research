"""Small deterministic baselines; no model downloads or perceptual-quality claims."""
from pathlib import Path
import numpy as np

PITCH_CLASSES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def pitch_class_entropy(chroma):
    """Entropy in bits of mean nonnegative chroma; NaN for zero mass."""
    x = np.asarray(chroma, dtype=float)
    if x.ndim != 2 or x.shape[0] != 12 or x.shape[1] == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Expected finite nonnegative chroma of shape (12, frames)')
    peak = x.max()
    mass = (x / peak).mean(axis=1) if peak > 0 else np.zeros(12)
    if mass.sum() == 0:
        return float('nan')
    p = mass / mass.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def estimate_key(chroma):
    """24 Pearson profile correlations. This baseline cannot detect modulation."""
    x = np.asarray(chroma, dtype=float)
    if x.ndim != 2 or x.shape[0] != 12 or x.shape[1] == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Expected finite nonnegative chroma of shape (12, frames)')
    peak = x.max()
    mean = (x / peak).mean(axis=1) if peak > 0 else np.zeros(12)
    mean -= mean.mean()
    if not np.any(mean):
        return None, float('nan')
    candidates = []
    for mode, profile in [('major', MAJOR_PROFILE), ('minor', MINOR_PROFILE)]:
        for tonic, name in enumerate(PITCH_CLASSES):
            p = np.roll(profile, tonic)
            p = p - p.mean()
            score = float(np.dot(mean, p) / (np.linalg.norm(mean) * np.linalg.norm(p)))
            candidates.append((f'{name}:{mode}', score))
    return max(candidates, key=lambda pair: pair[1])


def chord_templates():
    """C=0; 24 unit-norm major/minor triad templates, for teaching only."""
    labels, columns = [], []
    for root, name in enumerate(PITCH_CLASSES):
        for mode, intervals in [('maj', (0, 4, 7)), ('min', (0, 3, 7))]:
            vector = np.zeros(12)
            vector[[(root + i) % 12 for i in intervals]] = 1 / np.sqrt(3)
            labels.append(f'{name}:{mode}')
            columns.append(vector)
    return labels, np.column_stack(columns)


def aligned_snr(reference, reconstructed):
    """SNR for already aligned arrays at the same sample rate and gain."""
    a, b = np.asarray(reference, dtype=float), np.asarray(reconstructed, dtype=float)
    if a.shape != b.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected nonempty finite arrays with identical shapes')
    signal_peak = np.max(np.abs(a))
    if signal_peak == 0:
        return float('nan')
    scale = max(signal_peak, np.max(np.abs(b)))
    residual = a / scale - b / scale
    error_peak = np.max(np.abs(residual))
    if error_peak == 0:
        return float('inf')
    log_signal = 2*np.log10(signal_peak) + np.log10(np.sum((a/signal_peak)**2))
    log_error = 2*(np.log10(scale) + np.log10(error_peak)) + np.log10(np.sum((residual/error_peak)**2))
    return float(10*(log_signal-log_error))


def preprocess_audio(input_path, output_path, target_sr=16000, target_duration=30.0):
    """Mono, resample, trim/pad, peak-scale non-silence; this is not LUFS normalization."""
    import librosa
    import soundfile as sf
    if not np.isfinite(target_sr) or target_sr <= 0 or int(target_sr) != target_sr or not np.isfinite(target_duration) or target_duration <= 0:
        raise ValueError('Positive integer sample rate and positive finite duration required')
    target_sr = int(target_sr)
    count = round(target_sr * target_duration)
    if count < 1:
        raise ValueError('Duration must contain at least one sample')
    wav, sr = sf.read(input_path, always_2d=True, dtype='float32')
    if wav.size == 0 or not np.isfinite(wav).all():
        raise ValueError('Empty or nonfinite audio')
    wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    wav = np.pad(wav[:count], (0, max(0, count-len(wav))))
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = 0.95 * wav / peak
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wav, target_sr, subtype='FLOAT')
    return wav
