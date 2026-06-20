# Music AI Cookbook

Practical, runnable code snippets for common tasks in AI music research. Each recipe includes dependencies, a minimal working example, and explanation of key parameters.

---

## Setup 环境配置

```bash
# Create environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Core dependencies
pip install torch torchaudio librosa numpy matplotlib
pip install transformers  # MERT, CLAP
pip install audiocraft    # MusicGen, EnCodec
pip install demucs        # Source separation
pip install fadtk         # FAD evaluation
pip install essentia      # Advanced MIR features
pip install pretty_midi   # MIDI handling
```

---

## 1. Feature Extraction 特征提取

### 1.1 Mel Spectrogram

```python
import torchaudio
import torchaudio.transforms as T

# Load audio (resample to 16kHz or 32kHz for music AI models)
waveform, sr = torchaudio.load("input.wav")
if sr != 16000:
    resampler = T.Resample(sr, 16000)
    waveform = resampler(waveform)

# Mel spectrogram
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    f_min=0,
    f_max=8000,
    power=2.0,
)
mel_spec = mel_transform(waveform)  # Shape: (channels, n_mels, time)
print(f"Mel shape: {mel_spec.shape}")

# Convert to dB (log scale)
mel_db = T.AmplitudeToDB()(mel_spec)
```

**Key parameters**:
- `n_fft=2048`: Frequency resolution. Larger = finer frequency bins.
- `hop_length=512`: Time resolution. Smaller = more time frames.
- `n_mels=128`: Number of mel bands. 64 for lightweight, 128 for detail.
- `f_max`: Cutoff frequency. 8kHz is standard for speech; 16kHz for full music.

### 1.2 Chroma Features

```python
import librosa

y, sr = librosa.load("input.wav", sr=22050)

# Chroma (12 pitch classes)
chroma = librosa.feature.chroma_stft(
    y=y, sr=sr, n_fft=2048, hop_length=512, n_chroma=12
)

# Chroma CQT (better for music, constant-Q)
chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)

# Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
librosa.display.specshow(chroma, sr=sr, x_axis="time", ax=ax[0])
ax[0].set_title("STFT Chroma")
librosa.display.specshow(chroma_cqt, sr=sr, x_axis="time", ax=ax[1])
ax[1].set_title("CQT Chroma")
plt.tight_layout()
plt.savefig("chroma.png", dpi=150)
```

### 1.3 CQT (Constant-Q Transform)

```python
# Better frequency resolution for music (logarithmic)
C = librosa.cqt(y=y, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12)
# 84 bins = 7 octaves (C1 to C7), 12 bins per octave

# Magnitude in dB
C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
```

---

## 2. Audio Codecs 音频编解码器

### 2.1 EnCodec (Meta)

```python
from audiocraft.models import CompressionModel

# Load pre-trained EnCodec
model = CompressionModel.get_pretrained("facebook/encodec_24khz")
model.eval()

# Load and preprocess audio
import torchaudio
wav, sr = torchaudio.load("input.wav")
if sr != 24000:
    wav = torchaudio.transforms.Resample(sr, 24000)(wav)
wav = wav.unsqueeze(0)  # Add batch dim: (1, channels, time)

# Encode to discrete tokens
with torch.no_grad():
    encoded = model.encode(wav)
    # encoded is a list of (B, K, T) tensors, one per quantizer level
    codes = encoded[0]  # Shape: (1, n_codebooks, n_frames)
    print(f"Codes shape: {codes.shape}")
    print(f"Tokens per second: {codes.shape[2] * 75}")  # 75 Hz frame rate

# Decode back to audio
with torch.no_grad():
    decoded = model.decode(codes)
    torchaudio.save("reconstructed.wav", decoded[0].cpu(), 24000)
```

### 2.2 DAC (Descript Audio Codec)

```python
# Install: pip install descript-audio-codec
from dac.model import DAC

dac = DAC.from_pretrained("descript/dac_44khz")
dac.eval()

wav, sr = torchaudio.load("input.wav")
if sr != 44100:
    wav = torchaudio.transforms.Resample(sr, 44100)(wav)
wav = wav.unsqueeze(0)

# Encode
with torch.no_grad():
    x = dac.preprocess(wav)  # Normalize
    z, codes, _, _ = dac.encode(x)
    # z: continuous latent
    # codes: discrete codes (1, n_codebooks, n_frames)
    print(f"DAC codes: {codes.shape}")

# Decode
with torch.no_grad():
    decoded = dac.decode(z)
    torchaudio.save("dac_reconstructed.wav", decoded[0].cpu(), 44100)
```

### 2.3 Compare Codecs: Reconstruction Quality

```python
import torch
import torchaudio
import numpy as np
from scipy.optimize import minimize

def codec_snr(reference, reconstructed):
    """Signal-to-Noise Ratio in dB."""
    noise = reference - reconstructed
    signal_power = torch.mean(reference ** 2)
    noise_power = torch.mean(noise ** 2)
    return 10 * torch.log10(signal_power / noise_power).item()

# Load original
original, sr = torchaudio.load("input.wav")

# Encode/decode with each codec, then:
snr = codec_snr(original, decoded)
print(f"Reconstruction SNR: {snr:.2f} dB")

# For proper comparison, use FAD (see Evaluation section)
```

---

## 3. Music Generation 音乐生成

### 3.1 Text-to-Music with MusicGen

```python
from audiocraft.models import MusicGen

# Load model (choose size: small, medium, large)
model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(
    duration=30,          # Generation length in seconds
    top_k=250,            # Top-k sampling
    top_p=0.0,            # Nucleus sampling (0 = disabled)
    temperature=1.0,      # Sampling temperature
    cfg_coef=3.0,         # Classifier-free guidance scale
)

# Generate from text
descriptions = ["An upbeat electronic dance track with driving synth bass"]
wav = model.generate(descriptions)

# Save output
torchaudio.save("generated.wav", wav[0].cpu(), 32000)

# Generate from melody + text
melody_wav, sr = torchaudio.load("melody.wav")
if sr != 32000:
    melody_wav = torchaudio.transforms.Resample(sr, 32000)(melody_wav)

model.set_generation_params(duration=30)
wav = model.generate(descriptions, melody_wav[None, :, :int(sr * 15)])
torchaudio.save("melody_continued.wav", wav[0].cpu(), 32000)
```

**Key parameters explained**:
- `duration`: Max generation length. Longer = more VRAM.
- `cfg_coef`: Higher = more prompt adherence, less diversity. Typical range: 2–8.
- `temperature`: Higher = more random. 1.0 = default, 0.7 = conservative.

### 3.2 Text-to-Audio with Stable Audio

```python
# Install: pip install stable-audio-tools
from stable_audio_tools.models.modeling_audiocraft import Model
import torch

model = Model.from_pretrained("stabilityai/stable-audio-open-1.0")
model.eval()

# Generate with conditioning
output = model.generate(
    prompt=["ambient pad, deep space, reverb-heavy"],
    duration=47.0,
    steps=100,           # Diffusion steps
    cfg_scale=7.0,
    sampler_type="dpmpp-2m",  # DPM++ 2M sampler
)
torchaudio.save("stable_audio_out.wav", output.cpu(), 44100)
```

---

## 4. Source Separation 音源分离

### 4.1 Demucs (Meta)

```python
from demucs import pretrained
from demucs.separate import load_track, encode_mu_law
import torch

# Load pre-trained model
model = pretrained.get_model("htdemucs")
model.eval()

# Separate (Demucs handles loading internally)
import subprocess
# CLI usage (simplest):
subprocess.run([
    "python", "-m", "demucs",
    "--two-stems", "vocals",  # Separate vocals from accompaniment
    "-n", "htdemucs",
    "input.wav"
])
# Output: separated/htdemucs/input/vocals.wav, no_vocals.wav
```

### 4.2 BS-RoFormer (Current SOTA)

```python
# Install: pip install bsroformer
# or use via AudioSep framework

# Using AudioSep (recommended):
# https://github.com/audioscope/audiosep

# CLI usage:
# python inference.py --model_path checkpoint.pt --input audio.wav --output_dir out/
```

### 4.3 Evaluate Separation Quality

```python
import museval
import numpy as np

# Evaluate against reference stems
results = museval.eval(
    reference_dir="reference_stems/",
    estimates_dir="separated/htdemucs/input/",
    output_dir="evaluation_results/"
)

# Print SDR/SIR/SAR for each track and stem
print(results)
print(f"Median SDR: {np.median(results.sdr):.2f} dB")
```

---

## 5. Music Understanding 音乐理解

### 5.1 MERT Feature Extraction

```python
from transformers import AutoModel, AutoFeatureExtractor
import torch

# Load MERT-330M
model = AutoModel.from_pretrained("microsoft/mert-330m")
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/mert-330m")
model.eval()

# Extract features
inputs = feature_extractor(
    "input.wav",
    sampling_rate=24000,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state  # (1, time, 1024)
    print(f"MERT features: {last_hidden.shape}")
```

### 5.2 CLAP: Audio-Text Embedding

```python
from clap import CLAP

clap = CLAP(version="2023")

# Audio embedding
audio_emb = clap.get_audio_embedding_from_file("input.wav")
print(f"Audio embedding: {audio_emb.shape}")  # (1, 512)

# Text embedding
text_emb = clap.get_text_embedding(["upbeat electronic dance music"])
print(f"Text embedding: {text_emb.shape}")  # (1, 512)

# Similarity
similarity = torch.nn.functional.cosine_similarity(audio_emb, text_emb)
print(f"Audio-text similarity: {similarity.item():.3f}")
```

### 5.3 Beat Tracking

```python
import librosa

y, sr = librosa.load("input.wav", sr=22050)

# Beat tracking
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
print(f"Estimated tempo: {tempo:.1f} BPM")
print(f"Beat frames: {beats.shape}")

# Convert to time
beat_times = librosa.frames_to_time(beats, sr=sr)
print(f"Beat times: {beat_times[:5]}")

# Downbeat tracking (requires madmom for best results)
# import madmom
# from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNBarProcessor
```

### 5.4 Chord Recognition

```python
import librosa
import numpy as np

y, sr = librosa.load("input.wav", sr=22050)

# Extract chroma features
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

# Simple chord detection using template matching
# (For production, use madmom or a trained model)
chord_template = {
    "C:maj": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "G:maj": [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    "Am":   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
}

# ... template matching per frame
```

### 5.5 Key Detection

```python
import librosa

y, sr = librosa.load("input.wav", sr=22050)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

# Krumhansl-Schmuckler key finding
key, correlation = librosa.feature.key_detect(chroma)
print(f"Estimated key: {key}")
```

---

## 6. Evaluation 评测

### 6.1 FAD (Fréchet Audio Distance)

```python
# Using fadtk
from fadtk import FAD

fad = FAD("vggish")  # Or "clap" for CLAP-based FAD
score = fad.score("real_audio_dir/", "generated_audio_dir/")
print(f"FAD score: {score:.4f}")  # Lower is better
```

### 6.2 CLAP Score

```python
from clap import CLAP
import torch

clap = CLAP(version="2023")

# Load generated audio and text prompts
audio_files = ["gen1.wav", "gen2.wav", "gen3.wav"]
text_prompts = ["electronic dance music", "sad piano melody", "jazz trio"]

audio_embs = clap.get_audio_embedding_from_filelist(audio_files)
text_embs = clap.get_text_embedding(text_prompts)

# CLAP Score: cosine similarity per pair
scores = torch.nn.functional.cosine_similarity(audio_embs, text_embs)
for f, s in zip(audio_files, scores):
    print(f"{f}: CLAP Score = {s.item():.3f}")

# Average CLAP Score
print(f"Average CLAP Score: {scores.mean().item():.3f}")
```

### 6.3 Precision and Recall

```python
# Using FAD toolkit
from fadtk import FAD

fad = FAD("vggish")

# Precision: how many generated samples fall within real distribution
precision = fad.precision("generated_audio_dir/", "real_audio_dir/")

# Recall: how much of real distribution is covered
recall = fad.recall("generated_audio_dir/", "real_audio_dir/")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
```

### 6.4 Music-Specific Metrics

```python
import librosa
import numpy as np

def compute_metrics(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    # Tempo stability
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
    tempo_stability = 1.0 / (np.std(beat_intervals) + 1e-8)

    # Pitch class entropy (tonal stability)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    pitch_entropy = -np.sum(
        np.mean(chroma, axis=1) * np.log(np.mean(chroma, axis=1) + 1e-8)
    )

    # Onset density (notes per second)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_density = len(onset_frames) / (len(y) / sr)

    # Spectral centroid (brightness)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    brightness = np.mean(cent)

    return {
        "tempo_bpm": tempo,
        "tempo_stability": tempo_stability,
        "pitch_entropy": pitch_entropy,
        "onset_density": onset_density,
        "brightness_hz": brightness,
    }

metrics = compute_metrics("generated.wav")
for k, v in metrics.items():
    print(f"{k}: {v:.3f}")
```

---

## 7. Singing Voice Synthesis 歌声合成

### 7.1 OpenDiffSinger Inference

```python
# Requires OpenDiffSinger setup:
# git clone https://github.com/openvpi/OpenDiffSinger
# Follow installation instructions

from diffsinger import DiffSinger
import torch

# Load model
ds = DiffSinger("checkpoints/your_model")
ds.eval()

# Input: phonemes + pitch + duration
# Format depends on the specific model checkpoint
phonemes = ["SP", "N", "a", "n", "q", "i", "n", "SP"]
pitches = [60, 60, 62, 64, 65, 67, 64, 60]  # MIDI note numbers
durations = [0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.2, 0.1]

# Generate (exact API varies by model version)
# Refer to OpenDiffSinger README for exact inference code
```

### 7.2 So-VITS-SVC Voice Conversion

```bash
# So-VITS-SVC workflow:
# 1. Preprocess training data (wav files)
python preprocess.py --path data/singer_wavs/ --sample_rate 44100

# 2. Train (requires GPU)
python train.py -c configs/config.json

# 3. Inference (convert voice)
python inference.py -m "logs/your_model/G_xxx.pth" -n "input.wav" -t 0 -s "target_speaker"
```

---

## 8. MIDI Processing MIDI 处理

### 8.1 MIDI to Piano Roll

```python
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

# Load MIDI
pm = pretty_midi.PrettyMIDI("input.mid")
print(f"Instruments: {len(pm.instruments)}")
print(f"Tempo changes: {len(pm.get_tempo_changes()[1])}")

# Piano roll (128 pitches x time)
piano_roll = pm.get_piano_roll(fs=100)  # 100 frames per second
print(f"Piano roll shape: {piano_roll.shape}")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(piano_roll, aspect="auto", origin="lower", cmap="gray")
ax.set_xlabel("Time (frames @ 100Hz)")
ax.set_ylabel("MIDI pitch")
ax.set_title("Piano Roll")
plt.savefig("piano_roll.png", dpi=150)
```

### 8.2 Extract Notes from MIDI

```python
for instrument in pm.instruments:
    print(f"Instrument: {instrument.name}, Program: {instrument.program}")
    for note in instrument.notes:
        print(f"  Pitch: {note.pitch}, Start: {note.start:.2f}s, "
              f"End: {note.end:.2f}s, Velocity: {note.velocity}")
```

### 8.3 MIDI to REMI Tokens

```python
# Using miditok (pip install miditok)
from miditok import REMI, TokenizerConfig

config = TokenizerConfig(
    use_chords=True,
    use_rests=True,
    use_tempos=True,
    beat_res={(0, 16): 8, (16, 128): 4},  # Resolution per beat range
)
tokenizer = REMI(config)

# Tokenize MIDI
midi = pretty_midi.PrettyMIDI("input.mid")
tokens = tokenizer(midi)
print(f"Token sequence: {tokens}")
print(f"Vocab size: {tokenizer.vocab_size}")

# Decode back to MIDI
midi_out = tokenizer.decode(tokens)
midi_out.write("output.mid")
```

---

## 9. Data Pipeline 数据管线

### 9.1 Download and Preprocess a Dataset

```python
import os
import subprocess
from pathlib import Path

def download_maestro(data_dir="./data"):
    """Download and prepare MAESTRO dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    # Download
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
    zip_path = data_dir / "maestro.zip"

    if not zip_path.exists():
        subprocess.run(["wget", url, "-O", str(zip_path)])

    # Extract
    subprocess.run(["unzip", str(zip_path), "-d", str(data_dir)])

    print(f"MAESTRO ready at {data_dir}")

# download_maestro()
```

### 9.2 Audio Preprocessing Pipeline

```python
import torch
import torchaudio
from pathlib import Path

def preprocess_audio(input_path, output_path, target_sr=16000, target_duration=30):
    """Normalize audio: resample, mono, trim/pad to target duration."""
    wav, sr = torchaudio.load(input_path)

    # Mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

    # Trim or pad
    target_samples = target_sr * target_duration
    if wav.shape[1] > target_samples:
        wav = wav[:, :target_samples]
    else:
        pad = target_samples - wav.shape[1]
        wav = torch.nn.functional.pad(wav, (0, pad))

    # Normalize
    wav = wav / wav.abs().max()

    torchaudio.save(output_path, wav, target_sr)
    return wav

# Batch process
for wav_file in Path("./raw_audio").glob("*.wav"):
    out_file = Path("./processed") / wav_file.name
    preprocess_audio(wav_file, out_file)
```

---

## 10. Common Pitfalls 常见陷阱

### Pitfall 1: Forgetting Loudness Normalization

```python
# WRONG: comparing raw audio without normalization
# Different recordings have different RMS levels
raw_a = torchaudio.load("track_a.wav")[0]
raw_b = torchaudio.load("track_b.wav")[0]
# direct comparison is meaningless

# RIGHT: normalize to same loudness
def normalize_loudness(wav, target_rms=0.1):
    rms = torch.sqrt(torch.mean(wav ** 2))
    return wav * (target_rms / (rms + 1e-8))

wav_a = normalize_loudness(raw_a)
wav_b = normalize_loudness(raw_b)
```

### Pitfall 2: Phase Mismatch in Spectrograms

```python
# WRONG: using magnitude spectrogram and expecting perfect reconstruction
spec = torch.stft(wav, n_fft=2048, return_complex=True)
mag = spec.abs()
# Can't reconstruct original waveform from magnitude alone

# RIGHT: keep phase for reconstruction, or use Griffin-Lim
phase = spec.angle()
spec_complex = mag * torch.exp(1j * phase)
reconstructed = torch.istft(spec_complex, n_fft=2048)
# Or use a neural vocoder (HiFi-GAN) for better quality
```

### Pitfall 3: Evaluation Metric Mismatch

```python
# WRONG: Using FAD with different embedding models interchangeably
fad_vggish = compute_fad(gen_dir, real_dir, embedding="vggish")
fad_clap = compute_fad(gen_dir, real_dir, embedding="clap")
# These can give VERY different rankings!

# RIGHT: Always specify and be consistent
print(f"FAD (VGGish): {fad_vggish:.4f}")
print(f"FAD (CLAP):  {fad_clap:.4f}")
# Report both, but don't mix them
```

### Pitfall 4: Memory Management with Large Models

```python
# WRONG: loading multiple large models simultaneously
model1 = MusicGen.get_pretrained("musicgen-large")   # 16GB
model2 = StableAudio.from_pretrained("...")           # 12GB
model3 = MERT.from_pretrained("mert-330m")           # 2GB
# Total: 30GB+ — will OOM on most consumer GPUs

# RIGHT: load one at a time, free when done
model = MusicGen.get_pretrained("musicgen-large")
# ... use model ...
del model
torch.cuda.empty_cache()

# Or use CPU offloading for large models
model = MusicGen.get_pretrained("musicgen-large").to("cuda")
# Generate with gradient checkpointing / mixed precision
with torch.autocast("cuda", dtype=torch.float16):
    wav = model.generate(descriptions)
```

---

## 11. Quick Reference 速查

| Task | Best tool | One-liner |
|------|-----------|-----------|
| Mel spectrogram | torchaudio | `T.MelSpectrogram()(wav)` |
| Chroma | librosa | `librosa.feature.chroma_cqt(y, sr)` |
| Beat tracking | librosa | `librosa.beat.beat_track(y, sr)` |
| Source separation | Demucs | `python -m demucs -n htdemucs input.wav` |
| Text-to-music | MusicGen | `model.generate(["prompt"])` |
| Audio embedding | MERT/CLAP | `model(**inputs).last_hidden_state` |
| FAD | fadtk | `FAD("vggish").score(real, gen)` |
| CLAP Score | clap | `cosine_sim(audio_emb, text_emb)` |
| MIDI tokens | miditok | `tokenizer = REMI(config); tokens = tokenizer(midi)` |
| Piano roll | pretty_midi | `pm.get_piano_roll(fs=100)` |

---

> **Note**: This cookbook covers common starting points. For production use, always check the latest documentation of each library — APIs change frequently in this fast-moving field.
