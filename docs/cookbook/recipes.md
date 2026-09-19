# Music AI Cookbook

API review: 2026-09-19. CPU examples are tested with synthetic audio and MIDI. Model examples below are checked against linked upstream APIs; model weights, GPU inference, training and listening quality have **not** been run as part of this review. Each model needs its own compatible environment and checkpoint licence.

## Setup 环境配置

For the CPU examples, run from the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-examples.txt
python -m unittest discover -s scripts -p 'test_examples.py'
```

Install AudioCraft, DAC, LAION CLAP and other model stacks in separate environments following their official instructions. A single unpinned `pip install` command for every framework is not a reproducible environment. Keep Torch and TorchAudio versions compatible, and record package versions and checkpoint revisions.

## 1. Feature extraction 特征提取

Feature arrays are CPU-tested. The optional plotting lines use Matplotlib (`pip install matplotlib==3.10.7`) and are API-checked separately.

```python
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load('input.wav', sr=22050, mono=True)
mel = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=2048, hop_length=512,
    n_mels=128, fmax=sr/2, power=2.0)
mel_db = librosa.power_to_db(mel, ref=np.max)
chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
C = librosa.cqt(y=y, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12)
C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
fig, ax = plt.subplots()
librosa.display.specshow(C_db, sr=sr, hop_length=512, x_axis='time', y_axis='cqt_note', ax=ax)
fig.savefig('cqt.png', dpi=150)
```

`n_fft` controls the frequency grid and window support; increasing it by zero-padding alone does not create finer true resolving power. `hop_length/sr` is the frame spacing, not the complete temporal resolution. `fmax` must not exceed Nyquist. With default C1 and 12 bins per octave, 84 CQT centers run from C1 through B7; C8 would be the 85th center.

These are analysis settings, not a universal preprocessor for pretrained models. Follow each model's sample rate, channel count and normalization. [librosa feature API](https://librosa.org/doc/latest/feature.html)

When passing a precomputed CQT to `chroma_cqt(C=...)`, pass its magnitude and the same `bins_per_octave`: `librosa.cqt` defaults to 12, while `chroma_cqt` defaults to 36. Omitting that match can assign a tone to the wrong pitch class.

## 2. Audio codecs 音频编解码器

### EnCodec via AudioCraft — API-checked model example

```python
import torch
import torchaudio
from audiocraft.models import CompressionModel
from audiocraft.data.audio_utils import convert_audio

codec = CompressionModel.get_pretrained('facebook/encodec_24khz').eval()
wav, sr = torchaudio.load('input.wav')
wav = convert_audio(wav, sr, codec.sample_rate, codec.channels)
wav = wav.unsqueeze(0)
with torch.inference_mode():
    codes, scale = codec.encode(wav)
    decoded = codec.decode(codes, scale)[..., :wav.shape[-1]]
seconds = wav.shape[-1] / codec.sample_rate
print('codes shape:', tuple(codes.shape))  # batch, codebooks, frames
print('code indices per second:', codes.shape[1] * codes.shape[2] / seconds)
torchaudio.save('reconstructed.wav', decoded[0].cpu(), codec.sample_rate)
```

Use the codebook count times the frame rate for aggregate code indices per second. `n_frames * 75` is not tokens/second. Scale metadata must be passed back to the decoder when present. [AudioCraft codec interface](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/encodec.py)

### DAC — API-checked model example

```python
import dac
import torch
from audiotools import AudioSignal

codec = dac.DAC.load(dac.utils.download(model_type='44khz')).eval()
signal = AudioSignal('input.wav')
# DAC's standard model is mono. Explicitly convert stereo input for this example.
signal.audio_data = signal.audio_data.mean(dim=1, keepdim=True)
signal.resample(codec.sample_rate)
length = signal.audio_data.shape[-1]
with torch.inference_mode():
    x = codec.preprocess(signal.audio_data, signal.sample_rate)
    z, codes, latents, commitment_loss, codebook_loss = codec.encode(x)
    decoded = codec.decode(z)[..., :length]
AudioSignal(decoded, sample_rate=codec.sample_rate).write('dac_reconstructed.wav')
```

The native API uses `dac.utils.download`, `DAC.load`, a sample-rate argument to `preprocess`, and five outputs from `encode`. `preprocess` validates the rate and pads to the codec hop; resample first and remove the padding after decoding. For long files use the upstream chunked `compress`/`decompress` workflow. [DAC documentation](https://github.com/descriptinc/descript-audio-codec)

For reconstruction SNR, first align time, sample rate, channels, duration and gain; then use `aligned_snr` from [audio_baselines.py](../../src/audio_baselines.py). SNR measures sample error, not perceptual quality. FAD compares embedding distributions and does not replace paired codec listening tests.

## 3. Music generation 音乐生成

### MusicGen — API-checked model example

```python
import torchaudio
from audiocraft.models import MusicGen

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=8, cfg_coef=3.0)
waveforms = model.generate(['A gentle piano melody with sustained chords'])
torchaudio.save('generated.wav', waveforms[0].cpu(), model.sample_rate)

# Melody conditioning requires a melody-capable checkpoint and a different API.
melody_model = MusicGen.get_pretrained('facebook/musicgen-melody')
melody_model.set_generation_params(duration=8)
melody, sr = torchaudio.load('melody.wav')
output = melody_model.generate_with_chroma(
    ['Acoustic guitar with a steady pulse'], melody[None, ..., :int(sr*8)], sr)
torchaudio.save('melody_conditioned.wav', output[0].cpu(), melody_model.sample_rate)
```

Run the text and melody examples separately if memory is limited. Melody conditioning transfers chroma information; it is not identical to audio continuation. Use `generate_continuation` for continuation. Guidance trades conditioning and diversity empirically; larger values do not guarantee better output. [MusicGen API](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)

### Stable Audio Open

Use the [official model card](https://huggingface.co/stabilityai/stable-audio-open-1.0) and [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools). The original cookbook's `modeling_audiocraft.Model.from_pretrained(...).generate(...)` was not the official interface and has been removed. Model access and licence acceptance may be required. Keep model configuration, sample size, conditioning and sampler parameters tied to the selected release.

## 4. Source separation 音源分离

The [Demucs CLI](https://github.com/facebookresearch/demucs) is a documented entry point:

```bash
python -m demucs -n htdemucs --two-stems vocals input.wav
```

It writes `vocals.wav` and `no_vocals.wav` under its output directory. The `--two-stems` option still uses the separation model internally; it is not a new two-source architecture or a guarantee of lower memory.

For evaluation, load references and estimates in an explicit, matching stem order:

```python
from pathlib import Path
import numpy as np
import soundfile as sf
import museval

stems = ['vocals', 'drums', 'bass', 'other']
pairs = [(sf.read(Path('reference_stems') / f'{stem}.wav', always_2d=True),
          sf.read(Path('estimated_stems') / f'{stem}.wav', always_2d=True))
         for stem in stems]
rates = {rate for pair in pairs for _, rate in pair}
if len(rates) != 1:
    raise ValueError('All references and estimates must have the same sample rate')
sr = rates.pop()
reference = np.stack([pair[0][0] for pair in pairs])
estimate = np.stack([pair[1][0] for pair in pairs])
if reference.shape != estimate.shape:
    raise ValueError('Reference and estimate lengths/channels must match')
sdr, isr, sir, sar = museval.evaluate(reference, estimate, win=sr, hop=sr, mode='v4')
print(dict(zip(stems, np.nanmedian(sdr, axis=1))))
```

This prints the median of one-second-window SDRs for each stem of one track, not a dataset-wide score. Record the BSS Eval mode, sample rate, windows, silence handling and track/stem aggregation. Do not assume a `.sdr` property or silently mix these scores with SI-SDR. Explicit filename alignment avoids depending on filesystem enumeration order. [museval implementation](https://github.com/sigsep/sigsep-mus-eval)

BS-RoFormer checkpoints differ in instruments, band partitioning and implementation. Use a checkpoint-compatible repository, not the original cookbook's fictitious universal `bsroformer`/AudioSep commands. [BS-RoFormer paper](https://arxiv.org/abs/2309.02612)

## 5. Music understanding 音乐理解

### MERT — API-checked model example

```python
import librosa
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

checkpoint = 'm-a-p/MERT-v1-330M'
# The checkpoint uses custom model code; inspect and pin its revision for reproducibility.
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).eval()
processor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
y, sr = librosa.load('input.wav', sr=processor.sampling_rate, mono=True)
inputs = processor(y[:5*sr], sampling_rate=sr, return_tensors='pt')
with torch.inference_mode():
    features = model(**inputs).last_hidden_state
print(features.shape)
```

The feature extractor receives waveform samples, not a filename. The official checkpoint belongs to `m-a-p`, not `microsoft/mert-330m`. Pooling hidden states is a feature baseline, not a trained genre classifier. [MERT model card](https://huggingface.co/m-a-p/MERT-v1-330M)

### CLAP — API-checked model example

```python
import torch
import laion_clap

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()
with torch.inference_mode():
    a = model.get_audio_embedding_from_filelist(x=['input.wav'], use_tensor=True)
    t = model.get_text_embedding(['A solo piano performance'], use_tensor=True)
    score = torch.nn.functional.cosine_similarity(a, t, dim=-1)
print(float(score[0]))
```

Microsoft CLAP and LAION CLAP have different packages, checkpoints and APIs. This example explicitly uses LAION CLAP. Report checkpoint and preprocessing with every similarity score. [LAION CLAP](https://github.com/LAION-AI/CLAP)

### Beat, chord and key baselines — CPU examples

```python
import librosa
import numpy as np
from src.audio_baselines import chord_templates, estimate_key

y, sr = librosa.load('input.wav', sr=22050)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(f'Tempo: {float(np.asarray(tempo).reshape(-1)[0]):.1f} BPM')
print(librosa.frames_to_time(beat_frames, sr=sr))
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
labels, templates = chord_templates()
strength = np.linalg.norm(chroma, axis=0)
scores = templates.T @ (chroma / np.maximum(strength, 1e-12))
chords = [labels[i] if level > 1e-8 else 'N'
          for i, level in zip(scores.argmax(axis=0), strength)]
print(chords[:10])
print('Global key baseline:', estimate_key(chroma))
```

Templates use C=0, G major = G–B–D, A minor = A–C–E. They do not cover seventh chords, inversions, tuning variation or reliable segmentation. The key baseline implements profile correlations; `librosa.feature.key_detect` does not exist. An uninformative chroma vector returns no key. Correlation is not a calibrated confidence score. [librosa beat API](https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html), [music21 key profiles](https://www.music21.org/music21docs/moduleReference/moduleAnalysisDiscrete.html)

## 6. Evaluation 评测

For FAD, the documented toolkit CLI is:

```bash
fadtk clap-laion-audio real_audio_dir generated_audio_dir
```

Report embedding model, reference set, durations, sample counts and aggregation. The original `FAD(...).precision()` / `.recall()` calls were not documented FADtk APIs and have been removed; distribution precision/recall requires a separately defined estimator. [FADtk](https://github.com/microsoft/fadtk)

Descriptive audio statistics can be computed independently:

```python
import librosa
import numpy as np
from src.audio_baselines import pitch_class_entropy

y, sr = librosa.load('input.wav', sr=22050)
_, beats = librosa.beat.beat_track(y=y, sr=sr)
intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
interval_cv = float(intervals.std()/intervals.mean()) if len(intervals) >= 2 and intervals.mean() > 0 else float('nan')
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
onsets = librosa.onset.onset_detect(y=y, sr=sr)
print('Beat-interval coefficient of variation:', interval_cv)
print('Pitch-class entropy (bits):', pitch_class_entropy(chroma))
print('Detected onsets per second:', len(onsets)/(len(y)/sr))
print('Spectral centroid (Hz):', float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()))
```

Entropy requires nonnegative values normalized to total probability one. Uniform 12-class mass has entropy `log2(12)`; zero mass has undefined entropy. Chroma entropy is not itself “tonal stability,” detected onsets are not necessarily notes, and intentional rubato is not a quality defect. Very few detected beats make interval variation unreliable.

## 7. Singing voice synthesis 歌声合成

Use the versioned setup and inference instructions from [OpenVPI DiffSinger](https://github.com/openvpi/DiffSinger) or [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc). DiffSinger synthesis and singing voice conversion are different tasks. Input phonemes, pitch, duration units, dictionaries, speaker identifiers and checkpoints must agree. The original universal `from diffsinger import DiffSinger` recipe and unverified training commands have been removed.

## 8. MIDI processing

```python
import pretty_midi
pm = pretty_midi.PrettyMIDI('input.mid')
print(pm.get_piano_roll(fs=100).shape)
for instrument in pm.instruments:
    for note in instrument.notes:
        print(instrument.program, note.pitch, note.start, note.end, note.velocity)
```

Modern MidiTok uses Symusic scores:

```python
from miditok import REMI, TokenizerConfig
from symusic import Score

tokenizer = REMI(TokenizerConfig(use_programs=True, use_tempos=True))
tokens = tokenizer(Score('input.mid'))
reconstructed = tokenizer.decode(tokens)
reconstructed.dump_midi('output.mid')
```

Tokenization quantizes time and velocity and may discard unsupported metadata; decoding is not guaranteed to reproduce the input bytes. [MidiTok](https://github.com/Natooz/MidiTok), [pretty_midi](https://craffel.github.io/pretty-midi/)

## 9. Data pipeline and common pitfalls

```python
from pathlib import Path
from src.audio_baselines import preprocess_audio

for source in Path('raw_audio').glob('*.wav'):
    preprocess_audio(source, Path('processed') / source.name,
                     target_sr=16000, target_duration=30.0)
```

This CPU function creates output directories and preserves finite silent audio. It performs mono conversion, resampling, trim/pad and peak scaling. Mono averaging can cancel antiphase stereo material; peak or RMS scaling is not perceptual loudness normalization. Keep preprocessing identical across compared systems, and do not normalize away gain errors when testing gain preservation.

For STFT reconstruction, retain complex values and use matching window, hop, centering and length in the inverse. Magnitude-only inversion needs phase estimation and is not exact; a vocoder must match its conditioning features.

`torch.cuda.empty_cache()` only releases unused cached allocations, not live tensors. Delete all references to model outputs and model objects when freeing memory. Gradient checkpointing is primarily a training technique, not a blanket inference-memory solution.

Download datasets from their [official entries](../../datasets/README.md), with explicit version and licence. Large audio archives are optional inputs, not prerequisites for this repository's CPU verification.
