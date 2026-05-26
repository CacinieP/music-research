# Audio Engineering for Music AI: Technical Notes

Comprehensive technical reference covering feature representations, neural audio codecs, diffusion models, real-time processing, and audio tokenization. Reflects the state of the art as of 2025-2026.

---

## Table of Contents

1. [Feature Representations](#1-feature-representations)
2. [Neural Audio Codecs](#2-neural-audio-codecs)
3. [Diffusion-Based Audio Models](#3-diffusion-based-audio-models)
4. [Real-Time Audio Processing](#4-real-time-audio-processing)
5. [Audio Tokenization & Discrete Representations](#5-audio-tokenization--discrete-representations)

---

## 1. Feature Representations

### 1.1 Mel-Spectrogram

The workhorse representation for modern music AI. Used by MusicGen, AudioLDM, Stable Audio, Jukebox, and most music generation/source separation models.

**Definition:**

The mel-spectrogram is the magnitude of the Short-Time Fourier Transform (STFT) passed through a mel-scale filterbank:

```
Mel_spec(t, m) = sum_k |STFT(t, k)|^2 * H_m(k)
```

where `H_m(k)` is the m-th mel filterbank triangular window. The mel scale approximates human pitch perception:

```
mel(f) = 2595 * log10(1 + f / 700)
```

**Key parameters:**
- `n_fft` (FFT window size): typically 2048 or 4096 for music (44.1/48 kHz)
- `hop_length`: typically `n_fft / 4` (e.g., 512)
- `n_mels` (number of mel bands): 64, 80, or 128
- `fmin`, `fmax`: frequency range (e.g., 0-8000 Hz or 0-22050 Hz)

**Trade-offs:**
- **Pros**: Perceptually aligned; compact (n_mels << n_fft/2); differentiable in PyTorch/Torchaudio; extensively validated.
- **Cons**: Phase information is discarded (cannot perfectly reconstruct waveform without Griffin-Lim or neural vocoder); fixed time-frequency resolution; mel scale is a rough approximation.

**When to use**: Default choice for classification, generation, tagging, and most supervised learning tasks. Use with a neural vocoder (HiFi-GAN) for waveform generation.

### 1.2 Chroma Features

Capture pitch class content independent of octave. Twelve bins corresponding to the 12 semitones (C, C#, D, ..., B).

**Definition:**

Chroma features map the frequency axis of a spectral representation (STFT, CQT) onto 12 pitch classes:

```
Chroma(t, c) = sum_{f in pitch_class(c)} |X(t, f)|
```

where `pitch_class(c)` contains all frequencies whose pitch class is `c`.

**Variants:**
- **Chroma STFT**: From STFT magnitude
- **Chroma CQT**: From CQT (better frequency alignment with musical pitches)
- **Chroma CENS (Chroma Energy Normalized Statistics)**: Quantized, smoothed, and normalized; robust to dynamics and timbre

**When to use**: Chord recognition, key detection, harmonic analysis, cover song detection, music similarity. Not suitable as a primary representation for generation.

### 1.3 Constant-Q Transform (CQT)

Provides geometrically spaced frequency bins, matching the logarithmic nature of musical pitch. The quality factor Q (center frequency / bandwidth) is constant across all bins.

**Definition:**

```
X_CQT[k] = sum_n x[n] * w_k[n] * exp(-j * 2pi * f_k * n / f_s)
```

where `Q = f_k / delta_f_k` is constant, and `f_k` are geometrically spaced.

**Key properties:**
- Bins per octave is a user parameter (typically 12, 24, or 48)
- Low frequencies have better frequency resolution, high frequencies have better time resolution
- Total number of bins: `n_bins = B * log2(fmax / fmin)` where B = bins per octave

**Computational considerations:**
- More expensive than STFT; efficient implementations use sparse kernel or FFT-based methods
- Libraries: `librosa.cqt()`, `nnAudio` (GPU-accelerated, differentiable)

**Key paper**: Brown (1991) "Calculation of a constant Q spectral transform" -- the foundational work. Schorkhuber & Klapuri (2010) for efficient computation.

**When to use**: Automatic music transcription (AMT), pitch estimation, chord recognition, any task where alignment with musical pitch structure matters.

### 1.4 Variable-Q Transform (VQT)

A generalization of CQT where Q can vary across frequency bins, allowing flexible trade-offs between time and frequency resolution at different spectral regions.

**Definition:**

The VQT relaxes the constant-Q constraint. In the limit where Q varies to match STFT behavior (constant bandwidth), the VQT becomes the STFT. The CQT is the special case where Q is constant.

The VQT is parameterized by a "gamma" value that interpolates between constant-Q (gamma=0) and constant-bandwidth (gamma=1) behavior:

```
Effective bandwidth_k = Q_k * f_k + gamma
```

**Implementation**: `librosa.vqt()` supports both CQT and VQT via the `gamma` parameter.

**When to use**: When neither purely logarithmic (CQT) nor purely linear (STFT) frequency spacing is ideal. Emerging in music transcription and analysis tasks where adaptive resolution is beneficial.

### 1.5 MFCCs (Mel-Frequency Cepstral Coefficients)

A compact representation derived from the mel-spectrogram via the Discrete Cosine Transform (DCT):

```
MFCC(t, d) = DCT(log(Mel_spec(t, m)))
```

Typically only the first 13-20 coefficients are kept (lower coefficients capture spectral shape; higher ones capture fine detail).

**Historical significance**: Dominant in speech processing (ASR, speaker identification) for decades. In music, used for genre classification, instrument recognition, and music similarity.

**Modern status**: Largely superseded by learned features (mel-spectrograms fed to deep networks) and self-supervised representations (MERT, MusicHuBERT, CLAP embeddings). MFCCs remain useful as lightweight features for low-resource scenarios.

**When to use**: Low-latency / low-compute tasks (e.g., real-time classification on embedded devices); traditional machine learning pipelines (GMMs, SVMs); as a compact descriptor for music similarity. Not recommended as primary input to modern neural networks for generation.

### 1.6 Raw Waveform

Using the raw audio samples directly as input, letting the network learn its own feature extraction.

**Approaches:**
- **Learnable frontends**: SincNet (Ravanelli & Bengio, 2018) replaces mel filterbanks with learnable bandpass filters parameterized by cutoff frequencies
- **1D convolutional encoders**: As used in EnCodec, SoundStream, DAC -- the encoder learns to extract features from raw samples
- **Sample-level autoregressive models**: WaveNet, SampleRNN

**Trade-offs:**
- **Pros**: No information loss from hand-crafted features; potentially learns task-optimal representations.
- **Cons**: Much higher computational cost (audio at 44.1 kHz = 44,100 time steps per second); requires more data and training time; less interpretable intermediate representations.

**When to use**: When training end-to-end audio codecs; when the task requires phase information; when sufficient compute and data are available.

### 1.7 Self-Supervised Learned Representations

An increasingly important category where models pre-trained on large audio corpora produce general-purpose feature embeddings.

| Model | Training | Size | Use Case |
|-------|----------|------|----------|
| **MERT** (Min et al., 2023) | Masked language modeling on music | up to 330M params | Music understanding, tagging, QA |
| **MusicHuBERT** (Huang et al., 2024) | HuBERT-style pre-training on music | ~95M | Music information retrieval |
| **CLAP** (Wu et al., 2023) | Contrastive audio-text | ~400M | Audio-language alignment, retrieval |
| **Jukebox-5B embeddings** | VQ-VAE autoregressive | 5B | Music generation features |

### Comparison Table

| Representation | Dimensionality | Differentiable | Phase Info | Music-Aligned | Compute Cost |
|---|---|---|---|---|---|
| Mel-spectrogram | n_mels x T | Yes | No | Partial (mel scale) | Low |
| Chroma | 12 x T | Yes | No | Yes (pitch class) | Low |
| CQT | n_bins x T | Yes (nnAudio) | No | Yes (log freq) | Medium |
| VQT | n_bins x T | Yes (nnAudio) | No | Adaptive | Medium |
| MFCC | 13-20 x T | Yes | No | Partial | Low |
| Raw waveform | 1 x T_samples | Yes | Yes | Learned | Very High |

### Key Papers

- McFee et al. (2015/2020) -- librosa: Audio and Music Signal Analysis in Python
- Brown (1991) -- Calculation of a constant Q spectral transform
- Schorkhuber & Klapuri (2010) -- Constant-Q Transform Toolbox for Music Processing
- Ravanelli & Bengio (2018) -- SincNet (speaker recognition with learnable filters)
- Min et al. (2023) -- MERT: Acoustic Music Understanding with Large-Scale Pre-training
- Wu et al. (2023) -- CLAP: Large-Scale Contrastive Language-Audio Pretraining

### Common Datasets for Feature Representation Research

- **AudioSet** (Gemmeke et al., 2017): 2M+ 10-second YouTube clips, 527 event classes
- **FMA** (Defferrard et al., 2017): ~100K tracks for genre classification
- **MUSDB18-HQ** (Rafii et al., 2019): 150 full tracks with stems for source separation
- **MusicCaps** (Agostinelli et al., 2023): 5,521 music segments with rich text captions
- **NSynth** (Engel et al., 2017): 306K individual notes for instrument/timbre tasks
- **MedleyDB** (Bittner et al., 2014): 122 multi-track recordings for MIR

---

## 2. Neural Audio Codecs

### 2.1 Architecture Overview

Neural audio codecs follow an **encoder-quantizer-decoder** paradigm:

```
Raw Audio -> [Encoder] -> Continuous Latent -> [Quantizer] -> Discrete Codes -> [Decoder] -> Reconstructed Audio
```

The encoder compresses the waveform into a lower-dimensional latent representation. The quantizer discretizes the continuous latents into a finite set of codes (essential for language modeling). The decoder reconstructs audio from the discrete codes.

All major codecs share this structure but differ in: (1) encoder/decoder architecture, (2) quantization method, (3) discriminator design, (4) loss functions.

### 2.2 SoundStream (Google, 2021)

**Paper**: Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec" (ICML 2021 / IEEE/ACM TASLP 2022)

**Architecture:**
- **Encoder**: 1D strided convolutions, downsampling by factor of 320 (at 24 kHz -> 75 Hz frame rate)
- **Quantizer**: Residual Vector Quantization (RVQ) with N codebooks (typically 4-12)
- **Decoder**: Transposed convolutions, upsampling back to original sample rate
- **Discriminator**: Multi-scale waveform discriminator

**Key contribution**: First neural audio codec to demonstrate that a single model can perform both audio compression and real-time streaming.

**Bitrates**: 3-18 kbps (variable by number of RVQ layers used)

### 2.3 EnCodec (Meta, 2022)

**Paper**: Defossez et al., "High Fidelity Neural Audio Compression" (arXiv: 2210.13438, 2022)

**Architecture (SEANet backbone):**
- **Encoder**: Convolutional encoder with strided convolutions
  - Downsampling factor: 320 (24 kHz model) or 640 (48 kHz model)
  - Residual units within each downsampling block
  - Two variants: **causal** (for streaming/real-time) and **non-causal** (for offline)
  - The causal variant pads only to the left (past context only), while non-causal uses symmetric padding
- **Quantizer**: Residual Vector Quantization (RVQ)
  - Codebook size: 1024 entries per level (10 bits per code)
  - Embedding dimension: 128 or 256
  - Number of quantizer levels: up to 32 (variable bitrate by truncating)
  - **Straight-Through Estimator (STE)**: Passes gradients through the non-differentiable nearest-neighbor lookup as identity during backpropagation
  - **Exponential Moving Average (EMA)**: Codebook embeddings updated via EMA of assigned encoder outputs (from VQ-VAE v2)
  - **Codebook reset**: Stale codebook entries (low usage) are reinitialized to encoder outputs to prevent codebook collapse
- **Decoder**: Mirror of encoder with transposed convolutions
- **Discriminators**:
  - Multi-Scale STFT Discriminator (operates on different spectral resolutions)
  - Multi-Scale Band Discriminator (MSBD, operates on waveform at multiple time scales)

**Loss functions:**
- **Reconstruction**: L1 distance + multi-scale spectral loss (STFT at multiple window sizes)
- **Adversarial**: Hinge loss from discriminators (least-squares in some variants)
- **Perceptual**: Balancing reconstruction fidelity vs. adversarial realism
- **Commitment loss**: `||z_e - sg(e)||^2` -- encourages encoder outputs to commit to codebook entries
- **Codebook loss**: Updates codebook embeddings toward encoder outputs

**Models released:**
- 24 kHz mono: supports 1.5, 3, 6, 12, 24 kbps
- 48 kHz stereo: supports 3, 6, 12, 24 kbps

**Frame rate**: 75 Hz at 24 kHz (320x downsampling)

### 2.4 DAC -- Descript Audio Codec (2023)

**Paper**: Kumar et al., "High-Fidelity Audio Compression with Improved RVQGAN" (arXiv: 2306.06546, 2023)

**Key improvements over EnCodec/SoundStream:**
- **Improved RVQ training**: Stop gradients on residual computations between RVQ levels, preventing early quantizer layers from being disrupted by later layers' errors
- **Quantizer dropout**: Randomly drops quantizer levels during training to ensure graceful degradation at lower bitrates
- **Multi-scale mel loss**: At multiple STFT window sizes for both short-term and long-term spectral fidelity
- **Larger codebooks**: 1024 entries, but with improved codebook utilization
- **44.1 kHz support**: Operates at 44.1 kHz (near-CD quality)

**Compression**: ~90x compression ratio (down to 8 kbps from 44.1 kHz/16-bit)

**Architecture**: 9 RVQ levels at ~86 Hz frame rate, producing ~774 tokens/second

**Discriminators**: Multi-scale STFT discriminator + multi-band discriminator

### 2.5 FunCodec (Alibaba/ModelScope, 2023)

**Paper**: Du et al., "Funcodec: A Fundamental, Reproducible and Integrable Open Source Toolkit for Neural Speech Codec"

An open-source research toolkit (not a single codec model) designed for:
- Audio quantization experiments
- Downstream applications: TTS, music generation
- Reproducible codec research

Provides modular components for building and training neural audio codecs with configurable encoder/decoder/quantizer architectures.

**GitHub**: https://github.com/modelscope/FunCodec

### 2.6 SemantiCodec (2024)

**Paper**: Liu et al., 2024

**Key innovation**: Dual-encoder architecture that decouples semantic and acoustic information:
- **Semantic encoder**: Captures high-level content (phonemes, musical notes, sound events)
- **Acoustic encoder**: Captures fine-grained audio details (timbre, room acoustics, speaker identity)
- Targets **ultra-low bitrate** applications (significantly fewer tokens per second than EnCodec/DAC)

This decoupling is motivated by the observation that for language modeling over audio, semantic tokens are more valuable than raw acoustic tokens (which can be predicted conditionally).

### 2.7 WavTokenizer (2024)

**Paper**: Pan et al., "WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer for Audio Language Models" (arXiv: 2408.16532, 2024)

**Radical simplification**: Uses **Lookup-Free Quantization (LFQ)** instead of RVQ.

**LFQ mechanism:**
- No learnable codebook embeddings. Instead, each dimension of the latent vector is independently quantized to a binary value (sign of each dimension).
- For a d-dimensional latent, this produces 2^d possible tokens. WavTokenizer uses d=12 -> 4,096 tokens.
- The **straight-through estimator** handles gradient flow: forward pass uses binary values, backward pass passes gradients through as if identity.
- **Entropy regularization**: Encourages uniform utilization of all possible tokens (avoids codebook collapse without needing codebook reset).
- **Commitment loss**: `||z_e - sg(z_q)||^2` similar to VQ-VAE.

**Architecture:**
- Single codebook (1 layer of LFQ), no residual stacking
- ~75 Hz frame rate -> ~75 tokens/second (vs. 600-2400 for EnCodec)
- VQGAN-style training with multi-scale discriminators

**Impact on language modeling**: A 10-second clip = ~750 tokens (single stream), vs. thousands with multi-codebook RVQ. This maps directly onto standard LLM next-token prediction.

### 2.8 Recent Codecs (2025-2026)

**TQCodec** (arXiv: 2603.01592): Proposes trellis-based quantization for high-bitrate, high-fidelity music streaming. Targets applications where quality is prioritized over extreme compression.

**SUNAC** (MERL, 2026): Source-aware Unified Neural Audio Codec. A single model that handles speech, music, and sound effects with domain awareness built into the codec itself.

**SDCodec**: Uses three domain-specific RVQ modules (speech, music, SFX) within a single codec framework.

**HiFi-Codec** (Yang et al., 2024): Proposes Grouped-RVQ (GRVQ) to reduce codebook requirements while maintaining quality.

### 2.9 Quantization Methods Comparison

| Method | Codebooks | Bits/code | Tokens/sec | Codebook Learning | Collapse Risk |
|--------|-----------|-----------|------------|-------------------|---------------|
| **VQ** (single) | 1 | 10 | 75-86 | EMA or SGD | High |
| **RVQ** (EnCodec) | 8-32 | 10 | 600-2,400 | EMA + codebook reset | Medium |
| **RVQ** (DAC) | 9 | 10 | ~774 | EMA + stop-grad residuals | Low |
| **LFQ** (WavTokenizer) | 1 | 12 | ~75 | None (binary) | Very Low |
| **GRVQ** (HiFi-Codec) | 4-8 (grouped) | 10 | ~300-600 | Grouped EMA | Low |

### 2.10 Bitrate-Quality Tradeoffs

General benchmarks (ViSQOL, PESQ, MOS):
- **1.5 kbps**: EnCodec -- intelligible speech, degraded music
- **3-6 kbps**: Acceptable speech quality; music lacks high-frequency detail
- **8 kbps**: DAC -- near-transparent for speech; music has minor artifacts
- **12-24 kbps**: EnCodec/DAC -- good music quality, approaching transparency
- **48+ kbps**: Near-transparent to transparent (approaching lossless)

Standard evaluation metrics:
- **ViSQOL** (Virtual Speech Quality Objective Listener): 1-5 scale, correlates with MOS
- **PESQ** (Perceptual Evaluation of Speech Quality): -0.5 to 4.5, ITU-T P.862
- **MOS** (Mean Opinion Score): Subjective 1-5 scale
- **STOI** (Short-Time Objective Intelligibility): 0-1, speech intelligibility

### Key Papers

- Zeghidour et al. (2021) -- SoundStream
- Defossez et al. (2022) -- EnCodec
- Kumar et al. (2023) -- DAC (Improved RVQGAN)
- Du et al. (2023) -- FunCodec
- Liu et al. (2024) -- SemantiCodec
- Pan et al. (2024) -- WavTokenizer
- Yang et al. (2024) -- HiFi-Codec
- TQCodec (2025/2026) -- arXiv: 2603.01592
- SUNAC (MERL, 2026)

### Common Datasets for Codec Training & Evaluation

- **DNS Challenge**: Speech + noise for speech codec evaluation
- **VCTK**: Multi-speaker English speech
- **LibriSpeech**: Read English speech (1000 hours)
- **MUSDB18-HQ**: Music tracks for music codec evaluation
- **AudioSet**: General audio for universal codec training
- **MusicCaps**: Music with text descriptions

---

## 3. Diffusion-Based Audio Models

### 3.1 Theoretical Foundations

#### DDPM (Denoising Diffusion Probabilistic Models)

**Paper**: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)

The forward diffusion process gradually adds Gaussian noise over T timesteps:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
```

where `beta_t` is a variance schedule. The reverse process learns to denoise:

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```

The model `mu_theta` is trained to predict the noise that was added (or equivalently, to predict `x_0`).

For audio, this framework can be applied to:
- **Spectrograms**: Treat the mel-spectrogram as a 2D image and apply image diffusion techniques
- **Waveforms**: Apply diffusion directly to 1D audio samples (much more computationally expensive)

#### Score-Based Generative Models (SDE Framework)

**Paper**: Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR 2021)

Unifies DDPM and score-matching (NCSN/SMLD) under a continuous-time SDE framework.

**Forward SDE** (data -> noise):
```
dx = f(x, t)dt + g(t)dw
```

**Reverse SDE** (noise -> data):
```
dx = [f(x, t) - g(t)^2 * nabla_x log p_t(x)] dt + g(t) dw_bar
```

where `nabla_x log p_t(x)` is the **score function**, estimated by a neural network `s_theta(x, t)`.

**Three SDE types:**
1. **Variance Exploding (VE)**: Corresponds to NCSN/SMLD. `f(x,t) = 0`, noise variance increases.
2. **Variance Preserving (VP)**: Corresponds to DDPM. Mean scales down as variance increases.
3. **Sub-VP**: A variant that bounds likelihood.

**Probability Flow ODE**: For each SDE, there exists a deterministic ODE with the same marginal distributions:
```
dx = [f(x, t) - (1/2) * g(t)^2 * nabla_x log p_t(x)] dt
```
This enables exact likelihood computation, latent manipulation, and deterministic sampling.

**Sampling methods:**
- DDIM (Denoising Diffusion Implicit Models): Deterministic variant, fewer steps
- DPM-Solver: Higher-order ODE solver, 10-20 steps for good quality
- Consistency Models: Single-step generation by distilling a diffusion model

### 3.2 Latent Diffusion for Audio

Rather than applying diffusion in the high-dimensional audio/spectrogram space, latent diffusion first compresses audio into a compact latent space (using a VAE or autoencoder), then runs diffusion in that space.

**Advantages:**
- Much more computationally efficient (latent space is 4-64x smaller than spectrogram space)
- Better audio quality for the same compute budget
- Easier to condition on text/embeddings

#### AudioLDM (Liu et al., 2023)

**Paper**: "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models" (ICML 2023)

- Uses a **VAE** trained on audio spectrograms to create a latent space
- **CLAP** (Contrastive Language-Audio Pretraining) embeddings for text conditioning
- Latent diffusion with classifier-free guidance
- Generates audio from text descriptions

#### AudioLDM 2 (Liu et al., 2023-2024)

**Paper**: "AudioLDM 2: Learning Holistic Audio Generation with Self-Supervised Pretraining" (IEEE TASLP 2024)

- **Unified framework** for speech, music, and sound effect generation
- Introduces a **"audio text"** intermediate representation that bridges text and audio
- Two-stage: text -> audio text representation -> audio via latent diffusion
- Self-supervised pretraining improves generation quality across all domains

#### Stable Audio (Stability AI, 2024)

**Paper**: Roberts et al., "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion" (arXiv: 2402.04825, 2024)

**Architecture:**
- **Autoencoder**: Trained on 44.1 kHz stereo audio, compresses to a latent representation
- **Latent diffusion model**: DiT (Diffusion Transformer) architecture in the latent space
- **Timing conditioning**: Model is conditioned on start time and total duration, enabling generation of specific segments
- **Text conditioning**: Via CLAP text embeddings + T5 text encoder

**Stable Audio 2.0** (2024): Generates full-length tracks with coherent musical structure (intro, development, outro).

**Stable Audio Open** (2024): Open-weight variant for research use.

**Key technical details:**
- Generates at 44.1 kHz stereo
- DiT backbone with cross-attention for text conditioning
- Classifier-free guidance during inference
- Sampling with DPM-Solver++ for efficiency

### 3.3 Classifier-Free Guidance (CFG)

**Paper**: Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022)

A technique for controlling the tradeoff between quality and diversity in conditional diffusion models:

1. **Training**: The model is trained both with and without conditioning information (e.g., text prompt). The conditioning is randomly dropped (e.g., 10% of the time) and replaced with a null embedding.

2. **Inference**: The prediction is extrapolated between conditional and unconditional outputs:

```
tilde_epsilon = (1 + w) * epsilon_theta(x_t, c) - w * epsilon_theta(x_t, empty)
```

where `w` is the guidance scale:
- `w = 0`: Standard conditional generation (no guidance)
- `w = 1`: No guidance effect (same as training)
- `w > 1`: Stronger adherence to conditioning, higher "quality" but lower diversity
- Typical values for audio: `w = 3-7`

**Advantages over classifier guidance**: No need to train a separate classifier in the noisy latent space.

### 3.4 Diffusion Applied to Spectrograms vs. Waveforms

| Aspect | Spectrogram Diffusion | Waveform Diffusion |
|--------|----------------------|--------------------|
| **Dimensionality** | Lower (e.g., 80 x T) | Much higher (44100 * T) |
| **Compute** | Feasible on single GPU | Very expensive |
| **Quality** | Requires vocoder for waveform | Direct waveform output |
| **Phase** | Not modeled (vocoder fills in) | Explicitly modeled |
| **Example models** | AudioLDM, Grad-TTS | DiffWave |
| **Latency** | Diffusion + vocoder | Diffusion only (but slower) |

### 3.5 Key Papers

- Ho et al. (2020) -- DDPM
- Song et al. (2021) -- Score-Based Generative Modeling through SDEs
- Ho & Salimans (2022) -- Classifier-Free Diffusion Guidance
- Liu et al. (2023) -- AudioLDM
- Liu et al. (2024) -- AudioLDM 2 (IEEE TASLP)
- Roberts et al. (2024) -- Stable Audio
- Dhariwal & Nichol (2021) -- DiffWave (diffusion on waveforms)
- Popov et al. (2021) -- Grad-TTS (diffusion for TTS)
- Kong et al. (2021) -- DiffWave: A Versatile Diffusion Model for Audio
- Chen et al. (2023) -- Make-An-Audio

### 3.6 Current Challenges

- **Sampling speed**: Standard diffusion requires 50-1000 denoising steps; even with DPM-Solver (10-20 steps), generation is slower than autoregressive approaches
- **Long-form generation**: Stable Audio 2.0 addresses this, but maintaining coherence over minutes remains hard
- **Fine-grained control**: Text conditioning provides coarse control; note-level or instrument-level control is an open problem
- **Real-time generation**: Consistency models and adversarial distillation are being explored for real-time diffusion

---

## 4. Real-Time Audio Processing

### 4.1 Streaming Architecture Requirements

Real-time audio processing imposes strict constraints:

- **Latency**: Typically < 10-20 ms for live performance applications (at 44.1 kHz, 10 ms = 441 samples)
- **Causality**: The model must not access future samples (no lookahead beyond a small buffer)
- **Computational budget**: Must process each audio block within its duration (real-time factor < 1.0)
- **Deterministic memory**: Fixed memory allocation, no dynamic allocation in the audio thread
- **No system calls**: Audio thread must not perform I/O, memory allocation, or blocking operations

### 4.2 Causal Convolutions

Standard convolutions are non-causal: the output at time `t` depends on both past and future inputs. For streaming, we need **causal convolutions** where output depends only on past and current inputs.

**Implementation**: Left-pad the input so that the convolution kernel only "sees" past context:

```
Standard:  y[t] = sum_k x[t + k - K//2] * w[k]   (centered)
Causal:    y[t] = sum_k x[t - k] * w[k]            (left-aligned)
```

In practice, for a convolution with kernel size K, the causal version introduces a delay of (K-1) samples.

**Dilated causal convolutions** (from WaveNet, van den Oord et al., 2016):
- Use dilation to exponentially increase receptive field without increasing parameters
- Dilation pattern: 1, 2, 4, 8, 16, ... (doubling)
- With L layers of kernel size K and max dilation D, receptive field = (K-1) * sum(2^l) for l=0..L-1

### 4.3 EnCodec Streaming Mode

EnCodec explicitly provides a **causal variant** of its SEANet architecture for streaming:
- All convolutions use causal (left-only) padding
- The model processes audio in small chunks (typically 1-10 ms)
- Internal state (convolution buffers) is maintained across chunks
- Introduces a small algorithmic latency equal to the encoder's total receptive field

**Latency breakdown for EnCodec 24 kHz:**
- Encoder stride: 320 samples = 13 ms at 24 kHz (this is the frame rate)
- Additional lookback from convolution context: varies by model depth
- Total latency: typically 20-50 ms (acceptable for streaming, too high for live monitoring)

### 4.4 Streamable Non-Causal Models

**Paper**: "Streamable Neural Audio Synthesis with Non-Causal Convolutions" (Semantic Scholar)

A key insight: non-causal (bidirectional) models generally produce higher quality than causal models because they can use future context. This paper proposes methods to make non-causal models streamable by:
- Processing audio in overlapping chunks
- Using a look-ahead buffer to provide the "future" context
- Trading off latency for quality: more look-ahead = higher quality but higher latency

**Practical tradeoff**: A 20 ms look-ahead buffer at 48 kHz = 960 samples of additional latency, but enables using non-causal convolutions in a streaming context.

### 4.5 Real-Time Neural Inference Frameworks

#### RTNeural (Jatin Chowdhury)

**GitHub**: https://github.com/jatinchowdhury18/RTNeural

A C++ library designed specifically for **real-time audio-rate neural network inference**:
- Loads models trained in TensorFlow, PyTorch, or ONNX
- Optimized for **sample-by-sample processing** (no batch dimension)
- Supports: Dense, Conv1D, LSTM, GRU layers
- Designed for **hard real-time constraints**: no memory allocation in the audio callback
- Used for: guitar amp modeling, audio effect emulation, neural synth plugins
- Deployed as VST/AU plugins via JUCE

**Workflow**: Train in Python -> Export weights (JSON/ONNX) -> Load in RTNeural C++ -> Run in audio callback

#### ANIRA (2025)

**Paper**: arXiv: 2506.12665

Benchmarks three neural network architectures for audio effect emulation in real-time settings, comparing inference backends and latency characteristics.

#### Other Frameworks

- **Neural Amp Modeler (NAM)**: Open-source guitar amp modeler using neural networks
- **Proteus**: Guitar amp/pedal emulation
- **TorchAudio + TorchScript**: Deploy PyTorch models with real-time constraints
- **ONNX Runtime**: General-purpose neural inference, applicable to audio models
- **Faust**: Functional audio programming language; can integrate neural network inference

### 4.6 Latency Considerations for Live Music

| Application | Max Acceptable Latency | Notes |
|-------------|----------------------|-------|
| Guitar amp modeling | 1-5 ms | Players feel latency above 5-10 ms |
| Vocal processing | 5-10 ms | Slightly more tolerant than guitar |
| Live monitoring | < 10 ms | General guideline |
| Live electronic music | 10-50 ms | Often more tolerant |
| Non-interactive (streaming) | 100+ ms | Acceptable for broadcast/telecom |

**Latency sources:**
1. **Algorithmic latency**: Model's inherent delay (from causal padding, downsampling)
2. **Computation latency**: Time to process one frame of audio (must be < frame duration)
3. **Buffer latency**: Audio interface buffer size (typically 64-256 samples = 1.5-6 ms at 44.1 kHz)
4. **Total round-trip**: A/D conversion + buffer + processing + buffer + D/A conversion

### 4.7 Key Papers

- van den Oord et al. (2016) -- WaveNet: A Generative Model for Raw Audio
- Defossez et al. (2022) -- EnCodec (causal/non-causal streaming architecture)
- "Streamable Neural Audio Synthesis with Non-Causal Convolutions" -- bridging non-causal models and real-time
- Chowdhury -- RTNeural: Real-time neural inference for audio
- "Fast Temporal Convolutions for Real-Time Audio Signal Processing" (DAFx 2020)

### 4.8 Current Challenges

- **Quality-latency tradeoff**: Higher quality models (more layers, non-causal) have higher latency
- **Model size**: Large models (EnCodec, MusicGen) cannot run at audio rate on embedded hardware
- **ARM/embedded deployment**: Limited compute on mobile and edge devices
- **Dynamic models**: Models with variable compute (e.g., attention mechanisms) can cause buffer underruns
- **Training-inference mismatch**: Streaming mode may perform differently than offline mode

---

## 5. Audio Tokenization & Discrete Representations

### 5.1 Why Discretize Audio?

Continuous audio waveforms are high-dimensional (44,100 samples/second at CD quality). For language model approaches (AudioLM, MusicLM, MusicGen, VALL-E), audio must be converted into **discrete token sequences** analogous to text tokens:

```
Audio waveform -> [Neural Codec] -> Token sequence -> [Language Model] -> Token sequence -> [Codec Decoder] -> Audio waveform
```

The properties of the tokenization determine:
- **Sequence length**: Fewer tokens = more efficient language modeling
- **Information preservation**: Tokens must capture enough for high-fidelity reconstruction
- **Semantic content**: Ideally, tokens should capture meaningful structure (phonemes, notes)
- **Language model compatibility**: Single-stream tokens are easier for standard LLMs than multi-stream

### 5.2 Vector Quantization (VQ)

The fundamental building block. Given a continuous vector `z` and a codebook `C = {e_1, ..., e_K}`:

```
VQ(z) = argmin_k ||z - e_k||^2
```

The quantized vector is the nearest codebook entry. This maps continuous space to K discrete codes.

**Training**: Codebook entries are learned jointly with the encoder/decoder. Two main update strategies:
1. **Gradient-based**: Update codebook embeddings via standard gradient descent (with STE for the non-differentiable argmin)
2. **EMA (Exponential Moving Average)**: Each codebook entry tracks the running average of encoder outputs assigned to it: `e_k = decay * e_k + (1 - decay) * mean(z_i where VQ(z_i) = k)`

**Codebook collapse**: A common failure mode where only a small fraction of codebook entries are used. The model "ignores" most of the codebook, reducing effective capacity.

**Solutions to collapse:**
- **Codebook reset**: Re-initialize unused entries (EnCodec)
- **Entropy regularization**: Add a loss term encouraging uniform codebook usage
- **LFQ**: Avoid learnable codebooks entirely (WavTokenizer)
- **Product quantization**: Split the vector and quantize sub-vectors independently

### 5.3 Residual Vector Quantization (RVQ)

The dominant quantization method in current codecs. Stacks multiple VQ layers, each quantizing the residual (error) of the previous layer:

```
Step 1: q_1 = VQ_1(z)           -> residual r_1 = z - q_1
Step 2: q_2 = VQ_2(r_1)         -> residual r_2 = r_1 - q_2
Step 3: q_3 = VQ_3(r_2)         -> residual r_3 = r_2 - q_3
...
Step N: q_N = VQ_N(r_{N-1})
```

**Reconstruction**: `z_hat = q_1 + q_2 + ... + q_N`

**Variable bitrate**: Use only the first K layers (K <= N) for lower bitrate; quality increases with more layers.

**Mathematical interpretation**: RVQ performs iterative refinement. Each layer captures a coarser-to-finer level of detail. The first few layers capture most of the energy/information; later layers add fine detail.

**In DAC (Kumar et al., 2023)**: The key improvement is applying **stop gradients** between RVQ levels during training. Without this, gradients from later quantizer levels can destabilize earlier levels. DAC also uses quantizer dropout during training.

**Parameters in practice:**
- **EnCodec**: N = 8-32 levels, K = 1024 codes/level, 10 bits/level
- **DAC**: N = 9 levels, K = 1024 codes/level
- **SoundStream**: N = 4-12 levels

### 5.4 Lookup-Free Quantization (LFQ)

Used in WavTokenizer (Pan et al., 2024). Instead of learning codebook embeddings, LFQ quantizes each dimension independently to a binary value:

```
LFQ(z_i) = sign(z_i)   (per dimension i)
```

For a d-dimensional vector, this produces 2^d possible codes. With d=12, that is 4,096 codes.

**Advantages:**
- **No codebook collapse**: All 2^d codes are reachable; no learnable parameters to collapse
- **Simplicity**: No codebook storage, no EMA updates
- **Efficiency**: Binary quantization is trivially fast

**Training losses:**
- **Commitment loss**: `||z - sg(z_q)||^2` -- encourages encoder to produce values close to the quantized binary values
- **Entropy loss**: `H(z_q)` -- encourages uniform distribution over all codes (maximizes codebook utilization)
- **Straight-through estimator**: Forward pass uses quantized values; backward pass passes gradients through unchanged

### 5.5 Grouped and Structured Variants

**Grouped RVQ (GRVQ)** -- HiFi-Codec (Yang et al., 2024):
- Divides the latent dimension into groups and quantizes each group with a separate RVQ
- Reduces the total number of codebooks needed

**Cross-Scale RVQ (CS-RVQ)** -- ESC (EMNLP 2024):
- Combines different quantization granularities across scales

**Trellis Quantization** -- TQCodec (2025/2026):
- Organizes codebook entries in a trellis structure for efficient sequential encoding

### 5.6 The Semantic vs. Acoustic Token Split

A key insight from AudioLM (Borsos et al., 2022) and MusicLM (Agostinelli et al., 2023):

**Semantic tokens** (high-level content):
- Extracted from self-supervised models (w2v-BERT, HuBERT, MERT)
- Capture phonemes, notes, rhythmic patterns
- Lower temporal resolution
- More relevant for language modeling (the "what")

**Acoustic tokens** (fine detail):
- From neural codec codebooks (SoundStream, EnCodec)
- Capture timbre, room acoustics, speaker identity
- Higher temporal resolution
- More relevant for reconstruction (the "how")

**Hierarchical generation** (AudioLM/MusicLM):
1. Generate semantic tokens autoregressively (models high-level structure)
2. Condition coarse acoustic tokens on semantic tokens (adds prosody, timbre)
3. Condition fine acoustic tokens on coarse tokens (adds waveform detail)

This hierarchy produces better results than directly modeling all codec tokens, because the language model can focus on structure at the semantic level before filling in detail.

### 5.7 From Multi-Codebook to Single-Stream

A major trend in 2024-2025 is reducing the number of token streams:

| Approach | Token Streams | Tokens/sec | LLM Complexity |
|----------|--------------|------------|-----------------|
| EnCodec RVQ (8 levels) | 8 parallel | 600 | Delay pattern interleaving |
| EnCodec RVQ (32 levels) | 32 parallel | 2,400 | Complex flattening |
| MusicGen | 4 parallel (with delay pattern) | 300 | Interleaved autoregressive |
| WavTokenizer (LFQ) | 1 | ~75 | Standard next-token prediction |

**MusicGen's approach**: Uses EnCodec's 4 codebook levels with a **delay pattern** where each codebook stream is offset by one position, then flattened into a single sequence. The transformer predicts all codebook levels for each time step in parallel.

**Single-stream codecs** (WavTokenizer, SemantiCodec): Eliminate the multi-stream problem entirely, enabling direct use of standard LLM architectures (LLaMA-style transformers) without special interleaving or delay patterns.

### 5.8 Key Papers

- van den Oord et al. (2017) -- VQ-VAE: Neural Discrete Representation Learning
- Razavi et al. (2019) -- Generating Diverse High-Fidelity Images with VQ-VAE-2
- Zeghidour et al. (2021) -- SoundStream
- Defossez et al. (2022) -- EnCodec
- Borsos et al. (2022) -- AudioLM: A Language Modeling Approach to Audio Generation
- Agostinelli et al. (2023) -- MusicLM: Generating Music From Text
- Kumar et al. (2023) -- DAC (Improved RVQGAN)
- Copet et al. (2023) -- MusicGen: Simple and Controllable Music Generation
- Pan et al. (2024) -- WavTokenizer
- Liu et al. (2024) -- SemantiCodec
- Yang et al. (2024) -- HiFi-Codec

### 5.9 Current Challenges

- **Token efficiency**: Balancing reconstruction quality with token count. Fewer tokens make language modeling easier but may lose audio fidelity.
- **Semantic alignment**: Codec tokens capture acoustic detail but may not align well with semantic concepts (notes, phonemes). SemantiCodec and related work address this.
- **Token inconsistency**: Small perturbations in audio can lead to large changes in discrete tokens (ACL 2025 paper analyzes this problem).
- **Domain generality**: Codecs trained on speech may not generalize to music and vice versa. SUNAC and SDCodec address multi-domain tokenization.
- **Reconstruction ceiling**: The quality of generated audio is fundamentally limited by codec reconstruction quality. Even perfect language model predictions will sound bad if the codec cannot faithfully reconstruct.
- **Stereo/spatial**: Most codecs operate in mono. Spatial audio tokenization is underexplored.

---

## Appendix: Tools and Libraries

| Tool | Purpose | Link |
|------|---------|------|
| **librosa** | Audio analysis, feature extraction | https://librosa.org |
| **torchaudio** | PyTorch audio, differentiable transforms | https://pytorch.org/audio |
| **nnAudio** | GPU-accelerated, differentiable audio transforms | https://github.com/KinWaiCheuk/nnAudio |
| **demucs** | Source separation (Meta) | https://github.com/facebookresearch/demucs |
| **audiocraft** | MusicGen, AudioGen, EnCodec (Meta) | https://github.com/facebookresearch/audiocraft |
| **descript-audio-codec** | DAC inference/training | https://github.com/descriptinc/descript-audio-codec |
| **stable-audio-tools** | Stable Audio training/inference | https://github.com/Stability-AI/stable-audio-tools |
| **diffusers** | HuggingFace diffusion pipelines (includes AudioLDM 2) | https://github.com/huggingface/diffusers |
| **RTNeural** | Real-time neural inference for audio | https://github.com/jatinchowdhury18/RTNeural |
| **WavTokenizer** | LFQ-based audio tokenizer | https://github.com/novelfm/WavTokenizer |
| **FunCodec** | Neural codec research toolkit | https://github.com/modelscope/FunCodec |
| **Faust** | Functional audio programming language | https://faust.grame.fr |

---

*Last updated: May 2026*
