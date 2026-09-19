# Audio Engineering for Music AI: Technical Notes

> 中文版：[audio-engineering-zh.md](audio-engineering-zh.md)

Technical reference for representations, neural codecs, diffusion, streaming, and tokenization. Content checked on **2026-09-19** against the linked papers and implementations. Model results below describe specific published configurations, rather than a comprehensive current leaderboard.

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

A mel-spectrogram aggregates STFT bins using a mel-spaced filterbank. A **power** mel-spectrogram is:

```text
M(t, m) = sum_k H_m(k) * |X(t, k)|^2
```

Here `X` is the complex STFT and `H_m` is a filter weight. Using `|X|` instead gives a magnitude mel-spectrogram. Log compression needs a positive floor, for example `log(max(M, epsilon))`. Specify the power, normalization, log convention, and reference level when reproducing a model.

The HTK mel convention is `2595 * log10(1 + f / 700)`. It is one convention; librosa defaults to the Slaney convention unless `htk=True`. `n_fft` controls FFT length, `win_length` controls the actual window, and `hop_length` controls frame spacing. Set `fmax <= sample_rate / 2`. [librosa mel-spectrogram documentation](https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html)

Mel features discard phase and aggregate frequency detail. Griffin–Lim or a neural vocoder can synthesize a waveform from an estimated spectrum, but neither guarantees recovery of the original waveform. Mel features are common in tagging and AudioLDM; MusicGen models codec tokens, and Jukebox uses waveform VQ-VAE codes. A spectral training loss does not make a model a mel-spectrogram generator.

### 1.2 Chroma Features

Chroma folds spectral energy across octaves into pitch classes, commonly 12 for twelve-tone equal-tempered music. A weighted formulation is:

```text
C(t, c) = sum_k W(c, k) * |X(t, k)|^p
```

The mapping `W` depends on tuning, frequency representation, and normalization; `p` is commonly 1 or 2. STFT chroma, CQT chroma, and smoothed/quantized CENS features serve different invariance goals. Chroma is useful for harmony, key, alignment, and version identification, but loses octave and timbral detail. Twelve-bin chroma is not a universal representation of every tuning system.

### 1.3 Constant-Q Transform (CQT)

CQT uses geometrically spaced center frequencies and approximately constant center-frequency-to-bandwidth ratio:

```text
f_k = f_min * 2^(k / B),  k = 0, ..., K-1
Q = f_k / bandwidth_k
X_CQT(t, k) = sum_n x[t*H + n] * w_k[n] * exp(-j*2*pi*f_k*n/f_s)
```

`B` is bins per octave and `H` is the hop. Window length varies with frequency: lower bins use longer windows. If both endpoint centers are included, `K = 1 + floor(B * log2(f_max / f_min))`, subject to Nyquist and filter support constraints. The common approximation without `+1` describes the interval span, not an exact inclusive bin count.

**Complex CQT coefficients retain phase.** Magnitude CQT features discard it. Inversion depends on the particular transform and sampling scheme; a magnitude-only representation is not automatically invertible. Brown's 1991 work is foundational; implementation details and inverse routines are documented in [librosa CQT](https://librosa.org/doc/0.11.0/generated/librosa.cqt.html).

### 1.4 Variable-Q Transform (VQT)

In the librosa formulation, filter bandwidth is:

```text
bandwidth_k = alpha * f_k + gamma
Q_k = f_k / bandwidth_k
```

`alpha` is a relative bandwidth factor; `gamma` is a bandwidth offset in Hz. `gamma=0` gives constant Q. Positive `gamma` broadens low-frequency filters relative to CQT and improves their time resolution. It is **not** a 0-to-1 interpolation parameter: `gamma=1` does not turn VQT into an STFT, and changing gamma does not turn geometric center spacing into linear spacing. The transform returns complex coefficients. [librosa VQT documentation](https://librosa.org/doc/0.11.0/generated/librosa.vqt.html)

### 1.5 MFCCs

MFCCs apply a discrete cosine transform to log-mel energies:

```text
MFCC(t, d) = DCT_m(log(max(M(t, m), epsilon)))[d]
```

Retaining a small number of coefficients gives a compact spectral-envelope descriptor. This can suit classical classifiers, clustering, and limited-compute systems. The coefficient count, inclusion of coefficient 0, DCT normalization, and log convention are part of the feature definition. MFCCs are not a lossless audio representation.

### 1.6 Raw Waveform

Waveform frontends preserve the available sampled signal before learned compression. Examples include SincNet's parameterized band-pass convolutions and the convolutional encoders in SoundStream, EnCodec, and DAC. WaveNet instead predicts waveform samples autoregressively. Compute depends on downsampling, architecture, and sequence length; a waveform input does not imply sample-by-sample autoregressive inference. [SincNet](https://arxiv.org/abs/1808.00158), [WaveNet](https://arxiv.org/abs/1609.03499)

### 1.7 Learned Representations

| Model | Training representation | Typical use |
|---|---|---|
| MERT (Li et al.; 2023 preprint, ICLR 2024) | Masked prediction with RVQ-VAE acoustic targets and CQT musical targets; 95M/330M original variants | Music understanding features |
| MusicFM (Won, Hung, Le; 2023 preprint) | Self-supervised audio representation learning | Frame-level and clip-level MIR |
| LAION-CLAP (Wu et al.; 2022 preprint) | Contrastive audio–text alignment | Retrieval and prompted zero-shot classification |
| Jukebox representations | Hidden states from a music-generation model | Transfer features; layer selection matters |

These are different training objectives: contrastive audio–text models use paired text supervision, while audio-only self-supervised models need a separate mechanism for text retrieval. Parameter counts and frame rates are checkpoint-specific. [MERT](https://arxiv.org/abs/2306.00107), [MusicFM](https://arxiv.org/abs/2311.03318), [CLAP](https://arxiv.org/abs/2211.06687)

| Representation | Shape, omitting batch/channel axes | Phase | Main limitation |
|---|---|---|---|
| Power mel | `n_mels × T` | Discarded | Frequency aggregation |
| Chroma | Commonly `12 × T` | Discarded | Octave/timbre information lost |
| Complex CQT/VQT | `n_bins × T`, complex | Retained | Resolution and inversion depend on implementation |
| MFCC | `n_coeffs × T` | Discarded | Spectral detail removed by truncation |
| Waveform | `channels × n_samples` | Present in signal | High sample rate before compression |

Differentiability is implementation-dependent. NumPy/librosa feature extraction is not a PyTorch autograd graph; tensor implementations such as nnAudio can supply differentiable transforms.

---

## 2. Neural Audio Codecs

### 2.1 Architecture and Rate Accounting

```text
waveform -> encoder -> latent vectors -> quantizer -> indices
indices -> codebook vectors -> decoder -> reconstructed waveform
```

For `N` codebooks of size `K`, sampled at frame rate `F`, the nominal fixed-length payload is:

```text
indices_per_second = F * N
bits_per_second = F * N * ceil(log2(K))
```

This excludes packet headers, scale metadata, padding, and entropy coding. Frame rate, number of indices, and autoregressive transformer steps are different quantities. Nominal PCM bitrate is `sample_rate * bits_per_sample * channels`.

### 2.2 SoundStream

Zeghidour et al., [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312), 2021 preprint / TASLP volume 30 (2022). It combines a convolutional encoder/decoder, residual vector quantization (RVQ), adversarial training, and reconstruction objectives. Quantizer dropout supports several bitrates in one trained model. The paper evaluates 24 kHz audio and 3–18 kbps configurations, with streaming and joint compression/enhancement experiments. It should not be credited as the first neural codec capable of real-time compression without a historical comparison.

### 2.3 EnCodec

Défossez et al., [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438), 2022 preprint / TMLR 2023. SEANet-style convolutions and LSTMs surround an RVQ bottleneck. Its key training components include a multi-scale STFT discriminator and a loss balancer, with waveform/spectral reconstruction and adversarial feature matching. The released RVQ uses EMA codebook updates and replacement of stale codes; a separate gradient-updated codebook loss should not be assumed.

The [released model definitions](https://github.com/facebookresearch/encodec/blob/main/encodec/model.py) distinguish:

| Released model | Causality | Encoder hop | Frame rate | Nominal bandwidths |
|---|---|---|---|---|
| 24 kHz mono | Causal | 320 samples | 75 Hz | 1.5, 3, 6, 12, 24 kbps |
| 48 kHz stereo | Non-causal | 320 samples | 150 Hz | 3, 6, 12, 24 kbps |

Each index represents one of 1,024 codes (10 bits). For 24 kHz at 6 kbps, eight codebooks give `75 * 8 * 10 = 6000` bits/s. The 48 kHz release uses normalization and overlapping segments; it is not the causal streaming model.

### 2.4 DAC — Descript Audio Codec

Kumar et al., [High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546), NeurIPS 2023. Key choices include low-dimensional factorized codebook lookup, L2-normalized lookup vectors, periodic Snake activations, quantizer dropout, and multi-scale spectral/adversarial objectives. Its codebook entries are learned with a codebook loss; this differs from EnCodec's EMA updates. A 1,024-entry codebook is not larger than EnCodec's.

For the [44.1 kHz default architecture](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/dac.py), hop 512 and nine 1,024-entry codebooks give approximately `86.13` frames/s, `775.20` indices/s, and `7.75` kbps before overhead. Relative to **mono** 44.1 kHz/16-bit PCM this is about 91-fold nominal compression. This arithmetic is not a claim of perceptual transparency or losslessness. [Quantizer implementation](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/quantize.py)

### 2.5 FunCodec

[FunCodec](https://github.com/modelscope/FunCodec) is a research toolkit for neural speech codecs, with modular training/inference components. Its capabilities and released checkpoints should be distinguished from the properties of any one codec. A speech-oriented toolkit is not by itself evidence of high-fidelity music performance.

### 2.6 SemantiCodec

Liu et al., [SemantiCodec](https://arxiv.org/abs/2405.00233), 2024. A semantic encoder uses AudioMAE features discretized with k-means; an acoustic encoder represents remaining detail. A diffusion decoder reconstructs audio from both. The paper describes 25/50/100 total tokens per second and approximately 0.31–1.40 kbps configurations. Semantic and acoustic codes remain distinct components: this is not simply a one-codebook tokenizer interchangeable with WavTokenizer.

### 2.7 WavTokenizer

Ji et al., [WavTokenizer](https://arxiv.org/abs/2408.16532), 2024 preprint / ICLR 2025. It uses **a learned vector-quantization codebook**, with 4,096 entries, and 40 or 75 tokens/s for 24 kHz audio. Expanded VQ space, code utilization techniques, attention, and a Fourier-based decoder support the single-quantizer design. It does **not** use lookup-free binary quantization. A 4,096-way index has 12 bits, but that does not mean the learned latent is a 12-dimensional binary vector. [Official implementation](https://github.com/jishengpeng/WavTokenizer)

### 2.8 Other Verified Codec Directions

- **HiFi-Codec** (Yang et al., 2023): grouped residual vector quantization, with a four-codebook system in the paper. Its evaluations emphasize speech/TTS datasets. [Paper](https://arxiv.org/abs/2305.02765)
- **TQCodec** (He et al., March 2026 preprint): 44.1 kHz music at 32–128 kbps, using SEANet, SimVQ, phase-aware loss, and perceptual band-wise bit allocation. The paper does not propose trellis quantization. [Paper](https://arxiv.org/abs/2603.01592)
- **SUNAC** (Aihara et al., 2025 preprint / ICASSP 2026): source-type prompts select sources to encode directly from mixtures, including multiple sources of the same type. Its purpose goes beyond generic domain awareness. [Paper](https://arxiv.org/abs/2511.16126)

### 2.9 Comparing Quality

There is no universal bitrate at which a neural codec becomes transparent, and perceptual transparency is not losslessness. Compare sample rate, channels, dataset, bitrate accounting, reconstruction protocol, and latency together.

- **MUSHRA / listening tests:** useful for intermediate-quality audio; report subjects, anchors, confidence intervals, and test material.
- **ViSQOL:** an objective perceptual-quality estimate; specify speech/audio mode and version. It does not replace listening tests. [Implementation](https://github.com/google/visqol)
- **PESQ and STOI:** designed for speech quality/intelligibility; do not treat them as universal music-fidelity measures.
- **Spectral errors:** diagnose reconstruction changes but do not fully capture perceived artifacts or stereo image quality.

Use music held out from training, such as appropriately separated MUSDB18-HQ evaluation tracks. AudioSet, speech datasets, and MusicCaps have different purposes; the presence of captions does not make a collection a standard codec benchmark.

---

## 3. Diffusion-Based Audio Models

### 3.1 DDPM and Score-Based Models

For a DDPM with `0 < beta_t < 1`, let `alpha_t = 1 - beta_t` and `alpha_bar_t = product_{s=1..t}(alpha_s)`:

```text
q(x_t | x_(t-1)) = Normal(sqrt(alpha_t)*x_(t-1), beta_t*I)
x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon
p_theta(x_(t-1) | x_t) = Normal(mu_theta(x_t,t), sigma_t^2*I)
```

The common network predicts `epsilon_theta`; the reverse mean `mu_theta` is computed from this prediction and the schedule. They are not the same output. Predicting `x_0` or velocity is another parameterization, with corresponding loss weighting. [Ho et al., DDPM](https://arxiv.org/abs/2006.11239)

For scalar, state-independent diffusion `g(t)`, the score-SDE formulation is:

```text
forward:      dx = f(x,t) dt + g(t) dw
reverse:      dx = [f(x,t) - g(t)^2 * grad_x log p_t(x)] dt + g(t) dw_bar
probability flow ODE:
              dx = [f(x,t) - 0.5*g(t)^2 * grad_x log p_t(x)] dt
```

The reverse SDE is integrated from large to small `t` (`dt < 0`). The ODE has the same one-time marginal distributions when the exact score is used. Learned scores and numerical solvers introduce approximation error; likelihood evaluation additionally requires divergence integration. VE, VP, and sub-VP are different noise processes within this framework. [Song et al., ICLR 2021](https://arxiv.org/abs/2011.13456)

### 3.2 Audio Model Families

| Model | Representation and conditioning | Source |
|---|---|---|
| AudioLDM (Liu et al., 2023) | Diffusion in a mel-spectrogram VAE latent space; CLAP audio embeddings during diffusion training and text embeddings at inference; vocoder reconstructs waveform | [Paper](https://arxiv.org/abs/2301.12503) |
| AudioLDM 2 (Liu et al., 2023/2024) | AudioMAE-derived **Language of Audio (LOA)**; a language model predicts the intermediate representation to condition latent diffusion | [Paper](https://arxiv.org/abs/2308.05734) |
| Original Stable Audio (Evans et al., 2024) | Waveform-autoencoder latents, a convolutional diffusion architecture, CLAP text features, timing conditioning | [Fast Timing-Conditioned Latent Audio Diffusion](https://arxiv.org/abs/2402.04825) |
| Stable Audio Open (Evans et al., 2024) | Waveform VAE, T5 text conditioning, diffusion transformer; 44.1 kHz stereo up to 47 seconds | [Paper](https://arxiv.org/abs/2407.14358) |
| DiffWave (Kong et al., ICLR 2021) | Waveform diffusion, including spectrogram-conditioned synthesis | [Paper](https://arxiv.org/abs/2009.09761) |

Stable Audio versions have different architectures and text encoders; CLAP and T5 should not be merged into one undocumented configuration. Open weights also have their own model license, separate from code licensing.

### 3.3 Classifier-Free Guidance (CFG)

Training drops the condition on a subset of examples. Using the common **scale `s`** convention:

```text
epsilon_guided = epsilon_uncond + s * (epsilon_cond - epsilon_uncond)
```

- `s=0`: unconditional prediction.
- `s=1`: ordinary conditional prediction.
- `s>1`: extrapolation toward the condition; may improve adherence while reducing diversity or introducing artifacts.

The original paper also writes `(1+w)*epsilon_cond - w*epsilon_uncond`; the mapping is **`s = 1 + w`**. In that convention `w=0` is ordinary conditional prediction, and `w=1` already adds guidance. Choose the scale for the specific checkpoint and sampler. [Ho & Salimans](https://arxiv.org/abs/2207.12598)

### 3.4 Sampling and Trade-offs

Spectrogram diffusion needs a waveform reconstruction stage; waveform diffusion models the sampled signal directly; latent waveform diffusion reduces sequence length through an autoencoder. The bottleneck trades representational capacity against compute.

DDIM can be deterministic when its stochasticity parameter is zero. DPM-Solver-type methods can reduce the number of evaluations. Consistency models can be trained or distilled for few-step generation. Speed depends on sequence length, model size, sampler, and hardware; diffusion is not intrinsically slower than autoregressive generation. Long-duration musical structure, precise event control, and consistent stereo imaging remain evaluation targets.

---

## 4. Real-Time Audio Processing

### 4.1 Deadlines and Causality

For a block of `B` samples at rate `f_s`, the processing deadline is `B/f_s` seconds. Average real-time factor below one is necessary but insufficient: deadline outliers can cause audible underruns. Live monitoring often needs total latency of only a few milliseconds; acceptable values depend on instrument, signal path, and performer.

The audio callback should avoid blocking I/O, locks with unbounded waits, allocation, and other operations with unpredictable duration. Preallocate buffers, prepare models outside the callback, and measure worst-case behavior on the target machine.

A causal system uses present and past input. A streamable system may instead use finite lookahead and delay its output. Streaming, strict causality, and low latency are separate properties.

### 4.2 Causal Convolutions

A stride-one causal convolution can be written:

```text
y[t] = sum_(k=0..K-1) w[k] * x[t - d*k]
receptive_field = 1 + sum_l (K_l - 1)*d_l
```

The receptive-field formula assumes a stack of stride-one layers. With kernel size `K` and dilations `1,2,...,2^(L-1)`, it becomes `1 + (K-1)*(2^L-1)`. Strided stacks need stride products in the calculation.

Past context is a cache requirement, not an automatic `(K-1)`-sample delay. A causal FIR can compute `y[t]` as soon as `x[t]` arrives. Lookahead, framing, resampling, and scheduling determine algorithmic latency. Linear-phase filter group delay is another concept and should not be inferred from causality alone.

### 4.3 Codec Streaming

EnCodec's 24 kHz model is causal, but the public whole-file interface is not by itself a stateful streaming implementation. A deployment must preserve convolution and recurrent state and match offline padding/boundary behavior.

A 320-sample hop at 24 kHz means **13.33 ms per latent frame**, or **75 frames/s**. It is neither a measured end-to-end latency nor the whole receptive field. Measure codec buffering, resampling, inference, packetization, and device latency separately. [EnCodec implementation](https://github.com/facebookresearch/encodec)

### 4.4 Non-Causal Streaming

Caillon & Esling's [Streamable Neural Audio Synthesis With Non-Causal Convolutions](https://arxiv.org/abs/2204.07064) (DAFx 2022) converts non-causal convolutional models after training using cached operations and inserted delays that preserve graph alignment, including parallel paths. This is more specific than generic overlap-add processing. Any needed future context becomes latency; 20 ms at 48 kHz corresponds to 960 samples.

### 4.5 Inference Frameworks

| Tool | Scope and deployment considerations |
|---|---|
| [RTNeural](https://github.com/jatinchowdhury18/RTNeural) | C++ inference for supported layer types; documented workflow exports TensorFlow/PyTorch weights to JSON, prepares the model, then runs its forward method. It is not a general ONNX importer. |
| [anira](https://arxiv.org/abs/2506.12665) | Ackva & Schulz; IS2 2024 paper, 2025 arXiv version. Decouples inference from the audio callback through a static thread pool and manages latency; paper evaluates ONNX Runtime, LibTorch, TensorFlow Lite. |
| General tensor/ONNX runtimes | Support many models, but do not automatically provide bounded callback execution time. Warmup, allocation behavior, scheduling, and thread count matter. |
| [Faust](https://faust.grame.fr/) | DSP language/toolchain; neural inference integration depends on the chosen backend or external component. |

### 4.6 Latency Budget and Verification

```text
round_trip = ADC + input_buffers + algorithmic_delay
             + processing/scheduling + output_buffers + DAC
```

Avoid double-counting computation already hidden within buffering. Measure the actual input-to-output path using an impulse or loopback, and report sample rate, block size, hardware, backend, and load. Check stream/offline agreement, state reset, startup/tail handling, and behavior when a computation misses its deadline. A fast offline benchmark does not establish safe live performance.

---

## 5. Audio Tokenization & Discrete Representations

### 5.1 VQ and RVQ

For codebook `C={e_1,...,e_K}`:

```text
index(z) = argmin_k ||z - e_k||^2
VQ(z) = e_index(z)
```

The index is discrete; the quantized vector is its codebook embedding. Straight-through estimation supplies a surrogate gradient to the encoder. Codebooks can use a dedicated gradient loss or EMA updates of assignment counts and sums. A naive average of per-batch means is not generally the same EMA algorithm. [VQ-VAE](https://arxiv.org/abs/1711.00937)

RVQ refines the residual:

```text
r_0 = z
q_i = VQ_i(r_(i-1))
r_i = r_(i-1) - q_i
z_hat = sum_i q_i
```

Truncating a model trained for variable bitrate reduces payload. More levels generally improve representation capacity, but perceived quality need not rise monotonically for every signal. Earlier levels are not guaranteed to correspond to explicit notes or phonemes. Low-dimensional normalized lookup (DAC), stale-code replacement (EnCodec), initialization, and usage regularizers address underused codebooks in different ways.

### 5.2 Lookup-Free and Grouped Quantization

Binary LFQ maps each of `d` components to a sign, giving up to `2^d` discrete combinations. Avoiding a learned codebook does not guarantee uniform token usage; entropy objectives and training design still matter. LFQ is a separate method from WavTokenizer's learned VQ. [LFQ reference, *Language Model Beats Diffusion — Tokenizer is Key to Visual Generation*](https://arxiv.org/abs/2310.05737)

Grouped RVQ splits channels into groups and applies residual quantization within each group, as in HiFi-Codec. The number of groups and levels determines the total number of code streams; gains must be checked at matched quality and bitrate.

### 5.3 Semantic and Acoustic Tokens

AudioLM/MusicLM use hierarchies of tokens from self-supervised audio features and neural codecs. Coarse semantic representations help model longer-range content; acoustic codes support waveform detail. This is a learned division of labor, not a clean symbolic decomposition into “notes” versus “timbre.” Semantic features may retain acoustics, and codec codes can contain semantics. [AudioLM](https://arxiv.org/abs/2209.03143), [MusicLM](https://arxiv.org/abs/2301.11325)

### 5.4 Token Rates and MusicGen's Delay Pattern

| Configuration | Frame rate | Code streams | Total indices/s |
|---|---|---|---|
| EnCodec 24 kHz, 6 kbps | 75 Hz | 8 | 600 |
| EnCodec 24 kHz, 24 kbps | 75 Hz | 32 | 2,400 |
| DAC 44.1 kHz, nine codebooks | ~86.13 Hz | 9 | ~775.20 |
| MusicGen's 32 kHz codec | 50 Hz | 4 | 200 |
| WavTokenizer | 40 or 75 Hz | 1 | 40 or 75 |

MusicGen uses a 32 kHz EnCodec configuration with four codebooks and a 50 Hz frame rate. It delays codebook streams and predicts several indices per transformer step using separate output heads. The resulting generation requires roughly **50 autoregressive steps per second**, plus boundary overhead; it is not a flattened 200-step sequence. These rates refer to the original mono configuration. [Copet et al., MusicGen](https://arxiv.org/abs/2306.05284)

### 5.5 Practical Evaluation

Evaluate reconstruction, language-model predictability, downstream semantics, domain coverage, and stereo behavior separately. Fewer tokens can reduce modeling cost while increasing decoder burden. Codec reconstruction fidelity is an important bottleneck, but it is not a strict perceptual upper bound on every generated sample: generated latents can differ from encoded test latents. No single codec metric establishes the musical usefulness of the complete system.

---

## Appendix: Tools and Libraries

| Tool | Purpose |
|---|---|
| [librosa](https://librosa.org/) | Audio analysis and feature extraction |
| [nnAudio](https://github.com/KinWaiCheuk/nnAudio) | Tensor-based audio transforms |
| [AudioCraft](https://github.com/facebookresearch/audiocraft) | MusicGen and related audio models |
| [DAC](https://github.com/descriptinc/descript-audio-codec) | Codec training and inference |
| [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) | Stable Audio model tooling |
| [RTNeural](https://github.com/jatinchowdhury18/RTNeural) | C++ neural inference for audio |
| [anira](https://github.com/anira-project/anira) | Inference scheduling for audio applications |
| [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) | Single-codebook audio tokenizer |
| [FunCodec](https://github.com/modelscope/FunCodec) | Neural speech codec toolkit |

For datasets, MIR metrics, and model evaluation context, see [Music Understanding / MIR](music-understanding-mir.md).
