# Singing Voice Synthesis (SVS): Technical Research Notes

State of the field in singing voice synthesis as of 2025--2026. Covers acoustic modeling, singing-specific representations, controllable singing generation, singing voice conversion, datasets, and evaluation.

---

## 1. What Is SVS? SVS 是什么

Singing voice synthesis (歌声合成) generates singing audio from symbolic input (lyrics + musical score) or text. It sits at the intersection of TTS, music generation, and audio synthesis.

**Why SVS is distinct from TTS**:

| Dimension | TTS | SVS |
|-----------|-----|-----|
| Pitch range | Narrow (speaking pitch) | Wide (musical pitch, 2+ octaves) |
| Duration | Prosodic, language-dependent | Musical (note durations, tied to score) |
| Vibrato | Minimal | Essential artistic expression |
| Phonation | Continuous speech | Sustained notes, vocal runs, melisma |
| F0 control | Coarse | Fine-grained (semitones + cents) |
| Style | Conversational | Artistic, genre-dependent |

**Why SVS is distinct from music generation**:

Music generation produces full arrangements. SVS focuses on the *singing voice* as a distinct timbral and expressive entity, with lyrics as a primary conditioning signal.

---

## 2. Pipeline Architecture 管线架构

A typical SVS system:

```
Input: Lyrics + Musical Score (pitch, duration, phonemes)
    ↓
Frontend: Text/phoneme processing + score parsing
    ↓
Acoustic Model: Generates acoustic features (mel-spectrogram, F0, s/δ)
    ↓
Vocoder: Converts acoustic features to waveform
    ↓
Output: Singing audio
```

### 2.1 Frontend

- **Lyrics-to-phoneme conversion**: Grapheme-to-phoneme (G2P) for language-specific pronunciation.
- **Score parsing**: Extract note pitch, duration, and phoneme boundaries from musicXML, MIDI, or custom formats.
- **Prosody prediction**: Some systems predict singing-specific prosody (phrase boundaries, breath marks).

### 2.2 Acoustic Model

Generates time-aligned acoustic features:

| Feature | Description | Role |
|---------|-------------|------|
| **Mel-spectrogram** | Time-frequency representation | Primary output, feeds vocoder |
| **F0 (fundamental frequency)** | Pitch contour over time | Controls singing pitch |
| **s/δ** | Voiced/unvoiced flag + duration delta | Breathiness, phonation control |

### 2.3 Vocoder

Converts mel-spectrogram to waveform:

| Vocoder | Type | Notes |
|---------|------|-------|
| **HiFi-GAN** | GAN-based | Standard baseline, fast inference |
| **DiffWave** | Diffusion-based | Higher quality, slower |
| **Vocos** | Transformer-based | Recent, good quality-speed tradeoff |
| **SoundStream / EnCodec** | Neural codec | Used in end-to-end systems |

---

## 3. Key Models 关键模型

### 3.1 Classical Approaches (Pre-2020)

- **HMM-based SVS** (2000s–2010s): Hidden Markov Models trained on singing databases. Produced robotic but intelligible singing. Limited expressiveness.
- **Statistical parametric**: Used DNNs to predict mel-spectrogram + F0 from phoneme/note sequences. Improved naturalness but still "singing-speech" quality.

### 3.2 Neural Vocoder Era (2019--2022)

Introduction of HiFi-GAN and other high-quality neural vocoders enabled significantly better output quality from the same acoustic models.

### 3.3 Diffusion-Based SVS (2022--Present)

Diffusion models generate mel-spectrograms by iterative denoising:

| Model | Description |
|-------|-------------|
| **DiffSinger (2021, updated 2022)** | Diffusion-based mel-spectrogram generation. Landmark open-source SVS system. Supports multi-speaker, multi-language. Architecture: U-Net diffusion conditioned on phoneme + pitch + duration. |
| **Grad-TTS** | Score-based diffusion for TTS, adapted for singing |
| **SingSong (Google, 2023)** | Separates singing voice from music, then re-synthesizes. Diffusion-based approach. |

### 3.4 End-to-End and Large Model Approaches (2024--2026)

| Model | Description |
|-------|-------------|
| **ACE Singer (ACE Studio)** | Commercial SVS system with DAW integration, multi-language, multiple synthesis engines |
| **OpenDiffSinger** | Open-source DiffSinger fork with community extensions (multi-speaker, multi-language, fast inference) |
| **Music-SOM (2024)** | Singing-oriented model with style control |
| **Singing voice in foundation models** | Some music foundation models (ACE-Step, MusicGen extended) include singing voice generation capability |
| **RDCM-based models (2025)** | Recurrent Diffusion Composition Models for long-form singing with structural coherence |

**对 AI 的意义**：SVS is converging with general audio generation. MusicGen, AudioLDM 2, and other text-to-music models can produce singing-like audio, but lack the fine-grained control (phoneme-level timing, exact pitch) that dedicated SVS systems provide. The frontier is *controllable* singing within general generation frameworks.

---

## 4. Singing-Specific Representations 歌声专用表示

### 4.1 Score Representations

| Format | Description | AI relevance |
|--------|-------------|-------------|
| **MusicXML** | Standard sheet music format | Pitch + duration + lyrics + dynamics |
| **MIDI** | Note events + timing | Pitch and timing, but no lyrics/phonemes |
| **SVS-specific formats** | Custom (e.g., DS format in DiffSinger) | Phoneme-level alignment + pitch + duration |

### 4.2 Pitch Representation

| Representation | Description | Use |
|----------------|-------------|-----|
| **MIDI note number** | Discrete semitone | Coarse conditioning |
| **Continuous F0** | Fundamental frequency in Hz | Fine pitch control |
| **F0 in cents** | Semitones + cents deviation | Microtonal expressiveness |
| **Pitch contour** | Normalized pitch trajectory | Pitch shape modeling |

**Vibrato modeling**: Vibrato is typically 5--7 Hz with depth of 20--50 cents. Most SVS systems model vibrato implicitly (learned from data) rather than explicitly.

---

## 5. Controllable Singing Generation 可控歌声生成

### 5.1 Controllable Dimensions

| Dimension | Control method | Notes |
|-----------|---------------|-------|
| **Pitch** | F0 conditioning, pitch shift | Most fundamental control |
| **Timbre** | Speaker embedding, reference audio | Identity of singer |
| **Style** | Genre/emotion tags, style embedding | Pop vs. opera vs. folk |
| **Expressiveness** | Vibrato depth, breathiness, dynamics | Artistic control |
| **Language** | Phoneme set, G2P model | Multi-lingual support |
| **Duration** | Note duration, tempo | Tempo control |
| **Vocal technique** | Belting, falsetto, vocal fry | Advanced control |

### 5.2 Controllable Generation Methods

| Approach | Method | Trade-off |
|----------|--------|-----------|
| **Score conditioning** | Pitch + phoneme + duration as input | Precise but needs accurate score |
| **Reference audio** | Encode reference singer's timbre + style | Natural timbre but less controllable |
| **Text + pitch** | Text lyrics + pitch contour | Intuitive but coarse |
| **Disentangled representations** | Separate pitch/timbre/style in latent space | Most flexible, hardest to train |

---

## 6. Singing Voice Conversion 歌声转换

Singing voice conversion (SVC) changes the singer's identity while preserving pitch and lyrics:

| Method | Description | Limitation |
|--------|-------------|-----------|
| **Acoustic feature conversion** | Convert mel-spectrogram/F0 with singer-specific models | Requires parallel data |
| **Diffusion-based conversion** | Diffusion model conditioned on target speaker | Better quality, needs speaker embeddings |
| **Content-style disentanglement** | Separate content (pitch/lyrics) from style (timbre) in latent space | Most flexible, challenging to train |
| **So-VITS-SVC** | Open-source VITS-based SVC, widely used | Popular but quality varies by speaker |

**Key challenge**: Preserving pitch accuracy and phoneme timing during voice conversion. Many SVC systems introduce pitch drift or timing artifacts.

---

## 7. Datasets 数据集

### 7.1 Public SVS Datasets

| Dataset | Language | Speakers | Content | Notes |
|---------|----------|----------|---------|-------|
| **Opencpop** | Chinese | 1 | ~100 songs, pop | Popular for Chinese SVS research |
| **M4Singer** | Chinese | 8 | 92 songs, pop/folk | Multi-speaker Chinese |
| **NUS48E** | English | 4 | 48 song excerpts | Multi-style |
| **JSUT Song** | Japanese | 1 | 100 songs | Japanese SVS benchmark |
| **Kiritan** | Japanese | 5 | 163 songs | Multi-speaker Japanese |
| **VCTK** | English | 110 | Speech (not singing) | Often used for pre-training |
| **MUSAN** | — | — | Noise/music augmentation | Data augmentation |

### 7.2 Data Challenges

- **Alignment**: Phoneme-to-audio alignment for singing is harder than speech (sustained notes, vibrato, melisma).
- **Multi-lingual scarcity**: High-quality singing data exists primarily for Chinese, Japanese, and English. Low-resource languages are underserved.
- **Copyright**: Singing datasets derived from commercial recordings have licensing restrictions.

---

## 8. Evaluation 评测

### 8.1 Objective Metrics

| Metric | Description |
|--------|-------------|
| **F0 RMSE** | Root mean square error of pitch contour (cents) |
| **F0 correlation** | Pearson correlation between predicted and ground-truth F0 |
| **Mel-spectrogram L1/L2** | Spectrogram reconstruction quality |
| **MCD** | Mel-cepstral distortion — voice quality similarity |
| **Phoneme accuracy** | Intelligibility of lyrics |
| **Speaker similarity** | Cosine similarity of speaker embeddings (for timbre) |

### 8.2 Subjective Metrics

| Metric | Description |
|--------|-------------|
| **MOS** | Overall quality rating |
| **PEMO** | Perceptual evaluation (same as audio quality) |
| **Pitch accuracy** | Human-rated pitch correctness |
| **Naturalness** | How natural the singing sounds |
| **Speaker similarity** | How similar to target singer |

### 8.3 Benchmark Challenges

| Challenge | Description |
|-----------|-------------|
| **SVS Challenge (ISCS 2022/2023)** | Annual challenge with multiple tracks (singing voice synthesis, conversion) |
| **Vocals-based SVS** | Using source-separated vocals as training data |

---

## 9. Open Problems 开放问题

### 9.1 Technical Challenges

- **Long-form coherence**: Maintaining consistent timbre and pitch accuracy over full songs (3--5 minutes). Most systems evaluate on 10--30s clips.
- **Expressive control**: Fine-grained control over vibrato, breathiness, dynamics, and vocal technique remains limited.
- **Zero-shot singing**: Generating singing in a new voice without any training data from that voice.
- **Multi-track singing**: Generating backing instruments + singing together with proper mixing.
- **Real-time SVS**: Low-latency singing synthesis for interactive applications (karaoke, games).

### 9.2 Data and Cultural Challenges

- **Low-resource languages**: Most SVS research focuses on Chinese, Japanese, and English. Languages with limited singing data are underserved.
- **Singing style diversity**: Opera, folk, pop, rap-singing, and other styles require different modeling approaches.
- **Phoneme alignment**: Accurate phoneme-to-audio alignment for singing is fundamentally harder than speech due to sustained notes and melisma.

### 9.3 Evaluation Challenges

- **No standard benchmark**: Unlike MIR (GTZAN, MAESTRO), there is no universally accepted SVS benchmark dataset and evaluation protocol.
- **Subjective preference dominates**: Objective metrics (F0 RMSE, MCD) correlate poorly with human preference for singing quality.
- **Missing dimensions**: Current evaluation rarely measures musicality, stylistic authenticity, or emotional impact.

---

## 10. SVS and the Broader Music AI Ecosystem SVS 与更广泛的 AI 音乐生态

SVS does not exist in isolation:

| Related field | Connection |
|---------------|-----------|
| **TTS** | Shared pipeline (frontend → acoustic model → vocoder). SVS can benefit from TTS advances in expressiveness. |
| **Music generation** | SVS is the "vocal layer" in full song generation (Suno, Udio). |
| **Source separation** | Singing voice extraction (vocals isolation) provides training data for SVS. |
| **Audio codecs** | Neural codecs (EnCodec, DAC) are enabling end-to-end singing generation in token space. |
| **Singing voice conversion** | SVC enables voice changing for existing recordings — related to voice conversion in speech. |

**Frontier**: The boundary between SVS and text-to-music is blurring. Models like MusicGen can sing (sort of), and dedicated SVS models are incorporating more musical context. The future likely involves unified models that can generate *any* musical audio including singing, with fine-grained control derived from specialized representations.

---

## 11. Key References

### Foundational

| Paper | Year | Contribution |
|-------|------|-------------|
| DiffSinger | 2021 | Landmark diffusion-based SVS, open-source |
| OpenDiffSinger | 2022--25 | Community fork with extensions |
| So-VITS-SVC | 2022 | Popular VITS-based voice conversion |
| SingSong (Google) | 2023 | Diffusion-based singing extraction + synthesis |

### Recent

| Paper | Year | Contribution |
|-------|------|-------------|
| ACE Singer (ACE Studio) | 2024--25 | Commercial multi-language SVS with DAW integration |
| Music-SOM | 2024 | Style-controllable singing generation |
| RDCM | 2025 | Recurrent diffusion for long-form singing |
| DiffSinger Acceleration | 2025 | Faster inference via distillation |
| SVS Challenge results | 2022--25 | Annual community benchmarks |

---

> This document covers singing voice synthesis. For related topics, see [music-generation.md](music-generation.md) (general generation), [music-understanding-mir.md](music-understanding-mir.md) (source separation for vocals), [audio-engineering.md](audio-engineering.md) (vocoders, acoustic features), and [music-evaluation.md](music-evaluation.md) (evaluation methodology).
