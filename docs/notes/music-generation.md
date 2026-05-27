# Music Generation: Technical Research Notes

Comprehensive technical survey of music generation as of 2025--2026. Covers symbolic generation, audio-level generation, singing voice synthesis, controllable generation, and evaluation.

---

## 1. Symbolic Music Generation

### 1.1 Representation Formats

Symbolic music operates on discrete representations (notes, timing, dynamics) rather than raw audio. The choice of representation fundamentally shapes what a model can learn.

#### MIDI (Musical Instrument Digital Interface)
- Event-based protocol: `Note On`, `Note Off`, `Velocity`, `Time Shift`, `Program Change`
- Polyphony handled via overlapping note-on/note-off events with delta-time offsets
- Standard "MIDI-Like" tokenization (Oore et al., 2018): sequential stream of events ordered by onset time
- Limitation: raw MIDI events lack explicit metric structure (bar lines, beat positions); timing is purely relative

#### Piano Roll
- 2D matrix representation: pitch (rows) x time frames (columns), values indicate velocity/activation
- Directly compatible with CNN architectures (treat as image)
- Loses explicit note-level event structure; boundary detection is implicit
- Resolution trade-off: fine temporal resolution = very large matrices

#### ABC Notation
- Text-based music notation common in folk/traditional music
- Compact, human-readable, naturally suited for language model tokenization
- Limited to monophonic or simple polyphonic music

#### Token-Based Representations (current standard for Transformers)

| Format | Key Idea | Reference |
|--------|----------|-----------|
| **MIDI-Like** | Raw MIDI events as tokens (Note On/Off, Time Shift) | Oore et al., 2018 |
| **REMI** | Revamped MIDI events with explicit Bar, Position, Tempo, Chord tokens | Huang & Yang, 2020 (Pop Music Transformer) |
| **Compound Word (CP)** | Groups multiple attributes per timestep into a single compound token | Hsiao et al., 2021 |
| **REMI+ / REMI-z** | Multi-track extensions of REMI with track-aware tokenization | NeurIPS 2025 |
| **Pianoroll-Event** | Hybrid spatial + sequential encoding bridging piano roll and token views | arXiv 2601.19951 (2025) |
| **MIDI-Token** | Various tokenization strategies compared in MIDITok package | Natole et al., ISMIR 2021 |

REMI (Huang & Yang, 2020) is the most influential: it adds explicit `Bar`, `Position`, `Tempo`, and `Chord` tokens so the model can learn metric structure directly, rather than inferring it from delta-times. REMI-z (2025) extends this to multi-track music with per-track token streams.

### 1.2 Key Models

#### Music Transformer (Huang, Vaswani et al., 2018/2019)
- **Paper**: "Music Transformer: Generating Music with Long-Term Structure" (arXiv:1809.04281, ICLR 2019)
- **Authors**: Cheng-Zhi Anna Huang, Ashish Vaswani, et al. (Google Magenta)
- **Architecture**: Decoder-only Transformer with modified **relative attention** mechanism
- **Key contribution**: Relative position representations in self-attention allow the model to learn relative intervals and recurrent patterns in music, rather than absolute positions. This is critical because music is inherently relational (intervals, chord progressions, rhythmic cycles)
- **Representation**: MIDI-Like tokenization (performance-level piano performances from the Maestro dataset)
- **Results**: Generates coherent minute-long piano compositions with long-term structural repetition (A-B-A form, recurring themes)
- **Limitation**: Single instrument (piano only); MIDI-Like tokenization lacks explicit metric structure

#### Pop Music Transformer (Huang & Yang, 2020)
- **Architecture**: Transformer with REMI tokenization
- **Key contribution**: Introduced REMI representation with explicit Bar/Position tokens
- **Training data**: Pop piano arrangements
- **Improvement over Music Transformer**: Better rhythmic structure and beat-level modeling due to explicit metric tokens

#### MuPT (2024--2025)
- Pre-trained Transformer for symbolic music generation
- Aims to improve structural consistency over longer compositions
- Focus on pre-training scale for symbolic music

#### Transformer-GAN for Symbolic Music (AAAI)
- Combines autoregressive Transformer generator with GAN discriminator
- Produces minute-long compositions with adversarial training for quality control

### 1.3 Autoregressive vs. Diffusion for Symbolic Music

**Autoregressive (AR) approaches** (Music Transformer, REMI-based models):
- Model music as a sequence of tokens, predicting next token given previous ones
- Strengths: Natural fit for sequential music; handles variable-length outputs; strong local coherence
- Weaknesses: Error accumulation over long sequences; slow sequential generation; may struggle with global structure

**Diffusion approaches** for symbolic music:
- Apply diffusion (progressive denoising) to piano roll or latent representations
- Strengths: Parallel generation; can model global structure directly; less error accumulation
- Weaknesses: Fixed output length; less natural fit for event-based music; may miss fine-grained sequential dependencies

**Hybrid approaches** are emerging: combining AR for sequential fine structure with diffusion for global planning.

A 2025 study (arXiv:2506.08570) provides the first controlled comparison of AR vs. flow-matching for text-to-music generation, finding that diffusion/flow-matching provides superior diversity and controllability, while AR tends to produce more stable training and coherent local structure.

### 1.4 Key Datasets

| Dataset | Content | Size |
|---------|---------|------|
| **MAESTRO** | Aligned MIDI-audio piano performances | ~200 hours |
| **Lakh MIDI** | Multi-instrument MIDI files | 180K files |
| **POP909** | Pop piano arrangements with melody/accompaniment | 909 songs |
| **AD Pianoforte** | Classical piano performances | ~100 hours |
| **MetaMIDI** | Large-scale MIDI collection | 436K files |

### 1.5 Open Problems
- **Long-form structure**: Maintaining global form (verse-chorus-bridge), thematic development, and harmonic direction over multi-minute compositions
- **Multi-track generation**: Coordinated generation across multiple instruments with appropriate voice leading, rhythm section coordination, and arrangement
- **Controllability**: Precise control over harmony, form, style, and dynamics beyond text conditioning
- **Representation gap**: No single representation captures all musical dimensions (pitch, rhythm, dynamics, timbre, form) equally well

---

## 2. Audio-Level Music Generation

### 2.1 Neural Audio Codecs (Foundation for Audio Generation)

Audio-level models require compressing high-dimensional waveforms (44,100 samples/sec) into tractable representations.

#### EnCodec (Defossez et al., Meta, 2022)
- **Paper**: "High Fidelity Neural Audio Compression" (arXiv:2210.13438)
- **Architecture**: Convolutional encoder-decoder with **Residual Vector Quantization (RVQ)** bottleneck
  - Encoder: strided convolutional network compresses audio to latent frames
  - RVQ: Multiple hierarchical codebooks quantize the continuous latent into discrete tokens. Residual quantization: first codebook captures coarse structure, subsequent codebooks capture progressively finer details
  - Decoder: Transposed convolutions reconstruct waveform from quantized latents
  - Discriminator: Multi-scale STFT + multi-period discriminators for adversarial training
- **Bitrates**: 1.5 to 24 kbps (24 kHz model); 3 to 24 kbps (48 kHz stereo model)
- **Frame rate**: ~50 Hz (75 Hz for 48 kHz model) across 4--8 codebook levels
- **Training**: Reconstruction loss + adversarial loss + quantization commitment loss
- **Role**: Serves as the audio tokenizer for MusicGen and many other models

#### SoundStream (Zeghidour et al., Google, 2021)
- Predecessor to EnCodec; introduced the RVQ approach for neural audio codecs
- Used in AudioLM and MusicLM pipelines
- 24 kHz, 3 kbps streaming audio codec

#### DAC (Descript Audio Codec, Kumar et al., 2023)
- Improved neural codec with snake activation functions and larger codebooks
- 44.1 kHz, used in some Stable Audio variants

### 2.2 Jukebox (OpenAI, 2020)

- **Paper**: "Jukebox: A Generative Model for Music" (Dhariwal, Jun, Payne, Kim, Radford, Sutskever; arXiv:2005.00341)
- **Architecture**: Hierarchical VQ-VAE with autoregressive Transformers
  - **3-level hierarchical VQ-VAE**: Compresses raw audio at three temporal resolutions
    - Top level: ~8 Hz -- captures long-range musical structure (melody, harmony, form)
    - Middle level: ~34 Hz -- captures timbre, vocal characteristics
    - Bottom level: ~65 Hz or raw -- fine-grained audio detail
  - Each level has its own codebook of discrete embeddings
  - **Autoregressive Transformers** (decoder-only, similar to GPT-2/GPT-3) model token distributions at each level
  - **Upsampling**: Top-level tokens are conditionally upsampled to middle, then bottom level
  - **Lyrics conditioning**: Lyrics aligned to audio via forced alignment, providing conditioning for the top-level Transformer
  - **Artist/genre metadata**: Additional conditioning via artist and genre embeddings
- **Training data**: 1.2 million songs (600K English) with lyrics and metadata, sourced from the web
- **Output**: 44.1 kHz stereo audio, up to several minutes
- **Quality**: Recognizable singing and genre-appropriate instrumentation, but notable artifacts and below-human audio quality
- **Limitations**: Extremely slow generation (hours for a single song); quality gap vs. human-composed music; hierarchical upsampling can introduce inconsistency between levels
- **Significance**: Demonstrated that scaling VQ-VAE + autoregressive models to raw audio was feasible, establishing the paradigm for subsequent work

### 2.3 MusicLM (Google Brain, 2023)

- **Paper**: "MusicLM: Generating Music From Text" (Agostinelli, Denk, Borsos, et al.; arXiv:2301.11325)
- **Architecture**: Hierarchical sequence-to-sequence with three stages
  - **Stage 1 -- MuLan text encoding**: Text description encoded via MuLan (contrastive audio-text embedding model, analogous to CLIP for audio). Maps text into a shared audio-text embedding space
  - **Stage 2 -- Semantic modeling**: Autoregressive Transformer generates **semantic tokens** (high-level musical structure) conditioned on MuLan text embedding. Uses the SoundStream codec's top-level codebook
  - **Stage 3 -- Acoustic modeling**: Autoregressive Transformer upsamples semantic tokens into fine-grained **acoustic tokens** (full audio detail) using SoundStream's full RVQ stack
  - Separation of semantic and acoustic modeling dramatically reduces effective sequence length at the top level
- **Key components**:
  - **MuLan**: Contrastive audio-text model aligning music and text embeddings
  - **SoundStream**: Neural audio codec providing discrete tokens via RVQ
  - **Transformers**: At both semantic and acoustic levels
- **Capabilities**: Text-to-music from rich descriptions; melody conditioning (humming a melody generates full arrangement); long-term structural coherence
- **Output**: 24 kHz, up to several minutes
- **Evaluation**: Introduced MusicCaps benchmark (5,521 10-second clips with human-written captions from AudioSet/YouTube). Evaluated with FAD, KL divergence, MuLan similarity, and human side-by-side comparisons
- **Not released publicly** due to copyright concerns with training data
- **Significance**: Established the hierarchical semantic-acoustic paradigm and the MusicCaps evaluation benchmark

### 2.4 MusicGen (Meta, 2023)

- **Paper**: "Simple and Controllable Music Generation" (Copet, Kreuk, Gat, Remez, Kant, Synnaeve, Adi, Defossez; arXiv:2306.05284)
- **Architecture**: **Single-stage autoregressive Transformer** (no hierarchical semantic/acoustic separation)
  - Operates on EnCodec discrete audio tokens (32 kHz, 4 codebooks at 50 Hz)
  - Single Transformer predicts all 4 codebook tokens per timestep using delayed pattern interleaving
  - Text conditioning via **T5 encoder** (frozen) and/or **chroma-based melody conditioning**
  - **Melody conditioning**: Extracts chroma features from an audio prompt and uses them as additional conditioning, enabling "generate music that follows this melody"
- **Key simplification**: Eliminates the hierarchical multi-stage approach of MusicLM/Jukebox by using a single Transformer with EnCodec's multi-codebook tokens
- **Training data**:
  - 20K hours of licensed music (Meta's internal dataset)
  - Shutterstock and Pond5 music libraries
  - MusicCaps for text-aligned evaluation
- **Model sizes**: 300M (small), 1.5B (medium), 3.3B (large)
- **Output**: 32 kHz, up to 30 seconds standard (extendable with sliding window)
- **Framework**: Released as part of Meta's **AudioCraft** library (open source)
- **Evaluation**: State-of-the-art FAD scores on MusicCaps benchmark at release; human evaluation showing preference over baselines
- **Controllability**: Text prompts + optional melody/audio conditioning
- **Companion model -- MAGNeT**: Non-autoregressive (masked) variant for faster parallel generation using the same EnCodec foundation

### 2.5 AudioLDM and AudioLDM 2 (2023--2024)

#### AudioLDM (Liu et al., 2023)
- **Paper**: "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models" (Haohe Liu et al.)
- **Architecture**: Latent Diffusion Model (LDM) adapted from Stable Diffusion to audio domain
  - **VAE encoder-decoder**: Compresses mel-spectrograms into lower-dimensional latent space (based on AudioMAE/HiFi-GAN components)
  - **U-Net diffusion backbone**: Operates in latent space, conditioned on text embeddings
  - **Text conditioning**: CLAP (Contrastive Language-Audio Pretraining) embeddings bridge text-audio modality gap
  - **Inference**: Diffusion denoising in latent space, then VAE decode to mel-spectrogram, then vocoder to waveform
- **Output**: Sound effects, ambient audio, short music clips (~10 seconds)
- **Advantage**: Computationally efficient (diffusion in compressed latent space vs. full spectrogram space); supports text-to-audio, audio-to-audio, and inpainting

#### AudioLDM 2 (Liu et al., 2023--2024)
- **Paper**: "AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining" (arXiv:2308.05734)
- **Key improvements over AudioLDM 1**:
  - **Unified framework**: Single architecture handles speech, music, and sound effects
  - **GPT-2 integration**: Joint finetuning of GPT-2 language model with latent diffusion, enhancing text understanding
  - **Self-supervised pretraining**: Improves quality and generalization
  - **Optimized architecture**: 16 kHz improved model with more training data
  - **Performance**: Matches state-of-the-art on text-to-audio, text-to-music, and text-to-speech benchmarks simultaneously
- **Three variants**: Specialized for audio, music, and speech respectively

### 2.6 Stable Audio (Stability AI, 2023--2024)

#### Stable Audio 1.0 (2023)
- **Architecture**: Latent Diffusion Model with three components
  1. **Autoencoder**: VAE-based encoder compresses 44.1 kHz stereo audio into compact latent representation
  2. **Text encoder**: Frozen text encoder (T5 or CLAP) for prompt conditioning
  3. **Diffusion model**: U-Net-based denoiser operating on latents with text cross-attention
- **Output**: Variable-length stereo audio up to ~47 seconds at 44.1 kHz
- **Training data**: Licensed music and audio

#### Stable Audio 2.0 (April 2024)
- **Architecture upgrade**: Replaces U-Net backbone with **Diffusion Transformer (DiT)**
  - **Highly compressed autoencoder**: Projects raw audio waveforms into continuous latent representation at a latent rate of **21.5 Hz** (significant compression)
  - **DiT backbone**: Transformer-based denoiser replaces U-Net, following the trend of DiT architectures (as in Stable Diffusion 3, Sora)
  - **T5 text encoder**: For text conditioning
  - **Latent diffusion**: Entire diffusion process occurs in the learned latent space
- **Output**: Full tracks up to **3 minutes** with coherent musical structure (intro-development-outro)
- **Significance**: First open-weight model to generate 3-minute structured music tracks
- **Released**: Open weights on Hugging Face (Stable Audio Open 1.0 variant)

#### Stable Audio Open (June 2024)
- Open-weights release (~486K royalty-free audio recordings + ~70K music tracks)
- Training data: Free Music Archive (FMA), Freesound, Creative Commons licensed audio
- Explicitly not trained on copyrighted music
- Supports fine-tuning on custom datasets
- Generates up to ~47 seconds at 44.1 kHz

#### Stable Audio 3 (2025, in development)
- **Semantic-acoustic autoencoder**: Novel autoencoder projecting audio into compact latent space with both semantic and acoustic information
- Diffusion Transformer operating on improved latent representations
- Conditioned on text and desired output characteristics

### 2.7 YuE (Multimodal Art Projection, 2025)

- **Paper**: "YuE: Scaling Open Foundation Models for Long-Form Music Generation" (Yuan et al.; arXiv:2503.08638, ICLR 2025)
- **Authors**: Ruibin Yuan, Hanfeng Lin, Haohe Liu, and ~50 collaborators from multiple institutions
- **Architecture**: Based on **LLaMA2** architecture, adapted for music
  - **Track-decoupled next-token prediction**: Separates vocal and accompaniment tracks during generation to overcome the challenge of dense mixture signals in raw audio tokens
  - **Structural progressive conditioning**: Enables long-context lyrical alignment by progressively conditioning on lyrics segments, maintaining coherence over minutes
  - **Multitask, multiphase pre-training recipe**: Multi-stage training for convergence and generalization across diverse musical tasks
  - **Redesigned in-context learning**: Enables versatile style transfer (e.g., converting Japanese city pop into English rap while preserving accompaniment) and bidirectional generation
- **Scale**: Trained on **trillions of tokens**
- **Output**: Up to **5 minutes** of music with vocals + accompaniment
- **Capabilities**:
  - Lyrics-to-song generation (primary task)
  - Style transfer across genres and languages
  - Bidirectional generation (forward and backward from a point)
  - Music understanding: Learned representations match or exceed state-of-the-art on MARBLE music understanding benchmark
- **Fine-tuning**: Supports additional controls and enhanced support for "tail" (underrepresented) languages
- **Hardware**: Can run on consumer GPUs (10 GB VRAM for ~1 minute of music)
- **Significance**: First open-source model to match or surpass proprietary systems (Suno, Udio) in musicality and vocal agility for full-song generation
- **Limitations**: Generation speed (real-time factor); occasional lyrical misalignment; quality varies by genre

### 2.8 Suno (Proprietary, 2023--2026)

- **Company**: Suno AI (Cambridge, MA)
- **Architecture**: Not publicly disclosed; believed to be a combination of diffusion and autoregressive models for lyrics, vocals, and accompaniment
- **Evolution**:
  - V1--V3 (2023--2024): Rapid iteration, improving from simple clips to structured songs
  - **V4** (late 2024): Major quality leap; professional-sounding output with coherent verse-chorus structure
  - **V4.5** (2025): Further quality improvements; considered best-in-class for complete songs with vocals
  - **V5** (anticipated 2025): In development; expected to bring further leaps in audio fidelity, vocal realism, and song structure
- **Capabilities**:
  - Text-to-song: Generate complete songs from text descriptions
  - Lyrics input: Custom or AI-generated lyrics
  - Genre/style control via prompting
  - Maximum track length: up to 4 minutes
  - Multi-genre support
- **Strengths**: Best overall quality for complete songs with vocals; beginner-friendly interface; strong genre adherence
- **Limitations**: Proprietary (no open weights); limited fine-grained control beyond prompting; occasional structural artifacts in longer generations

### 2.9 Udio (Proprietary, 2024--2026)

- **Company**: Udio (formerly Uncharted Labs)
- **Architecture**: Not publicly disclosed
- **Capabilities**:
  - Text-to-song generation
  - Maximum track length: up to 2 minutes
  - Known for more musical variety: tempo changes, syncopation, melodic variation
  - Output often passes as non-AI more readily than competitors
- **Evolution**: Pivoting toward being an "AI playground for superfans" rather than purely a generation tool
- **Strengths**: More musical surprises and variation; quality perceived as more "human-like"
- **Limitations**: Shorter max length than Suno; less accessible for beginners

### 2.10 Other Notable Models

#### TangoFlux (2024)
- 515M parameters; generates up to 30 seconds of 44.1 kHz audio
- Uses **flow matching** (rectified flow) instead of diffusion -- a promising alternative architecture
- Faster inference than comparable diffusion models

#### TVC-MusicGen (INTERSPEECH 2025)
- Time-Varying Structure Control for MusicGen
- Novel approach to background music generation with dynamic structural control over time
- Addresses the limitation of static conditioning in MusicGen

#### MusicLDM
- Adapts Stable Diffusion + AudioLDM for music generation
- Focus on reducing plagiarism risk in generated outputs

#### ACE-Step (ACE Studio + StepFun / 阶跃星辰, 2025)
- **Open-source** step-by-step diffusion model for music generation, co-led by ACE Studio and StepFun (阶跃星辰)
- Novel paradigm: decomposes generation into sequential stages (melody → harmony → rhythm → full arrangement) rather than single end-to-end diffusion pass
- Strong **Chinese-language (Mandarin) lyrics and vocal** support — a notable advantage over Western-developed models
- Supports text prompts, genre tags, and intermediate editing at each generation step
- Built on PyTorch with inference optimization for consumer hardware
- Represents a growing trend of Chinese tech companies and AI studios releasing open-source music generation models

#### MusicFlow (ICLR 2025)
- **Cascading flow matching** framework for music generation
- Multi-stage generative process where flow matching models are arranged in a cascade, progressively refining output
- Analogous to cascaded diffusion models but adapted for the flow matching paradigm
- Addresses the challenge of multi-track, multi-instrument, and temporal structure in music
- Flow matching offers simpler training and faster inference compared to traditional diffusion

#### SongCreator (2025)
- **Lyric-to-song generation** with in-context learning capabilities
- Dual-track architecture handling vocal (singing) and accompaniment tracks with musical coherence
- Supports controllable generation: style/genre, singer voice characteristics, accompaniment style, tempo and mood
- Conditioning on reference audio or style prompts for diverse and controllable song creation
- Addresses the challenging problem of aligning lyrics with melody and rhythm while producing natural vocals and accompaniment

#### MusicFX (Google, 2023--2025)
- Consumer-facing music generation tool powered by MusicLM technology
- Released as part of Google's AI Test Kitchen
- Generates short music clips from text prompts
- Applies **SynthID watermarking** to generated audio for identification
- Restricts generation of music mimicking specific artists (copyright safeguard)
- Updated periodically with quality and diversity improvements

#### SkyMusic (昆仑万维, 2024--2025)
- Chinese text-to-music generation platform from Kunlun Tech
- Supports multilingual output (Chinese, English, and others)
- Integrated with昆仑万维's broader AI ecosystem (Skywork models)
- Part of the emerging Chinese AI music generation landscape targeting domestic users

---

## 3. Singing Voice Synthesis (SVS)

### 3.1 Overview

SVS generates a singing voice from a musical score input (lyrics phoneme sequence + pitch/duration annotations). Unlike Text-to-Speech (TTS), SVS must handle sustained vowels, vibrato, pitch glides, breath control, and the expressive timing of singing.

### 3.2 Key Models

#### DiffSinger (Liu, Ren et al., 2021/2022)
- **Paper**: "DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism" (arXiv:2105.02446, AAAI 2022)
- **Authors**: Jinglin Liu, Zhi Ren, Yi Ren, Chen Zhang, Zhou Zhao (Zhejiang University)
- **Architecture**:
  - **Diffusion probabilistic model** as acoustic model: Parameterized Markov chain iteratively converts noise into **mel-spectrogram** conditioned on music score
  - **Shallow diffusion mechanism**: Key innovation. Instead of running the full diffusion chain from pure Gaussian noise to mel-spectrogram, DiffSinger starts the diffusion from an intermediate representation provided by a simple baseline model (e.g., Feed-Forward Transformer prediction). This:
    - Reduces the number of diffusion steps needed
    - Improves training stability
    - Results in higher-quality output than full-chain diffusion
  - **Input**: Music score (phonemes + F0 pitch contour + note durations)
  - **Output**: Mel-spectrogram, converted to waveform via neural vocoder (HiFi-GAN or similar)
- **Training**: Reconstruction loss on mel-spectrogram + diffusion denoising objective
- **Evaluation**: MOS scores competitive with or exceeding prior neural SVS systems
- **Significance**: Established diffusion as a powerful approach for SVS; the "shallow diffusion" trick became widely adopted

#### VISinger / VISinger 2 (Xia et al., NWPU, 2022--2023)

**VISinger** (ICASSP 2022):
- **Paper**: "VISinger: Variational Inference with Adversarial Learning for End-to-End Singing Voice Synthesis"
- **Authors**: Yiwei Xia et al. (Northwestern Polytechnical University)
- **Architecture**:
  - **Fully end-to-end**: Directly generates waveform from music score, eliminating the traditional multi-stage pipeline (separate duration model, acoustic model, vocoder)
  - **Variational Inference (VI)**: Models complex acoustic distributions needed for natural singing
  - **Adversarial training**: Discriminator ensures realistic audio output
  - **Advantage**: Fewer parameters than multi-stage systems; simpler training pipeline
- **Limitation**: Phase prediction issues caused glitches and jitter in voiced segments

**VISinger 2** (INTERSPEECH 2023):
- **Paper**: "VISinger 2: High-Fidelity End-to-End Singing Voice Synthesis"
- **Key improvement**: Integrates a **Digital Signal Processing (DSP) synthesizer** into the end-to-end framework
  - DSP synthesizer handles phase-related components, avoiding the problematic direct text-to-phase mapping that caused artifacts in VISinger 1
  - Results in higher-fidelity output at higher sampling rates
- **Performance**: Achieves better quality than two-stage models with fewer parameters

#### XiaoiceSing (Microsoft, 2021)
- **Paper**: "XiaoiceSing: A High-Quality and Integrated Singing Voice Synthesis System" (Microsoft Xiaoice team)
- **Architecture**: Sequence-to-sequence (seq2seq) model adapted from TTS
  - **Encoder**: Encodes phoneme/musical note sequences into hidden representations
  - **Attention mechanism**: Aligns encoder outputs with decoder frames
  - **Decoder**: Autoregressively generates acoustic features (mel-spectrograms)
  - **Variance adaptor**: Predicts duration, pitch (F0), and energy for expressive singing control
  - **Vocoder**: Neural vocoder (HiFi-GAN/WaveNet) converts mel-spectrograms to waveform
- **Singing-specific features**:
  - Explicit F0 prediction for precise pitch following
  - Duration control for correct musical timing
  - Breath and vibrato modeling for expressive singing
- **Training data**: Hours of high-quality singing recordings with precise phoneme-level pitch and duration annotations
- **Significance**: One of the first high-quality neural SVS systems, building on Microsoft's TTS expertise

#### ExpressiveSinger (2024)
- Cascade of diffusion models for multilingual and multi-style SVS
- Supports multiple singing styles and languages

#### RDSinger (2024)
- Reference-based diffusion network for high-fidelity SVS
- Uses reference audio conditioning to control timbre and style

#### DITSinger (2025)
- Investigates scaling effects on SVS quality
- Addresses unclear scaling laws and systematic methodology for SVS

#### OpenDiffSinger (Community, 2022--2025)
- Open-source community fork and extension of the DiffSinger framework
- Major ongoing contributions: multi-speaker/multi-language support expansions, improved phoneme dictionary and duration modeling
- New GUI tools for dataset preparation and training pipeline simplification
- Vocoder improvements integrating NSF (Neural Source Filter) and HiFi-GAN variants
- The OpenVPI ecosystem provides companion tools for SVS dataset creation, phoneme alignment, and training

#### ACE Studio / ACE Singer (2024--2025)
- Commercial SVS system by ACE Studio (Chinese AI music technology company)
- Generates realistic singing from score/lyrics input — true SVS, not voice conversion
- Offers multi-language support (Chinese, English, Japanese), expressive parameter control (breath, vibrato, dynamics), multiple voicebanks, and DAW integration via VST plugins
- Competes with Synthesizer V (Dreamtonics) and XiaoiceSing in the commercial SVS market

#### DiffSinger Acceleration (2025)
- Multiple ongoing efforts to accelerate DiffSinger inference, including consistency distillation, progressive distillation, and fewer-step ODE solvers
- Reduces diffusion sampling steps while maintaining quality, enabling near-real-time inference for interactive SVS applications
- No single canonical "Lite" paper; acceleration techniques are adapted from broader diffusion model acceleration literature

### 3.3 SVS Evaluation

- **MOS (Mean Opinion Score)**: Gold standard; human raters score naturalness on 1--5 scale
- **Pitch accuracy**: Measured as F0 RMSE between generated and target pitch contours
- **Duration accuracy**: Alignment of generated phoneme durations with musical score
- **Spectral quality**: Metrics like log-spectral distance between generated and reference spectrograms
- **Subjective listening tests**: Comparative preference tests between systems
- **Challenge**: No single automated metric captures the full perceptual quality of singing (intonation, expressiveness, timbre naturalness, breath control)

### 3.4 Open Problems
- **Expressive control**: Fine-grained control over vibrato, dynamics, phrasing, and emotional expression
- **Zero-shot speaker adaptation**: Generating singing in an arbitrary voice from a short reference clip
- **Multilingual support**: Most SVS systems are language-specific (Mandarin, English, Japanese)
- **Real-time SVS**: Low-latency generation for interactive applications
- **Singing technique modeling**: Belting, falsetto, growl, and other vocal techniques

---

## 4. Controllable Generation

### 4.1 Conditioning Modalities

Controllable music generation allows users to specify musical attributes beyond free-form text.

#### Text-Based Conditioning
- **Free-form text**: Natural language description of desired music (used by MusicGen, Stable Audio, Suno, Udio)
- **Text encoders**: T5 (MusicGen, Stable Audio), CLAP (AudioLDM), MuLan (MusicLM)
- **Limitation**: Text is inherently imprecise for musical attributes; "upbeat jazz" means different things to different models

#### Music-Specific Attribute Control
- **Mustango** (Melechovsky, Guo, Ghosal, Majumder, Herremans, Poria; NAACL 2024)
  - **Architecture**: Latent diffusion model with structured text conditioning
  - **Key innovation**: Parses music-specific attributes from text prompts -- genre, key, tempo, chords, instruments
  - Uses a language model to extract structured musical parameters from text, then conditions generation on these explicit attributes
  - Achieves state-of-the-art controllability for Western music
  - **Limitation**: Controllability limited to Western music theory concepts

#### Melody Conditioning
- MusicGen: Chroma-based melody extraction from audio prompt; generates music that follows the provided melody
- MusicLM: Humming-to-arrangement capability
- YuE: Style transfer preserving accompaniment while changing vocal style

#### Emotion Conditioning
- **LARA-Gen**: Enables continuous emotion control for music generation
- **EBS (Emotion-Based Sampling)**: Algorithm for controlling generation process with emotion labels (IEEE TMM)
- **Symbolic music with continuous-valued emotions**: Controlling texture and emotional arc in symbolic generation
- Approaches: Valence-arousal space mapping; emotion embedding vectors injected into conditioning

#### Instrument and Timbre Control
- Specifying particular instruments or timbral qualities via text or audio reference
- AudioLDM: Audio-to-audio style transfer
- MusicGen: Can be conditioned on audio prompts that establish timbral references

### 4.2 Fine-Grained Control Approaches

#### SegTune (2025)
- **Paper**: "SegTune: Structured and Fine-Grained Control for Song Generation" (arXiv:2510.18416)
- Enables segment-level control over different aspects of musical output
- Supports structured generation where different sections (intro, verse, chorus) can have different attributes
- Fine-grained conditioning for controllable long-form text-to-audio generation

#### TVC-MusicGen (INTERSPEECH 2025)
- Time-Varying Structure Control for MusicGen
- Dynamic structural control that changes over the course of generation
- Addresses the limitation that standard MusicGen applies uniform conditioning throughout

#### In-Context Learning (YuE)
- YuE redesigns in-context learning for music generation
- Enables style transfer, bidirectional generation, and few-shot adaptation
- Can convert between genres/languages while preserving structural elements

### 4.3 Taxonomy of Control Methods

| Method | Granularity | Example Systems |
|--------|-------------|-----------------|
| Free-form text | Coarse | MusicGen, Stable Audio, Suno |
| Structured text attributes | Medium | Mustango |
| Melody conditioning | Medium | MusicGen, MusicLM |
| Audio reference | Medium | AudioLDM, YuE |
| Emotion labels/continuous | Medium | LARA-Gen, EBS |
| Time-varying/segment | Fine | SegTune, TVC-MusicGen |
| Score-level | Fine | Symbolic models (REMI-based) |

### 4.4 Open Problems
- **Precise harmonic control**: Specifying exact chord progressions, modulations, and voice leading
- **Form-level control**: Controlling verse-chorus-bridge structure, song length, and transitions
- **Real-time interactive control**: Adjusting generation parameters during playback
- **Multi-attribute control**: Simultaneously controlling multiple attributes without interference
- **Non-Western music**: Most controllable systems assume Western tonal harmony

---

## 5. Video-to-Music Generation

Cross-modal music generation from visual inputs (video, images) is an emerging research direction that bridges computer vision and audio generation.

### 5.1 Key Models

#### MuVi (arXiv:2410.07840, 2024)
- **Paper**: "MuVi: Video-to-Music Generation with Rhythmic Alignment"
- Generates music from video input with **rhythmic alignment** between visual motion and musical beat structure
- **Visual Rhythm Extractor**: Extracts rhythmic cues from video (motion intensity, scene transitions) to form a "visual rhythm" representation
- **Music Generation Module**: Uses extracted visual rhythm as conditioning to generate music that aligns temporally with video dynamics
- Evaluation via both objective rhythmic alignment scores and subjective human evaluations

#### CMT (Contrastive Multimodal Transformer)
- Cross-modal generation framework using contrastive learning to align video and music representations
- Transformer-based architecture processes video frames and generates corresponding music via cross-modal attention
- Note: multiple papers use similar naming; verify specific arXiv ID for the exact work referenced

#### M2UGen (Multi-modal Music Understanding and Generation)
- Unified framework bridging music understanding (captioning, QA, analysis) and generation (text-to-music, image-to-music)
- LLM-based architecture integrating specialized audio and visual encoders with a music generation decoder
- Handles text, image, and audio inputs for cross-modal music creation

#### Video2Music
- Generates background music conditioned on video semantics and motion
- Extracts multi-modal features (visual, motion, semantic) from video frames as conditioning signals
- Addresses temporal alignment: ensuring generated music rhythm and mood match scene transitions

#### MuVi
- Visual-to-music generation with attention to rhythmic alignment between video motion and musical beat structure

### 5.2 Evaluation Challenges
- **CMMD (Contrastive Music-Video Metric)**: Evaluation metric for assessing alignment between generated music and video content
- **Temporal synchronization**: How well do musical beats align with visual scene transitions?
- **Emotional congruence**: Does the mood of generated music match the visual content?
- No standardized benchmark exists for video-to-music evaluation

### 5.3 Emerging Trends
- **Diffusion-based V2M**: Higher-fidelity audio generation from video using latent diffusion
- **LLM-conditioned V2M**: Using large language models as music generation backbones with visual feature conditioning
- **Emotion-driven generation**: Mapping visual emotional content (color, motion, facial expression) to musical parameters

---

## 6. Human Preference Alignment

Aligning music generation models with human aesthetic preferences, analogous to RLHF in language models, is a rapidly emerging research area.

### 6.1 RLHF for Music

- **MusicRLHF**: Reward model trained on human preference data for music generation
  - Introduces curated dataset of pairwise human comparisons across musical attributes (harmony, melody, rhythm, overall quality)
  - Enables fine-tuning of music language models via reinforcement learning from human feedback
  - Key challenge: music's temporal structure and multi-attribute nature make reward modeling harder than for text

### 6.2 Direct Preference Optimization (DPO)

- Applies the DPO framework (originally developed for LLMs) to music generation
- Directly optimizes from preference pairs without needing a separate reward model
- Often found to be more stable and efficient than RLHF for music alignment
- Demonstrated improved musical quality and stylistic consistency when applied to MusicGen/AudioCraft

### 6.3 Evaluation Benchmarks

- **MusicBench**: Benchmark specifically designed to measure alignment with human preferences across dimensions: musicality, genre adherence, emotional expressiveness
- **SongBench** (Make-It-Music, 2025): Song-level evaluation benchmark going beyond short clip evaluation to assess full songs
  - Covers melody quality, harmony, rhythm, vocal clarity, lyrics adherence, structural coherence, and overall musicality
  - Includes both objective metrics (audio signal analysis, lyrics alignment scores) and subjective/human evaluations
- **FakeMusicCaps**: Dataset of AI-generated music for detection and attribution tasks

### 6.4 Open Problems
- **Reward modeling**: Music's multi-attribute nature (harmony, rhythm, melody, timbre) and high subjectivity make reward modeling uniquely challenging
- **Preference data collection**: Scalable and reliable collection of pairwise human comparisons for music
- **Multi-dimensional alignment**: Aligning across multiple musical dimensions simultaneously without trade-offs
- **Cultural sensitivity**: Human preferences vary significantly across cultures and musical traditions

---

## 7. Evaluation

### 7.1 Automated Metrics

#### Frechet Audio Distance (FAD)
- **Origin**: Kilgour et al., INTERSPEECH 2019; adapted from Frechet Inception Distance (FID) for images
- **Method**: Computes Frechet distance between Gaussian distributions of audio embeddings from a reference set and a generated set
- **Embedding model**: Typically uses VGGish, PANNs (Pre-trained Audio Neural Networks), or CLAP audio encoder
- **Properties**:
  - **Reference-free** (at distribution level): Does not require pairwise comparison; compares distribution statistics
  - Measures both quality and diversity
  - Lower is better
- **Per-song FAD** (Microsoft, 2023): Extension that computes FAD for individual samples, showing moderate-to-strong correlation with human perceptual quality (MOS). Available in `microsoft/fadtk`
- **Limitation**: Depends on embedding model quality; may not capture all perceptually relevant dimensions; sensitive to dataset bias

#### KL Divergence on Label Distributions
- Computes KL divergence between label distributions (e.g., genre, instrument tags) predicted by a classifier on reference vs. generated audio
- Measures whether generated audio has similar high-level attributes as the reference distribution
- **Limitation**: Less robust than FAD; can behave inconsistently on degenerate outputs (e.g., silent audio). Noted in AudioLDM evaluation toolkit as unreliable

#### MuLan Similarity / CLAP Score
- Computes cosine similarity between text and audio embeddings using contrastive models (MuLan, CLAP)
- Measures text-audio alignment: how well does generated audio match the text prompt
- Higher is better
- Used in MusicLM and subsequent models

#### FrEchet Music Distance (FMD) (2024)
- **Paper**: arXiv:2412.07948 (December 2024)
- Adaptation of FAD specifically for symbolic music evaluation
- Operates on symbolic representations rather than audio
- Addresses the lack of evaluation metrics for generative symbolic music models

#### Other Automated Metrics
- **IS (Inception Score)**: Measures quality and diversity of generated samples using classifier predictions
- **FID (Frechet Inception Distance)**: Image-domain metric occasionally adapted for spectrogram representations
- **Log-spectral distance**: Measures spectral similarity between generated and reference audio

### 7.2 Human Evaluation

#### Mean Opinion Score (MOS)
- Gold standard for perceptual quality assessment
- Human raters score audio on 1--5 scale (bad to excellent)
- Typically measures overall quality, naturalness, or fidelity
- Used as ground truth to validate automated metrics

#### Comparative Preference Tests
- Side-by-side comparison of outputs from different models
- Raters choose which sample is better on specific dimensions (quality, relevance to prompt, musicality)
- Used in MusicLM, MusicGen, and most major model evaluations

#### MusicCaps Benchmark
- **Dataset**: 5,521 10-second music clips with human-written captions (Google, released with MusicLM)
- **Source**: YouTube videos via AudioSet; diverse genres
- **Use**: Standard benchmark for text-to-music models
- **Evaluation protocol**: Generate audio from MusicCaps captions, compute FAD and other metrics against reference clips
- **Limitation**: Only 10-second clips; limited cultural/genre diversity despite broad coverage

### 7.3 Comprehensive Evaluation Survey (2025)

- **Paper**: "A Survey on Evaluation Metrics for Music Generation" (arXiv:2509.00051, 2025)
- Consolidates evaluation approaches across the field
- Key findings:
  - FAD and KL divergence are most commonly used automated metrics
  - Per-song FAD shows best correlation with human perceptual quality
  - No single metric captures all dimensions of music quality
  - Human evaluation remains essential for reliable assessment
  - Benchmark datasets (MusicCaps, AudioSet) introduce their own biases

### 7.4 Challenges in Evaluating Generated Music

1. **Multidimensionality**: Music quality encompasses melody, harmony, rhythm, timbre, form, dynamics, and expressiveness. No single metric captures all dimensions.

2. **Subjectivity**: Musical quality is inherently subjective; different listeners, cultures, and genres have different standards.

3. **Long-form evaluation**: Most metrics operate on short clips (10--30 seconds). Evaluating structural coherence over minutes is underserved by existing metrics.

4. **Text-audio alignment**: Measuring how well generated audio matches a text prompt requires understanding both modalities, which is itself an open research problem.

5. **Originality vs. quality trade-off**: Should models prioritize generating novel music or high-quality music that may closely resemble training data? Metrics often reward the latter.

6. **Distribution vs. sample-level**: Most metrics (FAD, KL) measure distributional properties, not individual sample quality. Per-song metrics are emerging but not yet standard.

7. **Cultural bias**: Benchmarks and evaluation protocols are predominantly Western-music-centric.

8. **Lack of standardized benchmarks**: Different papers use different datasets, splits, and protocols, making cross-paper comparison difficult.

### 7.5 Emerging Evaluation Approaches

- **LLM-as-judge**: Using large language models to evaluate music quality from descriptions and audio features
- **Music understanding benchmarks**: MARBLE benchmark (used by YuE) evaluates learned representations on music understanding tasks
- **Human preference alignment**: AAAI 2025 paper (arXiv:2511.15038) on aligning generative music AI with human preferences
- **FakeMusicCaps**: Dataset of AI-generated music for detection and attribution tasks, derived from MusicCaps
- **SongBench** (Make-It-Music, 2025): Song-level evaluation benchmark covering melody quality, harmony, rhythm, vocal clarity, lyrics adherence, structural coherence, and overall musicality. Goes beyond short clip evaluation to assess full songs
- **MusicBench**: Benchmark for measuring alignment with human preferences across musicality, genre adherence, and emotional expressiveness

---

## 8. Architecture Comparison Summary

| Model | Year | Approach | Tokenizer/Codec | Max Length | Text Cond. | Open Source |
|-------|------|----------|-----------------|------------|------------|-------------|
| Jukebox | 2020 | Hierarchical AR (VQ-VAE) | VQ-VAE (3-level) | Several min | Lyrics + metadata | Yes |
| MusicLM | 2023 | Hierarchical AR (semantic + acoustic) | SoundStream | Several min | MuLan text emb. | No |
| MusicGen | 2023 | Single-stage AR Transformer | EnCodec (32 kHz, 4 CB) | ~30 sec | T5 + melody | Yes |
| AudioLDM | 2023 | Latent Diffusion (U-Net) | VAE (mel-spectrogram) | ~10 sec | CLAP | Yes |
| AudioLDM 2 | 2024 | Latent Diffusion + GPT-2 | VAE + improved | ~30 sec | CLAP + GPT-2 | Yes |
| Stable Audio 2.0 | 2024 | Latent Diffusion (DiT) | Compressed autoencoder (21.5 Hz) | 3 min | T5 | Partial |
| Stable Audio Open | 2024 | Latent Diffusion (DiT) | Autoencoder | ~47 sec | T5 | Yes |
| YuE | 2025 | LLaMA2-based AR Transformer | Learned music tokens | 5 min | Lyrics + style | Yes |
| Suno V4.5 | 2025 | Undisclosed | Undisclosed | 4 min | Text + lyrics | No |
| Suno V5 | Anticipated | Undisclosed | Undisclosed | TBD | Text + lyrics | No |
| Udio | 2024--25 | Undisclosed | Undisclosed | 2 min | Text + lyrics | No |
| TangoFlux | 2024 | Flow Matching | Learned | 30 sec | Text | Yes |
| ACE-Step | 2025 | Step-by-step Diffusion | Learned (step-wise) | ~3 min | Text + lyrics | Yes (ACE Studio + StepFun) |
| MusicFlow | 2025 | Cascading Flow Matching | Learned | ~2 min | Text | Yes |
| SongCreator | 2025 | Dual-track AR | Learned | Full song | Lyrics + style | Yes |
| MusicFX | 2023--25 | MusicLM-based (Google) | SoundStream | Short clips | Text | No |
| SkyMusic | 2024--25 | Undisclosed | Undisclosed | Full song | Text (multilingual) | No |

---

## 9. Key Open Problems and Future Directions

1. **Long-form coherence**: Maintaining musical structure, thematic development, and global form over multi-minute compositions remains the primary challenge. YuE's 5-minute output is a milestone but quality remains inconsistent.

2. **Evaluation**: No comprehensive, standardized evaluation framework exists. FAD is the de facto standard but has known weaknesses. Human evaluation is expensive and subjective.

3. **Controllability**: The gap between what users can specify (text prompts) and what they want to control (specific harmonies, structures, timbres) remains large. Structured control (Mustango, SegTune) is promising but limited.

4. **Copyright and ethics**: Training on copyrighted music raises legal and ethical questions. Open models (Stable Audio Open) that explicitly avoid copyrighted data show the path forward but may sacrifice quality.

5. **Real-time generation**: Current models are far from real-time for high-quality output. Interactive music generation for games, live performance, and creative tools requires low-latency inference.

6. **Cultural diversity**: Most models are biased toward Western popular music. Support for non-Western scales, instruments, forms, and singing styles is limited.

7. **Multimodal integration**: Combining music generation with video, dance, and other modalities is nascent but growing (ISMIR 2025 survey on vision-to-music generation).

8. **Data scale and quality**: Proprietary systems (Suno, Udio) likely train on far larger and more diverse datasets than academic/open models, contributing to their quality advantage.

9. **Convergence of AR and diffusion**: Hybrid architectures combining the strengths of autoregressive (sequential coherence, variable length) and diffusion/flow-matching (parallel generation, global structure, diversity) are a major trend.

10. **Preference alignment**: Aligning generative models with human musical preferences at scale is an emerging research area (AAAI 2025), analogous to RLHF in language models.

---

## References (Key Papers)

- Huang & Vaswani: "Music Transformer" (arXiv:1809.04281, ICLR 2019)
- Huang & Yang: "Pop Music Transformer / REMI" (arXiv:2002.00212, 2020)
- Dhariwal et al.: "Jukebox: A Generative Model for Music" (arXiv:2005.00341, 2020)
- Zeghidour et al.: "SoundStream" (arXiv:2107.03312, 2021)
- Defossez et al.: "EnCodec" (arXiv:2210.13438, 2022)
- Liu et al.: "DiffSinger" (arXiv:2105.02446, AAAI 2022)
- Xia et al.: "VISinger" (ICASSP 2022) and "VISinger 2" (INTERSPEECH 2023)
- Agostinelli et al.: "MusicLM" (arXiv:2301.11325, 2023)
- Liu et al.: "AudioLDM" (2023) and "AudioLDM 2" (arXiv:2308.05734, 2023--2024)
- Copet et al.: "MusicGen" (arXiv:2306.05284, 2023)
- Stability AI: "Stable Audio 2.0" (April 2024)
- Melechovsky et al.: "Mustango" (NAACL 2024, arXiv:2311.08355)
- Yuan et al.: "YuE" (arXiv:2503.08638, ICLR 2025)
- "A Survey on Evaluation Metrics for Music Generation" (arXiv:2509.00051, 2025)
- "Auto-Regressive vs Flow-Matching: A Comparative Study" (arXiv:2506.08570, 2025)
- "Aligning Generative Music AI with Human Preferences" (arXiv:2511.15038, AAAI 2025)
- "Pianoroll-Event" (arXiv:2601.19951, 2025)
- "REMI-z: Track-Aware Tokenization" (NeurIPS 2025)
- "SegTune: Structured and Fine-Grained Control" (arXiv:2510.18416, 2025)
- ACE-Step: "Step-by-Step Diffusion for Music Generation" (ACE Studio + StepFun, 2025)
- MusicFlow: "Cascading Flow Matching for Music Generation" (ICLR 2025)
- SongCreator: "Text-based Song Generation with In-context Learning" (2025)
- "MusicRLHF: Reward Modeling for Music Generation" (2025)
- "Direct Preference Optimization for Music Generation" (2025)
- ACE Studio / ACE Singer: Commercial SVS system (2024--2025)
- DiffSinger Acceleration: Consistency distillation and progressive distillation for faster SVS inference (2025)
- CMT: "Contrastive Multimodal Transformer for Video-to-Music" (verify specific arXiv ID)
- M2UGen: "Multi-Modal Music Understanding and Generation" (2025)
- "Make-It-Music / SongBench: Song-Level Music Generation Benchmark" (2025)
- OpenDiffSinger community: GitHub (ongoing, 2022--2025)
- "Discrete Audio Tokens: More Than a Survey" (arXiv:2506.10274, 2025)
