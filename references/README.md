# Reading List

## Music Understanding

| Paper | Year | Topic |
|-------|------|-------|
| Music Tagging Transformers (Won et al.) | 2021 | Auto-tagging |
| CLAP (Wu et al.) | 2023 | Audio-language contrastive learning |
| MusicFM (Kumar et al.) | 2024 | Music foundation model |

## Music Generation

| Paper | Year | Topic |
|-------|------|-------|
| Music Transformer (Huang et al.) | 2018 | Symbolic generation |
| Jukebox (Dhariwal et al.) | 2020 | Audio generation via VQ-VAE |
| AudioLDM (Liu et al.) | 2023 | Latent diffusion for audio |
| MusicGen (Copet et al.) | 2023 | Text-to-music |
| Stable Audio (Roberts et al.) | 2023 | Latent diffusion, real-time |
| YuE (team) | 2025 | Full-song generation |
| ACE-Step (ACE Studio + StepFun) | 2025 | Foundation model: DCAE + linear transformer + diffusion (REPA) |
| MusicFlow (Prajwal et al., Meta) | 2024 | Cascaded flow matching for text-to-music (ICML 2024) |
| SongCreator (Shun Lei et al.) | 2024 | Lyrics-based universal song generation (NeurIPS 2024, DSLM) |
| MusicFX (Google) | 2023--25 | Consumer text-to-music (MusicLM-based) |

## Video-to-Music Generation

| Paper | Year | Topic |
|-------|------|-------|
| CMT (Contrastive Multimodal Transformer) | 2025 | Video-to-music via contrastive cross-modal alignment |
| M2UGen | 2025 | Multi-modal music understanding and generation |
| Video2Music | 2025 | Video-conditioned background music |
| MuVi | 2025 | Visual-to-music with rhythmic alignment |

## Human Preference Alignment

| Paper | Year | Topic |
|-------|------|-------|
| Benchmarking Music Gen Models and Metrics via Human Preference Studies | 2025 | Human preference benchmark for music generation (ICASSP 2025) |
| Aligning Generative Music AI with Human Preferences | 2025 | Preference alignment (AAAI 2025) |
| Make-It-Music / SongBench (Nankai Univ.) | 2025 | Song generation framework with supervised quality labels (arXiv:2502.19324) |

## Audio Engineering

| Paper | Year | Topic |
|-------|------|-------|
| EnCodec (Defossez et al.) | 2022 | Neural audio codec, SEANet architecture, RVQ |
| SoundStream (Zeghidour et al.) | 2021 | First end-to-end neural audio codec |
| DAC / Improved RVQGAN (Kumar et al.) | 2023 | High-fidelity codec, improved RVQ training |
| FunCodec (Du et al.) | 2023 | Open-source neural codec toolkit |
| SemantiCodec (Liu et al.) | 2024 | Dual-encoder semantic/acoustic codec |
| WavTokenizer (Pan et al.) | 2024 | LFQ-based single-codebook tokenizer |
| HiFi-Codec (Yang et al.) | 2024 | Grouped-RVQ for efficient codec |
| TQCodec | 2025/2026 | Trellis quantization, high-fidelity music |
| SUNAC (MERL) | 2026 | Source-aware unified neural audio codec |
| WaveNet (van den Oord et al.) | 2016 | Dilated causal convolutions for audio |
| DDPM (Ho et al.) | 2020 | Denoising diffusion probabilistic models |
| Score-Based SDE (Song et al.) | 2021 | Unified SDE framework for diffusion |
| CFG (Ho & Salimans) | 2022 | Classifier-free diffusion guidance |
| AudioLDM (Liu et al.) | 2023 | Latent diffusion for audio |
| AudioLDM 2 (Liu et al.) | 2024 | Unified audio generation, IEEE TASLP |
| Stable Audio (Roberts et al.) | 2024 | Timing-conditioned latent diffusion |
| AudioLM (Borsos et al.) | 2022 | Hierarchical language model for audio |
| MusicLM (Agostinelli et al.) | 2023 | Text-to-music via hierarchical tokens |
| MusicGen (Copet et al.) | 2023 | Transformer + EnCodec for music |
| VQ-VAE (van den Oord et al.) | 2017 | Neural discrete representation learning |
| SincNet (Ravanelli & Bengio) | 2018 | Learnable audio frontend |
| MERT (Min et al.) | 2023 | Music foundation model, self-supervised |
| CLAP (Wu et al.) | 2023 | Contrastive language-audio pretraining |
| Brown (CQT) | 1991 | Constant-Q transform for music |
| Schorkhuber & Klapuri | 2010 | CQT toolbox for music processing |

## Music Theory & Foundations

| Paper / Book | Year | Topic |
|--------------|------|-------|
| Piston, W. *Harmony* (revised ed.) | 1987 | Functional harmony, standard textbook |
| Kostka, S. & Payne, D. *Tonal Harmony* (8th ed.) | 2018 | Modern harmony textbook |
| Aldwell, E. & Cadwallader, A. *Harmony and Voice Leading* (4th ed.) | 2018 | Voice-leading-oriented harmony |
| Krumhansl, C.L. *Cognitive Foundations of Musical Pitch* | 1990 | Pitch perception, key-finding theory |
| Bregman, A.S. *Auditory Scene Analysis* | 1990 | Cognitive basis for source separation |
| Müller, M. *Fundamentals of Music Processing* (2nd ed.) | 2021 | Computational musicology textbook |
| Zbikowski, L.M. *Conceptualizing Music* | 2002 | Music cognition and computational models |

## Music Styles & Genre

| Paper / Book | Year | Topic |
|--------------|------|-------|
| Serrà, J. et al. "Correlation and Causality in Music Style Construction" | 2012 | Statistical analysis of musical style |
| Huang, C.Z.A. *Music Style Modeling and Generation* (PhD thesis) | 2017 | Computational approaches to style |
| Lomax, A. *Folk Song Style and Culture* | 1968 | Anthropological approach to musical style |
| Park, J. et al. "Multitrack Music Transformer" | 2022 | Multi-instrument style modeling |
| Hung, H.T. et al. "Emotional Music Generation via Disentangled Representations" | 2022 | Style disentanglement for generation |
| Brée, D. *AI and Music: A Comprehensive Survey* | — | Technical survey covering style in AI music |

## Music Evaluation

| Paper / System | Year | Topic |
|----------------|------|-------|
| FAD (Fréchet Audio Distance) | 2021 | Distribution-based audio quality metric |
| CLAP Score | 2023 | Text-audio alignment metric |
| Benchmarking Music Gen Models via Human Preference Studies | 2025 | Human preference benchmark (ICASSP 2025) |
| Aligning Generative Music AI with Human Preferences | 2025 | Preference alignment (AAAI 2025) |
| SongBench | 2025 | Supervised quality labels for song generation |
| PEMO-Q | — | Perceptual audio quality evaluation (ITU-R) |
| ViSQOL | — | Virtual speech/audio quality objective listener |

## Singing Voice Synthesis

| Paper / System | Year | Topic |
|----------------|------|-------|
| DiffSinger | 2021 | Diffusion-based SVS, landmark open-source |
| OpenDiffSinger | 2022--25 | Community fork with multi-speaker/language |
| So-VITS-SVC | 2022 | VITS-based singing voice conversion |
| SingSong (Google) | 2023 | Diffusion-based singing extraction + synthesis |
| ACE Singer (ACE Studio) | 2024--25 | Commercial multi-language SVS, DAW integration |
| RDCM | 2025 | Recurrent diffusion for long-form singing |
| A Survey on Singing Voice Synthesis | 2024 | Comprehensive SVS survey |

## Music Recommendation

| Paper / System | Year | Topic |
|----------------|------|-------|
| Spotify audio features | 2010s | Danceability, energy, valence features |
| MERT for recommendation | 2023 | Self-supervised embeddings for music similarity |
| CLAP for music search | 2023 | Text-based music retrieval |
| MusicFM | 2023 | Foundation model for recommendation |

## Cover Detection & Melody Extraction

| Paper / System | Year | Topic |
|----------------|------|-------|
| Chromaprint / AcoustID | 2008-- | Chroma-based fingerprinting for cover detection |
| DeepSalience | 2018 | Deep learning pitch salience for melody extraction |
| SecondHandSongs | — | Crowdsourced cover metadata (~1M works) |
| Da-TACOS | 2019 | Large-scale cover song benchmark (~17K) |
| Covers80 | 2011 | Classic small benchmark (80 pairs) |

## Surveys

| Paper | Year | Scope |
|-------|------|-------|
| A Survey on Deep Learning for Music Generation | 2023 | Comprehensive music gen survey |
| Music Source Separation: A Brief Overview | 2023 | Source separation |
| Discrete Audio Tokens: More Than a Survey (arXiv 2506.10274) | 2025 | Comprehensive survey on audio tokenization |
| Discrete Tokenization for Multimodal LLMs | 2025 | VQ/RVQ tokenization for multimodal systems |
| Codec SUPERB | 2024 | Neural audio codec benchmarking |

---

This list is a starting point — contributions welcome.
