# Survey Reading Guide: Key Literature Reviews in AI Music

A curated reading guide to the most important survey papers in AI music research as of 2025--2026. Each entry includes scope, key insights, gaps, and mapping to this repository's notes.

---

## 1. A Survey on Deep Learning for Music Generation (2023)

**Scope**: Comprehensive coverage of symbolic and audio-level music generation. The most-cited general survey in the field.

**Key sections**:
- Symbolic music generation (RNN, Transformer, GAN approaches)
- Audio-level generation (WaveNet, GAN, diffusion)
- Representation learning (MIDI, piano roll, audio features)
- Evaluation metrics
- Challenges and future directions

**Key insights**:
- Autoregressive models dominate symbolic generation; diffusion dominates audio-level generation
- The "representation gap" between symbolic and audio-level approaches remains largely unbridged
- Evaluation is the weakest link: no standardized protocol, metrics don't correlate well with human judgment

**Gaps**:
- Published in 2023, missing 2024–2025 advances (MusicGen, Stable Audio, ACE-Step, YuE, diffusion-based symbolic generation)
- Limited coverage of controllable generation
- No coverage of singing voice synthesis as a distinct topic
- Sparse on foundation models and pre-training paradigms

**Read this for**: Foundational understanding of the generation landscape before diving into specific model papers. Best used as a historical baseline (what was known by 2023).

**→ Maps to**: [music-generation.md](music-generation.md) §1–§5, [music-evaluation.md](music-evaluation.md)

---

## 2. Discrete Audio Tokens: More Than a Survey (arXiv 2506.10274, 2025)

**Scope**: The most comprehensive and recent survey on audio tokenization — the backbone of modern text-to-music systems.

**Key sections**:
- Historical evolution: From VQ-VAE to neural codecs to large-scale tokenization
- Taxonomy of tokenization approaches (RVQ, LFQ, product quantization, semantic tokenization)
- Codec architectures compared (EnCodec, DAC, SoundStream, WavTokenizer, SemantiCodec, TQCodec)
- Downstream applications: language modeling, generation, compression
- Open problems and future directions

**Key insights**:
- Tokenization quality directly limits generation quality — it's not just a "compression" step
- RVQ remains dominant but LFQ and semantic tokenization are gaining ground
- The field needs standardized benchmarks for codec evaluation beyond reconstruction quality
- Music-specific requirements (harmonic structure, temporal resolution) differ from speech

**Gaps**:
- Very recent — some 2025 models may not be included
- Heavy on technical comparison, lighter on musical implications
- Limited discussion of how tokenization design affects generation controllability

**Read this for**: Deep understanding of audio codecs and tokenization — the foundation for understanding how MusicGen, AudioLDM, and similar systems work. Essential reading before implementing any token-based generation system.

**→ Maps to**: [audio-engineering.md](audio-engineering.md) §2–§5, [music-generation.md](music-generation.md) §1.1

---

## 3. Foundation Models for Music: A Survey (arXiv 2408.14340, 2024)

**Scope**: Comprehensive survey of music foundation models — large-scale pre-trained models for music understanding and generation.

**Key sections**:
- Pre-training objectives (masked modeling, contrastive learning, autoregressive)
- Model families (MERT, MusicFM, JukeMIR, CLAP, CLaMP, MusicBERT, etc.)
- Downstream task performance (tagging, transcription, generation)
- Cross-modal alignment (audio-text, audio-MIDI, audio-image)
- Evaluation benchmarks (MARBLE, SUPERB)

**Key insights**:
- Self-supervised pre-training on large music corpora is the dominant paradigm
- Foundation models transfer well to downstream tasks but vary in which tasks they excel at
- Multi-modal alignment (CLAP, CLaMP 3) enables zero-shot music understanding
- Key detection and fine-grained harmonic analysis remain weak spots for all foundation models

**Gaps**:
- Rapidly evolving field — 2025 models (MERT v2, MusicFM v2, newer variants) may not be covered
- Limited discussion of generation capabilities of foundation models (most focus on understanding)
- No coverage of computational cost and accessibility for smaller research groups

**Read this for**: Understanding the foundation model landscape — which models exist, how they're trained, and what they're good at. Critical for anyone doing transfer learning or zero-shot MIR.

**→ Maps to**: [music-understanding-mir.md](music-understanding-mir.md) §5, [music-styles.md](music-styles.md) §7

---

## 4. A Survey on Singing Voice Synthesis (2024)

**Scope**: Comprehensive survey of SVS — from classical HMM-based approaches to modern diffusion and end-to-end systems.

**Key sections**:
- SVS pipeline components (frontend, acoustic model, vocoder)
- Model evolution (statistical parametric → neural vocoder → diffusion → end-to-end)
- Controllable singing (pitch, timbre, style, expressiveness)
- Singing voice conversion
- Datasets and evaluation
- Open challenges

**Key insights**:
- Diffusion models have become the dominant approach for mel-spectrogram generation in SVS
- Controllability (exact pitch, phoneme timing) is what distinguishes SVS from general TTS
- Zero-shot singing voice conversion is an emerging frontier
- No standard benchmark exists — evaluation is fragmented

**Gaps**:
- Some very recent models (2025) not included
- Limited coverage of SVS within general audio generation systems (MusicGen, AudioLDM)
- Evaluation discussion is brief — this remains an under-addressed problem

**Read this for**: Complete SVS background before diving into specific systems (DiffSinger, OpenDiffSinger, ACE Singer).

**→ Maps to**: [music-singing-synthesis.md](music-singing-synthesis.md)

---

## 5. A Survey on Evaluation Metrics for Music Generation (arXiv 2509.00051, 2025)

**Scope**: Systematic survey of evaluation methodology specifically for music generation — the most directly applicable survey to this repository's evaluation notes.

**Key sections**:
- Human evaluation protocols (MOS, MUSHRA, pairwise, etc.)
- Distribution-based metrics (FAD, precision/recall)
- Embedding-based metrics (CLAP Score, FSD)
- Perceptual metrics (PEMO-Q, ViSQOL)
- Music-specific metrics (harmonic, rhythmic, structural)
- Text-audio alignment metrics
- Benchmarking platforms

**Key insights**:
- No single metric captures all quality dimensions — multi-metric reporting is essential
- The correlation gap between automatic metrics and human judgment persists across all metric types
- Music-specific metrics (harmonic coherence, rhythmic alignment) are underdeveloped compared to general audio metrics
- Evaluation methodology is more inconsistent than model architecture — standardization is urgently needed

**Gaps**:
- Very recent — may not include 2025's newest metrics
- Heavy on comparison tables, lighter on practical guidance for choosing metrics
- Limited discussion of evaluation cost vs. signal trade-offs

**Read this for**: The definitive guide to choosing and implementing evaluation metrics. Use alongside [music-evaluation.md](music-evaluation.md) for practical implementation guidance.

**→ Maps to**: [music-evaluation.md](music-evaluation.md)

---

## 6. AI and Music: A Comprehensive Survey (Brée)

**Scope**: Broad survey covering the full spectrum of AI music — from early expert systems to modern deep learning. Focus on style and aesthetics.

**Key sections**:
- Historical development of AI music (1950s–present)
- Symbolic generation approaches
- Audio generation
- Style and genre modeling
- Interaction and improvisation
- Philosophical and aesthetic considerations

**Key insights**:
- AI music has oscillated between "rule-based" and "learning-based" paradigms multiple times
- Style modeling is fundamentally different from quality modeling — a model can produce stylistically correct but musically boring output
- The aesthetic dimension of AI music is under-researched compared to technical performance

**Gaps**:
- Older publication — missing 2022–2025 advances (diffusion models, foundation models, large-scale generation)
- Limited technical depth on modern architectures
- Less relevant for practitioners building systems, more for researchers understanding the field's trajectory

**Read this for**: Historical and philosophical context. Best read first to understand where the field came from, then read more recent surveys for current state.

**→ Maps to**: [music-styles.md](music-styles.md), [music-generation.md](music-generation.md) §historical context

---

## 7. Music Source Separation: A Brief Overview (2023)

**Scope**: Focused survey on music source separation — separating mixed audio into individual stems (vocals, drums, bass, other).

**Key sections**:
- Problem formulation and evaluation (BSS Eval, SDR/SIR/SAR)
- Model evolution (spectrogram → waveform → hybrid → band-split)
- Open-unmix scenario (separating arbitrary sources)
- Datasets (MUSDB18, MUSDB18-HQ, MoisesDB)

**Key insights**:
- Waveform-domain approaches (Demucs) surpassed spectrogram-domain (Spleeter)
- Hybrid Transformers (HT Demucs) and band-split architectures (BSRNN, BS-RoFormer) represent the current frontier
- Band-split processing (dividing spectrum into bands, processing independently) is the key architectural innovation of 2023–2024
- SDR ~12 dB on MUSDB18-HQ is the current ceiling; further gains are diminishing

**Gaps**:
- Brief by design — not comprehensive
- Missing some 2024–2025 models (Band-SCNet, SUNAC)
- Limited discussion of real-time separation and mobile deployment

**Read this for**: Quick orientation to source separation before diving into specific model papers.

**→ Maps to**: [music-understanding-mir.md](music-understanding-mir.md) §3, [audio-engineering.md](audio-engineering.md)

---

## 8. Additional Notable Surveys

| Survey | Year | Scope | Why it matters |
|--------|------|-------|----------------|
| **Music Information Retrieval: A Survey** (McKinney, 2009) | 2009 | Classic MIR survey | Historical baseline for MIR tasks |
| **Neural Audio Synthesis: A Survey** | 2023 | Audio synthesis (TTS + music) | Cross-pollination between speech and music synthesis |
| **A Survey on Music Generation with Diffusion Models** | 2024 | Diffusion for music specifically | Deep dive on the dominant generation paradigm |
| **Multimodal Music Understanding: A Survey** | 2024 | Cross-modal music AI (audio+text+image+video) | Emerging direction |
| **Culture-Aware MIR: A Survey** | 2024 | Non-Western music, cultural bias | Addresses the cross-cultural gap |

---

## 9. Recommended Reading Order 推荐阅读顺序

### For beginners:

1. **AI and Music (Brée)** — historical and philosophical context
2. **Deep Learning for Music Generation (2023)** — foundational generation landscape
3. **Music Source Separation (2023)** — focused, accessible introduction to one key MIR task
4. This repository's notes (乐理基础 → 音乐风格 → 音乐理解 → 音频工程)

### For generation researchers:

1. **Discrete Audio Tokens (2025)** — understand tokenization before anything else
2. **Deep Learning for Music Generation (2023)** — baseline landscape
3. **Evaluation Metrics for Music Generation (2025)** — evaluation methodology
4. **Foundation Models for Music (2024)** — pre-trained models for generation
5. This repository's notes (音乐生成 → 音乐评测 → 歌声合成)

### For MIR researchers:

1. **Foundation Models for Music (2024)** — the current state of music understanding
2. **Music Source Separation (2023)** — one of MIR's most successful tasks
3. **McKinney MIR Survey (2009)** — for task definitions and historical context
4. This repository's notes (音乐理解 → 乐理基础 → 音乐风格)

### For SVS researchers:

1. **Singing Voice Synthesis (2024)** — comprehensive SVS background
2. **Deep Learning for Music Generation (2023)** — for broader generation context
3. **Foundation Models for Music (2024)** — for pre-trained representations
4. This repository's notes (歌声合成 → 音乐生成)

---

## 10. What This Repository Adds 本仓库的独特价值

These surveys are invaluable, but they share common limitations that this repository addresses:

| Limitation | How this repo helps |
|-----------|---------------------|
| Surveys are snapshots (become outdated) | Continuously updated with 2025–2026 developments |
| Surveys focus on "what" not "why for AI" | Every section connects theory/practice to AI implications |
| Surveys are English-only | Bilingual (EN + ZH) for accessibility |
| Surveys don't provide a learning path | Structured from fundamentals (乐理) → style → technology → practice |
| Surveys lack practical guidance | Includes code structure, GPU requirements, dataset catalogs, reproduction guides |

---

> This reading guide complements the individual notes in this repository. For detailed coverage of each topic, see the corresponding notes: [music-generation.md](music-generation.md), [music-understanding-mir.md](music-understanding-mir.md), [audio-engineering.md](audio-engineering.md), [music-evaluation.md](music-evaluation.md), [music-singing-synthesis.md](music-singing-synthesis.md), [music-styles.md](music-styles.md), [music-theory-fundamentals.md](music-theory-fundamentals.md).
