# Music Generation Evaluation: Metrics, Protocols, and Open Problems

A practitioner's guide to evaluating AI-generated music. Covers human evaluation protocols, automatic metrics (distribution-based, embedding-based, perceptual), text-audio alignment, benchmarking frameworks, and the persistent gap between automatic metrics and human judgment.

---

## 1. Why Music Evaluation Is Hard 为什么音乐评测难

Music is fundamentally multi-dimensional:

| Dimension | What varies | Why it matters |
|-----------|-------------|----------------|
| **Quality** | Clarity, fidelity, noise level | Basic audio quality |
| **Coherence** | Structural logic, phrase flow | Does it sound like *music*? |
| **Style fidelity** | Genre match, timbral accuracy | Does it match the target style? |
| **Novelty** | Originality, surprise, boredom | Is it interesting to listen to? |
| **Text-audio alignment** | Prompt adherence | Did it follow instructions? |
| **Emotional impact** | Arousal, valence conveyance | Does it evoke the intended feeling? |
| **Musicality** | Harmonic validity, rhythmic feel | Does it follow music conventions? |

No single metric captures all dimensions. A track can score high on FAD (distribution match) while being musically incoherent, or be harmonically interesting but have poor audio quality.

**The core tension**: Automatic metrics are fast and reproducible but miss musical meaning. Human evaluation captures musical meaning but is slow, expensive, and subjective.

---

## 2. Human Evaluation 人类评测

Human evaluation remains the gold standard. Key design decisions:

### 2.1 Common Protocols

| Protocol | Description | Use case |
|----------|-------------|----------|
| **MOS** (Mean Opinion Score) | Rate overall quality 1–5 | Quick quality check |
| **MUSHRA** | Rate against hidden references | High-quality comparison |
| **Pairwise comparison** | A vs. B, which is better? | Model comparison |
| **ABX** | A vs. B with hidden reference X | Perceptual difference |
| **Ranking** | Sort 3+ samples by preference | Preference ordering |
| **Attribute rating** | Rate specific dimensions (style, coherence, etc.) | Multi-dimensional eval |

### 2.2 Best Practices

- **Listeners**: Use musically trained listeners for fine-grained evaluation. Use crowdsourcing (MTurk, Prolific) for large-scale preference studies.
- **Listening environment**: Use studio headphones (ATH-M50x, HD-650) or monitors. Normalize loudness across samples (ITU-R BS.1770).
- **Sample duration**: 10–30s for quality/style checks. 30s–3min for coherence. Full tracks for end-to-end evaluation.
- **Randomization**: Randomize sample order to avoid position bias.
- **Inter-annotator agreement**: Report Cohen's kappa or ICC. Low agreement means the dimension is inherently subjective.
- **Number of listeners**: Minimum 20–30 for reliable MOS. More for fine-grained attribute rating.

### 2.3 Large-Scale Benchmarks

| Benchmark | Description | Scale |
|-----------|-------------|-------|
| **MUSECA** | Comparative listening tests for music generation | Multiple models, multiple judges |
| **SongBench** | Supervised quality labels for song generation | 1K+ tracks with human ratings |
| **ICASSP 2025 Human Preference Study** | Benchmarking music gen models via human preference | Systematic pairwise comparison |
| **AAAI 2025 Preference Alignment** | Human preference data for alignment training | Preference pairs |

---

## 3. Automatic Metrics: Overview 自动指标总览

### 3.1 The Metric Landscape

```
Automatic Metrics
├── Distribution-based    (FAD, FD, precision/recall)
├── Embedding-based       (CLAP Score, FSD)
├── Perceptual            (PEMO-Q, ViSQOL)
├── Music-specific        (Chroma-based, harmonic, rhythmic)
└── Text-audio alignment  (CLAP Score, FA-CLAP)
```

### 3.2 Metric Properties to Consider

| Property | Question |
|----------|----------|
| **Correlation with humans** | Does high score = human preference? |
| **Reference requirement** | Does it need ground-truth audio? |
| **Computation cost** | Forward pass through large model? |
| **Dimension coverage** | Quality? Style? Alignment? Coherence? |
| **Robustness** | Stable across different music types? |

No metric scores well on all five dimensions simultaneously.

---

## 4. Distribution-Based Metrics 分布指标

### 4.1 Fréchet Audio Distance (FAD)

Computes Fréchet distance between multivariate Gaussian distributions of audio embeddings:

$$
FAD = ||\mu_r - \mu_g||^2 + Tr(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
$$

- $\mu_r, \Sigma_r$: mean and covariance of real audio embeddings
- $\mu_g, \Sigma_g$: mean and covariance of generated audio embeddings

**Embedding backbones**: VGGish, YAMNet, CLAP, EnCodec features, music-specific models.

**Interpretation**: Lower FAD = generated distribution closer to real distribution. Lower is better.

**Common pitfalls**:
- FAD is sensitive to embedding choice. VGGish-based FAD and CLAP-based FAD can give very different rankings.
- FAD measures distribution match, not individual track quality. A model can have good FAD but produce occasional bad samples.
- FAD does not measure text-audio alignment.

### 4.2 Fréchet Distance (FD)

Same formula as FAD but applied to specific embedding spaces (e.g., CLAP audio embeddings, music foundation model features).

### 4.3 Precision and Recall

| Metric | Description |
|--------|-------------|
| **Precision** | % of generated samples that fall within the real distribution (avoiding outliers) |
| **Recall** | % of real distribution covered by generated samples (avoiding mode collapse) |
| **Density/Coverage** | Refined versions using k-nearest-neighbor |

A model with high precision but low recall is "safe" but boring. High recall but low precision is diverse but unreliable. Both should be reported.

---

## 5. Embedding-Based Metrics 嵌入指标

### 5.1 CLAP Score

Uses the CLAP model's audio-text alignment score:

$$
CLAPScore(a, t) = cosine\_similarity(CLAP_{audio}(a), CLAP_{text}(t))
$$

- **Average CLAP Score**: Mean over generated audio and their text conditions. Higher = better alignment.
- **CLAP Score by genre/style**: Decompose by style tag to check if alignment is uniform across styles.
- **Limitation**: CLAP was trained on general audio, not music-specific. CLAP Score correlates with human judgment for *some* dimensions but not all.

### 5.2 FA-CLAP (Fréchet Audio-CLAP Distance)

FAD computed in the CLAP embedding space. Captures both distribution match and text conditioning quality.

### 5.3 Frechet Style Distance (FSD)

FAD computed in a style-specific embedding space (e.g., genre classifier features, style embedding). Measures how well the generated output matches the target *style* distribution rather than the overall real distribution.

---

## 6. Perceptual Audio Quality Metrics 感知音频质量指标

### 6.1 Overview

| Metric | Full name | What it captures | Requires reference |
|--------|-----------|------------------|-------------------|
| **PEMO-Q** | Perceptual Evaluation of Audio Quality | Perceptual quality (ITU-R standard) | Yes |
| **ViSQOL** | Virtual Speech/audio Quality Object Listener | Perceptual similarity | Yes |
| **POLQA** | Perceptual Objective Listening Quality Analysis | Speech quality (less music) | Yes |
| **PESQ** | Perceptual Evaluation of Speech Quality | Speech quality only | Yes |
| **STFT-based** | L1/L2 loss on spectrograms | Spectral match | Yes |
| **Mel-distance** | L1/L2 on mel-spectrograms | Mel-spectral match | Yes |

**Key insight**: Reference-based perceptual metrics (PEMO-Q, ViSQOL) correlate better with human judgment of *audio quality* than spectrogram losses. But they require high-quality reference audio, which may not exist for creative generation.

### 6.2 When Reference Is Not Available

For creative generation (no ground-truth reference), use:
- **No-reference metrics**: BRISQUE, NIQE (image-derived, adapted for spectrograms)
- **Self-supervised features**: EnCodec reconstruction quality, audio codec bitrate efficiency
- **Distribution metrics**: FAD, precision/recall (no reference needed for distribution)

---

## 7. Music-Specific Metrics 音乐特有指标

These capture musical properties beyond audio quality:

### 7.1 Harmonic Metrics

| Metric | Description |
|--------|-------------|
| **Chord similarity** | Compare estimated chords of generated vs. reference (if condition is chord-based) |
| **Harmonic change rate** | Rate of chord changes per bar — matches target style |
| **Key consistency** | % of time spent in the expected key |
| **Tonal stability** | Pitch-class entropy over time — lower = more tonal |

### 7.2 Rhythmic Metrics

| Metric | Description |
|--------|-------------|
| **Beat alignment** | Correlation between generated and expected beat positions |
| **Groove consistency** | Swing ratio, micro-timing standard deviation |
| **Tempo stability** | BPM variance over time |
| **Onset density** | Notes/events per beat — matches genre norms |

### 7.3 Melodic Metrics

| Metric | Description |
|--------|-------------|
| **Pitch histogram match** | Pitch-class distribution similarity to target style |
| **Interval distribution** | Frequency of each interval size — Zipf-like for tonal music |
| **Pitch range** | Span in semitones |
| **Contour match** | Melodic contour classification match |

### 7.4 Structural Metrics

| Metric | Description |
|--------|-------------|
| **Repetition score** | Self-similarity matrix density — matches genre expectations |
| **Section boundary detection** | Automatic section change detection (boundary F-measure) |
| **Section proportion** | Intro/verse/chorus/bridge lengths match genre templates |

---

## 8. Text-Audio Alignment 文本-音频对齐

### 8.1 Why It Matters

For text-conditioned generation, *following the prompt* is as important as musical quality. Current failure modes:

| Failure | Example |
|---------|---------|
| Ignoring style | Prompt: "fast metal" → output: slow ambient |
| Ignoring mood | Prompt: "sad piano" → output: neutral piano |
| Ignoring structure | Prompt: "verse-chorus" → output: no clear sections |
| Hallucinating instruments | Prompt: "violin solo" → output: no violin |

### 8.2 Measuring Alignment

| Method | Description |
|--------|-------------|
| **CLAP Score** | Cosine similarity between CLAP audio and text embeddings |
| **FA-CLAP** | FAD in CLAP space (distribution + alignment) |
| **LLM-as-judge** | Use an LLM to compare generated audio description with prompt |
| **Attribute accuracy** | Classify generated audio for specific attributes (BPM, key, genre) and compare with prompt |
| **Human alignment rating** | "Did the generated music follow the prompt?" |

### 8.3 The Alignment-Perfection Trade-off

Stronger text conditioning (higher CFG scale) can improve prompt adherence but hurt musical quality and diversity. Finding the right CFG scale is task-dependent.

---

## 9. Benchmarking Frameworks 评测框架

### 9.1 Benchmark Platforms

| Platform | Description |
|----------|-------------|
| **MusicCaps** | 5.5K music clips with detailed captions (text quality evaluation) |
| **SynthSeg** | Synthetic evaluation for text-to-music alignment |
| **COMP** | Comprehensive benchmarking for music generation |
| **Codec SUPERB** | Benchmarking neural audio codecs (downstream task performance) |

### 9.2 Evaluation Protocol Template

A minimal but complete evaluation protocol:

```
1. Dataset: [description of test set, size, diversity]
2. Models: [models compared, checkpoints]
3. Conditions: [text prompts, BPM/key/meter conditions]
4. Samples: [number of generated samples per condition]
5. Automatic metrics: [FAD, CLAP Score, precision/recall]
6. Music-specific metrics: [harmonic, rhythmic, structural]
7. Human evaluation: [protocol, number of listeners, dimensions rated]
8. Baselines: [prior models, ground-truth references if available]
```

---

## 10. The Metric-Human Judgment Gap 指标与人类判断的鸿沟

### 10.1 Known Discrepancies

| Scenario | Automatic metric | Human judgment | Reason |
|----------|-----------------|----------------|--------|
| High FAD + human preference for generated | Good FAD | Generated preferred | FAD sensitive to embedding choice; humans judge musicality |
| Low FAD + human preference for real | Good FAD | Real preferred | FAD measures distribution, not individual quality |
| High CLAP Score + wrong style | Good alignment | Wrong style | CLAP captures coarse semantics, not fine-grained style |
| Good FAD + musically incoherent | Good distribution | Incoherent | FAD doesn't model musical structure |

### 10.2 Why the Gap Persists

1. **Metrics optimize for correlation, not causation**: Metrics are fitted to human data but don't model *why* humans prefer something.
2. **Music is cultural and subjective**: What sounds "good" depends on listener background, cultural context, and listening purpose.
3. **Metrics miss temporal structure**: FAD and CLAP Score operate on clip-level embeddings, missing phrase-level and section-level musical coherence.
4. **No metric captures "surprise"**: Musical interest comes from expectation violation (Meyer's theory), which is inherently hard to quantify.
5. **Reference dependency**: Best metrics need reference audio, which doesn't exist for creative generation tasks.

### 10.3 Current Research Directions

| Direction | Approach |
|-----------|----------|
| Music-specific embeddings | Train embeddings that capture musical structure (MERT variants, MusicFM probes) |
| Multi-dimensional metrics | Separate metrics per dimension instead of one score |
| LLM-based evaluation | Use LLMs with music knowledge to assess coherence, style, etc. |
| Perceptual metrics for music | Adapt speech quality metrics (PEMO-Q) for music-specific perception |
| Human-AI hybrid | Use metrics for screening + humans for final judgment |

---

## 11. Practical Recommendations 实践建议

### For model developers:

- **Always report FAD + CLAP Score** as minimum viable automatic evaluation. They are standard in the field.
- **Report precision AND recall** to catch mode collapse.
- **Decompose FAD/CLAP Score by style** to check if performance is uniform across genres.
- **Add music-specific metrics** relevant to your conditioning signals (e.g., chord accuracy for chord-conditioned generation, BMA for beat-conditioned).
- **Include human evaluation** for any paper or benchmark — even 20 listeners provides signal.
- **Report inter-annotator agreement** to contextualize human evaluation results.

### For benchmark designers:

- Use diverse test sets covering multiple genres, moods, and structural complexities.
- Include both simple and challenging conditions.
- Provide reference audio when possible (even if synthetic/rendered).
- Release evaluation code and generated samples for reproducibility.

### For metric developers:

- Target correlation with human judgment on *musical* dimensions, not just audio quality.
- Evaluate metrics across diverse musical traditions, not just Western pop.
- Develop reference-free or low-reference metrics for creative evaluation.
- Report metric performance on multiple datasets to assess robustness.

---

## 12. Summary: Evaluation Checklist 评测清单

Before publishing generation results:

- [ ] **FAD** reported (with embedding type specified)?
- [ ] **CLAP Score** reported (average + by style)?
- [ ] **Precision and recall** reported?
- [ ] **Music-specific metrics** included where relevant?
- [ ] **Human evaluation** conducted with minimum 20 listeners?
- [ ] **Inter-annotator agreement** reported?
- [ ] **Test set diversity** covers target genres/moods?
- [ ] **Baselines** compared (prior models, ground-truth)?
- [ ] **Evaluation code and samples** released?

---

## Further Reading 延伸阅读

- "Benchmarking Music Gen Models and Metrics via Human Preference Studies," ICASSP 2025
- "Aligning Generative Music AI with Human Preferences," AAAI 2025
- "Evaluation of Music Generation Systems: A Comprehensive Review," arXiv 2024
- "On the Evaluation of Conditional Music Generation," ISMIR 2024
- "FAD: Fréchet Audio Distance," arXiv 2021
- "CLAP Score," Wu et al., 2023

---

> This document covers evaluation methodology for AI music generation. For related topics, see [music-generation.md](music-generation.md) (generation architectures), [music-styles.md](music-styles.md) (style conditioning), and [music-understanding-mir.md](music-understanding-mir.md) (MIR evaluation).
