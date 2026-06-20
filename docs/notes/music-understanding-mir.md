# Music Understanding / Music Information Retrieval (MIR)

State of the field as of 2025-2026. Focused on technical architectures, datasets, benchmarks, and open problems.

---

## 1. Auto-Tagging and Classification

### Problem Definition

Automatically assigning descriptive labels (genre, mood, instrumentation, acoustic qualities) to audio recordings. Formulated as multi-label classification over a tag vocabulary of 50--100+ tags.

### Key Architectures

#### CNN-based (2016--2020)
- **Choi et al., "Automatic Tagging Using Deep Convolutional Neural Networks," ISMIR 2016.** FCN with 2D convolutions on mel-spectrograms. Established that deep learning could outperform hand-crafted features for tagging.
- **Won et al., "Data-driven Hybrid Approaches," ISMIR 2020.** Compared ResNet, SENet (squeeze-and-excitation), and other CNN architectures for music tagging. Music tagging CNN based on ResNet-ish architecture with 4--5 convolutional blocks remains a strong baseline.

#### Transformer-based (2021--)
- **Won, Chun, Nieto, Serra, "Semi-Supervised Music Tagging Transformer," ISMIR 2021.** Applies a Vision Transformer (ViT)-style architecture to audio spectrograms. Shallow layers capture local acoustic features; deeper self-attention layers model global temporal structure. Trained semi-supervised using unlabeled data. Achieved ROC-AUC of **~0.914** on MagnaTagATune, surpassing CNN baselines.
- The key insight: Transformers capture long-range temporal dependencies (e.g., song structure, repeated motifs) that CNNs with limited receptive fields miss.

#### CLAP / Contrastive Audio-Language Models (2023--)
- **Wu et al., "Large-scale Contrastive Language-Audio Pretraining," arXiv 2211.06687, 2022 (CLAP).** Dual-encoder contrastive learning aligning audio and text in a shared latent space. Trained on ~630K audio-text pairs. Enables zero-shot audio classification using natural language prompts. Not music-specific but widely applied to music tasks. Code: [github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP).
- **T-CLAP (2024):** Temporal-Enhanced CLAP for improved temporal reasoning in audio.
- **CLaMP 3 (ACL 2025 Findings, [aclanthology.org/2025.findings-acl.133](https://aclanthology.org/2025.findings-acl.133/)):** Aligns all major music modalities (sheet music, MIDI, audio, images, text) in a shared representation space via contrastive learning. Supports cross-modal and cross-lingual retrieval. Current SOTA on many MIR tasks. Code: [github.com/sanderwood/clamp3](https://github.com/sanderwood/clamp3).

### Representative Datasets

| Dataset | Size | Tags | Notes |
|---------|------|------|-------|
| **MagnaTagATune** | ~25,863 clips (30s each) | 50 tags | Standard benchmark. Noisy labels (crowd-sourced). Curated by Law et al., 2009. |
| **MTG-Jamendo** | ~55,000 full tracks | 700+ tags (genre, mood, instrument) | Higher quality, per-track annotations. Bogdanov et al., 2019. |
| **GTZAN** | 1,000 tracks (30s) | 10 genre classes | Small but widely used for genre classification. Tzanetakis & Cook, 2002. |
| **FMA (Free Music Archive)** | ~106,574 tracks | Genre hierarchy (8/16/161 genres) | Defferrard et al., 2017. Larger-scale genre classification. |
| **NSynth** | ~305,000 single notes | Instrument (11 families), pitch, velocity | Google Magenta, Engel et al., 2017. Note-level, not song-level. |

### Approximate SOTA Performance

- **MagnaTagATune (ROC-AUC):** Music Tagging Transformer ~0.914; foundation models (MERT finetuned, CLAP) push toward 0.92--0.93.
- **MTG-Jamendo (ROC-AUC):** ~0.92--0.94 depending on tag subset.
- **GTZAN genre classification (accuracy):** >93% with modern models; dataset has known issues (replications, length).
- **NSynth instrument classification (accuracy):** >95% with CNN/Transformer on mel-spectrograms; near-solved for this dataset.

### Open Problems

- **Noisy labels:** MagnaTagATune labels are crowd-sourced and inconsistent. New benchmarks with expert annotations emerging (e.g., MGPHot, 2025).
- **Long-form music:** Most models operate on fixed-length clips (10--30s). Handling full tracks with variable structure is under-explored.
- **Fine-grained tags:** Genre taxonomies are culturally specific and contested. Few-shot and zero-shot tagging via language models (CLAP, CLaMP 3) is a promising direction.
- **Cross-cultural bias:** Models trained on Western popular music do not generalize well to non-Western traditions.

---

## 2. Music Transcription

### Problem Definition

Converting raw audio into symbolic note representations (MIDI-like): onset time, offset time, pitch, velocity, and optionally instrument label. The "speech recognition of music."

### Piano Transcription

The most-studied case due to the availability of aligned MIDI/audio datasets.

#### Key Models

- **Onsets and Frames (Hawthorne et al., "Onsets and Frames: Dual-Objective Piano Transcription," ISMIR 2018).**
  - Architecture: CNN on spectrogram feeding two separate LSTM stacks: one for onset detection, one for frame-level pitch classification. Onset detections condition the frame predictions.
  - Trained on MAPS: note-level F1 ~50% on MAESTRO test. Trained on MAESTRO: note-level F1 ~67%.
  - Google Magenta implementation. Significant leap over prior HMM-based approaches.

- **High-Resolution Piano Transcription (Kong et al., 2020--2021).**
  - Architecture: Regression-based onset/offset detection + pitch classification using CNNs with high-resolution feature maps.
  - Reported note-level F1 of ~90--93% on MAESTRO test set (with offset tolerance). Among the first to break 90%.
  - Regressed onset/offset times rather than binary classification.

- **Piano Transcription with Transformers (Hawthorne et al., 2022).**
  - Transformer encoder applied to spectrograms for joint onset/offset/pitch/velocity prediction. Further improvements on MAESTRO.

- **Onsets and Velocities (2023).** Lightweight model achieving SOTA onset performance with more efficient architecture, trained on MAESTRO v3.

#### Modern systems (2023--2025) achieve note-level F1 scores pushing 93--97% on MAESTRO (frame-level F1 >95%) for piano.

### Multi-Instrument Transcription

- **MT3 (Gardner et al., "MT3: Multi-Task Multitrack Music Transcription," ICLR 2022, [arXiv 2111.03017](https://arxiv.org/abs/2111.03017)).**
  - Architecture: T5 encoder-decoder Transformer. Treats transcription as sequence-to-sequence: audio spectrogram in, token sequence of note events out. Each note token includes instrument label + pitch + onset/offset/velocity.
  - Trained on multiple datasets simultaneously (multi-task), enabling transcription of arbitrary instrument combinations with a single model.
  - Built on Google's T5X framework. Code: [github.com/magenta/mt3](https://github.com/magenta/mt3).
  - Does not require separate models per instrument.

- **2025 AMT Challenge (NeurIPS 2025, [arxiv.org/html/2603.27528v1](https://arxiv.org/html/2603.27528v1)).**
  - Community benchmark for multi-instrument transcription. New test set, cloud-based evaluation.
  - MT3 used as baseline. Multiple teams submitted improvements.
  - Extends evaluation beyond piano to realistic multi-instrument mixes.

- **YourMT3+ (2024).** Toolkit building on MT3 for training multi-task multi-track models.

- **CountEM (ISMIR 2025).** Uses note event histograms as supervision instead of requiring exact time-aligned MIDI. Reduces dependence on costly aligned data.

### Drum Transcription

- Separate sub-field focused on transcribing drum hits (kick, snare, hi-hat, etc.) with onset times and kit piece labels.
- Datasets: ENST-Drums, RBMA-13, SoundBrush.
- CNN and CRNN approaches on mel-spectrograms; typically framed as frame-level multi-label classification.
- Less progress than piano transcription due to smaller datasets and greater timbral variety.

### Representative Datasets

| Dataset | Content | Size | Notes |
|---------|---------|------|-------|
| **MAPS** | Piano (synthesized + recorded) | ~240 pieces | Aligned MIDI/audio. Emiya et al., 2010. First standard benchmark. |
| **MAESTRO** | Piano (real performances) | ~200 hours, ~1,282 performances | Aligned MIDI/audio from International Piano-e-Competition. Hawthorne et al., 2019. Gold standard for piano transcription. |
| **MusicNet** | Various instruments (ensemble) | 330 recordings | Thickstun et al., 2017. Multi-instrument, but alignment quality varies. |
| **SLAKH** | Multi-instrument (synthesized) | ~2,100 mixtures | Manilow et al., 2019. Built from individual synthesized instrument tracks. |
| **URMP** | Multi-instrument (duets to quintets) | 44 performances | Li et al., 2018. Small but high-quality multi-instrument data. |

### Approximate SOTA Performance

- **Piano (MAESTRO, note-level F1 with offset):** 93--97% for top systems. Frame-level F1 >95%.
- **Piano (MAESTRO, note-level F1 without offset):** Lower, ~85--90%, as offset detection is harder.
- **Multi-instrument (MT3 on Slakh):** Onset F1 varies by instrument: piano/violin ~80--85%, bass/guitar ~70--75%. Overall lower than piano-only.
- **Drums:** Onset F1 ~75--85% depending on kit piece and dataset.

### Open Problems

- **Multi-instrument transcription in realistic mixes** remains far from solved. Overlapping harmonics, room acoustics, and diverse timbres make this much harder than piano-only.
- **Expressive performance transcription:** Velocity, articulation, pedal (sustain, soft pedal), micro-timing.
- **Vocal transcription:** Pitch tracking of singing voice in the presence of accompaniment is still challenging.
- **Training data bottleneck:** Aligned MIDI/audio is expensive to collect. Self-supervised and weakly-supervised approaches (CountEM) are promising.
- **Generalization:** Models trained on one dataset often degrade significantly on different instruments or recording conditions.

---

## 3. Source Separation

### Problem Definition

Decomposing a mixed audio signal into its constituent sources (e.g., vocals, drums, bass, other). Most commonly formulated as 4-stem separation (vocals, drums, bass, other) following the MUSDB18 benchmark.

### Key Models

#### Spectrogram-Domain Approaches
- **Spleeter (Henaff et al., "Spleeter: A Fast and Efficient Music Source Separation Tool," ISMIR 2019).** U-Net operating on mel-spectrograms. Predicts magnitude masks for each stem. Fast (100x realtime) but quality limited by spectrogram phase reconstruction. Pre-trained on internal Deezer dataset. Code: [github.com/deezer/spleeter](https://github.com/deezer/spleeter).

#### Waveform-Domain Approaches
- **Demucs (Defossez et al., "Music Source Separation in the Waveform Domain," arXiv 1911.13254, 2019).** U-Net-style encoder-decoder operating directly on raw waveforms. Bidirectional LSTM in bottleneck. Surpassed SOTA by 0.3+ dB SDR. No spectrogram domain needed.

- **Hybrid Demucs (Defossez et al., 2021).** Combined spectral and waveform branches. The spectral branch handles fine frequency structure; the waveform branch handles temporal patterns. Improved SDR across all stems.

#### Hybrid Transformer Approaches
- **HT Demucs (Rouard & Massa, "Hybrid Transformers for Music Source Separation," ICASSP 2023).** Adds Transformer layers to Hybrid Demucs for long-range context modeling in both temporal and spectral domains. Fine-tuned version achieves SDR ~9.2--10.5 dB on MUSDB18-HQ. Code: [github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs).

#### Band-Split Architecture Family (2023--)
- **BandSplit RNN (BSRNN) (Luo et al., "Music Source Separation with Band-Split RNN," ICASSP 2023, [arXiv 2209.15174](https://arxiv.org/abs/2209.15174)).**
  - Divides the frequency spectrum into non-overlapping bands. Each band processed by a shared RNN. Band-level features are then combined.
  - Won the URGENT 2025 Challenge.
  - Key insight: Processing frequency bands independently before combining captures both local spectral patterns and global structure.

- **BS-RoFormer (Band-Split RoPE Transformer) (current SOTA).**
  - Replaces BSRNN's RNN modules with Rotary Position Embedding (RoPE) attention / Transformer layers.
  - **Current overall SOTA on MUSDB18-HQ**: SDR of **~12.0 dB (median)** and **~13.3 dB (mean)** with the L=12 configuration.
  - Won relevant competitions and leads all public leaderboards.

- **Band-SCNet (Interspeech 2025).** Causal, lightweight model achieving SDR of 7.79 dB in real-time scenarios.

#### Other Notable Models
- **SCNet:** Sparse compression network. SDR ~9.0--9.7 dB on MUSDB18-HQ.

### Evaluation Metrics (BSS Eval Framework)

Defined by **Vincent et al., "Performance Measurement in Blind Audio Source Separation," IEEE Trans. Audio, 2006.** Implemented in `museval` ([github.com/sigsep/sigsep-mus-eval](https://github.com/sigsep/sigsep-mus-eval)).

| Metric | Full Name | What It Measures |
|--------|-----------|-----------------|
| **SDR** | Source-to-Distortion Ratio | Overall separation quality (global). Higher = better. Unit: dB. |
| **SIR** | Source-to-Interference Ratio | How well other sources are suppressed. Measures cross-talk / bleed from other instruments. |
| **SAR** | Source-to-Artifact Ratio | Level of algorithmic artifacts (metallic sounds, musical noise, warbling). |
| **ISR** | Image-to-Spatial Distortion Ratio | Spatial fidelity of the separated source image (for stereo). |

SDR is the primary metric. Typical values on MUSDB18-HQ: SOTA models achieve ~9--12 dB SDR overall; vocals tend to be easiest (SDR ~10--14 dB), bass hardest (SDR ~6--9 dB).

### Representative Datasets

| Dataset | Size | Stems | Notes |
|---------|------|-------|-------|
| **MUSDB18** | 150 full songs | 4 stems (vocals, drums, bass, other) | Standard benchmark. Rafii et al., 2017. 50 dev + 100 test. |
| **MUSDB18-HQ** | Same 150 songs at 44.1 kHz | Same | Higher quality version. Standard for reporting SDR numbers. |
| **MoisesDB** | Larger multi-stem dataset | Variable stems | Emerging as a next-generation benchmark. |
| **Slakh2100** | 2,100 synthesized mixtures | Multi-track MIDI synthesis | Manilow et al., 2019. Used for training/eval in academic settings. |

### SOTA Performance Snapshot (MUSDB18-HQ, Overall SDR dB)

| Model | Median SDR | Mean SDR | Notes |
|-------|-----------|----------|-------|
| **BS-RoFormer (L=12)** | ~12.0 | ~13.3 | Current SOTA |
| **BS-RoFormer (L=6)** | ~9.8 | ~11.3 | |
| **HT Demucs (fine-tuned)** | ~9.2 | ~10.5 | Strong, widely used baseline |
| **BSRNN** | Comparable to HT Demucs | | URGENT 2025 winner |
| **Spleeter** | ~5--6 | | Fast but lower quality |

### Open Problems

- **Generalization to real-world audio:** Models trained on MUSDB18 (mostly Western pop/rock) degrade on non-Western music, classical, electronic, etc.
- **More than 4 stems:** Separating individual instruments within "other" (e.g., separating two guitars). Open-unmix scenario.
- **Real-time / low-latency separation:** Band-SCNet (7.79 dB SDR) shows the quality gap vs. offline models.
- **Artifact perception:** SDR does not fully capture perceptual quality. Some models with lower SDR sound better to humans. Research into better metrics is ongoing (see "SDR -- Half-Baked or Well Done?", MERL, 2019, and the "Musical Source Separation Bake-Off" paper, 2025).
- **Singing voice separation with lyrics conditioning:** Using language to guide separation.

---

## 4. Music Emotion Recognition (MER)

### Problem Definition

Predicting the emotional content of music, either as categorical labels (happy, sad, angry, relaxed, etc.) or as continuous values on the valence-arousal (V-A) circumplex model (Russell, 1980).

### Approaches

#### Dimensional (Valence-Arousal Regression)
- Regress continuous valence (positive/negative) and arousal (calm/energetic) scores from audio features.
- Typically uses CNN, LSTM, or Transformer on mel-spectrograms or learned audio representations.
- Evaluated using MSE, Pearson correlation (r), or R-squared.
- Typical Pearson r: 0.3--0.7 for valence (harder), 0.4--0.8 for arousal (easier), depending on model and dataset.

#### Categorical (Emotion Class Classification)
- Classify into discrete emotion categories (e.g., 4-class: happy, sad, angry, relaxed).
- Higher accuracies achievable than fine-grained dimensional prediction.
- Approaches: CNN classifiers on spectrograms, often with transfer learning from music tagging models.

#### Multi-modal (Audio + Lyrics + Metadata)
- **BEE-MER (SMC 2025):** Bimodal Embeddings Ensemble combining audio and lyrics representations for static MER.
- **Music2Emo** ([huggingface.co/amaai-lab/music2emo](https://huggingface.co/amaai-lab/music2emo)): Unified multi-task framework integrating both categorical and dimensional labels.
- Text (lyrics) provides semantic content; audio provides acoustic expression. Combining both improves performance.

#### Transformer-based MER
- **Transformer Encoder approaches (ACM 2025):** Directly apply Transformer encoders to music features, mapping to emotional states.
- **Semi-supervised Multi-Task MER (TISMIR 2025):** Leverages large amounts of unlabeled music with weak emotion labels; multi-task learning over categorical and dimensional targets.

### Representative Datasets

| Dataset | Size | Annotations | Notes |
|---------|------|-------------|-------|
| **DEAM** | ~1,800 songs | Continuous V-A per-second (dynamic) | Aljanaki et al., 2017. Crowd-sourced via 2D emotion plane. Standard benchmark for dimensional MER. |
| **PMEmo** | ~794 songs | Static + dynamic V-A; includes physiological signals (EEG, ECG, GSR) | Zhang et al., 2018. Richer annotations. |
| **Emotify** | 400 tracks | 8 categorical emotions | Crowd-sourced categorical labels. |
| **MER-Arena** | Emerging (2025) | Preference-based comparisons | New evaluation paradigm. |

### Open Problems

- **Subjectivity:** Emotion perception is highly personal and culturally dependent. Inter-annotator agreement is low.
- **Cultural bias:** Models trained on Western listeners' annotations do not transfer to other cultural contexts. See research on cultural bias in MER (JCBI, 2025).
- **Dynamic vs. static emotion:** Most models predict per-song emotion, but real music has time-varying emotional arcs. Dynamic MER (per-second prediction) is harder and less standardized.
- **Beyond valence-arousal:** The 2D V-A model captures only part of emotional experience. More nuanced models (e.g., 13 emotions in Hevner's model, or continuous multi-dimensional spaces) are under-explored.
- **Ground truth quality:** Crowd-sourced emotion annotations are noisy and reflect the annotators' cultural background, musical training, and listening context.

---

## 5. Foundation Models for Music

### Overview

The paradigm shift from task-specific models to large-scale pre-trained models that learn general music audio representations, which can then be fine-tuned for downstream tasks. Analogous to BERT/GPT for NLP.

### Key Models

#### MERT (Music Audio Representation with Transformer)
- **Li, Yuan et al., "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training," ICLR 2023 ([arXiv 2306.00107](https://arxiv.org/abs/2306.00107)).**
- Self-supervised pre-training on ~160K music tracks. Uses masked audio modeling: mask portions of the audio, train the Transformer to predict masked regions.
- Incorporates teacher models (distillation from audio models like HuBERT) to provide pseudo-labels for more stable pre-training.
- Available in 95M and 330M parameter variants.
- Evaluated on MIR benchmarks: achieves SOTA on many tasks including instrument classification, genre classification, and music tagging when fine-tuned.
- Code: [github.com/yizhilll/MERT](https://github.com/yizhilll/MERT).
- **CultureMERT (ISMIR 2025):** Continual pre-training of MERT for cross-cultural music understanding, addressing bias toward Western music.

#### MusicFM
- **Won et al., "MusicFM: A Foundation Model for Music Informatics," arXiv 2311.03318, 2023.**
- Self-supervised foundation model specifically designed for music informatics. Pre-trained on large-scale music data.
- Addresses data scarcity and generalization challenges in MIR.
- Consistently reports relatively low performance on key detection (a known hard task for audio SSL models).
- Code: [github.com/minzwon/musicfm](https://github.com/minzwon/musicfm).

#### JukeMIR
- Uses representations extracted from OpenAI's **Jukebox** (Dhariwal et al., 2020) music generation model for downstream MIR tasks.
- Jukebox is a hierarchical VQ-VAE trained on 1.2M songs. Its internal representations encode rich musical structure.
- **Castellon et al., "Codified Audio Audio-Driven MIR," 2021.** Showed Jukebox representations are useful for emotion, genre, and tag prediction.

#### MuQ
- A newer self-supervised music audio model that has been compared alongside MusicFM and MERT in benchmarking studies.

#### SoniDo
- **"Music Foundation Model as Generic Booster," OpenReview 2024.** Proposes SoniDo as a new foundation model for audio, characterizing its encoding capabilities across downstream music tasks.

#### LLark (Spotify)
- **"LLark: A Multimodal Foundation Model for Music," Spotify Research, 2023 ([research.atspotify.com](https://research.atspotify.com/2023/10/llark-a-multimodal-foundation-model-for-music)).**
- A multimodal language model combining audio and text for flexible music understanding and reasoning.
- Can answer questions about music content, perform tagging, and describe musical features in natural language.

#### Qwen-Audio
- Part of the Qwen family of models. Trained on 30+ diverse audio tasks including classification, speech recognition, and emotion recognition. Not music-specific but applicable.

### Benchmarking: MARBLE
- **"MARBLE: Music Audio Representation Benchmark for Evaluation," 2024.**
- Comprehensive benchmark for evaluating music audio representations across a wide range of MIR tasks: tagging, instrument recognition, genre, mood, pitch, beat tracking, source separation, key detection, segmentation.
- Enables fair comparison of MERT, MusicFM, JukeMIR, CLAP, and other models.

### Multi-Modal Approaches

| Approach | Modalities | Key Idea |
|----------|-----------|----------|
| **CLAP** | Audio + Text | Contrastive alignment. Zero-shot classification via text prompts. |
| **CLaMP 3** | Audio + MIDI + Sheet Music + Text + Images | Universal cross-modal MIR. Contrastive pre-training across all modalities. |
| **LLark** | Audio + Text | LLM-based music reasoning and Q&A. |
| **MusicFM** | Audio only (with optional text) | Self-supervised audio representations. |
| **MERT** | Audio only | Self-supervised audio representations via masked modeling. |
| **Audio+Video** | Audio + Visual | Music video understanding, performance analysis. Less developed than audio+text. |

### What Foundation Models Enable

- **Linear probing:** Freeze the pre-trained encoder, train a linear classifier on top. Tests representation quality.
- **Fine-tuning:** Update all parameters on a downstream task. Usually gives best performance.
- **Zero-shot / few-shot:** Use CLAP-style models for classification without any task-specific training data.
- **Feature extraction:** Use representations as input features for downstream models.

### Open Problems

- **Key detection remains hard:** All foundation models (MERT, MusicFM, JukeMIR) report relatively low performance on key detection, suggesting the representations do not capture tonal structure well.
- **Temporal resolution:** Foundation models typically operate at frame rates of 50--75 Hz, which may be too coarse for fine-grained rhythm tasks.
- **Compute cost:** Pre-training requires significant GPU resources (weeks on 8--32 GPUs).
- **Evaluation methodology:** MARBLE is a step forward, but evaluation on diverse musical traditions and real-world scenarios is still limited.
- **Data contamination:** Pre-training datasets are often opaque; it is hard to verify no overlap with downstream evaluation data.

---

## 6. Beat/Tempo Tracking and Chord/Key Recognition

### Beat and Tempo Tracking

#### Problem Definition
- **Beat tracking:** Detect the times of musical beats (the pulse a listener would tap to).
- **Downbeat tracking:** Detect the times of downbeats (the first beat of each measure).
- **Tempo estimation:** Estimate the BPM (beats per minute) of a recording.

#### Key Models and Evolution

- **Ellis, "Beat Tracking by Dynamic Programming," JNMR 2007.** Classical approach using onset detection functions and dynamic programming. Fast but limited by hand-crafted features.

- **Bock & Davies, "Temporal Convolutional Networks for Musical Audio Beat Tracking," ISMIR 2020.**
  - Architecture: Temporal Convolutional Network (TCN) with dilated convolutions. Bi-directional processing.
  - Processes mel-spectrograms directly. Outputs beat/downbeat activation functions.
  - Typically followed by a Dynamic Bayesian Network (DBN) post-processing step for final beat times.
  - Achieved F-measure >85% on most standard benchmarks.

- **Beat Transformer (Zhao et al., "Beat Transformer: Dilated Self-Attention for Joint Beat and Downbeat Tracking," ISMIR 2022).**
  - Dilated self-attention mechanism allowing the model to attend to both local and long-range temporal patterns.
  - Joint beat and downbeat tracking in a single model.
  - Removed the need for a separate DBN post-processing step in some configurations.

- **"Beat This!" (ISMIR 2024, [github.com/CPJKU/beat_this](https://github.com/CPJKU/beat_this)).**
  - Highly accurate beat tracker that eliminates the need for DBN post-processing entirely.
  - Current go-to model for beat tracking. Clean, efficient implementation.
  - Achieves strong F-measure across standard benchmarks.

- **Dual-Path TCN+Transformer (2024).** Combines TCNs (local temporal detail) with Transformers (global sequence modeling). Reduces model complexity while maintaining accuracy.

- **Beat-U (MIREX 2025).** Multi-task U-shape Transformer for music understanding across multiple timescales. Jointly addresses beat tracking, downbeat tracking, and related sequential MIR tasks.

- **End-to-End Transformer for Performance MIDI (SMC 2025).** Encoder-decoder Transformer for beat/downbeat tracking in MIDI (not audio) performances.

#### Key Datasets

| Dataset | Content | Notes |
|---------|---------|-------|
| **Ballroom** | 698 excerpts | Standard beat tracking benchmark. Dance music. |
| **Beatles** | 178 Beatles tracks | Annotated beats and chords. Widely used. |
| **Hainsworth** | 222 excerpts | Varied genres. |
| **SMC (Soleym, MIREX, CMU)** | 210+ excerpts | Includes difficult cases (rubato, expressive timing). |
| **GTZAN Rhythm** | 1,000 tracks | Tempo/beat annotations for GTZAN tracks. |
| **GiantSteps** | Electronic music tempo annotations | 660 tracks, primarily electronic dance music. |

#### Approximate SOTA Performance

- **Beat tracking F-measure (standard window, e.g., 70ms):** 85--92% on Ballroom; 80--88% on Beatles; 70--80% on SMC (harder set). "Beat This!" and TCN-based models lead.
- **Downbeat tracking F-measure:** Typically 5--10 percentage points lower than beat tracking on the same dataset.
- **Tempo estimation accuracy (within 4% of ground truth):** >90% on Ballroom; 80--85% on more diverse datasets.

### Chord Recognition (Automatic Chord Estimation, ACE)

#### Problem Definition
Identifying the chord label (e.g., C:maj, G:min7, F#:dim) at each time step in a recording.

#### Approaches

- **Chordino / NNLS Chroma (Mauch, 2010):** Classic approach using NNLS chroma features + HMM decoding. Still a useful baseline. Available as a Vamp plugin.

- **Deep Learning (2015--):** CNNs and CRNNs on chroma or spectrogram features. ISMIR 2015 papers (McLeod & Wyse, etc.) showed deep learning could match or exceed HMM-based approaches.

- **CNN-LSTM Hybrids:** CNNs extract local spectral features; LSTMs model temporal chord sequences. Achieved >80% weighted accuracy on standard benchmarks.

- **Transformer-based:** Self-attention captures long-range harmonic context (e.g., recognizing a chord based on the surrounding harmonic progression). Achieves weighted accuracy of 80--85%+ on Billboard and Isophonics datasets.

- **Foundation model features:** Using MERT, MusicFM, or JukeMIR representations as input features for chord recognition. Improves performance, especially on complex chords (7ths, diminished, augmented).

- **Training on artificially generated audio (2025, [arxiv.org/html/2508.05878v1](https://arxiv.org/html/2508.05878v1)):** Compares two Transformer models trained on synthesized audio for chord recognition, addressing the data scarcity problem.

#### Key Datasets

| Dataset | Content | Chord Vocabulary | Notes |
|---------|---------|-----------------|-------|
| **Billboard (McVicar et al.)** | ~200 pop/rock songs from Billboard Hot 100 | 24 maj/min + 7ths | Standard ACE benchmark. Harte's chord vocabulary. |
| **Isophonics** | Beatles, Queen, Carole King albums | maj, min, 7, maj7, min7, etc. | Widely used. |
| **Robbie Williams** | 55 Robbie Williams tracks | maj, min, 7, maj7, etc. | |
| **ChoTo** | Various | maj/min | Smaller benchmark. |

#### Approximate SOTA Performance

- **Weighted accuracy (Billboard, maj/min):** 80--87% for SOTA systems.
- **Weighted accuracy (Isophonics, maj/min):** 82--88%.
- **With extended chord vocabulary (7ths, etc.):** Performance drops significantly, to 60--75%.

### Key Detection

- Estimate the global key (e.g., C major, A minor) of a recording.
- **Classical approach:** Krumhansl-Schmuckler key-finding algorithm using key profiles.
- **Deep learning:** CNN classifiers on chroma features; foundation model probing.
- **Datasets:** GiantSteps Key (electronic music), meters.tsv (classical).
- **SOTA accuracy:** ~70--85% on standard datasets. Remains a challenging task, especially for foundation models (as noted above).

### Open Problems

- **Expressive timing:** Beat tracking degrades significantly on music with rubato, tempo changes, and expressive timing (classical, jazz).
- **Complex meters:** 5/4, 7/8, and irregular meters are poorly handled by most models trained on 4/4 pop music.
- **Hierarchical rhythm:** Modeling beat, downbeat, and higher-level rhythmic structure (phrase, section) jointly.
- **Chord vocabulary:** Most systems handle maj/min well but struggle with complex chords (9ths, 11ths, altered chords, slash chords).
- **Key detection:** Surprisingly, foundation models do not excel at this. Tonal center estimation may require specialized representations beyond general audio features.
- **Data annotation:** Chord labels are subjective (especially for ambiguous harmonies). Inter-annotator agreement sets a ceiling on achievable accuracy.

---

## Cross-Cutting Trends (2024--2026)

1. **Self-supervised pre-training dominates.** MERT, MusicFM, and related models show that large-scale SSL on music audio produces representations that transfer well to nearly all MIR tasks.

2. **Band-split architectures lead source separation.** BSRNN and BS-RoFormer represent the current frontier, with SDR ~12 dB on MUSDB18-HQ. The paradigm of processing frequency bands independently has proven highly effective.

3. **Transformers are replacing RNNs/CNNs across all tasks.** Beat tracking, chord recognition, source separation, and transcription are all moving toward Transformer-based architectures. The trend is away from hybrid models with HMM/DBN post-processing toward end-to-end neural approaches.

4. **Multi-modal foundation models (CLAP, CLaMP 3) enable zero-shot MIR.** Text-audio alignment allows classification without task-specific training data, opening MIR to open-vocabulary and cross-lingual settings.

5. **Evaluation methodology is evolving.** MARBLE (foundation model benchmarking), the 2025 AMT Challenge (multi-instrument transcription), and new expert-annotated tagging benchmarks are raising evaluation standards.

6. **Cultural bias is a growing concern.** Models trained on Western popular music do not generalize to other traditions. CultureMERT and similar efforts aim to address this.

7. **Data remains a bottleneck.** For transcription, source separation, and emotion recognition, high-quality labeled data is scarce and expensive to collect. Self-supervised, weakly-supervised, and synthetic data generation are key research directions.

---

## 7. Music Recommendation 音乐推荐

### Problem Definition

Recommending music to users based on audio content, user preferences, or context. Audio-based recommendation is a direct application of MIR — it requires extracting meaningful representations from audio and computing similarity.

### Approaches

#### Content-Based Filtering

| Method | Features | Notes |
|--------|----------|-------|
| Hand-crafted features | MFCC, chroma, spectral features + distance metrics | Classic approach, interpretable but limited |
| Embedding-based | Pre-trained embeddings (MERT, CLAP, MusicFM) + nearest neighbor | Current standard |
| Metric learning | Learn distance function from preference data | Better for fine-grained similarity |

#### Audio Embedding for Recommendation

Modern recommendation systems use pre-trained audio embeddings:

- **MERT embeddings**: Self-supervised music representations. Encode harmonic, rhythmic, and timbral information. Can be used directly for similarity search.
- **CLAP embeddings**: Audio-text joint space. Enables text-based music search ("find music like X with mood Y").
- **MusicFM embeddings**: Large-scale pre-trained representations with strong transfer to recommendation tasks.

**相似度计算**：Cosine similarity on embeddings is the standard. For large catalogs, approximate nearest neighbor (ANN) search (FAISS, ScaNN) is essential.

#### Hybrid Systems

| System | Content signal | Collaborative signal | Notes |
|--------|---------------|---------------------|-------|
| Spotify (internal) | Audio features + NLP | User listening history | Industry standard, proprietary |
| Spotify public API | Audio features (danceability, energy, valence) | — | Limited but accessible |
| Content-based only | MERT/CLAP embeddings | None | Works without user data |

### Evaluation

| Metric | Description |
|--------|-------------|
| **Precision@K** | % of top-K recommendations that are relevant |
| **Recall@K** | % of relevant items retrieved in top-K |
| **NDCG@K** | Normalized discounted cumulative gain — ranks relevant items higher |
| **Coverage** | % of catalog that can be recommended |
| **Serendipity** | Novelty of recommendations (not just popular items) |

### Open Problems

- **Cold start**: Recommending new tracks without listening history.
- **Long-tail**: Most recommendation data is concentrated on popular tracks; niche music is underserved.
- **Context-awareness**: Time of day, activity, mood as additional signals.
- **Cross-cultural**: Recommendation systems trained on Western music fail for non-Western listeners.

---

## 8. Cover Detection and Version Identification 翻唱检测与版本识别

### Problem Definition

Identifying when two audio recordings are different performances of the same underlying musical work. A "cover" is a new performance/arrangement of an existing song.

### Why It Matters

- **Music rights management**: Identifying unlicensed covers for royalty distribution.
- **Music discovery**: Finding different versions of songs a user likes.
- **Cultural analysis**: Studying how musical works evolve across performances.

### Approaches

#### Traditional Approaches (Pre-2020)

- **Chromaprint / AcoustID**: Chroma-based fingerprinting. Robust to tempo changes, key transposition. Still widely used.
- **Cover Song Identification (CSI) systems**: Chroma + dynamic time warping (DTW) for alignment-invariant comparison.

#### Deep Learning Approaches

| Method | Description |
|--------|-------------|
| **2D-CNN on chromagrams** | Learn chroma patterns robust to arrangement changes |
| **Siamese networks** | Learn similarity metric for cover pairs |
| **Triplet loss** | Train with anchor-positive (same work) and anchor-negative (different works) |
| **Self-supervised pretraining** | Pre-train on large audio corpora, fine-tune for cover detection |

### Datasets

| Dataset | Content | Size | Notes |
|---------|---------|------|-------|
| **SecondHandSongs** | Crowdsourced cover metadata | ~1M works | Largest metadata source |
| **Covers80** | 80 original + cover pairs | 160 tracks | Classic small benchmark |
| **Da-TACOS** | Cover + original pairs | ~17K | Large-scale benchmark |
| **Covers2001** | Query-cover pairs | ~1K queries | Standard evaluation set |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Mean Rank** | Average rank of the correct match |
| **mAP** | Mean average precision across queries |
| **MRR** | Mean reciprocal rank |
| **Top-K accuracy** | % of queries where correct cover is in top-K |

### Open Problems

- **Arrangement changes**: Same song with completely different instrumentation (orchestral → electronic).
- **Structural variations**: Covers may reorder sections, add/remove verses.
- **Medleys**: Multiple songs combined in one performance.
- **Humming/whistling queries**: Query by humming (QBH) — finding covers from a sung query.

---

## 9. Melody Extraction 主旋律提取

### Problem Definition

Extracting the dominant melody pitch contour from polyphonic music. Also called "vocal melody extraction" when focused on singing voice, or "pitch tracking" more broadly.

### Why It Matters

- **Downstream task**: Input for cover detection, query-by-humming, music transcription, singing voice analysis.
- **Music production**: Automatic melody isolation for remixing, karaoke track generation.
- **Music education**: Pitch visualization for learning.

### Approaches

#### Traditional Methods

- **SALAMI** + pitch tracking: Source separation + F0 estimation on separated melody source.
- **P. Rao (2010)**: Salience-based melody extraction using pitch salience functions from harmonic structure.

#### Deep Learning Methods

| Method | Architecture | Notes |
|--------|-------------|-------|
| **CNN-based** | CNN on mel-spectrogram predicting pitch per frame | Fast but limited temporal context |
| **CRNN (CNN + LSTM)** | CNN features + LSTM temporal modeling | Better temporal coherence |
| **Transformer-based** | Self-attention over spectrogram frames | State-of-art, captures long-range |
| **Segmentation-free** | End-to-end pitch tracking without note segmentation | Current trend |

### Key Systems

| System | Year | Notes |
|--------|------|-------|
| **Melodia** | 2013 | Classic pitch contour extraction, widely used |
| **DeepSalience** | 2018 | Deep learning for pitch salience |
| **MelodyCNN** | 2019 | CNN for melody extraction |
| **pYIN** | 2015 | Probabilistic YIN, standard baseline |

### Evaluation

| Metric | Description |
|--------|-------------|
| **RPA** (Raw Pitch Accuracy) | % of frames where estimated pitch is within 50 cents of ground truth |
| **RCA** (Raw Chroma Accuracy) | RPA ignoring octave errors |
| **OA** (Overall Accuracy) | RPA + correct voicing decisions |
| **VR** (Voicing Recall) | % of voiced frames correctly identified |
| **VF** (Voicing False Alarm) | % of unvoiced frames incorrectly labeled voiced |

Typical SOTA: RPA ~80–85% on MIREX datasets. Performance degrades on complex mixtures (orchestral, heavy polyphony).

### Open Problems

- **Melody in complex mixtures**: When melody is not the loudest source (e.g., soft vocal in full band).
- **Polyphonic melody**: Multiple melodic lines (e.g., counterpoint in Bach).
- **Non-vocal melody**: Instrumental melodies (saxophone, guitar lead) have different timbral characteristics.
- **Real-time extraction**: Low-latency melody extraction for interactive applications.

---

## 10. Cross-Cutting MIR Challenges 跨领域 MIR 挑战

### 10.1 Cultural Bias in MIR

All major MIR benchmarks (GTZAN, MAESTRO, MUSDB18) are dominated by Western popular and classical music. Consequences:

- **Poor generalization**: Models trained on Western music fail on non-Western traditions.
- **Measurement bias**: Evaluation results are not representative of global musical diversity.
- **Reinforcement loop**: AI-generated music trained on Western data perpetuates Western idioms.

**Efforts to address**: CultureMERT (multilingual/cultural pre-training), diverse benchmark collections, community-led dataset creation.

### 10.2 The Evaluation Bottleneck

MIR evaluation is bottlenecked by:

- **Label quality**: Crowd-sourced labels (MagnaTagATune) are noisy. Expert annotations are expensive.
- **Dataset size**: Many MIR datasets are small (hundreds to thousands of examples), limiting model capacity.
- **Benchmark saturation**: Some tasks (genre classification on GTZAN, piano transcription on MAESTRO) approach ceiling performance, reducing research incentive.
- **Lack of standardization**: Different papers use different train/test splits, metrics, and preprocessing, making comparison difficult.

### 10.3 From Lab to Production

| Challenge | Description |
|-----------|-------------|
| **Real-time requirements** | Production systems need low latency (<100ms for some applications) |
| **Robustness** | Must handle noisy recordings, phone microphones, compressed audio |
| **Scalability** | Billions of tracks to process; efficient inference essential |
| **Continuous learning** | New music styles, languages, and artists emerge continuously |
| **Explainability** | Why did the model tag this as "jazz"? Needed for trust and debugging |

---

## 11. Summary: MIR Task Landscape MIR 任务全景

| Task | Input | Output | Current SOTA | Open challenge |
|------|-------|--------|-------------|----------------|
| Auto-tagging | Audio | Tags | ~0.93 ROC-AUC | Fine-grained, cross-cultural |
| Transcription | Audio | MIDI notes | 93–97% F1 (piano) | Multi-instrument realistic mixes |
| Source separation | Mixed audio | Isolated stems | ~12 dB SDR | >4 stems, real-time |
| Emotion recognition | Audio | Arousal/valence | Moderate | Standardized evaluation |
| Beat tracking | Audio | Beat times | 85–92% F1 | Rubato, complex meters |
| Chord recognition | Audio | Chord labels | 80–87% | Complex chords, ambiguous harmony |
| Key detection | Audio | Key label | 70–85% | Foundation model weakness |
| Recommendation | Audio + user | Track list | Industry practice | Cold start, long-tail |
| Cover detection | Audio pair | Match/none | Moderate | Arrangement changes |
| Melody extraction | Audio | Pitch contour | 80–85% RPA | Complex mixtures |

---

> This document covers music understanding and MIR as of 2025--2026. For related topics, see [music-generation.md](music-generation.md) (generation), [audio-engineering.md](audio-engineering.md) (signal processing), [music-theory-fundamentals.md](music-theory-fundamentals.md) (theoretical foundations), and [music-styles.md](music-styles.md) (style analysis).


### Auto-Tagging
- Choi et al., "Automatic Tagging Using Deep Convolutional Neural Networks," ISMIR 2016.
- Won, Chun, Nieto, Serra, "Semi-Supervised Music Tagging Transformer," ISMIR 2021.
- Wu et al., "Large-scale Contrastive Language-Audio Pretraining" (CLAP), arXiv 2211.06687, 2022.
- Wu et al., "CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages," ACL 2025 Findings.

### Transcription
- Hawthorne et al., "Onsets and Frames: Dual-Objective Piano Transcription," ISMIR 2018.
- Kong et al., "High-Resolution Piano Transcription," 2020--2021.
- Gardner et al., "MT3: Multi-Task Multitrack Music Transcription," ICLR 2022 (arXiv 2111.03017).
- "Advancing Multi-Instrument Music Transcription: Results from the 2025 AMT Challenge," NeurIPS 2025 (arXiv 2603.27528).

### Source Separation
- Henaff et al., "Spleeter," ISMIR 2019.
- Defossez et al., "Music Source Separation in the Waveform Domain" (Demucs), arXiv 1911.13254, 2019.
- Luo et al., "Music Source Separation with Band-Split RNN" (BSRNN), ICASSP 2023 (arXiv 2209.15174).
- Rouard & Massa, "Hybrid Transformers for Music Source Separation" (HT Demucs), ICASSP 2023.
- Vincent et al., "Performance Measurement in Blind Audio Source Separation" (BSS Eval), IEEE Trans. Audio, 2006.

### Music Emotion
- Russell, "A Circumplex Model of Affect," J. Personality & Social Psychology, 1980.
- Aljanaki et al., "DEAM: Developing an Emotion Annotation Dataset for Music," ISMIR 2017.
- Zhang et al., "PMEmo: A Multimodal Dataset for Perceived Emotion Recognition," ICASSP 2018.
- "BEE-MER: Bimodal Embeddings Ensemble for Music Emotion Recognition," SMC 2025.

### Foundation Models
- Li, Yuan et al., "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training," ICLR 2023 (arXiv 2306.00107).
- Won et al., "MusicFM: A Foundation Model for Music Informatics," arXiv 2311.03318, 2023.
- Castellon et al., "Codified Audio Audio-Driven MIR" (JukeMIR), 2021.
- "MARBLE: Music Audio Representation Benchmark for Evaluation," 2024.
- "Foundation Models for Music: A Survey," arXiv 2408.14340, 2024.

### Beat/Tempo/Chord
- Ellis, "Beat Tracking by Dynamic Programming," JNMR 2007.
- Bock & Davies, "Temporal Convolutional Networks for Musical Audio Beat Tracking," ISMIR 2020.
- Zhao et al., "Beat Transformer: Dilated Self-Attention for Joint Beat and Downbeat Tracking," ISMIR 2022.
- "Beat This!" ISMIR 2024 ([github.com/CPJKU/beat_this](https://github.com/CPJKU/beat_this)).
- Harte et al., "Symbolic Representation of Musical Chords," ISMIR 2005 (chord vocabulary).
- Mauch, "Automatic Chord Transcription from Audio," PhD Thesis, 2010 (Chordino).
