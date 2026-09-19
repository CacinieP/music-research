# Music Understanding / Music Information Retrieval (MIR)

> 中文版：[music-understanding-mir-zh.md](music-understanding-mir-zh.md)

Methods, datasets, evaluation definitions, and open problems. Content checked on **2026-09-19** against primary papers, dataset releases, and evaluation documentation. Published examples are identified by their experimental setting; this page does not assert an exhaustive current SOTA ranking.

---

## 1. Auto-Tagging and Classification

### Problem Definition

Assign labels such as genre, instrumentation, mood, or acoustic attributes to recordings. Tagging is often multi-label; a fixed-genre benchmark may instead be multiclass. The label vocabulary is a property of the dataset and task, not a universal 50–100-tag limit.

### Representative Architectures

- **CNNs:** operate on time-frequency features and pool local patterns into clip-level predictions. Convolutional and recurrent variants remain useful baselines; larger receptive fields can also capture long-range context.
- **Music Tagging Transformer** (Won, Choi, Serra, ISMIR 2021): a convolutional frontend followed by temporal self-attention, with noisy-student semi-supervised training. The authors also introduce an artist-disjoint Million Song Dataset split. This is not simply a patch-only vision transformer. [Paper](https://arxiv.org/abs/2111.13457), [official implementation](https://github.com/minzwon/semi-supervised-music-tagging-transformer)
- **LAION-CLAP** (Wu et al., 2022 preprint / ICASSP 2023): dual audio/text encoders learn contrastive alignment, enabling retrieval and prompted zero-shot classification. Its LAION-Audio-630K collection is general audio, not exclusively music. [Paper](https://arxiv.org/abs/2211.06687)
- **CLaMP 3** (Wu et al., Findings of ACL 2025): aligns symbolic sheet music, performance representations, audio, and multilingual text, using text to bridge unaligned modalities. Its modality claims should not be extended to arbitrary images. [Paper](https://aclanthology.org/2025.findings-acl.133/)

### Datasets and Protocols

| Dataset | Scope | Evaluation caveat |
|---|---|---|
| MagnaTagATune | About 25.9K short music excerpts; top-50 tags are a common benchmark subset | The original annotations have a larger vocabulary; specify filtering and split |
| MTG-Jamendo | Cleaned base: 55,609 tracks, 195 tags | Published splits retain 55,525 tracks and 183 tags; genre/instrument/mood and top-50 subsets differ |
| GTZAN | 1,000 excerpts, ten genre labels | Repetitions, artist overlap, and label issues affect evaluation |
| FMA | 106,574 tracks in the full release | Small/medium/large/full differ in duration and labels; do not conflate them |
| NSynth | 305,979 isolated notes, 11 instrument families | Family, instrument identity, and sound-source classification are different tasks |

Sources: [MagnaTagATune](https://mirg.city.ac.uk/datasets/magnatagatune/index1.html), [MTG-Jamendo release](https://github.com/MTG/mtg-jamendo-dataset), [FMA release](https://github.com/mdeff/fma), [NSynth release](https://magenta.tensorflow.org/datasets/nsynth).

Report ROC-AUC and PR-AUC/average precision with their exact averaging convention. With rare positive tags, ROC-AUC alone can hide poor retrieval precision. Fix the tag subset, artist split, clip duration, and pretraining-data policy before comparing systems. An isolated high GTZAN accuracy does not establish that genre recognition is solved. [GTZAN dataset audit](https://arxiv.org/abs/1306.1461)

### Open Problems

Noisy or missing labels, culturally specific genre taxonomies, long recordings, rare instruments, and distribution shift. Evaluate language-prompt sensitivity and unseen genres separately from supervised tagging.

---

## 2. Music Transcription

### Problem Definition

Convert audio into symbolic events: pitch, onset, offset, and optionally velocity, instrument, and pedal events. A pitch contour, a piano roll, and a complete MIDI transcription are different outputs.

### Piano and Multi-Instrument Models

| Model | Main idea | Reference |
|---|---|---|
| Onsets and Frames (Hawthorne et al., ISMIR 2018) | Joint onset and frame prediction; onset detections constrain when notes may begin | [Paper](https://arxiv.org/abs/1710.11153) |
| High-resolution piano transcription (Kong et al., 2020 preprint / TASLP 2021) | Regress onset/offset timing and decode fine-grained note and pedal events | [Paper](https://arxiv.org/abs/2010.01815) |
| MT3 (Gardner et al., ICLR 2022) | T5-style encoder-decoder maps spectrogram segments to event-token sequences; trained jointly across transcription datasets | [Paper](https://arxiv.org/abs/2111.03017), [code](https://github.com/magenta/mt3) |

MT3 represents timing, pitches, instruments, and note state through separate token types; a note is not one token containing every attribute. Its multi-instrument setup does not recover full expressive velocity merely because the event vocabulary can contain velocity-related state.

The **2025 AMT Challenge** report was accepted to the **AI for Music Workshop at NeurIPS 2025**, and posted to arXiv in March 2026. It reports eight valid teams, two improving on the MT3 baseline, with remaining polyphony and timbre challenges. It is not a NeurIPS main-track benchmark paper. [Challenge report](https://arxiv.org/abs/2603.27528)

Drum transcription predicts hit times and drum classes. ENST-Drums and similar percussion datasets have different label mappings and recording conditions; compare at a shared drum vocabulary and onset tolerance.

### Representative Datasets

| Dataset | Content | Important distinction |
|---|---|---|
| MAPS | Piano recordings and synthesized piano with aligned symbolic labels | Split by recording condition and musical content |
| MAESTRO | Around 200 hours of piano performance with tightly aligned MIDI | Versions differ; v3 removes six recordings with string accompaniment from v2 |
| MusicNet | 330 classical ensemble recordings | Instrument/note labels with alignment uncertainty |
| Slakh2100 | 2,100 synthesized multitrack mixtures | Synthesized audio is not equivalent to real ensemble recording |
| URMP | 44 small-ensemble performances | Isolated/combined performances with instrument annotations |

Sources: [MAESTRO release](https://magenta.tensorflow.org/datasets/maestro), [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html), [Slakh](http://www.slakh.com/), [URMP](https://labsites.rochester.edu/air/projects/URMP.html).

### Evaluation: Do Not Mix F1 Definitions

- **Frame F1:** compares framewise active pitches.
- **Note F1 without offsets:** matches pitch and onset.
- **Note F1 with offsets:** also requires an acceptable end time.
- **Velocity/instrument-aware scores:** add further conditions and must be named explicitly.

In `mir_eval.transcription`, default note matching uses onset tolerance 50 ms and pitch tolerance 50 cents. With offset scoring enabled, offset tolerance is `max(50 ms, 20% of reference-note duration)`. Set `offset_ratio=None` to ignore offsets. For the same predictions and matching protocol, adding an offset requirement cannot increase F1. [Evaluation documentation](https://mir-eval.readthedocs.io/latest/api/transcription.html)

As a **historical example**, Kong et al. report **96.72% onset F1** on their MAESTRO evaluation. This is not 96.72% note F1 with offsets, and not a universal current piano-transcription score. Their paper supplies the dataset version and comparison protocol. [Paper](https://arxiv.org/abs/2010.01815)

### Open Problems

Real ensemble mixtures, overlapping harmonics, vocals, expressive velocity and articulation, pedal effects, annotation alignment, and transfer between recording conditions. High piano benchmark scores do not establish equivalent multi-instrument performance.

---

## 3. Source Separation

### Problem Definition

Estimate source waveforms from a mixture. MUSDB18 standardizes four stems: vocals, drums, bass, and **other**. “Other” can contain multiple instruments; four-stem separation is not complete instrument isolation.

### Representative Models

- **Spleeter** (Hennequin, Khlif, Voituret, Moussallam; JOSS 2020, software released in 2019): U-Net-style separation on **linear-frequency STFT magnitudes**, with masks and waveform reconstruction using mixture phase. It does not operate on mel-spectrograms. Throughput claims depend on hardware and configuration. [Paper](https://joss.theoj.org/papers/10.21105/joss.02154), [code](https://github.com/deezer/spleeter)
- **Demucs / Hybrid Demucs:** waveform encoder-decoder models, later combined with a spectral branch. [Official repository](https://github.com/facebookresearch/demucs)
- **HT Demucs** (Rouard, Massa, Défossez; ICASSP 2023): temporal/spectral branches communicate through self-attention and cross-attention. [Paper](https://arxiv.org/abs/2211.08553)
- **BSRNN** (Luo, Yu; 2022 preprint / TASLP 2023): divides the spectrum into subbands and alternates sequence-level and band-level recurrent modeling. It is not just independent RNNs with a final concatenation. [Paper](https://arxiv.org/abs/2209.15174)
- **BS-RoFormer** (Lu, Wang, Kong, Hung; 2023 preprint): band-split features, hierarchical attention, rotary position embeddings, and complex mask estimation. Its reported competition win is the music-separation track of **SDX23**, not an URGENT 2025 win by the original BSRNN paper. [Paper](https://arxiv.org/abs/2309.02612)

### Datasets

| Dataset | Release facts | Use |
|---|---|---|
| MUSDB18 | 150 songs: **100 train, 50 test**; stereo, 44.1 kHz | Four-stem benchmark; original compressed release |
| MUSDB18-HQ | Same tracks/split, uncompressed WAV sources | Avoid conflating HQ training/evaluation with compressed-release preprocessing |
| MoisesDB | Multitrack recordings with a more detailed instrument hierarchy | Finer source definitions than four fixed stems |
| Slakh2100 | Synthesized multitrack mixtures | Controlled mixing and joint symbolic tasks |

The training set can be divided again into training/validation; those validation tracks do not replace the 50-song test set. [MUSDB18 official description](https://sigsep.github.io/datasets/musdb.html), [MoisesDB](https://github.com/moises-ai/moises-db)

### Metrics and Published Examples

BSS Eval decomposes estimation error to compute SDR (signal-to-distortion), SIR (signal-to-interference), SAR (signal-to-artifacts), and, for source images, ISR (image-to-spatial distortion). SI-SDR, whole-track SDR, and framewise BSS Eval SDR are different metrics. Specify implementation/version, allowed filtering, windowing, silent-reference handling, track aggregation, and source averaging. [museval](https://github.com/sigsep/sigsep-mus-eval)

| Published system | Reported result | Setting |
|---|---|---|
| HT Demucs | 9.20 dB SDR | Paper's sparse-attention/per-source-fine-tuned setup with 800 extra training songs |
| Smaller BS-RoFormer | 9.80 dB average SDR | Paper's MUSDB18-HQ benchmark without extra training data |

These figures are reported examples, not a matched-data ranking. Trackwise means, medians, and source averages are not interchangeable; comparisons must retain the original aggregation protocol. [HT Demucs paper](https://arxiv.org/abs/2211.08553), [BS-RoFormer paper](https://arxiv.org/abs/2309.02612)

### Open Problems

Separate similar instruments, handle unseen arrangements, preserve stereo phase and transients, reduce artifacts, and meet streaming latency constraints. Perceptual quality is not fully described by SDR, and the hardest stem depends on the model/data rather than a universal “bass is hardest” rule. “Open-unmix” names a particular baseline; it is not the term for arbitrary-source separation.

---

## 4. Music Emotion Recognition (MER)

### Tasks and Methods

Distinguish **perceived emotion in the music** from **emotion induced in a listener**. Targets may be categorical labels or continuous valence (pleasantness) and arousal (activation), at track or time-varying level. Listener context and annotation instructions determine what the model is learning.

CNNs, recurrent models, Transformers, and pretrained embeddings can drive classification or regression. Lyrics and metadata may add information, but multimodal gains require comparison on the same split; they are not guaranteed. Categorical accuracy and dimensional correlation measure different tasks and cannot be ranked directly.

### Representative Datasets

| Dataset | Content | Annotation scope |
|---|---|---|
| DEAM | 1,802 excerpts/full songs | Static and continuous valence/arousal annotations |
| PMEmo | 794 songs with selected chorus excerpts | Static/dynamic valence-arousal and **electrodermal activity (EDA)**; not a general EEG/ECG dataset |
| Emotify | 400 one-minute excerpts across four genres | Nine GEMS emotion categories; induced-emotion annotations |

Sources: [DEAM — Aljanaki, Yang, Soleymani, PLOS ONE 2017](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0173392), [PMEmo official dataset description](https://www.next.zju.edu.cn/cn/archive/pmemo/), [Emotify official annotations](https://www2.projects.science.uu.nl/memotion/emotifydata/). PMEmo's dataset paper is **ICMR 2018**, not ICASSP 2018.

### Evaluation and Limitations

Report MSE/MAE, Pearson correlation, or concordance correlation for dimensional tasks, and class-aware metrics for categorical tasks. Match label scaling, temporal alignment, annotator aggregation, and split policy. Avoid splitting adjacent segments from one song between train and test.

Disagreement can reflect real differences in experience, rather than annotation error alone. Valence-arousal is not exhaustive; GEMS provides a music-oriented alternative. Hevner's historical adjective-circle work should not be described as a “13-emotion model.” No cross-dataset universal correlation range is claimed.

---

## 5. Foundation Models for Music

### Model Families

| Model | Verified method/scope | Reference |
|---|---|---|
| MERT | Li et al., ICLR **2024**; masked pretraining with RVQ-VAE acoustic and CQT musical teachers; original 95M/330M variants | [Paper](https://arxiv.org/abs/2306.00107) |
| MusicFM | Won, Hung, Le; *A Foundation Model for Music Informatics*, 2023 preprint; self-supervised music-audio representation learning | [Paper](https://arxiv.org/abs/2311.03318) |
| JukeMIR | Castellon, Donahue, Liang, ISMIR 2021; investigates representations extracted from Jukebox | [Codified Audio Language Modeling Learns Useful Representations for MIR](https://arxiv.org/abs/2107.05677) |
| CLAP | Audio/text contrastive alignment; general audio rather than exclusively musical training | [Paper](https://arxiv.org/abs/2211.06687) |
| CLaMP 3 | Shared cross-modal/cross-lingual music representation; text bridges modalities | [Paper](https://aclanthology.org/2025.findings-acl.133/) |

Audio-only pretraining does not itself enable text-based zero-shot classification. MERT's selected teachers are not simply HuBERT distillation, and MusicFM should not be labeled “audio with optional text” without a specific multimodal extension.

### Evaluation Modes

- **Linear probe:** freeze the encoder and train a linear head.
- **Frozen-feature downstream model:** train a potentially nonlinear task head; this is not necessarily a linear probe.
- **Fine-tuning:** update some or all pretrained parameters, with an explicit data/compute budget.
- **Zero-shot:** use a specified decision rule, prompts, and candidate labels without task training; prompt selection can still leak test information.

**MARBLE** is Yuan et al.'s *Music Audio Representation Benchmark for Universal Evaluation*, **NeurIPS 2023**. Its conference version includes **18 tasks on 12 datasets** and multiple evaluation regimes. Earlier preprint counts differ; cite the version. It covers sequence tasks as well as clip-level understanding. [Conference paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7cbeec46f979618beafb4f46d8f39f36-Paper-Datasets_and_Benchmarks.pdf)

### Open Problems

Pretraining overlap with evaluation recordings, domain shift, prompt/language coverage, compute cost, and temporal resolution. Frame rates and strengths vary by checkpoint and chosen layer. A weak key-detection result on one benchmark does not prove that every foundation model fails to represent tonality. Fine-tuning can overfit small datasets and is not guaranteed to beat frozen features.

---

## 6. Beat/Tempo Tracking and Chord/Key Recognition

### Beat, Downbeat, and Tempo

Beat tracking predicts pulse times; downbeat tracking identifies the first beat of a bar; tempo estimation predicts a rate. They are related but distinct: a correct global BPM does not ensure correct beat phase or changing-tempo tracking.

Classical onset-strength/dynamic-programming methods coexist with CNN/TCN/RNN/Transformer activation models, often followed by a dynamic Bayesian network (DBN). **Beat This!** (Foscarin, Schlüter, Widmer; ISMIR 2024) combines convolutions and Transformers without DBN postprocessing. Its paper reports stronger F1 but weaker continuity metrics in some comparisons, illustrating why one score is insufficient. [Paper](https://arxiv.org/abs/2407.21658), [code and dataset configurations](https://github.com/CPJKU/beat_this)

Common evaluation collections include Ballroom, Hainsworth, SMC, GTZAN rhythm annotations, Beatles/Isophonics, and GiantSteps tempo. Verify the exact annotation release and split rather than assuming all recordings have identical rhythm annotations.

For beat F1, the `mir_eval` default tolerance is **70 ms**. Continuity measures assess longer correct sequences; metrical-level metrics may allow half/double tempo alternatives. Tempo evaluation must state whether octave errors are accepted, along with the relative tolerance. [Beat evaluation](https://mir-eval.readthedocs.io/latest/api/beat.html)

### Automatic Chord Estimation

Predict chord labels and time intervals, such as `C:maj`, `G:min7`, or no-chord. Chroma/NNLS features and temporal decoding provide classical baselines; neural methods learn spectral features and chord sequence context. Major/minor chord labels describe chord quality, not the global major/minor key.

The **McGill Billboard** release described by its authors contains annotations for **890 chart slots, 740 distinct songs**, rather than approximately 200 songs. Repeated slots must be handled explicitly when constructing splits. Isophonics is another common annotation collection. [McGill Billboard project](https://ddmal.ca/research/The_McGill_Billboard_Project_%28Chord_Analysis_Dataset%29/)

Report the chord vocabulary reduction, duration-weighted overlap score, treatment of inversions and no-chord segments, and split. `majmin`, `triads`, `tetrads`, and root-only scores evaluate different criteria; “80% chord accuracy” without a vocabulary and protocol is underspecified. [Chord evaluation](https://mir-eval.readthedocs.io/latest/api/chord.html)

### Key Detection

Estimate tonic and mode, globally or over time. Key-profile matching and learned classifiers are different approaches. GiantSteps Key is a known dataset; a filename such as `meters.tsv` is not a verifiable key benchmark on its own.

Distinguish exact accuracy from MIREX-style weighted score: the latter gives partial credit for related keys, including fifth, relative major/minor, and parallel major/minor relationships. Atonality, modulation, tuning, and non-Western modal systems can violate the assumptions of a fixed 24-key task. [Key evaluation](https://mir-eval.readthedocs.io/latest/api/key.html)

### Open Problems

Rubato, irregular meters, meter changes, ambiguous harmony, extended/inverted chords, and appropriate tonal systems. Evaluate these cases directly rather than extrapolating from fixed-meter popular-music benchmarks.

---

## 7. Music Recommendation

### Approaches

Content-based retrieval uses hand-crafted features or learned audio embeddings. Collaborative filtering learns from interaction histories; hybrid recommendation combines content, interaction, and context. Audio similarity can support new-track cold start but is not equivalent to listener preference.

Cosine similarity is a useful embedding baseline. Approximate-nearest-neighbor systems such as FAISS or ScaNN can improve retrieval efficiency at scale; the index and similarity function require validation for the actual embedding/task. Metadata and catalog rights are separate requirements from audio analysis.

Do not assume Spotify's audio-features, audio-analysis, or recommendation endpoints are generally available to a new application. Spotify announced restrictions for new Web API use cases in November 2024; current access must be checked for the actual application. [Official change notice](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api)

### Evaluation

| Metric | Meaning |
|---|---|
| Precision@K | Fraction of the K retrieved items that are relevant |
| Recall@K | Fraction of all relevant items retrieved in the top K |
| NDCG@K | Discounted relevance gain normalized against the ideal ranking |
| Coverage | Fraction of the catalog exposed/recommended under a defined policy |
| Serendipity | Useful and unexpected discovery relative to an expectation baseline; more than novelty alone |

Use temporal splits, record the candidate set and negative-sampling policy, and distinguish offline ranking from online engagement. Popularity bias, feedback loops, cold-start listeners, long-tail coverage, and cultural/contextual differences remain important.

---

## 8. Cover Detection and Version Identification

### Task and Methods

Identify different performances of the same underlying work despite changes in key, tempo, instrumentation, or arrangement. This differs from identifying duplicate or near-identical recordings.

**Chromaprint/AcoustID** is designed for **near-identical audio identification**, not general cover-song recognition under arbitrary transposition or tempo changes. For version identification, compare chroma/harmonic sequences with explicit alignment/transposition handling, or learn embeddings with same-work/different-work supervision. Siamese/triplet training can encode useful invariances but does not guarantee them. [Chromaprint's stated scope](https://github.com/acoustid/chromaprint)

### Datasets and Evaluation

- **Covers80:** 80 pairs / 160 recordings, a small historical benchmark. [Dataset](https://labrosa.ee.columbia.edu/projects/coversongs/covers80/)
- **Da-TACOS:** 15,000-track benchmark subset and a separate 10,000-track cover-analysis subset. The release provides features and metadata, **not audio files**. [Official release](https://github.com/MTG/da-tacos)
- **SecondHandSongs:** work/performance metadata that can support labeling; work counts, performance counts, and downloadable audio are different things.

Report mAP, MRR, or recall/hit rate at K with a defined query/gallery split, relevant versions, and self-match exclusion. Split by work when testing generalization to unseen compositions. A high similarity score is a candidate match, not a determination of ownership or authorization.

Structural changes, medleys, live versions, and major rearrangements remain challenging. Query-by-humming is a related retrieval task with a different input distribution.

---

## 9. Melody Extraction

### Task and Methods

Estimate the predominant melody's time-varying fundamental frequency and whether a melody is present. It does not directly output an isolated vocal waveform or karaoke accompaniment; those are source-separation tasks.

- **MELODIA / Salamon & Gómez (2012):** harmonic salience and pitch-contour selection for polyphonic melody. [Author's method/resources](https://www.justinsalamon.com/melody-extraction.html)
- **Neural salience or sequence models:** predict pitch/voicing from time-frequency features, optionally using temporal decoding or source separation.
- **pYIN (Mauch & Dixon, ICASSP 2014):** probabilistic fundamental-frequency estimation for a monophonic input. It is not a complete polyphonic melody extractor by itself. [Paper](https://webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2014/MauchDixon-PYIN-ICASSP2014.pdf)
- **CREPE (Kim et al., ICASSP 2018):** neural monophonic pitch tracker; using it on a separated stem does not remove separation errors. [Official implementation](https://github.com/marl/crepe)

[SALAMI](https://ddmal.ca/research/salami/annotation/) provides music-structure annotations; it is not a source-separation or melody-extraction algorithm.

### Evaluation

| Metric | Definition for binary voicing annotations |
|---|---|
| Raw Pitch Accuracy (RPA) | Fraction of reference melody/voiced frames whose estimated pitch is within the tolerance, commonly 50 cents |
| Raw Chroma Accuracy (RCA) | Pitch accuracy after octave equivalence is applied |
| Overall Accuracy (OA) | Correct voiced-pitch frames plus correctly unvoiced frames, divided by all frames |
| Voicing Recall (VR) | Correctly detected voiced frames divided by reference voiced frames |
| Voicing False Alarm (VFA) | Incorrectly voiced frames divided by reference unvoiced frames |

OA is not “RPA plus a voicing score”: the denominators and conditions differ. State pitch tolerance, time-grid alignment, voicing convention, and dataset. Instruments can also be “voiced” in this evaluation terminology; it means melody-present, not necessarily human voice. [Melody evaluation documentation](https://mir-eval.readthedocs.io/latest/api/melody.html)

### Open Problems

Weak melodies within dense mixtures, multiple simultaneous melodic lines, octave errors, instrumental timbre diversity, expressive ornamentation, and real-time constraints. There is no single “MIREX dataset” whose RPA summarizes all of these settings.

---

## 10. Cross-Cutting MIR Challenges

1. **Representativeness:** widely used datasets cover limited musical traditions, instruments, recording conditions, and listeners. Measure transfer by domain instead of asserting that every model fails on all non-Western music.
2. **Data leakage:** separate artists, works, recordings, and derived clips where the task requires it. Check overlap with pretraining data and keep test data out of model selection.
3. **Comparable evaluation:** publish dataset version, split, metric implementation, thresholds, aggregation, and extra data. Similar-looking percentages or decibels may encode different tasks.
4. **Deployment:** distinguish batch throughput, streaming support, and measured end-to-end latency. Test realistic codecs, microphones, noise, and recording lengths.
5. **Annotation and uncertainty:** preserve disagreement where meaningful. A model's prediction is an estimate, especially for genre, emotion, harmony, and ambiguous melody.

## 11. MIR Task Landscape

| Task | Output | Evaluation focus | Main challenge |
|---|---|---|---|
| Auto-tagging | Labels/scores | Per-label ROC/PR metrics and split | Rare labels and domain shift |
| Transcription | Symbolic events | Separate onset, offset, frame, instrument, velocity criteria | Real multi-instrument mixtures |
| Source separation | Waveform stems | Defined SDR protocol plus listening tests | Similar sources and artifacts |
| Emotion recognition | Labels or trajectories | Annotation target and listener/split policy | Subjectivity and context |
| Beat tracking | Beat/downbeat times | F1 plus continuity and meter conventions | Rubato and irregular meter |
| Chord recognition | Labeled intervals | Vocabulary and duration weighting | Ambiguous/extended harmony |
| Key detection | Tonic/mode | Exact versus weighted score | Modulation and tonal-system assumptions |
| Recommendation | Ranked items | Candidate set, relevance, user/time split | Cold start and popularity bias |
| Version identification | Ranked matching recordings | Work-disjoint retrieval protocol | Arrangement and structural changes |
| Melody extraction | F0/voicing contour | Pitch and voicing scores separately | Dense or multiple melodies |

Related notes: [Music Generation](music-generation.md), [Audio Engineering](audio-engineering.md), [Music Theory Fundamentals](music-theory-fundamentals.md), [Music Styles](music-styles.md).
