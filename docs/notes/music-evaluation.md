# Music Generation Evaluation: Metrics, Protocols, and Open Problems

Research notes checked on 2026-09-19. The protocols below distinguish creative generation, conditional generation, and reconstruction; their metrics are not interchangeable.

> 中文版：[music-evaluation-zh.md](music-evaluation-zh.md)

---

## 1. Why Music Evaluation Is Hard

| Dimension | Evaluation question |
|-----------|---------------------|
| Audio quality | Are there clipping, noise, distortion, or mixing artifacts? |
| Musical coherence | Do phrases, transitions, repetition, and development make sense? |
| Conditioning | Does the output follow the requested text, score, melody, or timing? |
| Diversity | Do multiple outputs cover different plausible solutions? |
| Novelty | Does the output reproduce training material or offer new material? |
| Expression | Does the performance convey the intended style or emotion? |

There is no universal musical-quality score. A **low** FAD can coexist with poor long-term structure. A high text-audio similarity can coexist with wrong notes or unclear lyrics. Conventions and listener preferences depend on genre, culture, and intended use.

First define the task. Reconstructing a particular recording permits aligned reference comparisons; composing a new piece from its description usually has many valid answers.

## 2. Human Evaluation

### 2.1 Protocols

| Protocol | What listeners do | Appropriate use |
|----------|-------------------|-----------------|
| MOS / attribute ratings | Rate a defined attribute, often on a 1–5 scale | Naturalness, quality, prompt adherence; report anchors and scale |
| Pairwise preference | Choose A or B for the same condition, optionally allowing ties | Comparing generators without an exact target recording |
| MUSHRA | Rate multiple versions on a 0–100 scale with reference, hidden reference, and anchors | Intermediate-quality audio-system comparisons with the same underlying content |
| ABX | Decide whether X is A or B | Detecting an audible difference; not measuring which version is better |
| Ranking | Order several candidates | Relative preference, with attention to listener workload |

[MUSHRA is specified in ITU-R BS.1534](https://www.itu.int/rec/R-REC-BS.1534). A listening test of different newly composed songs without an appropriate reference and anchors should be described as an adapted multi-stimulus test, rather than claimed to follow standard MUSHRA.

### 2.2 Experimental Design

- Recruit listeners representative of the target audience; use expert listeners for expert judgments and report musical experience.
- Blind model identities, randomize presentation and A/B positions, and balance prompts across systems.
- Define listening conditions and loudness treatment. [ITU-R BS.1770](https://www.itu.int/rec/R-REC-BS.1770) specifies loudness and true-peak measurement; it does not choose a universal listening level. Loudness matching should not erase dynamics that are themselves under evaluation.
- Match duration to the claim: short excerpts for local artifacts, complete sections or tracks for structural coherence. Do not infer full-song quality from a few seconds.
- Choose sample and listener counts from effect size, variability, power or confidence-interval requirements. There is no universal “20 listeners makes MOS reliable” rule.
- Report numbers of prompts, outputs, unique listeners, ratings per item, exclusions, uncertainty, and aggregation. Repeated ratings by one listener or repeated outputs from one prompt are not independent observations; account for this in bootstrap or mixed-effects analyses.
- Choose agreement statistics for the data: Cohen's kappa is for two categorical raters; weighted kappa, suitable multi-rater measures, or a specified ICC may fit other designs. Low agreement may reflect ambiguous instructions or unreliable annotation as well as subjective taste.

### 2.3 A Verified Preference Dataset

[Benchmarking Music Generation Models and Metrics via Human Preference Studies](https://arxiv.org/abs/2506.19085) reports approximately 6,000 generated examples from 12 systems, 15,000 pairwise comparisons, and 2,500 participants. Its [AIME dataset](https://huggingface.co/datasets/disco-eth/AIME) supports studying metric–preference relationships. Rankings apply to the tested model versions, prompts, and listeners; they are not a current universal leaderboard.

## 3. Choose Metrics by Reference Requirement

| Metric family | Needs paired target audio? | Needs a reference collection? | Main use |
|---------------|---------------------------|-------------------------------|----------|
| FAD / audio-embedding Fréchet distance | No | Yes | Distribution comparison |
| Generative precision / recall | No | Yes | Coverage and support in an embedding space |
| CLAP / MuLan audio-text similarity | No | No; needs the condition text | Semantic prompt alignment |
| ViSQOL, PEAQ, spectral reconstruction errors | Yes, for meaningful reconstruction assessment | No separate collection | Distortion of corresponding content |
| Score / beat / chord adherence | Needs a target score or control annotation | No | Task-specific controllability |
| Human preference | No exact target needed | No | Listener judgments |

“No paired reference” and “no reference data” mean different things. FAD falls into the first category.

## 4. Distribution-Based Metrics

### 4.1 Fréchet Audio Distance (FAD)

For reference and generated audio embeddings, estimate means and covariance matrices. The Gaussian squared 2-Wasserstein distance commonly called FAD is

$$
\operatorname{FAD}=\lVert\mu_r-\mu_g\rVert_2^2+
\operatorname{tr}\!\left(\Sigma_r+\Sigma_g-
2\left(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}\right)^{1/2}\right).
$$

Here $r$ denotes the reference collection and $g$ the generated collection; square roots are positive-semidefinite matrix square roots. The symmetric form avoids treating the generally nonsymmetric product $\Sigma_r\Sigma_g$ as a symmetric matrix. Audio embeddings need not actually be Gaussian: Gaussian fitting is the metric's approximation.

Lower means closer distributions **under this embedding and protocol**. The [original FAD paper](https://arxiv.org/abs/1812.08466) appeared as a 2018 preprint and at INTERSPEECH 2019, using VGGish features for music enhancement evaluation.

Practical limits:

- Record encoder/checkpoint, feature layer, pooling, window length, resampling, reference corpus, sample counts, and implementation. Values from different setups are not directly comparable.
- Finite-sample bias and small-sample covariance estimates can alter rankings. Compare matched sample sizes and report uncertainty; tiny per-genre subsets may be unreliable.
- FAD combines distribution differences into one value; it does not separately identify fidelity, diversity, or memorization.
- It does not use prompt–audio pairs, so it does not test individual prompt adherence. Permuting outputs across prompts leaves ordinary FAD unchanged.
- Aggregating window embeddings discards their global ordering. Local features may encode some timing, but FAD alone does not establish full-song structure.

### 4.2 FAD-CLAP and Per-Song Scores

**FAD-CLAP** means FAD using CLAP **audio** embeddings. It still compares audio distributions and does not automatically become a text-alignment metric. Avoid introducing “FA-CLAP” as a separate standard without a specific definition and source.

[Microsoft's FAD toolkit](https://github.com/microsoft/fadtk) supports multiple embedding models, sample-size-aware estimation, and individual-song scoring. Per-song procedures still use a background distribution and a specified sampling/embedding method. They should not be confused with fitting a covariance to one global embedding, or assumed to be the best perceptual metric for every domain.

### 4.3 Generative Precision, Recall, Density, and Coverage

These estimate support/coverage using finite samples and an explicit embedding-space neighborhood rule. Precision asks how generated samples relate to estimated reference support; recall asks how reference samples are covered by generated support. They are not exact percentages of an unknown “true distribution.” Report the implementation, neighborhood parameters, and sample counts. High coverage or precision does not independently establish creativity or musical validity.

### 4.4 Label KL and Inception Score

Classifier-based KL can compare label posteriors for corresponding reference/generated clips or compare aggregate label distributions. State which protocol is used, the KL direction, classifier, normalization, and zero-probability handling. A multi-label sigmoid vector is not automatically a categorical probability distribution.

Inception Score rewards confident per-example classifier predictions and varied aggregate predictions. Neither score directly measures musicality, and both depend on classifier calibration and label coverage.

## 5. Text-Audio Alignment

### 5.1 CLAP Score

For audio $a$ and prompt $t$, with nonzero embeddings:

$$
\operatorname{CLAPScore}(a,t)=
\frac{f_a(a)^\top f_t(t)}{\lVert f_a(a)\rVert_2\lVert f_t(t)\rVert_2}.
$$

Average scores across matched prompt–output pairs, and document any scaling (for example multiplying by 100). Different CLAP implementations/checkpoints use different training corpora; [LAION-CLAP](https://github.com/LAION-AI/CLAP) includes music-trained variants. “CLAP is never trained on music” is incorrect.

This measures learned semantic association, not exact BPM, harmony, lyric transcription, negation handling, or ordering of sections. MuLan similarity serves a related role in the [MusicLM evaluation](https://arxiv.org/abs/2301.11325); scores from distinct encoders are not interchangeable.

### 5.2 Complementary Tests

| Condition | Useful additional test |
|-----------|------------------------|
| Tempo / beats | Tempo error with half/double-tempo policy; beat F-measure with declared tolerance |
| Key / chords | Time-aligned key/chord adherence with a specified vocabulary |
| Lyrics | Transcription error, omissions/repetitions, and human intelligibility |
| Melody | Pitch-class or F0 contour adherence, with explicit octave handling |
| Song form | Requested versus realized section order, timing, and transitions |
| Instruments | Validated instrument tags plus listening checks for mixtures |

A text-only LLM judging an automatically generated caption evaluates a proxy that inherits captioning errors. Audio-capable judges also require validation against listeners and held-out systems. Do not treat either as ground truth.

Classifier-free guidance can trade off adherence, artifacts, and diversity. Evaluate a range of settings; increasing guidance does not guarantee monotonically better alignment.

## 6. Perceptual and Reconstruction Metrics

| Metric | Scope and caveat |
|--------|------------------|
| PEAQ | Objective audio-quality assessment standardized by [ITU-R BS.1387](https://www.itu.int/rec/R-REC-BS.1387) |
| PEMO-Q | Auditory-model-based objective comparison; distinct from PEAQ, and not itself the BS.1387 standard |
| ViSQOL | Full-reference speech/audio similarity estimator; select and report speech or audio mode ([official implementation](https://github.com/google/visqol)) |
| PESQ / POLQA | Speech-oriented quality estimators; do not assume validation for full musical mixtures |
| STFT / mel distance | Reconstruction differences dependent on alignment, gain, and analysis parameters |

These are useful for codecs, vocoders, restoration, and aligned reconstruction. Comparing two valid but different compositions against each other measures their difference, not which is musically better. Perceptual-model correlations must be established on the relevant distortions and data.

Without a paired reference, use condition adherence, a suitable reference-distribution comparison, and listeners. BRISQUE/NIQE are image metrics; applying them to spectrogram pictures does not establish audio-quality validity. Codec reconstruction error or bitrate is not a standalone measure of the quality of a newly generated composition.

## 7. Music-Specific Descriptors

| Dimension | Possible descriptors | Interpretation limit |
|-----------|----------------------|----------------------|
| Harmony | Chord transitions, pitch-class histograms, key-profile similarity | Pitch-class entropy alone is not tonal stability: a constant note has low entropy |
| Rhythm | Onset density, inter-onset intervals, swing ratio, tempo curve | Tempo variation and microtiming may be intentional |
| Melody | Interval distribution, range, contour, motif recurrence | No universal Zipf law or preferred interval distribution across traditions |
| Structure | Self-similarity, repetition, annotated section boundaries | High repetition need not be good; boundary F-measure requires target boundaries and a tolerance |

State whether descriptors come from symbolic notes or estimated audio annotations. Transcription, chord, beat, and source-separation errors propagate into evaluation. A genre template is a chosen target, not a universal definition of valid music.

For symbolic generation, [Fréchet Music Distance (FMD)](https://arxiv.org/abs/2412.07948) compares symbolic-music embedding distributions. Its 2024 preprint is separate from audio FAD and does not assess rendering fidelity.

## 8. Benchmarks and Reproducibility

| Resource | Verified role |
|----------|---------------|
| [MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) | 5,521 ten-second music examples with human descriptions; commonly used for text-to-music evaluation |
| [AIME](https://huggingface.co/datasets/disco-eth/AIME) | Generated music and human pairwise judgments for model/metric comparison |
| [MARBLE](https://arxiv.org/abs/2306.10548) | Music-representation understanding benchmark; not a direct generated-song quality test |

MusicCaps provides annotations and source identifiers; source-audio availability can change. Record the actually retrieved subset and preprocessing. It is a short-clip benchmark, so it cannot establish multi-minute structure. Use disjoint training/evaluation data and check song-, artist-, and recording-level overlap where possible.

A reproducible protocol records:

1. Task, test split, target genres/languages, reference requirements, and missing examples.
2. Model/checkpoint or service version, access date, conditioning, sampler, guidance, seeds, and outputs per prompt.
3. Duration, sample rate, channel treatment, normalization, and failure handling, including silence and truncated outputs.
4. Metrics with exact checkpoints/settings, baselines, aggregation, and uncertainty.
5. Listener recruitment, randomized test design, rated dimensions, counts, exclusions, and statistical analysis.
6. Generation time, hardware, batch size, and total cost if efficiency is claimed.
7. Released code, prompts, samples, and annotations where redistribution is permitted.

## 9. Open Problems and Practical Priorities

For text-to-music, FAD plus prompt similarity is a useful starting point when suitable reference data exists, followed by condition-specific tests and listening evaluation. For symbolic generation, SVS, and reconstruction, select metrics appropriate to those tasks instead of mandating CLAP everywhere.

Separate sample-level and system-level metric correlations. A score that ranks model averages well need not choose the better of two songs. Validate on new generators, languages, and styles, and inspect failure cases rather than only reporting a single correlation.

Long-term form, emotional expression, novelty, and cultural diversity remain difficult to summarize automatically. Preference optimization can exploit weaknesses in reward models; keep held-out human evaluation separate from training rewards. [Aligning Generative Music AI with Human Preferences: Methods and Challenges](https://arxiv.org/abs/2511.15038) is a November 2025 preprint accepted in the **AAAI 2026 Senior Member Track**.

Further reading: [A Survey on Evaluation Metrics for Music Generation](https://arxiv.org/abs/2509.00051) (2025).

> Related: [generation architectures](music-generation.md), [singing synthesis](music-singing-synthesis.md), [music styles](music-styles.md), and [MIR](music-understanding-mir.md).
