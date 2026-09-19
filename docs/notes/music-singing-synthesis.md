# Singing Voice Synthesis (SVS): Technical Research Notes

Research notes checked on 2026-09-19. Covers score-conditioned singing, acoustic models, waveform generation, controllability, voice conversion, datasets, and evaluation.

> 中文版：[music-singing-synthesis-zh.md](music-singing-synthesis-zh.md)

---

## 1. Task Definition

In score-conditioned **singing voice synthesis**, lyrics and musical notes specify what should be sung. A system predicts pronunciation timing and expressive acoustics, then produces a singing waveform. Text-to-song models may also compose melody and accompaniment; that broader task does not imply exact score control.

| Aspect | Typical speech synthesis | Score-conditioned singing synthesis |
|--------|--------------------------|-------------------------------------|
| Pitch | Linguistic intonation and expression | Note targets plus expressive deviations |
| Timing | Linguistic prosody | Musical timing plus phoneme-level articulation |
| Sustained sound | Usually shorter vowels | Long vowels and note transitions are central |
| Ornamentation | Speech-dependent pitch movement | May include vibrato, portamento, and melisma |
| Control | Text, voice, prosody | Lyrics, notes, timing, voice, and expression |

These are tendencies, not hard boundaries. Singing need not span two octaves or use vibrato. **Melisma** means several notes on one syllable. Music generation also includes monophonic or instrumental-only outputs, so “music generation always produces full arrangements” is too narrow.

## 2. Pipeline Architecture

```text
Lyrics + notes + tempo / timing
    → language-specific pronunciation and score frontend
    → note–syllable–phoneme alignment and duration modeling
    → acoustic model (spectral features, F0, voicing or other controls)
    → compatible vocoder / waveform decoder
    → singing audio
```

End-to-end models can train several of these stages jointly. This does not remove the need to model timing or synthesize a waveform.

### 2.1 Frontend and Alignment

- Convert lyrics to phonemes with language-specific pronunciation rules; handle lexical tone, pronunciation variants, and cross-language phones explicitly.
- Parse note pitch, onset, duration, rests, ties, slurs, tempo, and lyrics where present.
- Standard MusicXML/MIDI input does not generally contain phoneme-to-audio boundaries. These must be annotated, aligned, predicted, or supplied in a specialized format.
- Handle one syllable over multiple notes and multiple phonemes within a note. Consonants may precede the notated vowel onset, so allocating equal phoneme durations is usually inappropriate.
- Reserve time for breaths and phrase boundaries where the training representation and system support them.

### 2.2 Acoustic Features

| Feature | Meaning | Important distinction |
|---------|---------|-----------------------|
| Mel-spectrogram / spectral envelope | Time-frequency energy or timbral shape | The required representation depends on the decoder |
| F0 | Fundamental frequency in voiced frames | Note pitch is a target; performed F0 includes transitions and ornamentation |
| Voiced/unvoiced flag | Whether periodic voicing is detected or requested | Not identical to silence or breathiness |
| Aperiodicity, energy, breathiness controls | Noise, level, or expression-related features | Model-specific; no standard universal “s/δ” feature |
| Phoneme duration | Alignment of pronunciation to time | A timing variable, not a phonation flag |

### 2.3 Waveform Generation

| Method | Architecture / role |
|--------|---------------------|
| WORLD | Signal-processing vocoder using F0, spectral envelope, and aperiodicity |
| HiFi-GAN / neural source-filter variants | Convolutional adversarial waveform generation; some variants explicitly use F0 |
| DiffWave | Iterative diffusion waveform model; speed and quality depend on checkpoint and sampling |
| Vocos | Convolutional backbone with Fourier-domain output and inverse STFT; not a Transformer vocoder by default |
| SoundStream / EnCodec / DAC decoders | Reconstruct audio from their own learned codec representations; not drop-in mel vocoders |

[Vocos paper and implementation](https://github.com/gemelo-ai/vocos) document its Fourier-based approach. Vocoder performance must be measured on the target singing range and sampling rate; a speech-trained checkpoint can fail on sustained or very high notes.

## 3. Representative Models

### 3.1 Earlier Methods and XiaoiceSing

Unit concatenation and statistical parametric systems predate modern neural SVS. HMM/DNN systems model acoustic features and timing from linguistic and musical inputs; their expressiveness and smoothing depend on the system and data.

[XiaoiceSing](https://arxiv.org/abs/2006.06261) (Lu et al., INTERSPEECH 2020) uses a **FastSpeech-style non-autoregressive** architecture. It predicts spectral features, F0, and durations with singing-specific pitch and duration constraints; the original system uses WORLD. It should not be described as an autoregressive attention decoder with HiFi-GAN.

### 3.2 DiffSinger

[DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446) is by Jinglin Liu, Chengxi Li, Yi Ren, Feiyang Chen, and Zhou Zhao; preprint 2021, AAAI 2022.

The acoustic model denoises mel-spectrograms conditioned on the score. Shallow diffusion starts from an intermediate noise level applied to a simpler decoder's prediction, using a learned/estimated boundary, and shortens the reverse process. A vocoder then synthesizes audio. This is not an unconditional promise of superior quality with any step count.

The [OpenVPI DiffSinger implementation](https://github.com/openvpi/DiffSinger) extends the original framework. Multi-speaker/language support, variance controls, accelerated sampling, and supported vocoders depend on the repository version and trained voicebank; community features should not be attributed automatically to the original paper.

### 3.3 VISinger and VISinger 2

[VISinger](https://arxiv.org/abs/2110.08813) (Yongmao Zhang et al.; preprint 2021, ICASSP 2022) adapts VITS to singing: variational inference, normalizing flows, adversarial waveform decoding, frame-level priors, pitch prediction, and a note-aware duration model. End-to-end training still contains explicit duration modeling.

[VISinger 2](https://arxiv.org/abs/2211.02903) (Yongmao Zhang et al.; preprint 2022, INTERSPEECH 2023) adds harmonic/noise DSP synthesis to guide waveform decoding and address phase-related artifacts, generating 44.1 kHz singing.

### 3.4 DiTSinger and Commercial Tools

[DiTSinger](https://arxiv.org/abs/2510.09016) (2025) studies diffusion Transformers and implicit phoneme alignment constrained by character-level spans, together with a synthetic-data construction pipeline. “Implicit alignment” does not mean that all musical timing information is absent.

Commercial score/lyrics editors such as [ACE Studio](https://acestudio.ai/) and [Synthesizer V](https://dreamtonics.com/synthesizerv/) are relevant engineering systems, but product claims are not peer-reviewed architecture descriptions. Language, voicebank, plug-in, and licensing support should be checked for the exact product version.

### 3.5 Adjacent Tasks: SingSong and Full-Song Models

[SingSong](https://arxiv.org/abs/2301.12662) (Donahue et al., 2023) generates **instrumental accompaniment from input singing**. It adapts AudioLM and uses source-separated vocal/instrumental pairs for training. It is neither a diffusion SVS model nor a singing extraction-and-resynthesis method.

Lyrics-to-song models such as YuE and ACE-Step produce vocals with accompaniment; see [music generation](music-generation.md). MusicGen's chroma conditioning is not a guarantee of intelligible lyrics, exact notes, or phoneme timing. Do not equate singing-like audio with a validated SVS interface.

## 4. Singing Representations and Controls

### 4.1 Score and Pitch

| Representation | What it can provide |
|----------------|---------------------|
| MusicXML | Notes, durations, notation, lyrics, and expression marks when encoded |
| Standard MIDI File | Note events and timing; optional lyric/text meta events, but no required phoneme alignment |
| System-specific formats | Explicit phones, note assignment, durations, F0 and variance curves as supported |

For equal temperament with A4 = 440 Hz,

$$
f(n)=440\,2^{(n-69)/12},\qquad
c(f_1,f_2)=1200\log_2(f_1/f_2),\quad f_1,f_2>0.
$$

MIDI note number $n$ specifies a semitone grid. A continuous F0 curve represents vibrato, glides, and intonation between notes. Cents express a **ratio to a stated reference**, not an absolute frequency unit; the formula is undefined for unvoiced frames represented by zero.

Vibrato rate and extent vary with singer, genre, register, and definition of extent (amplitude versus peak-to-peak). Avoid treating “5–7 Hz and 20–50 cents” as a universal target. Straight-tone singing can be intentional.

### 4.2 Control Dimensions

| Dimension | Possible conditioning | Limitation |
|-----------|-----------------------|------------|
| Pitch and timing | Notes, F0, tempo, phone durations | Controls can conflict or exceed the singer's trained range |
| Singer identity | Singer embedding or reference recording | Identity/style leakage and domain mismatch |
| Style and emotion | Labels, reference audio, continuous controls | Labels do not guarantee perceptually independent factors |
| Breathiness and dynamics | Model-specific variance curves | A voiced/unvoiced switch alone is insufficient |
| Technique | Falsetto, belting, growl, vocal fry labels or examples | Requires relevant data and evaluation |
| Language | Pronunciation frontend and training coverage | Shared phone symbols do not guarantee pronunciation transfer |

## 5. Singing Voice Conversion (SVC)

SVC changes a recorded singer's identity/timbre while aiming to preserve lyrical content and timing. Pitch may be preserved or intentionally transposed; this must be specified. It takes a source performance, unlike score-conditioned SVS.

Feature mapping, content/pitch/singer disentanglement, adversarial decoders, and diffusion are possible components. Some methods need parallel recordings; many use non-parallel data and self-supervised content features. Neither a diffusion architecture nor a target-speaker embedding guarantees better quality.

[So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc) is a community implementation family; identify the exact version/fork. Evaluate content preservation, pitch, pronunciation timing, artifacts, and target-singer similarity separately.

## 6. Public Singing Datasets

Counts below refer to the cited release, and distinguish songs, recordings, and singers.

| Dataset | Language | Singers | Content |
|---------|----------|---------|---------|
| [Opencpop](https://wenet-e2e.github.io/opencpop/) | Mandarin | 1 | 100 songs, 3,756 utterances, about 5.2 hours; note/phoneme annotations |
| [M4Singer](https://openreview.net/forum?id=qiDmAaG6mP) | Mandarin | 20 | 700 songs, about 30 hours; annotated scores and SATB voice types |
| [NUS-48E](https://smcnus.comp.nus.edu.sg/archive/pdf/2012-2013/2013_05-Pub-NUS-48E.pdf) | English | 12 | 48 sung recordings of 20 unique songs; about 115 minutes singing plus 54 minutes spoken lyrics |
| [JSUT-song](https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song) | Japanese | 1 | 27 children's songs, about 25 minutes, 48 kHz |
| [Tohoku Kiritan](https://www.jstage.jst.go.jp/article/ast/42/3/42_E2074/_article) | Japanese | 1 | 50 songs, about 57 minutes of studio vocals |

VCTK is speech data, and MUSAN contains speech/music/noise for auxiliary uses; neither is a score-aligned SVS corpus. A corpus suitable for SVC may require new score annotations for supervised SVS. Read each release's usage and redistribution conditions separately.

Split by song and, for unseen-singer evaluation, by singer. Segmenting the same recording across training and test sets causes leakage. Source-separated vocals can increase coverage but retain accompaniment leakage, artifacts, and uncertain timing; they are not equivalent to clean studio stems.

## 7. Evaluation

### 7.1 Objective Metrics

| Metric | Required specification |
|--------|------------------------|
| F0 RMSE | Hz or log-frequency/cents, alignment, pitch extractor, voiced-frame mask, and octave-error treatment |
| F0 correlation | Common voiced frames; correlation does not detect a constant pitch offset and is undefined for zero variance |
| Voicing error | Separate voiced/unvoiced accuracy or error, rather than hiding unvoiced frames in pitch RMSE |
| Note/phoneme timing | Target annotations, onset/offset tolerances, and handling of melisma |
| Mel / STFT error or MCD | Aligned reference, feature definition, normalization, coefficient selection and any time warping |
| Lyric error | Validated singing transcription, WER/CER or phone error, plus manual intelligibility checks |
| Singer similarity | Embedding model validated on singing; compare against reference singer recordings |

On jointly voiced, aligned frames $V$, one possible cents RMSE is

$$
\operatorname{RMSE}_{\mathrm{cent}}=
\sqrt{\frac{1}{|V|}\sum_{t\in V}
\left(1200\log_2\frac{\hat f_0(t)}{f_0(t)}\right)^2}.
$$

Report an undefined/missing result if $|V|=0$, not a perfect zero. F0 error against a performed reference and deviation from an ideal note are different measurements: expressive vibrato can increase the latter without indicating poor singing.

### 7.2 Listening Tests

Rate naturalness, audio quality, pronunciation, musical timing, expression, and singer similarity separately where relevant. MOS and pairwise preference are methods; **PEMO-Q is an objective model, not a subjective listening protocol**. Listen to both isolated vocals and their musical context when the claim concerns integration into a song.

Specify listener population, randomization, examples, confidence intervals, and independent test songs/singers. Objective spectral and pitch errors measure particular deviations; their correlation with overall preference must be established for the evaluated setting. See [evaluation methodology](music-evaluation.md).

## 8. Open Problems

- Maintaining voice identity, intelligibility, and expressive phrasing through full songs.
- Learning controllable techniques and emotion without changing singer identity unintentionally.
- Generalizing to unseen singers from a short reference, with a clear distinction between zero-shot inference and fine-tuning.
- Robust phoneme/note alignment across languages, sustained vowels, and melisma.
- Low-latency interaction: report first-audio latency as well as total real-time factor and hardware.
- Evaluation across styles and cultures; common datasets such as Opencpop do not create a universal protocol for all SVS tasks.
- Integrating lead vocals, harmonies, and accompaniment with control over each part.

> Related: [music generation](music-generation.md), [source separation and MIR](music-understanding-mir.md), [audio engineering](audio-engineering.md), and [evaluation](music-evaluation.md).
