# Music Generation: Technical Research Notes

Selected research on symbolic generation, audio generation, singing, control, video conditioning, and evaluation. Content checked on 2026-09-19. Model specifications below refer to the named paper/release, not every later version of that product family.

> 中文版：[music-generation-zh.md](music-generation-zh.md)

---

## 1. Symbolic Music Generation

### 1.1 Representations

Symbolic models generate notes, durations, timing, instrumentation, or notation. A synthesizer or renderer is needed to turn the result into audio.

| Representation | Encoded information | Trade-off |
|----------------|---------------------|-----------|
| MIDI / MIDI-like events | Note on/off, pitch, velocity, timing, programs and optional controls | Compact performance events; tokenization must define ordering and time resolution |
| Piano roll | Pitch × time activation or velocity, optionally separated by track | Convenient grid; resolution increases size and adjacent repeated notes need onset handling |
| ABC notation | Text notation with meter, pitch, duration, voices and other notation | Compact and human-readable; supported features depend on parser and corpus |
| REMI | Explicit bar/position structure with pitch, duration, velocity and selected metadata | Makes metric position explicit; quantization trades timing detail for structure |
| Compound Word | Groups token attributes with separate attribute embeddings/predictions | Reduces sequence length; compound events need a defined schema |

**MIDI versus tokenization:** `Time Shift` is a model token, not a native MIDI channel-message type, and velocity is an attribute of note messages. Standard MIDI Files store delta times and may contain tempo/time-signature metadata; bars can be derived from that metadata. It is a chosen event vocabulary—not MIDI as a whole—that may omit meter.

ABC supports multiple voices and polyphony; it is not inherently restricted to monophonic music. The [MuPT paper](https://arxiv.org/abs/2404.06393) uses synchronized multi-track ABC (SMT-ABC) to address alignment of measures across tracks.

Verified tokenization references:

- [Pop Music Transformer / REMI](https://arxiv.org/abs/2002.00212), Yu-Siang Huang and Yi-Hsuan Yang, 2020.
- [Compound Word Transformer](https://arxiv.org/abs/2101.02402), Hsiao et al., 2021.
- [REMI-z implementation and paper citation](https://github.com/Sonata165/REMI-z), *Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization*, NeurIPS 2025. REMI-z groups notes by track within bar structure; REMI+ and REMI-z are distinct schemes, not two names for one 2025 format.
- [Pianoroll-Event](https://arxiv.org/abs/2601.19951), Qian et al., **January 2026**: frame, gap, pattern, and musical-structure events combine grid structure with compact event coding.
- [MidiTok](https://miditok.readthedocs.io/en/latest/) is a tokenization library supporting multiple schemes, not a single representation called “MIDI-Token.” Pin its version and tokenizer configuration for reproducibility.

### 1.2 Representative Models

**Music Transformer** (Huang et al., ICLR 2019) introduces an efficient implementation of relative attention for long musical sequences. The original experiments use **JSB Chorales and Piano-e-Competition**, not MAESTRO. Relative sequence positions help repetition and timing modeling; they do not by themselves encode transposition-invariant pitch intervals. The efficiency improvement concerns intermediate relative-position tensors, not removal of all quadratic attention costs. [Paper](https://arxiv.org/abs/1809.04281).

**Pop Music Transformer** combines a Transformer-XL-style model with REMI for beat-aware pop-piano generation. It demonstrates rhythmic benefits in the authors' evaluation, rather than guaranteeing better music for every genre. [Paper](https://arxiv.org/abs/2002.00212).

**MuPT** (2024) studies pretraining symbolic-music Transformers, SMT-ABC, context length, and scaling. Its results depend on its corpus and representation; “larger symbolic models always preserve form” is not a general result. [Paper](https://arxiv.org/abs/2404.06393).

### 1.3 Autoregression, Diffusion, and Flow Matching

An autoregressive model factorizes a token sequence as

$$
p(x\mid c)=\prod_{t=1}^{T}p(x_t\mid x_{<t},c).
$$

It naturally supports sequential continuation but incurs sequential sampling cost. Context limits and training/inference mismatch may harm long-term coherence.

Diffusion and flow-matching models iteratively transform noise into a piano roll, latent sequence, or other representation. Positions within a step can be processed in parallel, but multiple sampling steps remain. Fixed output length is common in particular implementations, not a mathematical requirement; duration conditioning, masking, and chunking can support varying lengths.

Flow matching learns a vector field for a chosen probability path; diffusion learns a denoising/score-related model. They are related continuous generative approaches, not synonyms, and neither guarantees greater diversity or better structure in every setting.

[Auto-Regressive vs Flow-Matching](https://arxiv.org/abs/2506.08570) (Tal, Kreuk, Adi, 2025) compares **text-conditioned audio music** under controlled data and training settings. Its quality, control, editing, and sampling results should not be presented as a proof about all symbolic-music models.

### 1.4 Datasets

| Dataset | Content | Release-scale reference |
|---------|---------|-------------------------|
| [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) | Closely aligned piano audio/MIDI performances | About 200 hours; use the release's official split |
| [Lakh MIDI v0.1](https://colinraffel.com/projects/lmd/) | Deduplicated MIDI collection | 176,581 files; LMD-matched is a smaller subset |
| [POP909](https://arxiv.org/abs/2008.07142) | Pop-piano arrangements with melody, bridge and piano tracks | 909 songs |

Counts of files, unique pieces, performances, and excerpts are different. Specify cleaning, duplicate handling, and song/artist separation when reporting a training or test set.

### 1.5 Open Problems

Long-form form, recurring themes, multi-track coordination, expressive performance timing, and control over harmony remain important. Symbolic structure does not directly specify recording timbre or production, and no single representation captures every musical dimension equally.

## 2. Audio-Level Music Generation

### 2.1 Codecs and Continuous Autoencoders

Discrete codecs turn waveforms into token sequences for language models. Continuous autoencoders provide latent sequences for diffusion/flow models. “Audio generation requires a discrete codec” is therefore incorrect.

| Codec | Verified design and settings |
|-------|------------------------------|
| [SoundStream](https://arxiv.org/abs/2107.03312) (2021) | Convolutional encoder/decoder with residual vector quantization (RVQ), adversarial training and quantizer dropout; supports multiple bitrates |
| [EnCodec](https://github.com/facebookresearch/encodec) (2022) | Convolutional/recurrent encoder-decoder, RVQ and multi-scale STFT adversarial training; original releases include 24 kHz mono and 48 kHz stereo |
| [DAC](https://arxiv.org/abs/2306.06546) (2023) | Improved RVQGAN codec with periodic activations and improved codebook learning; includes high-fidelity 44.1 kHz audio compression |

For the original EnCodec configurations, the 24 kHz model has a **75 Hz** frame rate and supports 1.5/3/6/12/24 kbps; the 48 kHz model has **150 Hz** frames and supports 3/6/12/24 kbps. Active codebook counts depend on bitrate. **MusicGen uses a separately trained 32 kHz, 50 Hz EnCodec configuration**, not either setting unchanged.

RVQ successively quantizes residual error. Its first/later codebooks can differ in information content, but they do not have guaranteed labels such as “harmony” versus “timbre.” Frame rate counts time positions; total tokens per second also depend on the number of codebooks and channel arrangement.

### 2.2 Jukebox (OpenAI, 2020)

[Jukebox](https://cdn.openai.com/papers/jukebox.pdf) uses a three-level VQ-VAE and autoregressive prior/upsamplers, conditioned on artist/genre metadata and, in a lyrics-conditioned model, text. Training uses 1.2 million songs, with a 600,000-song English subset described in the paper.

The 44.1 kHz **mono** waveform is compressed with hop factors **8, 32, and 128**, corresponding to approximately **5,512.5, 1,378.1, and 344.5 tokens/s** at bottom, middle, and top levels. These are downsampling factors, not token rates of 8/34/65 Hz. Each level has a 2,048-entry codebook.

Generation proceeds from the top level through conditional upsampling. Sliding windows permit multi-minute output, but the top-level context is about 24 seconds; generating minutes does not prove repeated choruses or whole-song form. Sampling is computationally expensive. Code and checkpoints are in the [official repository](https://github.com/openai/jukebox); check its license for the chosen use.

### 2.3 MusicLM (Google, 2023)

[MusicLM](https://arxiv.org/html/2301.11325v1) combines three independently pretrained representations:

1. **MuLan** provides music-text conditioning. The generator trains with quantized MuLan **audio** embeddings and substitutes MuLan **text** embeddings at inference.
2. **w2v-BERT features clustered by k-means** provide semantic tokens at 25 Hz.
3. **SoundStream** provides acoustic tokens: 24 kHz mono, 50 Hz frames, 12 RVQ levels in this system.

Autoregressive stages predict semantic tokens, coarse acoustic tokens, then fine acoustic tokens. Semantic tokens are not SoundStream's first codebook. The system demonstrates text and melody conditioning and multi-minute examples. Original model weights were not released with the paper; later consumer access is a separate question.

MusicLM introduced **MusicCaps**, 5,521 ten-second examples with human music descriptions. This is an evaluation resource, not the generator's 280,000-hour training collection. See [the original paper](https://arxiv.org/abs/2301.11325).

### 2.4 MusicGen (Meta, 2023)

[MusicGen](https://arxiv.org/abs/2306.05284) uses a single autoregressive Transformer over codec streams. Released models condition on a frozen T5 text encoder; melody variants add chroma features.

The mono setup uses 32 kHz audio, four 50 Hz codebooks, and delayed interleaving. At an autoregressive step, different codebooks refer to **offset codec times**, not all the same audio frame. This reduces the number of sequential model steps compared with flattening every codebook token.

The [official documentation](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md) lists 300M, 1.5B and 3.3B model sizes; 20,000 hours of licensed training music, including an internal collection and Shutterstock/Pond5; and separate stereo variants. Stereo uses separate left/right code streams. Standard training context is 30 seconds, with continuation/extension in supported implementations.

Chroma collapses octave information and may include harmonic content; it is a melodic guide, not exact score or lyric control. Code and weights have separate terms: AudioCraft code is MIT, released MusicGen weights use CC-BY-NC 4.0 in the [model card](https://github.com/facebookresearch/audiocraft/blob/main/model_cards/MUSICGEN_MODEL_CARD.md). “Open weights” does not imply unrestricted commercial use.

[MAGNeT](https://arxiv.org/abs/2401.04577) uses masked, non-autoregressive audio-token modeling; it is a related generation approach, not simply a faster sampler for an unchanged MusicGen checkpoint.

### 2.5 AudioLDM and AudioLDM 2

**AudioLDM** (2023) trains a latent diffusion model over a VAE's mel-spectrogram latents, followed by VAE decoding and a HiFi-GAN vocoder. It trains using CLAP audio embeddings and can use CLAP text embeddings at sampling. AudioMAE is not the defining VAE component of AudioLDM 1. The original setting targets general short audio, including sound effects and music. [Paper](https://arxiv.org/abs/2301.12503).

**AudioLDM 2** (2023 preprint; 2024 journal version) introduces the **language of audio (LOA)** based on AudioMAE. A GPT-2-based model predicts LOA from conditioning modalities; latent diffusion synthesizes audio conditioned on LOA. This is more specific than “jointly fine-tuning GPT-2 improves text understanding.” Speech, music and general-audio checkpoints should be distinguished; sample rate and duration depend on the checkpoint and implementation. [Paper](https://arxiv.org/abs/2308.05734).

### 2.6 Stable Audio Versions

| Release | Documented behavior | Distinction |
|---------|---------------------|-------------|
| Stable Audio 1.0 (September 2023) | Launch offered 45-second free and 90-second Pro generation | 47 seconds is not its universal limit ([launch](https://stability.ai/news-updates/stable-audio-using-ai-to-generate-music)) |
| Stable Audio 2.0 (April 2024) | Up to 3 minutes of 44.1 kHz stereo; text/audio conditioning | Commercial release, trained on licensed AudioSparx data; not the Open 1.0 checkpoint ([announcement](https://stability.ai/news/stable-audio-2-0)) |
| Stable Audio Open 1.0 (June 2024) | Up to 47 seconds, 44.1 kHz stereo; continuous autoencoder, T5, DiT | Separate open-weight model with its own license and limitations ([model card](https://huggingface.co/stabilityai/stable-audio-open-1.0)) |

The [Stable Audio Open paper](https://arxiv.org/html/2407.14358v1) reports **486,492 recordings total**: 472,618 from Freesound plus 13,874 from FMA, about 7,300 hours, with CC0/CC-BY/CC-Sampling+ source licenses. It is not “486K recordings plus 70K music tracks.” Its autoencoder runs at about 21.5 latent frames/s. Licensed Creative Commons works may still be copyrighted; “no copyrighted data” is an incorrect description.

By the check date, official documentation also describes **Stable Audio 3.0**. The old “Stable Audio 3 is in development in 2025” prediction is obsolete; consult the [official version guide](https://stability.ai/guides/stable-audio-3-prompt-guide) for that separate product family. Historical specifications above should not be silently applied to later models.

### 2.7 YuE (2025)

[YuE](https://arxiv.org/abs/2503.08638), by Ruibin Yuan et al., is a LLaMA2-based family for long-form lyrics-to-song generation. The paper describes track-decoupled next-token prediction, progressive structural conditioning, multi-task/multi-phase training at trillion-token scale, and examples up to five minutes.

It also studies audio-reference/in-context conditioning and representation evaluation on MARBLE. The authors report competitive results against selected proprietary systems; this is not an independently established universal ranking or proof of being the “first to surpass Suno/Udio.”

Use the [official repository](https://github.com/multimodal-art-projection/YuE) for checkpoint, stage, context and hardware requirements. A 10 GB VRAM claim is not a general requirement for the standard pipeline; memory and latency depend on implementation, offloading, precision, and output duration.

### 2.8 Commercial Song Generators

**Suno and Udio** offer text/lyrics-conditioned song generation, but their full architectures and training recipes are not publicly specified in the sources cited here. Do not infer AR/diffusion components, dataset scale, or a universal quality ranking from listening impressions.

Suno's official history records **v5 on 2025-09-23**, **v5.5 on 2026-03-26**, and **v6 on 2026-09-09**. Thus “v5 anticipated” is outdated. See [v5 release](https://suno.com/release-notes/introducing-v5-the-world-s-best-music-model) and [release history](https://suno.com/release-notes).

For [Udio](https://www.udio.com/), distinguish the duration of a newly generated segment from an extended complete song. Product limits, access, editing and export features should be reported with the exact version/plan and access date. This survey does not assign an unsupported universal two-minute maximum or claim its output is inherently more human-like.

### 2.9 Additional Research Systems

| System | Verified contribution |
|--------|-----------------------|
| [TangoFlux](https://arxiv.org/abs/2412.21037) (2024 preprint) | Flow-matching text-to-audio generation and CLAP-ranked preference optimization; audio generation is broader than music |
| [MusicLDM](https://arxiv.org/abs/2308.01546) (2023 preprint, ICASSP 2024) | Music-adapted latent diffusion and beat-synchronous audio/latent mixup; novelty improvements do not guarantee absence of copying |
| [MusicFlow](https://proceedings.mlr.press/v235/prajwal24a.html) (ICML 2024) | Cascaded flow matching for semantic and acoustic features; masked conditioning supports infilling and continuation |
| [SongCreator](https://arxiv.org/abs/2409.06029) (2024) | Dual-Sequence Language Model with configurable attention masks for vocal/accompaniment tasks |
| [ACE-Step](https://arxiv.org/abs/2506.00045) (2025 report) | Diffusion, music-adapted DCAE, linear Transformer, and MERT/m-HuBERT representation alignment during training |

ACE-Step's report describes up to four minutes generated in about 20 seconds on an A100 in its setup. That is an author-reported benchmark, not a hardware-independent latency promise. MERT/m-HuBERT are representation-alignment teachers, not the audio codec. The [project](https://ace-step.github.io/) and report should be distinguished from later ACE-Step releases when comparing capabilities.

## 3. Singing Voice Synthesis

Score-conditioned SVS predicts singing from lyrics, notes, and timing. It requires sustained-vowel modeling, note transitions, pronunciation alignment, and expressive pitch. It is distinct from both changing a recorded singer's voice and generating an entire song from a caption.

| System | Correct description |
|--------|---------------------|
| [XiaoiceSing](https://arxiv.org/abs/2006.06261) (2020) | FastSpeech-style non-autoregressive spectrum/F0/duration prediction with WORLD in the original system |
| [DiffSinger](https://arxiv.org/abs/2105.02446) (2021 preprint, AAAI 2022) | Liu, Li, Ren, Chen, Zhao; score-conditioned mel diffusion with a shallow starting point, followed by a vocoder |
| [VISinger](https://arxiv.org/abs/2110.08813) (2021 preprint, ICASSP 2022) | Yongmao Zhang et al.; variational/flow/adversarial end-to-end synthesis with pitch and duration modeling |
| [VISinger 2](https://arxiv.org/abs/2211.02903) (2022 preprint, INTERSPEECH 2023) | DSP harmonic/noise synthesis guides waveform decoding and improves phase handling |
| [DiTSinger](https://arxiv.org/abs/2510.09016) (2025) | Diffusion Transformer scaling and implicit alignment constrained by character-level spans |
| [OpenVPI DiffSinger](https://github.com/openvpi/DiffSinger) | Community framework; capabilities depend on version and voicebank |

Evaluate pitch on aligned voiced frames, timing against score/phone annotations, lyrics intelligibility, naturalness, expression, and singer identity separately. See [the SVS notes](music-singing-synthesis.md) for corrected datasets and metric definitions.

## 4. Controllable Generation

### 4.1 Conditioning Modalities

| Input | Intended control | Limitation |
|-------|------------------|------------|
| Free text | Genre, instrumentation, mood, production | Ambiguous and often too coarse for note-level requirements |
| Melody/chroma | Pitch-class movement and melodic guidance | Chroma loses octave and does not uniquely specify notes/voicing |
| Score/chord/beat sequence | Explicit musical content or temporal targets | Depends on annotation and supported vocabulary |
| Reference audio | Style, timbre, continuation or editing context | May mix identity, style and content |
| Segment descriptions and boundaries | Attributes that change over time | Boundary adherence and transition quality require separate tests |
| Emotion labels/continuous axes | Perceived valence/arousal or other affect | Annotation and interpretation vary across listeners/cultures |

**Mustango** predicts/conditions on music-specific information such as tempo, beat locations, key, and chords, with a music-informed diffusion denoiser. Its **MusicBench** is a music-text training resource, not a human-preference leaderboard. [Paper](https://arxiv.org/abs/2311.08355).

### 4.2 Time-Varying Control

[TVC-MusicGen](https://www.isca-archive.org/interspeech_2025/yang25f_interspeech.html) (INTERSPEECH 2025) conditions generation on segment boundaries and descriptions using self-supervised structure information. The study evaluates the control approach on language- and diffusion-based models; the name does not mean only the original Meta MusicGen checkpoint is involved.

[SegTune](https://arxiv.org/abs/2510.18416) (2025 preprint; revised for ACL 2026) is a non-autoregressive song-generation framework with local prompts aligned to temporal segments and global prompts for whole-song style. Its duration predictor produces sentence-level timestamped lyrics. Local control still requires checking lyrical alignment, transitions and interaction among attributes.

### 4.3 Remaining Challenges

Exact chord voicing, modulations, form, simultaneous controls, live editing, and non-Western musical systems require more than richer prose prompts. Distinguish “accepts a control” from “reliably follows it,” and evaluate only the musical constraints actually specified.

## 5. Video-to-Music Generation

### 5.1 Representative Models

| Model | Modality and contribution |
|-------|---------------------------|
| [CMT](https://arxiv.org/abs/2111.08380) (2021) | **Controllable Music Transformer**: connects video timing/motion cues to symbolic music rhythm, density and strength; not “Contrastive Multimodal Transformer” |
| [Video2Music](https://arxiv.org/abs/2311.00968) (2023) | Affective multimodal Transformer using semantic, scene, motion and emotion features; generates symbolic/chord content with dynamic rendering |
| [M²UGen](https://arxiv.org/abs/2311.11255) (2023 preprint) | Multimodal encoders and an LLM connected to music generators for understanding and generation/editing |
| [MuVi](https://arxiv.org/abs/2410.12957) (2024) | Visual adaptation, contrastive music-visual pretraining and flow-matching generation for semantic and rhythmic alignment |

MuVi's full title is *Video-to-Music Generation with Semantic Alignment and Rhythmic Synchronization*.

### 5.2 Evaluation

Evaluate semantic match, mood congruence, temporal synchronization, and audio/music quality independently. Declare the visual events, beat extractor, time tolerances, and negative pairs. A good soundtrack need not put a beat on every cut; intended synchrony is task-dependent.

A visual/audio embedding score alone does not establish fine timing or narrative suitability. Named metrics and benchmarks must be tied to a specific paper and implementation; a generic “CMMD = Contrastive Music-Video Metric” should not be assumed to be standard.

## 6. Human Preference Alignment

RLHF-style approaches learn a reward from listener preferences and optimize generation against it. DPO-style methods optimize from preferred/dispreferred pairs relative to a reference policy; applying them to continuous diffusion or flow models requires the corresponding objective, not simply importing an LLM loss unchanged.

[Human preference benchmarking](https://arxiv.org/abs/2506.19085) provides model comparisons and metric analysis; it is not itself proof that a trained reward will generalize. [Aligning Generative Music AI with Human Preferences: Methods and Challenges](https://arxiv.org/abs/2511.15038) is a 2025 preprint accepted at **AAAI 2026 Senior Member Track**.

Separate actual human comparisons from proxy labels such as CLAP-ranked outputs. Reward optimization can improve one attribute while reducing diversity or exploiting evaluator weaknesses. Keep independent listener tests, unseen prompts/styles, and source-overlap checks.

## 7. Evaluation

| Metric / protocol | Measures | Important limit |
|-------------------|----------|-----------------|
| FAD | Distance between fitted reference/generated audio embedding distributions; lower is closer | Requires a reference collection; no paired recording needed; not prompt alignment |
| CLAP / MuLan similarity | Learned prompt–audio association | Coarse semantics, not exact notes or lyrics |
| Classifier KL | Difference in specified label distributions | Paired or aggregate protocols must be identified |
| FMD | Symbolic-music embedding distribution difference | Not waveform fidelity |
| Pitch / chord / beat adherence | Agreement with musical conditions | Requires target controls and reliable extraction |
| MOS / pairwise preference | Listener ratings or choices | Requires a defined population, task, design and uncertainty |

[MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) is a short-clip text-to-music evaluation resource. [AIME](https://huggingface.co/datasets/disco-eth/AIME) contains preference data. MARBLE measures music understanding representations, not direct song quality.

No single metric establishes quality, originality, structure, or cultural appropriateness. FAD results depend on encoder, reference collection, sample size and preprocessing. Per-song FAD variants need their own validation; they are not universally the best proxy for listeners. See [evaluation notes](music-evaluation.md) for formulas, protocols and sources.

## 8. Architecture Comparison

Durations below are paper/release settings or demonstrated examples, not uniform architectural maxima.

| System | Generative approach | Representation / conditioning | Documented scope |
|--------|---------------------|-------------------------------|------------------|
| Jukebox (2020) | Hierarchical AR | Three VQ-VAE levels; metadata/lyrics | 44.1 kHz mono; windowed multi-minute generation |
| MusicLM (2023) | Semantic/coarse/fine AR | w2v-BERT + SoundStream; MuLan | 24 kHz mono; multi-minute examples |
| MusicGen (2023) | Single-stage AR | 32 kHz EnCodec; T5/chroma | 30-second context; mono and separate stereo variants |
| AudioLDM (2023) | Latent diffusion | Mel VAE; CLAP | General short audio |
| AudioLDM 2 (2023–2024) | LOA prediction + latent diffusion | AudioMAE/GPT-2 plus acoustic decoder | Speech/music/audio variants |
| Stable Audio 2.0 (2024) | Latent diffusion | Commercial model; text/audio inputs | Up to 3 minutes, 44.1 kHz stereo |
| Stable Audio Open 1.0 (2024) | DiT latent diffusion | Continuous autoencoder; T5 | Up to 47 seconds, 44.1 kHz stereo |
| YuE (2025) | Track-decoupled AR | Music tokens; lyrics/style/reference | Up to five-minute examples |
| ACE-Step (2025 report) | Diffusion with linear Transformer | Music DCAE; lyrics/text | Up to four minutes in the report |
| MusicFlow (2024) | Cascaded flow matching | Semantic and acoustic features | Text conditioning, infilling, continuation |
| SongCreator (2024) | Dual-sequence LM | Vocal/accompaniment streams and attention masks | Multiple song-generation/editing tasks |
| Suno / Udio | Not specified here | Service-provided text/lyrics controls | Version- and plan-dependent |

Code availability, model-weight availability, training-data availability, and license are separate axes. A binary “open source: yes” column obscures those differences.

## 9. Open Problems and Research Practice

- **Long-form music:** Evaluate motif development, repeated sections, transitions, and endings over the complete output, not only its nominal duration.
- **Precise control:** Measure adherence and interaction among melody, harmony, form, instrumentation and expression.
- **Efficiency:** Separate first-audio latency, throughput and total generation time. Some systems run faster than audio duration on suitable hardware; that alone does not establish interactive streaming.
- **Data and originality:** Audit training/evaluation overlap and nearest-neighbor copying separately from quality. Licensed data and open weights do not automatically establish novel output.
- **Cultural coverage:** Evaluate supported languages, tuning systems, instruments and traditions with appropriate listeners and annotations.
- **Evidence:** Separate paper findings, official product specifications, author-reported benchmarks and hypotheses. Avoid unverified leaderboard, venue, hardware, or architectural claims.

> Related: [singing synthesis](music-singing-synthesis.md), [evaluation](music-evaluation.md), [music understanding](music-understanding-mir.md), and [audio engineering](audio-engineering.md).
