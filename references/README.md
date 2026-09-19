# Verified Reading List

[中文版](README-zh.md)

Checked on 2026-09-19. Papers link to original sources. Years denote first preprint submission unless a conference/publication is explicitly distinguished; product launch, preprint, and proceedings dates can differ. This list verifies bibliographic identity and brief subject descriptions, not independent reproduction of experimental results.

## Music understanding

| Paper and source | Authors | Year | Topic |
|---|---|---|---|
| [Semi-Supervised Music Tagging Transformer](https://arxiv.org/abs/2111.13457) | Minz Won, Keunwoo Choi, Xavier Serra | 2021 | Music tagging with convolution and self-attention |
| [Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687) | Yusong Wu et al. | 2022 | LAION-CLAP; audio-text contrastive learning |
| [A Foundation Model for Music Informatics](https://arxiv.org/abs/2311.03318) | Minz Won, Yun-Ning Hung, Duc Le | 2023 | MusicFM; self-supervised music representations |
| [MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training](https://arxiv.org/abs/2306.00107) | Yizhi Li et al. | 2023 | Acoustic and musical teacher targets; ICLR 2024 |

## Music generation

| Paper and source | Authors | Year | Topic |
|---|---|---|---|
| [Music Transformer](https://arxiv.org/abs/1809.04281) | Cheng-Zhi Anna Huang et al. | 2018 | Relative attention for symbolic music |
| [Jukebox: A Generative Model for Music](https://arxiv.org/abs/2005.00341) | Prafulla Dhariwal et al. | 2020 | VQ-VAE and autoregressive audio modeling |
| [AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/abs/2209.03143) | Zalán Borsos et al. | 2022 | Hierarchical semantic/acoustic token modeling |
| [MusicLM: Generating Music From Text](https://arxiv.org/abs/2301.11325) | Andrea Agostinelli et al. | 2023 | Text-conditioned music audio generation |
| [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://arxiv.org/abs/2301.12503) | Haohe Liu et al. | 2023 | Continuous latent diffusion for audio |
| [AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) | Haohe Liu et al. | 2023 | Shared audio representation and latent diffusion |
| [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) | Jade Copet et al. | 2023 | MusicGen; text/melody-conditioned codec language model |
| [Fast Timing-Conditioned Latent Audio Diffusion](https://arxiv.org/abs/2402.04825) | Zach Evans et al. | 2024 | Stable Audio research model; text and timing conditions |
| [MusicFlow: Cascaded Flow Matching for Text Guided Music Generation](https://arxiv.org/abs/2410.20478) | K R Prajwal et al. | 2024 | Cascaded semantic/acoustic flow matching |
| [SongCreator: Lyrics-based Universal Song Generation](https://arxiv.org/abs/2409.06029) | Shun Lei et al. | 2024 | Dual-sequence language model for vocals/accompaniment |
| [YuE: Scaling Open Foundation Models for Long-Form Music Generation](https://arxiv.org/abs/2503.08638) | Ruibin Yuan et al. | 2025 | Long-form lyrics-to-song generation |
| [ACE-Step: A Step Towards Music Generation Foundation Model](https://arxiv.org/abs/2506.00045) | Junmin Gong et al. | 2025 | DCAE, linear transformer, diffusion and REPA |
| [ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation](https://arxiv.org/abs/2602.00744) | Junmin Gong et al. | 2026 | Distinct follow-up with LM planning and DiT generation |
| [Multitrack Music Transformer](https://arxiv.org/abs/2207.06983) | Hao-Wen Dong et al. | 2022 | Multitrack symbolic representation; ICASSP 2023 |

## Video and multimodal conditioning

| Paper and source | Authors | Year | Topic |
|---|---|---|---|
| [Video Background Music Generation with Controllable Music Transformer](https://arxiv.org/abs/2111.08380) | Shangzhe Di et al. | 2021 | CMT means Controllable Music Transformer |
| [M²UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models](https://arxiv.org/abs/2311.11255) | Shansong Liu et al. | 2023 | Multimodal understanding and music generation |
| [Video2Music: Suitable Music Generation from Videos using an Affective Multimodal Transformer model](https://arxiv.org/abs/2311.00968) | Jaeyong Kang, Soujanya Poria, Dorien Herremans | 2023 | Video-conditioned symbolic music framework |
| [MuVi: Video-to-Music Generation with Semantic Alignment and Rhythmic Synchronization](https://arxiv.org/abs/2410.12957) | Ruiqi Li et al. | 2024 | Video-conditioned audio with semantic/rhythmic alignment |

## Codecs and generative foundations

| Paper and source | Authors | Year | Topic |
|---|---|---|---|
| [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312) | Neil Zeghidour et al. | 2021 | End-to-end neural codec with residual vector quantization |
| [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) | Alexandre Défossez et al. | 2022 | EnCodec; neural audio compression |
| [High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546) | Rithesh Kumar et al. | 2023 | Descript Audio Codec (DAC) |
| [FunCodec: A Fundamental, Reproducible and Integrable Open-source Toolkit for Neural Speech Codec](https://arxiv.org/abs/2309.07405) | Zhihao Du et al. | 2023 | Speech-codec toolkit and reproducible recipes |
| [HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec](https://arxiv.org/abs/2305.02765) | Dongchao Yang et al. | 2023 | Group-residual vector quantization |
| [SemantiCodec: An Ultra Low Bitrate Semantic Audio Codec for General Sound](https://arxiv.org/abs/2405.00233) | Haohe Liu et al. | 2024 | Semantic/acoustic encoders and diffusion decoder |
| [WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling](https://arxiv.org/abs/2408.16532) | Shengpeng Ji et al. | 2024 | Single-quantizer VQ codec; not LFQ |
| [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) | Aaron van den Oord et al. | 2016 | Autoregressive raw-audio modeling |
| [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) | Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu | 2017 | VQ-VAE |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | Jonathan Ho, Ajay Jain, Pieter Abbeel | 2020 | DDPM |
| [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | Yang Song et al. | 2020 | Score-based SDE framework; ICLR 2021 |
| [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | Jonathan Ho, Tim Salimans | 2022 | Conditional/unconditional guidance |

## Evaluation and human preference

| Paper and source | Authors | Year | Topic |
|---|---|---|---|
| [Fréchet Audio Distance: A Metric for Evaluating Music Enhancement Algorithms](https://arxiv.org/abs/1812.08466) | Kevin Kilgour et al. | 2018 | FAD; audio-embedding distribution comparison |
| [MusicRL: Aligning Music Generation to Human Preferences](https://arxiv.org/abs/2402.04229) | Geoffrey Cideron et al. | 2024 | Reward-based and human-feedback music alignment |
| [Benchmarking Music Generation Models and Metrics via Human Preference Studies](https://arxiv.org/abs/2506.19085) | Florian Grötschla et al. | 2025 | Human preference comparisons and metric evaluation |
| [Aligning Text-to-Music Evaluation with Human Preferences](https://arxiv.org/abs/2503.16669) | Yichen Huang et al. | 2025 | MusicPrefs and MAUVE Audio Divergence (MAD) |
| [Aligning Generative Music AI with Human Preferences: Methods and Challenges](https://arxiv.org/abs/2511.15038) | Dorien Herremans, Abhinaba Roy | 2025 | Perspective on alignment methods and challenges |

## Singing and accompaniment

| Paper and source | Authors | Year | Topic |
|---|---|---|---|
| [DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446) | Jinglin Liu et al. | 2021 | Score-conditioned SVS; AAAI 2022 |
| [SingSong: Generating musical accompaniments from singing](https://arxiv.org/abs/2301.12662) | Chris Donahue et al. | 2023 | AudioLM-based accompaniment from input vocals |

## Style, datasets, and cultural analysis

| Paper and source | Authors | Year | Topic |
|---|---|---|---|
| [Measuring the evolution of contemporary western popular music](https://arxiv.org/abs/1205.5651) | Joan Serrà et al. | 2012 | Corpus-based pitch, timbre, and loudness analysis |
| [The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use](https://arxiv.org/abs/1306.1461) | Bob L. Sturm | 2013 | Genre-dataset audit and evaluation methodology |
| [Da-TACOS: A Dataset for Cover Song Identification and Understanding](https://archives.ismir.net/ismir2019/paper/000038.pdf) | Furkan Yesiler et al. | 2019 | Original ISMIR paper; distinct analysis/benchmark subsets |

## Surveys

| Paper and source | Authors | Year | Topic |
|---|---|---|---|
| [Deep Learning Techniques for Music Generation — A Survey](https://arxiv.org/abs/1709.01620) | Jean-Pierre Briot, Gaëtan Hadjeres, François-David Pachet | 2017 | Generation objectives, representations, architectures, and strategies |
| [A Comprehensive Survey on Deep Music Generation: Multi-level Representations, Algorithms, Evaluations, and Future Directions](https://arxiv.org/abs/2011.06801) | Shulei Ji, Jing Luo, Xinyu Yang | 2020 | Score, performance, and audio generation |
| [A Survey on Recent Deep Learning-driven Singing Voice Synthesis Systems](https://arxiv.org/abs/2110.02511) | Yin-Ping Cho et al. | 2021 | Historical neural SVS survey |
| [Foundation Models for Music: A Survey](https://arxiv.org/abs/2408.14340) | Yinghao Ma et al. | 2024 | Representation, generation, multimodality, control, agents |
| [Discrete Audio Tokens: More Than a Survey!](https://arxiv.org/abs/2506.10274) | Pooneh Mousavi et al. | 2025 | Survey and benchmarks for speech/music/general audio tokenizers |
| [A Survey on Evaluation Metrics for Music Generation](https://arxiv.org/abs/2509.00051) | Faria Binte Kader, Santu Karmaker | 2025 | Symbolic/audio evaluation taxonomy and limitations |

## Textbooks, tutorials, and software

- Meinard Müller, [*Fundamentals of Music Processing*, second edition](https://www.audiolabs-erlangen.de/fau/professor/mueller/bookFMP) (2021): author-maintained book page and computational examples.
- [Open Music Theory](https://viva.pressbooks.pub/openmusictheory/) and Robert Hutchinson’s [Music Theory for the 21st-Century Classroom](https://musictheory.pugetsound.edu/mt21c/): open theory textbooks.
- [CompMusic](https://compmusic.upf.edu/): culturally specific music-information research project.
- [Open-Source Tools & Data for Music Source Separation](https://source-separation.github.io/tutorial/intro/src_sep_101.html): a tutorial, not a survey paper of the same title.
- [Chromaprint](https://acoustid.org/chromaprint): fingerprinting for near-identical recordings, not a general cover-song identification algorithm.

## Citation corrections

Entries whose supplied title/author/year combination could not be identified are not retained, including Brée’s “AI and Music: A Comprehensive Survey,” Huang’s “Music Style Modeling and Generation” thesis, Serrà’s “Correlation and Causality in Music Style Construction,” Hung’s “Emotional Music Generation via Disentangled Representations,” and the McKinney MIR survey. Identifiable publications above cover related topics. Unverified does not mean proven nonexistent.

The former Make-It-Music / SongBench entry used arXiv:2502.19324, which actually identifies a [cosmic-ray anisotropy paper](https://arxiv.org/abs/2502.19324), not music research. Other unexplained acronyms, product capabilities, and ambiguously cited textbook editions have been removed from this verified list; they can be reintroduced with exact original sources.

For reading order and scope, see the [survey guide](../docs/surveys/reading-guide.md).
