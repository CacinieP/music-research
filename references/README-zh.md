# 参考文献阅读列表

[English](README.md)

核对日期：2026-09-19。论文均链接到原始来源；年份默认表示所链接预印本的首次提交年份，若同时列会议年份则明确区分。产品发布、预印本与正式出版可能不同年。这里核对文献身份与简要主题，不代表重新复现其实验结果。

## 音乐理解

| 论文与来源 | 作者 | 年份 | 主题 |
|---|---|---|---|
| [Semi-Supervised Music Tagging Transformer](https://arxiv.org/abs/2111.13457) | Minz Won, Keunwoo Choi, Xavier Serra | 2021 | 卷积与自注意力音乐标签标注 |
| [Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687) | Yusong Wu et al. | 2022 | LAION-CLAP，音频文本对比学习 |
| [A Foundation Model for Music Informatics](https://arxiv.org/abs/2311.03318) | Minz Won, Yun-Ning Hung, Duc Le | 2023 | MusicFM，自监督音乐表示 |
| [MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training](https://arxiv.org/abs/2306.00107) | Yizhi Li et al. | 2023 | 声学与音乐教师目标，ICLR 2024 |

## 音乐生成

| 论文与来源 | 作者 | 年份 | 主题 |
|---|---|---|---|
| [Music Transformer](https://arxiv.org/abs/1809.04281) | Cheng-Zhi Anna Huang et al. | 2018 | 符号音乐相对注意力 |
| [Jukebox: A Generative Model for Music](https://arxiv.org/abs/2005.00341) | Prafulla Dhariwal et al. | 2020 | VQ-VAE 与自回归音频建模 |
| [AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/abs/2209.03143) | Zalán Borsos et al. | 2022 | 分层语义与声学 token 建模 |
| [MusicLM: Generating Music From Text](https://arxiv.org/abs/2301.11325) | Andrea Agostinelli et al. | 2023 | 文本条件音乐音频生成 |
| [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://arxiv.org/abs/2301.12503) | Haohe Liu et al. | 2023 | 音频连续潜在扩散 |
| [AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) | Haohe Liu et al. | 2023 | 共享音频表示与潜在扩散 |
| [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284) | Jade Copet et al. | 2023 | MusicGen，文本／旋律条件编解码器语言模型 |
| [Fast Timing-Conditioned Latent Audio Diffusion](https://arxiv.org/abs/2402.04825) | Zach Evans et al. | 2024 | Stable Audio 研究模型，文本与时间条件 |
| [MusicFlow: Cascaded Flow Matching for Text Guided Music Generation](https://arxiv.org/abs/2410.20478) | K R Prajwal et al. | 2024 | 级联语义与声学流匹配 |
| [SongCreator: Lyrics-based Universal Song Generation](https://arxiv.org/abs/2409.06029) | Shun Lei et al. | 2024 | 人声与伴奏双序列语言模型 |
| [YuE: Scaling Open Foundation Models for Long-Form Music Generation](https://arxiv.org/abs/2503.08638) | Ruibin Yuan et al. | 2025 | 长时歌词到歌曲生成 |
| [ACE-Step: A Step Towards Music Generation Foundation Model](https://arxiv.org/abs/2506.00045) | Junmin Gong et al. | 2025 | DCAE、线性 Transformer、扩散与 REPA |
| [ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation](https://arxiv.org/abs/2602.00744) | Junmin Gong et al. | 2026 | 后续版本，语言模型规划与 DiT 生成 |
| [Multitrack Music Transformer](https://arxiv.org/abs/2207.06983) | Hao-Wen Dong et al. | 2022 | 多轨符号表示，ICASSP 2023 |

## 视频与多模态条件

| 论文与来源 | 作者 | 年份 | 主题 |
|---|---|---|---|
| [Video Background Music Generation with Controllable Music Transformer](https://arxiv.org/abs/2111.08380) | Shangzhe Di et al. | 2021 | CMT 指 Controllable Music Transformer（可控音乐 Transformer） |
| [M²UGen: Multi-modal Music Understanding and Generation with the Power of Large Language Models](https://arxiv.org/abs/2311.11255) | Shansong Liu et al. | 2023 | 多模态音乐理解与生成 |
| [Video2Music: Suitable Music Generation from Videos using an Affective Multimodal Transformer model](https://arxiv.org/abs/2311.00968) | Jaeyong Kang, Soujanya Poria, Dorien Herremans | 2023 | 视频条件符号音乐框架 |
| [MuVi: Video-to-Music Generation with Semantic Alignment and Rhythmic Synchronization](https://arxiv.org/abs/2410.12957) | Ruiqi Li et al. | 2024 | 语义与节奏对齐的视频条件音频 |

## 编解码器与生成基础

| 论文与来源 | 作者 | 年份 | 主题 |
|---|---|---|---|
| [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312) | Neil Zeghidour et al. | 2021 | 采用残差向量量化的端到端神经编解码器 |
| [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) | Alexandre Défossez et al. | 2022 | EnCodec，神经音频压缩 |
| [High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546) | Rithesh Kumar et al. | 2023 | Descript Audio Codec（DAC） |
| [FunCodec: A Fundamental, Reproducible and Integrable Open-source Toolkit for Neural Speech Codec](https://arxiv.org/abs/2309.07405) | Zhihao Du et al. | 2023 | 语音编解码器工具与可复现配方 |
| [HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec](https://arxiv.org/abs/2305.02765) | Dongchao Yang et al. | 2023 | 分组残差向量量化 |
| [SemantiCodec: An Ultra Low Bitrate Semantic Audio Codec for General Sound](https://arxiv.org/abs/2405.00233) | Haohe Liu et al. | 2024 | 语义／声学编码器与扩散解码器 |
| [WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling](https://arxiv.org/abs/2408.16532) | Shengpeng Ji et al. | 2024 | 单量化器 VQ 编解码器，不是 LFQ |
| [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) | Aaron van den Oord et al. | 2016 | 自回归原始音频建模 |
| [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) | Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu | 2017 | VQ-VAE |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | Jonathan Ho, Ajay Jain, Pieter Abbeel | 2020 | DDPM |
| [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | Yang Song et al. | 2020 | 基于分数的 SDE 框架，ICLR 2021 |
| [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | Jonathan Ho, Tim Salimans | 2022 | 条件与无条件预测引导 |

## 评测与人类偏好

| 论文与来源 | 作者 | 年份 | 主题 |
|---|---|---|---|
| [Fréchet Audio Distance: A Metric for Evaluating Music Enhancement Algorithms](https://arxiv.org/abs/1812.08466) | Kevin Kilgour et al. | 2018 | FAD，音频嵌入分布比较 |
| [MusicRL: Aligning Music Generation to Human Preferences](https://arxiv.org/abs/2402.04229) | Geoffrey Cideron et al. | 2024 | 基于奖励与人工反馈的音乐对齐 |
| [Benchmarking Music Generation Models and Metrics via Human Preference Studies](https://arxiv.org/abs/2506.19085) | Florian Grötschla et al. | 2025 | 人工偏好比较与指标评估 |
| [Aligning Text-to-Music Evaluation with Human Preferences](https://arxiv.org/abs/2503.16669) | Yichen Huang et al. | 2025 | MusicPrefs 与 MAUVE Audio Divergence（MAD） |
| [Aligning Generative Music AI with Human Preferences: Methods and Challenges](https://arxiv.org/abs/2511.15038) | Dorien Herremans, Abhinaba Roy | 2025 | 偏好对齐方法与挑战的观点论文 |

## 歌声与伴奏

| 论文与来源 | 作者 | 年份 | 主题 |
|---|---|---|---|
| [DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446) | Jinglin Liu et al. | 2021 | 乐谱条件歌声合成，AAAI 2022 |
| [SingSong: Generating musical accompaniments from singing](https://arxiv.org/abs/2301.12662) | Chris Donahue et al. | 2023 | 基于 AudioLM，从输入歌声生成伴奏 |

## 风格、数据集与文化分析

| 论文与来源 | 作者 | 年份 | 主题 |
|---|---|---|---|
| [Measuring the evolution of contemporary western popular music](https://arxiv.org/abs/1205.5651) | Joan Serrà et al. | 2012 | 基于语料的音高、音色与响度分析 |
| [The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use](https://arxiv.org/abs/1306.1461) | Bob L. Sturm | 2013 | 流派数据集审计与评测方法 |
| [Da-TACOS: A Dataset for Cover Song Identification and Understanding](https://archives.ismir.net/ismir2019/paper/000038.pdf) | Furkan Yesiler et al. | 2019 | ISMIR 原始论文，区分分析与基准子集 |

## 综述

| 论文与来源 | 作者 | 年份 | 主题 |
|---|---|---|---|
| [Deep Learning Techniques for Music Generation — A Survey](https://arxiv.org/abs/1709.01620) | Jean-Pierre Briot, Gaëtan Hadjeres, François-David Pachet | 2017 | 生成目标、表示、架构与策略 |
| [A Comprehensive Survey on Deep Music Generation: Multi-level Representations, Algorithms, Evaluations, and Future Directions](https://arxiv.org/abs/2011.06801) | Shulei Ji, Jing Luo, Xinyu Yang | 2020 | 乐谱、演奏与音频生成 |
| [A Survey on Recent Deep Learning-driven Singing Voice Synthesis Systems](https://arxiv.org/abs/2110.02511) | Yin-Ping Cho et al. | 2021 | 神经 SVS 历史综述 |
| [Foundation Models for Music: A Survey](https://arxiv.org/abs/2408.14340) | Yinghao Ma et al. | 2024 | 表示、生成、多模态、控制与 agent |
| [Discrete Audio Tokens: More Than a Survey!](https://arxiv.org/abs/2506.10274) | Pooneh Mousavi et al. | 2025 | 语音、音乐、通用音频 tokenizer 综述与基准 |
| [A Survey on Evaluation Metrics for Music Generation](https://arxiv.org/abs/2509.00051) | Faria Binte Kader, Santu Karmaker | 2025 | 符号／音频评测分类与限制 |

## 教材、教程与软件

- Meinard Müller，[*Fundamentals of Music Processing*，第二版](https://www.audiolabs-erlangen.de/fau/professor/mueller/bookFMP)（2021）：作者维护的教材页与计算示例。
- [Open Music Theory](https://viva.pressbooks.pub/openmusictheory/) 与 Robert Hutchinson 的 [Music Theory for the 21st-Century Classroom](https://musictheory.pugetsound.edu/mt21c/)：开放乐理教材。
- [CompMusic](https://compmusic.upf.edu/)：重视文化特殊性的音乐信息研究项目。
- [Open-Source Tools & Data for Music Source Separation](https://source-separation.github.io/tutorial/intro/src_sep_101.html)：教程，不是同名综述论文。
- [Chromaprint](https://acoustid.org/chromaprint)：用于近乎相同录音的指纹匹配，不是通用翻唱识别算法。

## 引用清理说明

未保留无法明确核对标题／作者／年份的条目，包括 Brée 的“AI and Music: A Comprehensive Survey”、Huang 的“Music Style Modeling and Generation”学位论文、Serrà 的“Correlation and Causality in Music Style Construction”、Hung 的“Emotional Music Generation via Disentangled Representations”和 McKinney 的 MIR 综述。相关主题改用上方可识别的真实文献；未核实不等于证明不存在。

原 Make-It-Music / SongBench 条目的 arXiv:2502.19324 实际对应[宇宙射线各向异性论文](https://arxiv.org/abs/2502.19324)，不能作为音乐论文引用。其他缺少明确原始来源的简称、产品能力和未标明版本的教材已移出这份已核对列表；补全来源后可重新收录。

阅读顺序与范围见[综述指南](../docs/surveys/reading-guide-zh.md)。
