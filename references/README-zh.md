# 参考文献阅读列表

> English version: [README.md](README.md)

## 音乐理解

| 论文 | 年份 | 主题 |
|------|------|------|
| Music Tagging Transformers (Won et al.) | 2021 | 自动标签标注 |
| CLAP (Wu et al.) | 2023 | 音频-语言对比学习 |
| MusicFM (Kumar et al.) | 2024 | 音乐基础模型 |

## 音乐生成

| 论文 | 年份 | 主题 |
|------|------|------|
| Music Transformer (Huang et al.) | 2018 | 符号音乐生成 |
| Jukebox (Dhariwal et al.) | 2020 | 基于 VQ-VAE 的音频生成 |
| AudioLDM (Liu et al.) | 2023 | 音频潜在扩散模型 |
| MusicGen (Copet et al.) | 2023 | 文本到音乐生成 |
| Stable Audio (Roberts et al.) | 2023 | 潜在扩散模型，实时生成 |
| YuE (team) | 2025 | 完整歌曲生成 |
| ACE-Step (ACE Studio + 阶跃星辰) | 2025 | 基础模型: DCAE + 线性 Transformer + 扩散 (REPA) |
| MusicFlow (Prajwal et al., Meta) | 2024 | 级联流匹配文本到音乐 (ICML 2024) |
| SongCreator (Shun Lei et al.) | 2024 | 基于歌词的通用歌曲生成 (NeurIPS 2024, DSLM) |
| MusicFX (Google) | 2023--25 | 消费级文本到音乐 (基于 MusicLM) |

## 视频到音乐生成

| 论文 | 年份 | 主题 |
|------|------|------|
| CMT (对比多模态 Transformer) | 2025 | 通过对比跨模态对齐的视频到音乐 |
| M2UGen | 2025 | 多模态音乐理解与生成 |
| Video2Music | 2025 | 视频条件背景音乐 |
| MuVi | 2025 | 视觉到音乐，节奏对齐 |

## 人类偏好对齐

| 论文 | 年份 | 主题 |
|------|------|------|
| 音乐生成模型与指标的人类偏好基准研究 | 2025 | 人类偏好基准 (ICASSP 2025) |
| Aligning Generative Music AI with Human Preferences | 2025 | 偏好对齐 (AAAI 2025) |
| Make-It-Music / SongBench (南开大学) | 2025 | 带监督质量标签的歌曲生成框架 (arXiv:2502.19324) |

## 歌声合成 (新增)

| 论文 | 年份 | 主题 |
|------|------|------|
| OpenDiffSinger (社区) | 2022--25 | DiffSinger 开源分支，多说话人/多语言 |
| ACE Studio / ACE Singer | 2024--25 | 商业 SVS 系统，多语言，DAW 集成 |
| DiffSinger 加速 | 2025 | 通过蒸馏加速 SVS 推理（多种方法） |

## 音频工程

| 论文 | 年份 | 主题 |
|------|------|------|
| EnCodec (Defossez et al.) | 2022 | 神经音频编解码器，SEANet 架构，RVQ |
| SoundStream (Zeghidour et al.) | 2021 | 首个端到端神经音频编解码器 |
| DAC / Improved RVQGAN (Kumar et al.) | 2023 | 高保真编解码器，改进的 RVQ 训练 |
| FunCodec (Du et al.) | 2023 | 开源神经编解码器工具包 |
| SemantiCodec (Liu et al.) | 2024 | 双编码器语义/声学编解码器 |
| WavTokenizer (Pan et al.) | 2024 | 基于 LFQ 的单码本分词器 |
| HiFi-Codec (Yang et al.) | 2024 | 分组 RVQ 高效编解码器 |
| TQCodec | 2025/2026 | 网格量化，高保真音乐 |
| SUNAC (MERL) | 2026 | 源感知统一神经音频编解码器 |
| WaveNet (van den Oord et al.) | 2016 | 用于音频的膨胀因果卷积 |
| DDPM (Ho et al.) | 2020 | 去噪扩散概率模型 |
| Score-Based SDE (Song et al.) | 2021 | 统一的扩散模型 SDE 框架 |
| CFG (Ho & Salimans) | 2022 | 无分类器扩散引导 |
| AudioLDM (Liu et al.) | 2023 | 音频潜在扩散模型 |
| AudioLDM 2 (Liu et al.) | 2024 | 统一音频生成，IEEE TASLP |
| Stable Audio (Roberts et al.) | 2024 | 时间条件潜在扩散模型 |
| AudioLM (Borsos et al.) | 2022 | 分层音频语言模型 |
| MusicLM (Agostinelli et al.) | 2023 | 基于分层 token 的文本到音乐生成 |
| MusicGen (Copet et al.) | 2023 | Transformer + EnCodec 音乐生成 |
| VQ-VAE (van den Oord et al.) | 2017 | 神经离散表示学习 |
| SincNet (Ravanelli & Bengio) | 2018 | 可学习的音频前端 |
| MERT (Min et al.) | 2023 | 音乐基础模型，自监督学习 |
| CLAP (Wu et al.) | 2023 | 对比语言-音频预训练 |
| Brown (CQT) | 1991 | 音乐恒定 Q 变换 |
| Schorkhuber & Klapuri | 2010 | 音乐处理 CQT 工具箱 |

## 乐理与基础

| 教材/论文 | 年份 | 主题 |
|-----------|------|------|
| Piston, W. *Harmony*（修订版） | 1987 | 功能和声经典教材 |
| Kostka, S. & Payne, D. *Tonal Harmony*（第 8 版） | 2018 | 现代和声学标准教材 |
| Aldwell, E. & Cadwallader, A. *Harmony and Voice Leading*（第 4 版） | 2018 | 声部进行导向的和声学 |
| Krumhansl, C.L. *Cognitive Foundations of Musical Pitch* | 1990 | 音高感知认知科学，调性检测理论基础 |
| Bregman, A.S. *Auditory Scene Analysis* | 1990 | 听觉场景分析，source separation 认知基础 |
| Müller, M. *Fundamentals of Music Processing*（第 2 版） | 2021 | 计算音乐学教材，含 MIR 算法 |
| Zbikowski, L.M. *Conceptualizing Music* | 2002 | 音乐认知与计算模型桥梁 |

## 综述文献

| 论文 | 年份 | 范围 |
|------|------|------|
| A Survey on Deep Learning for Music Generation | 2023 | 全面音乐生成综述 |
| Music Source Separation: A Brief Overview | 2023 | 音源分离 |
| Discrete Audio Tokens: More Than a Survey (arXiv 2506.10274) | 2025 | 音频离散化全面综述 |
| Discrete Tokenization for Multimodal LLMs | 2025 | 面向多模态系统的 VQ/RVQ 分词化 |
| Codec SUPERB | 2024 | 神经音频编解码器基准评测 |

---

本列表为起点内容——欢迎补充贡献。
