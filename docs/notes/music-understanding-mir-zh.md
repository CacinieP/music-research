# 音乐理解 / 音乐信息检索 (MIR)

截至 2025--2026 年领域现状。聚焦技术架构、数据集、基准测试与开放性问题。

> English version: [music-understanding-mir.md](music-understanding-mir.md)

---

## 1. 自动标注与分类

### 问题定义

自动为音频录音分配描述性标签（流派、情绪、乐器编制、声学特征）。通常表述为在 50--100+ 个标签词表上的多标签分类问题。

### 关键架构

#### 基于 CNN（2016--2020）
- **Choi et al., "Automatic Tagging Using Deep Convolutional Neural Networks," ISMIR 2016.** 在 mel-spectrogram 上使用 2D 卷积的 FCN。确立了深度学习可以超越手工特征用于标注任务。
- **Won et al., "Data-driven Hybrid Approaches," ISMIR 2020.** 比较了 ResNet、SENet（squeeze-and-excitation）等 CNN 架构用于音乐标注。基于 ResNet 风格架构、包含 4--5 个卷积块的标注 CNN 仍是强基线。

#### 基于 Transformer（2021--）
- **Won, Chun, Nieto, Serra, "Semi-Supervised Music Tagging Transformer," ISMIR 2021.** 将 Vision Transformer（ViT）风格架构应用于音频频谱图。浅层捕获局部声学特征；深层自注意力层建模全局时间结构。半监督训练利用未标注数据。在 MagnaTagATune 上取得 ROC-AUC **~0.914**，超越 CNN 基线。
- 核心洞察：Transformer 能捕获 CNN 因感受野有限而遗漏的长程时间依赖（如歌曲结构、重复动机）。

#### CLAP / 对比式音频-语言模型（2023--）
- **Wu et al., "Large-scale Contrastive Language-Audio Pretraining," arXiv 2211.06687, 2022 (CLAP).** 双编码器对比学习，将音频和文本对齐到共享潜在空间。在约 63 万音频-文本对上训练。支持使用自然语言提示进行零样本音频分类。并非音乐专用，但广泛应用于音乐任务。代码：[github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)。
- **T-CLAP (2024):** 时序增强 CLAP，改进音频中的时间推理能力。
- **CLaMP 3 (ACL 2025 Findings, [aclanthology.org/2025.findings-acl.133](https://aclanthology.org/2025.findings-acl.133/)):** 通过对比学习将所有主要音乐模态（乐谱、MIDI、音频、图像、文本）对齐到共享表示空间。支持跨模态和跨语言检索。在多项 MIR 任务上达到当前 SOTA。代码：[github.com/sanderwood/clamp3](https://github.com/sanderwood/clamp3)。

### 代表性数据集

| 数据集 | 规模 | 标签 | 备注 |
|--------|------|------|------|
| **MagnaTagATune** | ~25,863 个片段（各 30s） | 50 个标签 | 标准基准。标签噪声较大（众包标注）。Law et al., 2009 整理。 |
| **MTG-Jamendo** | ~55,000 首完整曲目 | 700+ 标签（流派、情绪、乐器） | 质量更高，逐曲标注。Bogdanov et al., 2019。 |
| **GTZAN** | 1,000 首曲目（各 30s） | 10 个流派类别 | 规模小但广泛用于流派分类。Tzanetakis & Cook, 2002。 |
| **FMA (Free Music Archive)** | ~106,574 首曲目 | 流派层级（8/16/161 个流派） | Defferrard et al., 2017。更大规模的流派分类。 |
| **NSynth** | ~305,000 个单音 | 乐器（11 个族）、音高、力度 | Google Magenta, Engel et al., 2017。音级标注，非曲级。 |

### 近似 SOTA 性能

- **MagnaTagATune (ROC-AUC):** Music Tagging Transformer ~0.914；基础模型（MERT 微调、CLAP）推至 0.92--0.93。
- **MTG-Jamendo (ROC-AUC):** ~0.92--0.94，取决于标签子集。
- **GTZAN 流派分类（准确率）:** 现代模型 >93%；该数据集存在已知问题（重复、长度不一致）。
- **NSynth 乐器分类（准确率）:** CNN/Transformer 在 mel-spectrogram 上 >95%；在该数据集上已接近解决。

### 开放性问题

- **标签噪声：** MagnaTagATune 标签为众包标注，一致性差。具有专家标注的新基准正在出现（如 MGPHot, 2025）。
- **长篇音乐：** 大多数模型处理固定长度片段（10--30s）。处理具有可变结构的完整曲目尚欠探索。
- **细粒度标签：** 流派分类体系具有文化特异性和争议性。基于语言模型的少样本和零样本标注（CLAP, CLaMP 3）是有前景的方向。
- **跨文化偏差：** 在西方流行音乐上训练的模型难以泛化到非西方音乐传统。

---

## 2. 音乐转录

### 问题定义

将原始音频转换为符号化音符表示（类 MIDI）：起始时间、结束时间、音高、力度，以及可选的乐器标签。被称为"音乐版的语音识别"。

### 钢琴转录

由于有对齐的 MIDI/音频数据集，这是研究最充分的场景。

#### 关键模型

- **Onsets and Frames (Hawthorne et al., "Onsets and Frames: Dual-Objective Piano Transcription," ISMIR 2018).**
  - 架构：频谱图上的 CNN 馈入两个独立的 LSTM 堆栈——一个用于起始检测，一个用于帧级音高分类。起始检测结果调节帧预测。
  - 在 MAPS 上训练：MAESTRO 测试集 note-level F1 ~50%。在 MAESTRO 上训练：note-level F1 ~67%。
  - Google Magenta 实现。相较之前的 HMM 方法有重大飞跃。

- **高分辨率钢琴转录 (Kong et al., 2020--2021).**
  - 架构：基于回归的起始/结束检测 + 音高分类，使用高分辨率特征图的 CNN。
  - 报告在 MAESTRO 测试集上 note-level F1 ~90--93%（含偏移容忍）。首批突破 90% 的模型之一。
  - 采用回归预测起始/结束时间，而非二值分类。

- **基于 Transformer 的钢琴转录 (Hawthorne et al., 2022).**
  - 将 Transformer 编码器应用于频谱图，联合预测起始/结束/音高/力度。在 MAESTRO 上进一步提升。

- **Onsets and Velocities (2023).** 轻量级模型，以更高效的架构在 MAESTRO v3 上达到 SOTA 起始检测性能。

#### 现代系统（2023--2025）在 MAESTRO 上的钢琴转录 note-level F1 已达 93--97%（frame-level F1 >95%）。

### 多乐器转录

- **MT3 (Gardner et al., "MT3: Multi-Task Multitrack Music Transcription," ICLR 2022, [arXiv 2111.03017](https://arxiv.org/abs/2111.03017)).**
  - 架构：T5 编码器-解码器 Transformer。将转录视为序列到序列任务：输入音频频谱图，输出音符事件的 token 序列。每个音符 token 包含乐器标签 + 音高 + 起始/结束/力度。
  - 同时在多个数据集上训练（多任务），使单一模型能转录任意乐器组合。
  - 基于 Google 的 T5X 框架。代码：[github.com/magenta/mt3](https://github.com/magenta/mt3)。
  - 无需为每种乐器训练独立模型。

- **2025 AMT Challenge (NeurIPS 2025, [arxiv.org/html/2603.27528v1](https://arxiv.org/html/2603.27528v1)).**
  - 多乐器转录社区基准。新测试集，云端评估。
  - MT3 作为基线。多个团队提交了改进方案。
  - 将评估从钢琴扩展到真实多乐器混合。

- **YourMT3+ (2024).** 基于 MT3 构建的多任务多轨模型训练工具包。

- **CountEM (ISMIR 2025).** 使用音符事件直方图作为监督信号，无需精确时间对齐的 MIDI。降低了对昂贵对齐数据的依赖。

### 鼓转录

- 专注于转录鼓击（底鼓、军鼓、踩镲等）的独立子领域，需输出起始时间和鼓件标签。
- 数据集：ENST-Drums, RBMA-13, SoundBrush。
- 在 mel-spectrogram 上使用 CNN 和 CRNN 方法；通常表述为帧级多标签分类。
- 由于数据集较小且音色变化更大，进展不如钢琴转录。

### 代表性数据集

| 数据集 | 内容 | 规模 | 备注 |
|--------|------|------|------|
| **MAPS** | 钢琴（合成 + 录制） | ~240 首作品 | 对齐的 MIDI/音频。Emiya et al., 2010。首个标准基准。 |
| **MAESTRO** | 钢琴（真实演奏） | ~200 小时, ~1,282 场演奏 | 来自 International Piano-e-Competition 的对齐 MIDI/音频。Hawthorne et al., 2019。钢琴转录的金标准。 |
| **MusicNet** | 多种乐器（合奏） | 330 首录音 | Thickstun et al., 2017。多乐器，但对齐质量参差不齐。 |
| **SLAKH** | 多乐器（合成） | ~2,100 个混合 | Manilow et al., 2019。由独立合成的乐器轨构建。 |
| **URMP** | 多乐器（二重奏到五重奏） | 44 场演奏 | Li et al., 2018。规模小但高质量的多乐器数据。 |

### 近似 SOTA 性能

- **钢琴 (MAESTRO, 含偏移的 note-level F1):** 顶级系统达 93--97%。Frame-level F1 >95%。
- **钢琴 (MAESTRO, 不含偏移的 note-level F1):** 较低，约 85--90%，因为偏移检测更难。
- **多乐器 (MT3 on Slakh):** 起始 F1 因乐器而异：钢琴/小提琴 ~80--85%，贝斯/吉他 ~70--75%。整体低于纯钢琴。
- **鼓：** 起始 F1 ~75--85%，取决于鼓件和数据集。

### 开放性问题

- **真实混合中的多乐器转录** 仍远未解决。重叠谐波、房间声学和多样的音色使其比纯钢琴难得多。
- **表现性演奏转录：** 力度、发音、踏板（延音、弱音踏板）、微时序。
- **人声转录：** 在有伴奏的情况下追踪歌声的音高仍具挑战性。
- **训练数据瓶颈：** 对齐的 MIDI/音频采集成本高。自监督和弱监督方法（CountEM）是有前景的方向。
- **泛化能力：** 在一个数据集上训练的模型在不同乐器或录音条件下往往性能显著下降。

---

## 3. 源分离

### 问题定义

将混合音频信号分解为其组成声源（如人声、鼓、贝斯、其他）。最常见的表述为 4 轨分离（人声、鼓、贝斯、其他），遵循 MUSDB18 基准。

### 关键模型

#### 频谱图域方法
- **Spleeter (Henaff et al., "Spleeter: A Fast and Efficient Music Source Separation Tool," ISMIR 2019).** 在 mel-spectrogram 上运行的 U-Net。为每个音轨预测幅度掩码。速度快（100x 实时），但质量受限于频谱图相位重建。在 Deezer 内部数据集上预训练。代码：[github.com/deezer/spleeter](https://github.com/deezer/spleeter)。

#### 波形域方法
- **Demucs (Defossez et al., "Music Source Separation in the Waveform Domain," arXiv 1911.13254, 2019).** 直接在原始波形上运行的 U-Net 风格编码器-解码器。瓶颈层使用双向 LSTM。SDR 超越 SOTA 0.3+ dB。无需频谱图域。

- **Hybrid Demucs (Defossez et al., 2021).** 结合频谱和波形分支。频谱分支处理精细频率结构；波形分支处理时间模式。所有音轨 SDR 均有提升。

#### 混合 Transformer 方法
- **HT Demucs (Rouard & Massa, "Hybrid Transformers for Music Source Separation," ICASSP 2023).** 在 Hybrid Demucs 基础上增加 Transformer 层，用于时间和频率域的长程上下文建模。微调版本在 MUSDB18-HQ 上达到 SDR ~9.2--10.5 dB。代码：[github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)。

#### 频带分割架构族（2023--）
- **BandSplit RNN (BSRNN) (Luo et al., "Music Source Separation with Band-Split RNN," ICASSP 2023, [arXiv 2209.15174](https://arxiv.org/abs/2209.15174)).**
  - 将频谱划分为不重叠的频带。每个频带由共享 RNN 处理。频带级特征随后合并。
  - 赢得 URGENT 2025 挑战赛。
  - 核心洞察：独立处理频带后再合并，能同时捕获局部频谱模式和全局结构。

- **BS-RoFormer (Band-Split RoPE Transformer)（当前 SOTA）.**
  - 用 Rotary Position Embedding (RoPE) 注意力 / Transformer 层替换 BSRNN 的 RNN 模块。
  - **MUSDB18-HQ 上当前整体 SOTA**: L=12 配置下 SDR 达 **~12.0 dB（中位数）** 和 **~13.3 dB（均值）**。
  - 赢得相关竞赛，领跑所有公开排行榜。

- **Band-SCNet (Interspeech 2025).** 因果、轻量级模型，实时场景下 SDR 达 7.79 dB。

#### 其他值得关注的模型
- **SCNet:** 稀疏压缩网络。MUSDB18-HQ 上 SDR ~9.0--9.7 dB。

### 评估指标（BSS Eval 框架）

由 **Vincent et al., "Performance Measurement in Blind Audio Source Separation," IEEE Trans. Audio, 2006** 定义。实现在 `museval`（[github.com/sigsep/sigsep-mus-eval](https://github.com/sigsep/sigsep-mus-eval)）。

| 指标 | 全称 | 衡量内容 |
|------|------|----------|
| **SDR** | Source-to-Distortion Ratio | 整体分离质量（全局）。越高越好。单位：dB。 |
| **SIR** | Source-to-Interference Ratio | 其他声源的抑制程度。衡量来自其他乐器的串音/泄漏。 |
| **SAR** | Source-to-Artifact Ratio | 算法引入的伪影水平（金属声、音乐噪声、颤抖感）。 |
| **ISR** | Image-to-Spatial Distortion Ratio | 分离声源的空间保真度（立体声）。 |

SDR 是主要指标。MUSDB18-HQ 上的典型值：SOTA 模型整体 SDR ~9--12 dB；人声通常最容易（SDR ~10--14 dB），贝斯最难（SDR ~6--9 dB）。

### 代表性数据集

| 数据集 | 规模 | 音轨 | 备注 |
|--------|------|------|------|
| **MUSDB18** | 150 首完整歌曲 | 4 轨（人声、鼓、贝斯、其他） | 标准基准。Rafii et al., 2017。50 开发集 + 100 测试集。 |
| **MUSDB18-HQ** | 同 150 首，44.1 kHz | 同上 | 更高质量版本。报告 SDR 数值的标准。 |
| **MoisesDB** | 更大的多轨数据集 | 可变音轨数 | 作为下一代基准正在兴起。 |
| **Slakh2100** | 2,100 个合成混合 | 多轨 MIDI 合成 | Manilow et al., 2019。学术环境中用于训练/评估。 |

### SOTA 性能概览（MUSDB18-HQ, 整体 SDR dB）

| 模型 | 中位数 SDR | 均值 SDR | 备注 |
|------|-----------|----------|------|
| **BS-RoFormer (L=12)** | ~12.0 | ~13.3 | 当前 SOTA |
| **BS-RoFormer (L=6)** | ~9.8 | ~11.3 | |
| **HT Demucs (fine-tuned)** | ~9.2 | ~10.5 | 强基线，广泛使用 |
| **BSRNN** | 与 HT Demucs 相当 | | URGENT 2025 冠军 |
| **Spleeter** | ~5--6 | | 速度快但质量较低 |

### 开放性问题

- **真实音频的泛化能力：** 在 MUSDB18（主要是西方流行/摇滚）上训练的模型在非西方音乐、古典、电子等类型上性能下降。
- **超过 4 轨的分离：** 分离"其他"中的单独乐器（如分离两把吉他）。开放混音场景。
- **实时/低延迟分离：** Band-SCNet（SDR 7.79 dB）显示了与离线模型之间的质量差距。
- **伪影感知：** SDR 不能完全反映感知质量。某些 SDR 较低的模型听起来更好。更好的评估指标正在研究中（见 "SDR -- Half-Baked or Well Done?", MERL, 2019，以及 "Musical Source Separation Bake-Off", 2025）。
- **结合歌词条件的歌声分离：** 使用语言信息指导分离。

---

## 4. 音乐情感识别 (MER)

### 问题定义

预测音乐的情感内容，可以是分类标签（快乐、悲伤、愤怒、放松等），也可以是效价-唤醒度 (valence-arousal, V-A) 环状模型上的连续值 (Russell, 1980)。

### 方法

#### 维度式（效价-唤醒度回归）
- 从音频特征回归连续的效价（积极/消极）和唤醒度（平静/激昂）分数。
- 通常在 mel-spectrogram 或学习的音频表示上使用 CNN、LSTM 或 Transformer。
- 使用 MSE、Pearson 相关系数 (r) 或 R² 评估。
- 典型 Pearson r：效价 0.3--0.7（更难），唤醒度 0.4--0.8（更容易），取决于模型和数据集。

#### 分类式（情感类别分类）
- 分入离散情感类别（如 4 类：快乐、悲伤、愤怒、放松）。
- 可达到比细粒度维度预测更高的准确率。
- 方法：频谱图上的 CNN 分类器，通常从音乐标注模型迁移学习。

#### 多模态（音频 + 歌词 + 元数据）
- **BEE-MER (SMC 2025):** 双模态嵌入集成，结合音频和歌词表示用于静态 MER。
- **Music2Emo** ([huggingface.co/amaai-lab/music2emo](https://huggingface.co/amaai-lab/music2emo)): 统一多任务框架，整合分类和维度标签。
- 文本（歌词）提供语义内容；音频提供声学表达。两者结合提升性能。

#### 基于 Transformer 的 MER
- **Transformer 编码器方法 (ACM 2025):** 直接将 Transformer 编码器应用于音乐特征，映射到情感状态。
- **半监督多任务 MER (TISMIR 2025):** 利用大量带有弱情感标签的未标注音乐；在分类和维度目标上进行多任务学习。

### 代表性数据集

| 数据集 | 规模 | 标注 | 备注 |
|--------|------|------|------|
| **DEAM** | ~1,800 首歌曲 | 逐秒连续 V-A（动态） | Aljanaki et al., 2017。通过二维情感平面众包标注。维度 MER 的标准基准。 |
| **PMEmo** | ~794 首歌曲 | 静态 + 动态 V-A；包含生理信号（EEG, ECG, GSR） | Zhang et al., 2018。更丰富的标注。 |
| **Emotify** | 400 首曲目 | 8 种分类情感 | 众包分类标签。 |
| **MER-Arena** | 新兴 (2025) | 偏好对比 | 新的评估范式。 |

### 开放性问题

- **主观性：** 情感感知高度个人化和文化依赖。标注者间一致性低。
- **文化偏差：** 在西方听众标注上训练的模型无法迁移到其他文化语境。参见 MER 中文化偏差的研究 (JCBI, 2025)。
- **动态 vs 静态情感：** 大多数模型预测整首曲目的情感，但真实音乐具有时变的情感弧线。动态 MER（逐秒预测）更难且缺乏标准化。
- **超越效价-唤醒度：** 二维 V-A 模型只捕获了情感体验的一部分。更细致的模型（如 Hevner 模型的 13 种情感，或连续多维空间）尚未充分探索。
- **标注质量：** 众包情感标注噪声较大，反映标注者的文化背景、音乐训练和聆听环境。

---

## 5. 音乐基础模型

### 概述

从任务专用模型到大规模预训练模型的范式转变——后者学习通用音乐音频表示，然后可在下游任务上微调。类似于 NLP 中的 BERT/GPT。

### 关键模型

#### MERT (Music Audio Representation with Transformer)
- **Li, Yuan et al., "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training," ICLR 2023 ([arXiv 2306.00107](https://arxiv.org/abs/2306.00107)).**
- 在约 16 万首音乐曲目上进行自监督预训练。使用掩码音频建模：遮蔽音频的部分区域，训练 Transformer 预测被遮蔽的内容。
- 引入教师模型（从 HuBERT 等音频模型蒸馏）提供伪标签，使预训练更稳定。
- 提供 95M 和 330M 参数版本。
- 在 MIR 基准上评估：微调后在乐器分类、流派分类和音乐标注等多项任务上达到 SOTA。
- 代码：[github.com/yizhilll/MERT](https://github.com/yizhilll/MERT)。
- **CultureMERT (ISMIR 2025):** 对 MERT 进行持续预训练以实现跨文化音乐理解，缓解对西方音乐的偏差。

#### MusicFM
- **Won et al., "MusicFM: A Foundation Model for Music Informatics," arXiv 2311.03318, 2023.**
- 专为音乐信息学设计的自监督基础模型。在大规模音乐数据上预训练。
- 解决 MIR 中数据稀缺和泛化挑战。
- 在调性检测任务上持续报告较低性能（这是音频自监督模型的已知难题）。
- 代码：[github.com/minzwon/musicfm](https://github.com/minzwon/musicfm)。

#### JukeMIR
- 使用从 OpenAI **Jukebox** (Dhariwal et al., 2020) 音乐生成模型提取的表示进行下游 MIR 任务。
- Jukebox 是在 120 万首歌曲上训练的层次化 VQ-VAE。其内部表示编码了丰富的音乐结构。
- **Castellon et al., "Codified Audio Audio-Driven MIR," 2021.** 证明了 Jukebox 表示对情感、流派和标签预测有用。

#### MuQ
- 较新的自监督音乐音频模型，已在基准研究中与 MusicFM 和 MERT 进行比较。

#### SoniDo
- **"Music Foundation Model as Generic Booster," OpenReview 2024.** 提出 SoniDo 作为新的音频基础模型，描述其在下游音乐任务上的编码能力。

#### LLark (Spotify)
- **"LLark: A Multimodal Foundation Model for Music," Spotify Research, 2023 ([research.atspotify.com](https://research.atspotify.com/2023/10/llark-a-multimodal-foundation-model-for-music)).**
- 结合音频和文本的多模态语言模型，用于灵活的音乐理解和推理。
- 能回答关于音乐内容的问题，执行标注，并用自然语言描述音乐特征。

#### Qwen-Audio
- Qwen 模型家族的一部分。在 30+ 多样化音频任务上训练，包括分类、语音识别和情感识别。非音乐专用但适用。

### 基准评估：MARBLE
- **"MARBLE: Music Audio Representation Benchmark for Evaluation," 2024.**
- 综合性基准，在广泛的 MIR 任务上评估音乐音频表示：标注、乐器识别、流派、情绪、音高、节拍追踪、源分离、调性检测、分段。
- 实现 MERT、MusicFM、JukeMIR、CLAP 等模型的公平比较。

### 多模态方法

| 方法 | 模态 | 核心思想 |
|------|------|----------|
| **CLAP** | 音频 + 文本 | 对比对齐。通过文本提示实现零样本分类。 |
| **CLaMP 3** | 音频 + MIDI + 乐谱 + 文本 + 图像 | 通用跨模态 MIR。跨所有模态的对比预训练。 |
| **LLark** | 音频 + 文本 | 基于 LLM 的音乐推理和问答。 |
| **MusicFM** | 仅音频（可选文本） | 自监督音频表示。 |
| **MERT** | 仅音频 | 通过掩码建模的自监督音频表示。 |
| **音频+视频** | 音频 + 视觉 | 音乐视频理解、演奏分析。发展不如音频+文本。 |

### 基础模型的能力

- **线性探测 (Linear probing):** 冻结预训练编码器，在其上训练线性分类器。测试表示质量。
- **微调 (Fine-tuning):** 在下游任务上更新所有参数。通常达到最佳性能。
- **零样本 / 少样本:** 使用 CLAP 风格模型在无需任何任务特定训练数据的情况下进行分类。
- **特征提取:** 使用表示作为下游模型的输入特征。

### 开放性问题

- **调性检测仍然困难：** 所有基础模型（MERT, MusicFM, JukeMIR）在调性检测上报告的性能都较低，表明这些表示未能很好地捕获调性结构。
- **时间分辨率：** 基础模型通常以 50--75 Hz 的帧率运行，对于细粒度节奏任务可能过粗。
- **计算成本：** 预训练需要大量 GPU 资源（8--32 块 GPU 运行数周）。
- **评估方法论：** MARBLE 是一个进步，但在多样化音乐传统和真实场景上的评估仍然有限。
- **数据污染：** 预训练数据集通常不透明；难以验证与下游评估数据是否有重叠。

---

## 6. 节拍/速度追踪与和弦/调性识别

### 节拍与速度追踪

#### 问题定义
- **节拍追踪 (Beat tracking):** 检测音乐节拍的时刻（听众会跟随打拍的脉冲）。
- **强拍追踪 (Downbeat tracking):** 检测强拍的时刻（每小节的第一拍）。
- **速度估计 (Tempo estimation):** 估计录音的 BPM（每分钟拍数）。

#### 关键模型与演进

- **Ellis, "Beat Tracking by Dynamic Programming," JNMR 2007.** 使用起始检测函数和动态规划的经典方法。快速但受限于手工特征。

- **Bock & Davies, "Temporal Convolutional Networks for Musical Audio Beat Tracking," ISMIR 2020.**
  - 架构：带膨胀卷积的时间卷积网络 (TCN)。双向处理。
  - 直接处理 mel-spectrogram。输出节拍/强拍激活函数。
  - 通常后接动态贝叶斯网络 (DBN) 后处理步骤得到最终节拍时间。
  - 在大多数标准基准上 F-measure >85%。

- **Beat Transformer (Zhao et al., "Beat Transformer: Dilated Self-Attention for Joint Beat and Downbeat Tracking," ISMIR 2022).**
  - 膨胀自注意力机制，使模型能同时关注局部和长程时间模式。
  - 单一模型联合执行节拍和强拍追踪。
  - 在部分配置中不再需要单独的 DBN 后处理。

- **"Beat This!" (ISMIR 2024, [github.com/CPJKU/beat_this](https://github.com/CPJKU/beat_this)).**
  - 高精度节拍追踪器，完全消除了对 DBN 后处理的需求。
  - 当前节拍追踪的首选模型。实现简洁高效。
  - 在标准基准上达到强 F-measure。

- **双路径 TCN+Transformer (2024).** 结合 TCN（局部时间细节）和 Transformer（全局序列建模）。降低模型复杂度的同时保持精度。

- **Beat-U (MIREX 2025).** 多任务 U 型 Transformer，跨多个时间尺度进行音乐理解。联合处理节拍追踪、强拍追踪及相关序列 MIR 任务。

- **端到端 Transformer 用于演奏 MIDI (SMC 2025).** 编码器-解码器 Transformer，用于 MIDI（非音频）演奏的节拍/强拍追踪。

#### 关键数据集

| 数据集 | 内容 | 备注 |
|--------|------|------|
| **Ballroom** | 698 个片段 | 标准节拍追踪基准。舞曲。 |
| **Beatles** | 178 首 Beatles 曲目 | 标注了节拍与和弦。广泛使用。 |
| **Hainsworth** | 222 个片段 | 多种流派。 |
| **SMC (Soleym, MIREX, CMU)** | 210+ 个片段 | 包含困难案例（自由速度、表现性时序）。 |
| **GTZAN Rhythm** | 1,000 首曲目 | GTZAN 曲目的速度/节拍标注。 |
| **GiantSteps** | 电子音乐速度标注 | 660 首曲目，主要是电子舞曲。 |

#### 近似 SOTA 性能

- **节拍追踪 F-measure（标准窗口，如 70ms）:** Ballroom 上 85--92%；Beatles 上 80--88%；SMC（更难的集合）上 70--80%。"Beat This!" 和 TCN 模型领先。
- **强拍追踪 F-measure:** 通常比同一数据集上的节拍追踪低 5--10 个百分点。
- **速度估计准确率（与真值偏差 4% 以内）:** Ballroom 上 >90%；更多样化数据集上 80--85%。

### 和弦识别 (Automatic Chord Estimation, ACE)

#### 问题定义
识别录音中每个时间步的和弦标签（如 C:maj, G:min7, F#:dim）。

#### 方法

- **Chordino / NNLS Chroma (Mauch, 2010):** 使用 NNLS 色度特征 + HMM 解码的经典方法。仍然是有用的基线。可作为 Vamp 插件使用。

- **深度学习 (2015--):** 色度或频谱图特征上的 CNN 和 CRNN。ISMIR 2015 的论文（McLeod & Wyse 等）表明深度学习可以匹配或超越基于 HMM 的方法。

- **CNN-LSTM 混合模型:** CNN 提取局部频谱特征；LSTM 建模时间序列上的和弦进程。在标准基准上加权准确率 >80%。

- **基于 Transformer:** 自注意力捕获长程和声上下文（如根据周围的和声进程识别和弦）。在 Billboard 和 Isophonics 数据集上加权准确率达 80--85%+。

- **基础模型特征:** 使用 MERT、MusicFM 或 JukeMIR 表示作为和弦识别的输入特征。提升性能，尤其对复杂和弦（七和弦、减和弦、增和弦）。

- **合成音频训练 (2025, [arxiv.org/html/2508.05878v1](https://arxiv.org/html/2508.05878v1)):** 比较两种在合成音频上训练的 Transformer 模型用于和弦识别，解决数据稀缺问题。

#### 关键数据集

| 数据集 | 内容 | 和弦词表 | 备注 |
|--------|------|----------|------|
| **Billboard (McVicar et al.)** | ~200 首 Billboard Hot 100 流行/摇滚歌曲 | 24 个大/小调和弦 + 七和弦 | 标准 ACE 基准。Harte 和弦词表。 |
| **Isophonics** | Beatles, Queen, Carole King 专辑 | 大调、小调、7, maj7, min7 等 | 广泛使用。 |
| **Robbie Williams** | 55 首 Robbie Williams 曲目 | 大调、小调、7, maj7 等 | |
| **ChoTo** | 多种 | 大/小调 | 较小的基准。 |

#### 近似 SOTA 性能

- **加权准确率 (Billboard, 大/小调):** SOTA 系统达 80--87%。
- **加权准确率 (Isophonics, 大/小调):** 82--88%。
- **扩展和弦词表（七和弦等）:** 性能显著下降至 60--75%。

### 调性检测

- 估计录音的全局调性（如 C 大调、A 小调）。
- **经典方法：** Krumhansl-Schmuckler 调性查找算法，使用调性轮廓。
- **深度学习：** 色度特征上的 CNN 分类器；基础模型探测。
- **数据集：** GiantSteps Key（电子音乐）、meters.tsv（古典）。
- **SOTA 准确率：** 标准数据集上 ~70--85%。仍然是一个挑战性任务，尤其是对基础模型而言（如上所述）。

### 开放性问题

- **表现性时序：** 在有自由速度、速度变化和表现性时序的音乐（古典、爵士）上，节拍追踪性能显著下降。
- **复杂节拍：** 5/4、7/8 和不规则节拍被大多数在 4/4 流行音乐上训练的模型处理得不好。
- **层次化节奏：** 联合建模节拍、强拍和更高层级的节奏结构（乐句、段落）。
- **和弦词表：** 大多数系统处理大/小调很好，但在复杂和弦（九和弦、十一和弦、变化和弦、斜线和弦）上表现不佳。
- **调性检测：** 令人意外的是，基础模型在这方面并不突出。调中心估计可能需要超越通用音频特征的专门表示。
- **数据标注：** 和弦标签具有主观性（尤其对于模糊和声）。标注者间一致性为可达到的准确率设定了上限。

---

## 跨领域趋势（2024--2026）

1. **自监督预训练占主导地位。** MERT、MusicFM 及相关模型表明，在大规模音乐音频上进行自监督学习产生的表示在几乎所有 MIR 任务上都具有良好的迁移性。

2. **频带分割架构引领源分离。** BSRNN 和 BS-RoFormer 代表当前前沿，在 MUSDB18-HQ 上 SDR ~12 dB。独立处理频带的范式已被证明非常有效。

3. **Transformer 正在所有任务中取代 RNN/CNN。** 节拍追踪、和弦识别、源分离和转录都在向基于 Transformer 的架构迁移。趋势是从 HMM/DBN 后处理的混合模型转向端到端神经方法。

4. **多模态基础模型 (CLAP, CLaMP 3) 实现零样本 MIR。** 文本-音频对齐允许在无需任务特定训练数据的情况下进行分类，将 MIR 拓展到开放词表和跨语言场景。

5. **评估方法论正在演进。** MARBLE（基础模型基准评估）、2025 AMT Challenge（多乐器转录）以及新的专家标注基准正在提高评估标准。

6. **文化偏差日益受到关注。** 在西方流行音乐上训练的模型无法泛化到其他音乐传统。CultureMERT 等努力旨在解决这一问题。

7. **数据仍然是瓶颈。** 对于转录、源分离和情感识别，高质量标注数据稀缺且采集成本高。自监督、弱监督和合成数据生成是关键研究方向。

---

---

## 7. 音乐推荐

### 问题定义

基于音频内容、用户偏好或上下文向用户推荐音乐。音频推荐是 MIR 的直接应用——需要从音频中提取有意义的表示并计算相似度。

### 方法

#### 基于内容的过滤

| 方法 | 特征 | 备注 |
|------|------|------|
| 手工特征 | MFCC、色度、频谱特征 + 距离度量 | 经典方法，可解释但有限 |
| 嵌入向量 | 预训练嵌入（MERT、CLAP、MusicFM）+ 最近邻 | 当前标准 |
| 度量学习 | 从偏好数据学习距离函数 | 适合细粒度相似性 |

#### 音频嵌入用于推荐

现代推荐系统使用预训练音频嵌入：
- **MERT 嵌入**：自监督音乐表示，编码和声、节奏、音色信息
- **CLAP 嵌入**：音频-文本联合空间，支持基于文本的音乐搜索
- **MusicFM 嵌入**：大规模预训练表示，迁移到推荐任务

**相似度计算**：嵌入上的余弦相似度是标准。大规模目录下需用近似最近邻（FAISS、ScaNN）。

#### 混合系统

| 系统 | 内容信号 | 协同信号 | 备注 |
|------|---------|---------|------|
| Spotify（内部） | 音频特征 + NLP | 用户收听历史 | 行业标准，专有 |
| Spotify 公开 API | 音频特征（舞蹈性、能量、效价） | — | 有限但可访问 |

### 评测

| 指标 | 描述 |
|------|------|
| **Precision@K** | Top-K 推荐中相关项目的比例 |
| **Recall@K** | Top-K 中检索到的相关项目比例 |
| **NDCG@K** | 归一化折损累计增益 |
| **覆盖率** | 可推荐目录的比例 |
| **惊喜度** | 推荐的新颖性 |

### 开放问题

- **冷启动**：无收听历史的新曲目推荐
- **长尾**：推荐数据集中在流行曲目；小众音乐服务不足
- **上下文感知**：时间、活动、情绪作为额外信号
- **跨文化**：在西方音乐上训练的推荐系统对非西方听众效果差

---

## 8. 翻唱检测与版本识别

### 问题定义

识别两个录音是否是同一底层音乐作品的不同表演版本。翻唱是对现有歌曲的新演奏/新编曲。

### 为什么重要

- **音乐版权管理**：识别未授权翻唱用于版税分配
- **音乐发现**：找到用户喜欢的歌曲的不同版本
- **文化分析**：研究音乐作品如何跨表演演变

### 方法

#### 传统方法（2020 年前）

- **Chromaprint / AcoustID**：基于色度的指纹。对速度变化和调性移调鲁棒。
- **翻唱识别系统**：色度 + 动态时间规整（DTW）进行对齐不变比较。

#### 深度学习方法

| 方法 | 描述 |
|------|------|
| **色度图上的 2D-CNN** | 学习对编曲变化鲁棒的色度模式 |
| **孪生网络** | 学习翻唱对的相似度度量 |
| **Triplet loss** | 用锚-正（同作品）和锚-负（不同作品）训练 |
| **自监督预训练** | 在大规模音频语料上预训练，微调用于翻唱检测 |

### 数据集

| 数据集 | 内容 | 规模 | 备注 |
|---------|------|------|------|
| **SecondHandSongs** | 众包翻唱元数据 | ~1M 作品 | 最大元数据来源 |
| **Covers80** | 80 原唱+翻唱对 | 160 曲目 | 经典小基准 |
| **Da-TACOS** | 翻唱+原唱对 | ~17K | 大规模基准 |
| **Covers2001** | 查询-翻唱对 | ~1K 查询 | 标准评测集 |

### 评测指标

| 指标 | 描述 |
|------|------|
| **平均排名** | 正确匹配的平均排名 |
| **mAP** | 查询的平均精度均值 |
| **MRR** | 平均倒数排名 |
| **Top-K 准确率** | 正确翻唱在 Top-K 中的查询比例 |

### 开放问题

- **编曲变化**：同一首歌完全不同的配器（管弦乐→电子）
- **结构变化**：翻唱可能重新排序段落、增删段落
- **串烧**：多个歌合成一个表演
- **哼唱查询**：从哼唱查询找翻唱

---

## 9. 主旋律提取

### 问题定义

从复调音乐中提取主导旋律的音高轮廓。聚焦歌声时叫"声乐旋律提取"，更广的范围叫"音高追踪"。

### 为什么重要

- **下游任务**：翻唱检测、哼唱查询、音乐转录、歌声分析的前置输入
- **音乐制作**：自动旋律分离用于混音、卡拉 OK 伴奏生成
- **音乐教育**：学习用的音高可视化

### 方法

#### 传统方法

- **SALAMI** + 音高追踪：音源分离 + 对分离的旋律源做 F0 估计
- **P. Rao (2010)**：基于音高显著性函数的旋律提取

#### 深度学习方法

| 方法 | 架构 | 备注 |
|------|------|------|
| **CNN-based** | CNN 在 mel 频谱图上逐帧预测音高 | 快但时序上下文有限 |
| **CRNN** | CNN 特征 + LSTM 时序建模 | 时序连贯性更好 |
| **Transformer-based** | 对频谱图帧的自注意力 | 最先进，捕捉长程关系 |
| **无分割** | 端到端音高追踪，无需音符分割 | 当前趋势 |

### 关键系统

| 系统 | 年份 | 备注 |
|------|------|------|
| **Melodia** | 2013 | 经典音高轮廓提取，广泛使用 |
| **DeepSalience** | 2018 | 深度学习的音高显著性 |
| **MelodyCNN** | 2019 | CNN 旋律提取 |
| **pYIN** | 2015 | 概率 YIN，标准基线 |

### 评测

| 指标 | 描述 |
|------|------|
| **RPA** (原始音高准确率) | 估计音高在真实值 ±50 音分内的帧比例 |
| **RCA** (原始色度准确率) | RPA 忽略八度错误 |
| **OA** (整体准确率) | RPA + 正确有声/无声决策 |
| **VR** (有声召回率) | 正确识别为有声的帧比例 |
| **VF** (有声误报率) | 错误标记为有声的无声帧比例 |

典型 SOTA：MIREX 数据集上 RPA ~80–85%。复杂混响（管弦乐、高度复调）下性能下降。

### 开放问题

- **复杂混响中的旋律**：旋律不是最响的音源时（如全乐队中的轻柔人声）
- **复调旋律**：多条旋律线（如巴赫对位）
- **非声乐旋律**：器乐旋律（萨克斯、吉他主音）有不同的音色特征
- **实时提取**：交互式应用的低延迟旋律提取

---

## 10. 跨领域 MIR 挑战

### 10.1 MIR 中的文化偏倚

所有主要 MIR 基准（GTZAN、MAESTRO、MUSDB18）都由西方流行和古典音乐主导。后果：

- **泛化差**：在西方音乐上训练的模型在非西方传统上表现差
- **测量偏倚**：评测结果不代表全球音乐多样性
- **强化循环**：在西方数据上训练的 AI 生成音乐进一步强化西方风格

**应对 effort**：CultureMERT（多语言/文化预训练）、多样化基准集合、社区主导的数据集创建。

### 10.2 评测瓶颈

MIR 评测受限于：
- **标签质量**：众包标签（MagnaTagATune）有噪声。专家标注昂贵
- **数据集规模**：许多 MIR 数据集小（数百到数千例），限制模型容量
- **基准饱和**：部分任务（GTZAN 流派分类、MAESTRO 钢琴转录）接近性能上限
- **缺乏标准化**：不同论文用不同的训练/测试划分、指标和预处理

### 10.3 从实验室到生产

| 挑战 | 描述 |
|-----------|-------------|
| **实时需求** | 生产系统需要低延迟（部分应用 <100ms） |
| **鲁棒性** | 必须处理噪声录音、手机麦克风、压缩音频 |
| **可扩展性** | 数十亿曲目需要处理；高效推理至关重要 |
| **持续学习** | 新音乐风格、语言和艺术家不断出现 |
| **可解释性** | 为什么模型把这个标记为"爵士"？需要可解释性 |

---

## 11. MIR 任务全景

| 任务 | 输入 | 输出 | 当前 SOTA | 开放挑战 |
|------|------|------|----------|---------|
| 自动标签 | 音频 | 标签 | ~0.93 ROC-AUC | 细粒度、跨文化 |
| 转录 | 音频 | MIDI 音符 | 93–97% F1（钢琴） | 多乐器真实混响 |
| 音源分离 | 混合音频 | 分离轨道 | ~12 dB SDR | >4 轨、实时 |
| 情绪识别 | 音频 | 唤醒/效价 | 中等 | 标准化评测 |
| 节拍追踪 | 音频 | 节拍时间 | 85–92% F1 | 弹性速度、复杂节拍 |
| 和弦识别 | 音频 | 和弦标签 | 80–87% | 复杂和弦、模糊和声 |
| 调性检测 | 音频 | 调性标签 | 70–85% | 基础模型弱点 |
| 音乐推荐 | 音频+用户 | 曲目列表 | 行业实践 | 冷启动、长尾 |
| 翻唱检测 | 音频对 | 匹配/无 | 中等 | 编曲变化 |
| 旋律提取 | 音频 | 音高轮廓 | 80–85% RPA | 复杂混响 |

## 关键参考文献（按领域分类）

### 自动标注
- Choi et al., "Automatic Tagging Using Deep Convolutional Neural Networks," ISMIR 2016.
- Won, Chun, Nieto, Serra, "Semi-Supervised Music Tagging Transformer," ISMIR 2021.
- Wu et al., "Large-scale Contrastive Language-Audio Pretraining" (CLAP), arXiv 2211.06687, 2022.
- Wu et al., "CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages," ACL 2025 Findings.

### 音乐转录
- Hawthorne et al., "Onsets and Frames: Dual-Objective Piano Transcription," ISMIR 2018.
- Kong et al., "High-Resolution Piano Transcription," 2020--2021.
- Gardner et al., "MT3: Multi-Task Multitrack Music Transcription," ICLR 2022 (arXiv 2111.03017).
- "Advancing Multi-Instrument Music Transcription: Results from the 2025 AMT Challenge," NeurIPS 2025 (arXiv 2603.27528).

### 源分离
- Henaff et al., "Spleeter," ISMIR 2019.
- Defossez et al., "Music Source Separation in the Waveform Domain" (Demucs), arXiv 1911.13254, 2019.
- Luo et al., "Music Source Separation with Band-Split RNN" (BSRNN), ICASSP 2023 (arXiv 2209.15174).
- Rouard & Massa, "Hybrid Transformers for Music Source Separation" (HT Demucs), ICASSP 2023.
- Vincent et al., "Performance Measurement in Blind Audio Source Separation" (BSS Eval), IEEE Trans. Audio, 2006.

### 音乐情感
- Russell, "A Circumplex Model of Affect," J. Personality & Social Psychology, 1980.
- Aljanaki et al., "DEAM: Developing an Emotion Annotation Dataset for Music," ISMIR 2017.
- Zhang et al., "PMEmo: A Multimodal Dataset for Perceived Emotion Recognition," ICASSP 2018.
- "BEE-MER: Bimodal Embeddings Ensemble for Music Emotion Recognition," SMC 2025.

### 基础模型
- Li, Yuan et al., "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training," ICLR 2023 (arXiv 2306.00107).
- Won et al., "MusicFM: A Foundation Model for Music Informatics," arXiv 2311.03318, 2023.
- Castellon et al., "Codified Audio Audio-Driven MIR" (JukeMIR), 2021.
- "MARBLE: Music Audio Representation Benchmark for Evaluation," 2024.
- "Foundation Models for Music: A Survey," arXiv 2408.14340, 2024.

### 节拍/速度/和弦
- Ellis, "Beat Tracking by Dynamic Programming," JNMR 2007.
- Bock & Davies, "Temporal Convolutional Networks for Musical Audio Beat Tracking," ISMIR 2020.
- Zhao et al., "Beat Transformer: Dilated Self-Attention for Joint Beat and Downbeat Tracking," ISMIR 2022.
- "Beat This!" ISMIR 2024 ([github.com/CPJKU/beat_this](https://github.com/CPJKU/beat_this)).
- Harte et al., "Symbolic Representation of Musical Chords," ISMIR 2005 (chord vocabulary).
- Mauch, "Automatic Chord Transcription from Audio," PhD Thesis, 2010 (Chordino).
