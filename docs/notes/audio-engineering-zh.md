> English version: [audio-engineering.md](audio-engineering.md)

# 音乐AI的音频工程：技术笔记

全面的技术参考，涵盖特征表示、神经音频编解码器、扩散模型、实时处理和音频标记化。反映了截至2025-2026年的最新技术水平。

---

## 目录

1. [特征表示](#1-特征表示)
2. [神经音频编解码器](#2-神经音频编解码器)
3. [基于扩散的音频模型](#3-基于扩散的音频模型)
4. [实时音频处理](#4-实时音频处理)
5. [音频标记化与离散表示](#5-音频标记化与离散表示)

---

## 1. 特征表示

### 1.1 梅尔频谱图

现代音乐AI的主力表示方法。被MusicGen、AudioLDM、Stable Audio、Jukebox以及大多数音乐生成/源分离模型所使用。

**定义：**

梅尔频谱图是短时傅里叶变换（STFT）的幅度经过梅尔尺度滤波器组后的结果：

```
Mel_spec(t, m) = sum_k |STFT(t, k)|^2 * H_m(k)
```

其中 `H_m(k)` 是第m个梅尔滤波器组的三角窗。梅尔尺度近似人类的音高感知：

```
mel(f) = 2595 * log10(1 + f / 700)
```

**关键参数：**
- `n_fft`（FFT窗口大小）：音乐（44.1/48 kHz）通常为2048或4096
- `hop_length`：通常为 `n_fft / 4`（例如512）
- `n_mels`（梅尔频带数）：64、80或128
- `fmin`、`fmax`：频率范围（例如0-8000 Hz或0-22050 Hz）

**权衡：**
- **优点**：感知对齐；紧凑（n_mels远小于n_fft/2）；在PyTorch/Torchaudio中可微分；经过充分验证。
- **缺点**：相位信息被丢弃（无法在没有Griffin-Lim或神经声码器的情况下完美重建波形）；固定的时频分辨率；梅尔尺度是一个粗略近似。

**何时使用**：分类、生成、标注和大多数监督学习任务的默认选择。配合神经声码器（HiFi-GAN）用于波形生成。

### 1.2 色度特征

捕获与八度无关的音高级别内容。十二个区间对应12个半音（C、C#、D、...、B）。

**定义：**

色度特征将频谱表示（STFT、CQT）的频率轴映射到12个音高类别：

```
Chroma(t, c) = sum_{f in pitch_class(c)} |X(t, f)|
```

其中 `pitch_class(c)` 包含所有音高类别为 `c` 的频率。

**变体：**
- **色度STFT**：基于STFT幅度
- **色度CQT**：基于CQT（与音乐音高的频率对齐更好）
- **CENS色度特征（色度能量归一化统计量）**：经量化、平滑和归一化处理；对动态和音色具有鲁棒性

**何时使用**：和弦识别、调性检测、谐波分析、翻唱歌曲检测、音乐相似度。不适合作为生成任务的主要表示。

### 1.3 恒定Q变换（CQT）

提供几何间距的频率区间，匹配音乐音高的对数特性。品质因数Q（中心频率/带宽）在所有区间保持恒定。

**定义：**

```
X_CQT[k] = sum_n x[n] * w_k[n] * exp(-j * 2pi * f_k * n / f_s)
```

其中 `Q = f_k / delta_f_k` 为常数，`f_k` 为几何间距排列。

**关键性质：**
- 每个八度的区间数是用户参数（通常为12、24或48）
- 低频具有更好的频率分辨率，高频具有更好的时间分辨率
- 总区间数：`n_bins = B * log2(fmax / fmin)`，其中B = 每八度区间数

**计算考量：**
- 计算开销高于STFT；高效实现使用稀疏核或基于FFT的方法
- 库：`librosa.cqt()`、`nnAudio`（GPU加速，可微分）

**关键论文**：Brown (1991) "Calculation of a constant Q spectral transform"——奠基性工作。Schorkhuber & Klapuri (2010) 针对高效计算。

**何时使用**：自动音乐转录（AMT）、音高估计、和弦识别，以及任何需要与音乐音高结构对齐的任务。

### 1.4 变量Q变换（VQT）

CQT的推广，其中Q可以在不同频率区间变化，允许在不同频谱区域灵活权衡时间和频率分辨率。

**定义：**

VQT放宽了恒定Q约束。当Q的变化趋向匹配STFT行为（恒定带宽）时，VQT变为STFT。CQT是Q为常数时的特殊情况。

VQT通过"gamma"值进行参数化，在恒定Q（gamma=0）和恒定带宽（gamma=1）行为之间插值：

```
Effective bandwidth_k = Q_k * f_k + gamma
```

**实现**：`librosa.vqt()` 通过 `gamma` 参数同时支持CQT和VQT。

**何时使用**：当纯粹的频率对数间距（CQT）或纯粹的线性间距（STFT）都不理想时。在自适应分辨率有益的音乐转录和分析任务中逐渐兴起。

### 1.5 MFCC（梅尔频率倒谱系数）

通过离散余弦变换（DCT）从梅尔频谱图导出的紧凑表示：

```
MFCC(t, d) = DCT(log(Mel_spec(t, m)))
```

通常仅保留前13-20个系数（低阶系数捕获频谱形状；高阶系数捕获细节）。

**历史意义**：在语音处理（ASR、说话人识别）中占主导地位数十年。在音乐中用于流派分类、乐器识别和音乐相似度。

**现状**：很大程度上已被学习特征（输入深度网络的梅尔频谱图）和自监督表示（MERT、MusicHuBERT、CLAP嵌入）所取代。MFCC仍可作为低资源场景下的轻量级特征使用。

**何时使用**：低延迟/低算力任务（例如嵌入式设备上的实时分类）；传统机器学习流水线（GMM、SVM）；作为音乐相似度的紧凑描述符。不建议作为现代神经网络生成任务的主要输入。

### 1.6 原始波形

直接使用原始音频采样点作为输入，让网络自行学习特征提取。

**方法：**
- **可学习前端**：SincNet（Ravanelli & Bengio, 2018）用由截止频率参数化的可学习带通滤波器替代梅尔滤波器组
- **一维卷积编码器**：如EnCodec、SoundStream、DAC所使用——编码器从原始采样点中学习提取特征
- **采样点级自回归模型**：WaveNet、SampleRNN

**权衡：**
- **优点**：没有手工特征造成的信息损失；有可能学习到任务最优的表示。
- **缺点**：计算开销大得多（44.1 kHz音频 = 每秒44,100个时间步）；需要更多数据和训练时间；中间表示的可解释性较差。

**何时使用**：端到端训练音频编解码器时；任务需要相位信息时；有充足算力和数据时。

### 1.7 自监督学习表示

日益重要的一类方法，在大规模音频语料上预训练的模型产生通用特征嵌入。

| 模型 | 训练方式 | 规模 | 应用场景 |
|-------|----------|------|----------|
| **MERT** (Min et al., 2023) | 音乐上的掩码语言建模 | 最高330M参数 | 音乐理解、标注、问答 |
| **MusicHuBERT** (Huang et al., 2024) | 音乐上的HuBERT风格预训练 | ~95M | 音乐信息检索 |
| **CLAP** (Wu et al., 2023) | 对比音频-文本 | ~400M | 音频-语言对齐、检索 |
| **Jukebox-5B嵌入** | VQ-VAE自回归 | 5B | 音乐生成特征 |

### 比较表

| 表示方法 | 维度 | 可微分 | 相位信息 | 音乐对齐 | 计算开销 |
|---|---|---|---|---|---|
| 梅尔频谱图 | n_mels x T | 是 | 否 | 部分（梅尔尺度） | 低 |
| 色度特征 | 12 x T | 是 | 否 | 是（音高类别） | 低 |
| CQT | n_bins x T | 是（nnAudio） | 否 | 是（对数频率） | 中 |
| VQT | n_bins x T | 是（nnAudio） | 否 | 自适应 | 中 |
| MFCC | 13-20 x T | 是 | 否 | 部分 | 低 |
| 原始波形 | 1 x T_samples | 是 | 是 | 学习得到 | 非常高 |

### 关键论文

- McFee et al. (2015/2020) -- librosa: Audio and Music Signal Analysis in Python
- Brown (1991) -- Calculation of a constant Q spectral transform
- Schorkhuber & Klapuri (2010) -- Constant-Q Transform Toolbox for Music Processing
- Ravanelli & Bengio (2018) -- SincNet (speaker recognition with learnable filters)
- Min et al. (2023) -- MERT: Acoustic Music Understanding with Large-Scale Pre-training
- Wu et al. (2023) -- CLAP: Large-Scale Contrastive Language-Audio Pretraining

### 特征表示研究的常用数据集

- **AudioSet** (Gemmeke et al., 2017)：超过200万个10秒YouTube片段，527个事件类别
- **FMA** (Defferrard et al., 2017)：约100K首曲目，用于流派分类
- **MUSDB18-HQ** (Rafii et al., 2019)：150首带有分轨的完整曲目，用于源分离
- **MusicCaps** (Agostinelli et al., 2023)：5,521个音乐片段，附带丰富文本描述
- **NSynth** (Engel et al., 2017)：306K个单音符，用于乐器/音色任务
- **MedleyDB** (Bittner et al., 2014)：122个多轨录音，用于MIR

---

## 2. 神经音频编解码器

### 2.1 架构概述

神经音频编解码器遵循**编码器-量化器-解码器**范式：

```
Raw Audio -> [Encoder] -> Continuous Latent -> [Quantizer] -> Discrete Codes -> [Decoder] -> Reconstructed Audio
```

编码器将波形压缩为低维潜在表示。量化器将连续潜在向量离散化为有限编码集（这对语言建模至关重要）。解码器从离散编码重建音频。

所有主要编解码器共享这一结构，但在以下方面有所不同：(1) 编码器/解码器架构，(2) 量化方法，(3) 判别器设计，(4) 损失函数。

### 2.2 SoundStream (Google, 2021)

**论文**：Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec" (ICML 2021 / IEEE/ACM TASLP 2022)

**架构：**
- **编码器**：一维步进卷积，下采样因子为320（24 kHz下 -> 75 Hz帧率）
- **量化器**：残差向量量化（RVQ），N个码本（通常4-12个）
- **解码器**：转置卷积，上采样回原始采样率
- **判别器**：多尺度波形判别器

**关键贡献**：首个证明单一模型可以同时执行音频压缩和实时流处理的神经音频编解码器。

**比特率**：3-18 kbps（根据使用的RVQ层数可变）

### 2.3 EnCodec (Meta, 2022)

**论文**：Defossez et al., "High Fidelity Neural Audio Compression" (arXiv: 2210.13438, 2022)

**架构（SEANet主干网络）：**
- **编码器**：使用步进卷积的卷积编码器
  - 下采样因子：320（24 kHz模型）或640（48 kHz模型）
  - 每个下采样块内的残差单元
  - 两种变体：**因果**（用于流式/实时处理）和**非因果**（用于离线处理）
  - 因果变体仅向左填充（仅使用过去上下文），非因果变体使用对称填充
- **量化器**：残差向量量化（RVQ）
  - 码本大小：每层1024个条目（每个编码10比特）
  - 嵌入维度：128或256
  - 量化器层数：最多32层（通过截断实现可变比特率）
  - **直通估计器（STE）**：在反向传播期间以恒等映射方式传递梯度，穿过不可微的最近邻查找
  - **指数移动平均（EMA）**：码本嵌入通过已分配编码器输出的EMA更新（源自VQ-VAE v2）
  - **码本重置**：过期（低使用率的）码本条目被重新初始化为编码器输出，以防止码本坍缩
- **解码器**：编码器的镜像，使用转置卷积
- **判别器**：
  - 多尺度STFT判别器（在不同频谱分辨率上操作）
  - 多尺度子带判别器（MSBD，在多个时间尺度上操作于波形）

**损失函数：**
- **重建损失**：L1距离 + 多尺度频谱损失（多个窗口大小的STFT）
- **对抗损失**：来自判别器的铰链损失（某些变体使用最小二乘损失）
- **感知损失**：平衡重建保真度与对抗真实性
- **承诺损失**：`||z_e - sg(e)||^2`——鼓励编码器输出"承诺"于码本条目
- **码本损失**：将码本嵌入向编码器输出更新

**发布模型：**
- 24 kHz单声道：支持1.5、3、6、12、24 kbps
- 48 kHz立体声：支持3、6、12、24 kbps

**帧率**：24 kHz下为75 Hz（320倍下采样）

### 2.4 DAC -- Descript Audio Codec (2023)

**论文**：Kumar et al., "High-Fidelity Audio Compression with Improved RVQGAN" (arXiv: 2306.06546, 2023)

**相对于EnCodec/SoundStream的关键改进：**
- **改进的RVQ训练**：在RVQ层之间的残差计算上停止梯度，防止早期量化器层被后期层的误差干扰
- **量化器丢弃**：训练期间随机丢弃量化器层，确保在较低比特率下优雅降级
- **多尺度梅尔损失**：在多个STFT窗口大小上计算，兼顾短期和长期频谱保真度
- **更大码本**：1024个条目，但具有改进的码本利用率
- **44.1 kHz支持**：在44.1 kHz下运行（接近CD品质）

**压缩**：约90倍压缩比（从44.1 kHz/16位降至8 kbps）

**架构**：9个RVQ层，约86 Hz帧率，产生约774个token/秒

**判别器**：多尺度STFT判别器 + 多子带判别器

### 2.5 FunCodec (阿里巴巴/ModelScope, 2023)

**论文**：Du et al., "Funcodec: A Fundamental, Reproducible and Integrable Open Source Toolkit for Neural Speech Codec"

一个开源研究工具包（而非单一编解码器模型），设计用于：
- 音频量化实验
- 下游应用：TTS、音乐生成
- 可复现的编解码器研究

提供用于构建和训练神经音频编解码器的模块化组件，具有可配置的编码器/解码器/量化器架构。

**GitHub**：https://github.com/modelscope/FunCodec

### 2.6 SemantiCodec (2024)

**论文**：Liu et al., 2024

**关键创新**：解耦语义和声学信息的双编码器架构：
- **语义编码器**：捕获高级内容（音素、音符、声音事件）
- **声学编码器**：捕获细粒度音频细节（音色、房间声学、说话人身份）
- 面向**超低比特率**应用（每秒token数显著少于EnCodec/DAC）

这种解耦的动机来自于观察到对于音频上的语言建模，语义token比原始声学token（可以有条件地预测）更有价值。

### 2.7 WavTokenizer (2024)

**论文**：Pan et al., "WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer for Audio Language Models" (arXiv: 2408.16532, 2024)

**激进的简化**：使用**无查找量化（LFQ）**替代RVQ。

**LFQ机制：**
- 没有可学习的码本嵌入。相反，潜在向量的每个维度独立量化为二值（每个维度的符号）。
- 对于d维潜在向量，这产生2^d个可能的token。WavTokenizer使用d=12 -> 4,096个token。
- **直通估计器**处理梯度流：前向传播使用二值，反向传播以恒等映射方式传递梯度。
- **熵正则化**：鼓励所有可能token的均匀利用（无需码本重置即可避免码本坍缩）。
- **承诺损失**：`||z_e - sg(z_q)||^2`，类似于VQ-VAE。

**架构：**
- 单码本（1层LFQ），无残差堆叠
- 约75 Hz帧率 -> 约75个token/秒（相比之下EnCodec为600-2400）
- VQGAN风格训练，使用多尺度判别器

**对语言建模的影响**：10秒片段 = 约750个token（单流），而使用多码本RVQ则需要数千个。这直接映射到标准LLM的下一个token预测。

### 2.8 最新编解码器 (2025-2026)

**TQCodec** (arXiv: 2603.01592)：提出基于网格的量化，用于高比特率、高保真音乐流传输。面向质量优先于极端压缩的应用。

**SUNAC** (MERL, 2026)：源感知统一神经音频编解码器。单一模型处理语音、音乐和音效，在编解码器内部构建领域感知能力。

**SDCodec**：在单一编解码器框架内使用三个领域特定的RVQ模块（语音、音乐、音效）。

**HiFi-Codec** (Yang et al., 2024)：提出分组RVQ（GRVQ），在保持质量的同时减少码本需求。

### 2.9 量化方法比较

| 方法 | 码本数 | 比特/编码 | token数/秒 | 码本学习 | 坍缩风险 |
|--------|-----------|-----------|------------|-------------------|---------------|
| **VQ**（单层） | 1 | 10 | 75-86 | EMA或SGD | 高 |
| **RVQ** (EnCodec) | 8-32 | 10 | 600-2,400 | EMA + 码本重置 | 中 |
| **RVQ** (DAC) | 9 | 10 | ~774 | EMA + 残差停止梯度 | 低 |
| **LFQ** (WavTokenizer) | 1 | 12 | ~75 | 无（二值） | 非常低 |
| **GRVQ** (HiFi-Codec) | 4-8（分组） | 10 | ~300-600 | 分组EMA | 低 |

### 2.10 比特率-质量权衡

一般基准测试（ViSQOL、PESQ、MOS）：
- **1.5 kbps**：EnCodec——语音可理解，音乐质量下降
- **3-6 kbps**：可接受的语音质量；音乐缺乏高频细节
- **8 kbps**：DAC——语音接近透明；音乐有轻微伪影
- **12-24 kbps**：EnCodec/DAC——良好的音乐质量，接近透明
- **48+ kbps**：接近透明到透明（接近无损）

标准评估指标：
- **ViSQOL** (Virtual Speech Quality Objective Listener)：1-5分制，与MOS相关
- **PESQ** (Perceptual Evaluation of Speech Quality)：-0.5到4.5，ITU-T P.862
- **MOS** (Mean Opinion Score)：主观1-5分制
- **STOI** (Short-Time Objective Intelligibility)：0-1，语音可理解度

### 关键论文

- Zeghidour et al. (2021) -- SoundStream
- Defossez et al. (2022) -- EnCodec
- Kumar et al. (2023) -- DAC (Improved RVQGAN)
- Du et al. (2023) -- FunCodec
- Liu et al. (2024) -- SemantiCodec
- Pan et al. (2024) -- WavTokenizer
- Yang et al. (2024) -- HiFi-Codec
- TQCodec (2025/2026) -- arXiv: 2603.01592
- SUNAC (MERL, 2026)

### 编解码器训练与评估的常用数据集

- **DNS Challenge**：语音+噪声，用于语音编解码器评估
- **VCTK**：多说话人英语语音
- **LibriSpeech**：朗读英语语音（1000小时）
- **MUSDB18-HQ**：音乐曲目，用于音乐编解码器评估
- **AudioSet**：通用音频，用于通用编解码器训练
- **MusicCaps**：带文本描述的音乐

---

## 3. 基于扩散的音频模型

### 3.1 理论基础

#### DDPM（去噪扩散概率模型）

**论文**：Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)

前向扩散过程在T个时间步中逐步添加高斯噪声：

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
```

其中 `beta_t` 是方差调度。反向过程学习去噪：

```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```

模型 `mu_theta` 被训练来预测所添加的噪声（或等价地，预测 `x_0`）。

对于音频，此框架可以应用于：
- **频谱图**：将梅尔频谱图视为2D图像，应用图像扩散技术
- **波形**：将扩散直接应用于1D音频采样点（计算开销大得多）

#### 基于分数的生成模型（SDE框架）

**论文**：Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR 2021)

在连续时间SDE框架下统一了DDPM和分数匹配（NCSN/SMLD）。

**前向SDE**（数据 -> 噪声）：
```
dx = f(x, t)dt + g(t)dw
```

**反向SDE**（噪声 -> 数据）：
```
dx = [f(x, t) - g(t)^2 * nabla_x log p_t(x)] dt + g(t) dw_bar
```

其中 `nabla_x log p_t(x)` 是**分数函数**，由神经网络 `s_theta(x, t)` 估计。

**三种SDE类型：**
1. **方差爆炸型（VE）**：对应NCSN/SMLD。`f(x,t) = 0`，噪声方差递增。
2. **方差保留型（VP）**：对应DDPM。均值随方差递增而缩小。
3. **次VP**：约束似然的变体。

**概率流ODE**：对于每个SDE，存在具有相同边缘分布的确定性ODE：
```
dx = [f(x, t) - (1/2) * g(t)^2 * nabla_x log p_t(x)] dt
```
这实现了精确的似然计算、潜在空间操控和确定性采样。

**采样方法：**
- DDIM（去噪扩散隐式模型）：确定性变体，所需步数更少
- DPM-Solver：高阶ODE求解器，10-20步即可获得良好质量
- 一致性模型：通过蒸馏扩散模型实现单步生成

### 3.2 音频的潜在扩散

不是在高维音频/频谱图空间中应用扩散，潜在扩散首先将音频压缩到紧凑的潜在空间（使用VAE或自编码器），然后在该空间中运行扩散。

**优势：**
- 计算效率大幅提高（潜在空间比频谱图空间小4-64倍）
- 相同计算预算下音频质量更好
- 更容易以文本/嵌入为条件进行生成

#### AudioLDM (Liu et al., 2023)

**论文**："AudioLDM: Text-to-Audio Generation with Latent Diffusion Models" (ICML 2023)

- 使用在音频频谱图上训练的**VAE**创建潜在空间
- **CLAP**（对比语言-音频预训练）嵌入用于文本条件
- 带有无分类器引导的潜在扩散
- 从文本描述生成音频

#### AudioLDM 2 (Liu et al., 2023-2024)

**论文**："AudioLDM 2: Learning Holistic Audio Generation with Self-Supervised Pretraining" (IEEE TASLP 2024)

- 面向语音、音乐和音效生成的**统一框架**
- 引入**"音频文本"**中间表示，桥接文本和音频
- 两阶段：文本 -> 音频文本表示 -> 通过潜在扩散生成音频
- 自监督预训练提高所有领域的生成质量

#### Stable Audio (Stability AI, 2024)

**论文**：Roberts et al., "Stable Audio: Fast Timing-Conditioned Latent Audio Diffusion" (arXiv: 2402.04825, 2024)

**架构：**
- **自编码器**：在44.1 kHz立体声音频上训练，压缩为潜在表示
- **潜在扩散模型**：潜在空间中的DiT（扩散Transformer）架构
- **时间条件**：模型以起始时间和总时长为条件，能够生成特定片段
- **文本条件**：通过CLAP文本嵌入 + T5文本编码器

**Stable Audio 2.0** (2024)：生成具有连贯音乐结构（前奏、展开、尾声）的完整曲目。

**Stable Audio Open** (2024)：用于研究用途的开放权重变体。

**关键技术细节：**
- 以44.1 kHz立体声生成
- DiT主干网络，使用交叉注意力进行文本条件
- 推理时使用无分类器引导
- 使用DPM-Solver++进行高效采样

### 3.3 无分类器引导（CFG）

**论文**：Ho & Salimans, "Classifier-Free Diffusion Guidance" (2022)

一种控制条件扩散模型中质量与多样性权衡的技术：

1. **训练**：模型同时在有条件信息（如文本提示）和无条件信息的情况下进行训练。条件信息以一定概率随机丢弃（如10%的时间）并替换为空嵌入。

2. **推理**：预测在条件输出和无条件输出之间进行外推：

```
tilde_epsilon = (1 + w) * epsilon_theta(x_t, c) - w * epsilon_theta(x_t, empty)
```

其中 `w` 是引导尺度：
- `w = 0`：标准条件生成（无引导）
- `w = 1`：无引导效果（与训练时相同）
- `w > 1`：更强地遵循条件，"质量"更高但多样性更低
- 音频的典型值：`w = 3-7`

**相对于分类器引导的优势**：无需在嘈杂的潜在空间中训练单独的分类器。

### 3.4 扩散应用于频谱图与波形的比较

| 方面 | 频谱图扩散 | 波形扩散 |
|--------|----------------------|--------------------|
| **维度** | 较低（例如80 x T） | 高得多（44100 * T） |
| **计算** | 单GPU可行 | 非常昂贵 |
| **质量** | 需要声码器转换为波形 | 直接输出波形 |
| **相位** | 未建模（由声码器填充） | 显式建模 |
| **代表性模型** | AudioLDM、Grad-TTS | DiffWave |
| **延迟** | 扩散 + 声码器 | 仅扩散（但更慢） |

### 3.5 关键论文

- Ho et al. (2020) -- DDPM
- Song et al. (2021) -- Score-Based Generative Modeling through SDEs
- Ho & Salimans (2022) -- Classifier-Free Diffusion Guidance
- Liu et al. (2023) -- AudioLDM
- Liu et al. (2024) -- AudioLDM 2 (IEEE TASLP)
- Roberts et al. (2024) -- Stable Audio
- Dhariwal & Nichol (2021) -- DiffWave (diffusion on waveforms)
- Popov et al. (2021) -- Grad-TTS (diffusion for TTS)
- Kong et al. (2021) -- DiffWave: A Versatile Diffusion Model for Audio
- Chen et al. (2023) -- Make-An-Audio

### 3.6 当前挑战

- **采样速度**：标准扩散需要50-1000个去噪步骤；即使使用DPM-Solver（10-20步），生成速度仍慢于自回归方法
- **长形式生成**：Stable Audio 2.0解决了部分问题，但在数分钟内保持连贯性仍然困难
- **细粒度控制**：文本条件提供粗粒度控制；音符级或乐器级控制仍是开放问题
- **实时生成**：一致性模型和对抗蒸馏正在被探索用于实时扩散

---

## 4. 实时音频处理

### 4.1 流式架构要求

实时音频处理施加了严格的约束：

- **延迟**：现场表演应用通常要求 < 10-20 ms（在44.1 kHz下，10 ms = 441个采样点）
- **因果性**：模型不得访问未来采样点（除小缓冲区外无前瞻）
- **计算预算**：必须在每个音频块的时长内完成处理（实时因子 < 1.0）
- **确定性内存**：固定内存分配，音频线程中不进行动态分配
- **无系统调用**：音频线程不得执行I/O、内存分配或阻塞操作

### 4.2 因果卷积

标准卷积是非因果的：时刻 `t` 的输出依赖于过去和未来的输入。对于流式处理，我们需要**因果卷积**，其输出仅依赖于过去和当前输入。

**实现**：对输入进行左填充，使卷积核仅"看到"过去上下文：

```
Standard:  y[t] = sum_k x[t + k - K//2] * w[k]   (centered)
Causal:    y[t] = sum_k x[t - k] * w[k]            (left-aligned)
```

实践中，对于核大小为K的卷积，因果版本引入(K-1)个采样点的延迟。

**膨胀因果卷积**（来自WaveNet, van den Oord et al., 2016）：
- 使用膨胀在不增加参数的情况下指数级扩大感受野
- 膨胀模式：1、2、4、8、16、...（倍增）
- 对于核大小K和最大膨胀D的L层，感受野 = (K-1) * sum(2^l)，l=0..L-1

### 4.3 EnCodec流式模式

EnCodec明确提供了其SEANet架构的**因果变体**用于流式处理：
- 所有卷积使用因果（仅左侧）填充
- 模型以小块处理音频（通常1-10 ms）
- 内部状态（卷积缓冲区）跨块维护
- 引入等于编码器总感受野的小额算法延迟

**EnCodec 24 kHz的延迟分解：**
- 编码器步幅：320个采样点 = 24 kHz下13 ms（即帧率）
- 来自卷积上下文的额外回溯：因模型深度而异
- 总延迟：通常20-50 ms（对流式处理可接受，对实时监听太高）

### 4.4 可流化的非因果模型

**论文**："Streamable Neural Audio Synthesis with Non-Causal Convolutions" (Semantic Scholar)

一个关键洞察：非因果（双向）模型通常比因果模型产生更高质量，因为它们可以使用未来上下文。本文提出通过以下方法使非因果模型可流化：
- 以重叠块处理音频
- 使用前瞻缓冲区提供"未来"上下文
- 在延迟和质量之间权衡：更多前瞻 = 更高质量但更高延迟

**实际权衡**：48 kHz下20 ms前瞻缓冲区 = 960个采样点的额外延迟，但使得在流式上下文中使用非因果卷积成为可能。

### 4.5 实时神经推理框架

#### RTNeural (Jatin Chowdhury)

**GitHub**：https://github.com/jatinchowdhury18/RTNeural

专为**实时音频速率神经网络推理**设计的C++库：
- 加载在TensorFlow、PyTorch或ONNX中训练的模型
- 针对逐采样点处理进行了优化（无批次维度）
- 支持：Dense、Conv1D、LSTM、GRU层
- 面向**硬实时约束**设计：音频回调中不进行内存分配
- 用于：吉他放大器建模、音频效果模拟、神经合成插件
- 通过JUCE部署为VST/AU插件

**工作流**：在Python中训练 -> 导出权重（JSON/ONNX） -> 在RTNeural C++中加载 -> 在音频回调中运行

#### ANIRA (2025)

**论文**：arXiv: 2506.12665

在实时设置中对三种神经网络架构进行音频效果模拟基准测试，比较推理后端和延迟特性。

#### 其他框架

- **Neural Amp Modeler (NAM)**：使用神经网络的开源吉他放大器建模器
- **Proteus**：吉他放大器/踏板模拟
- **TorchAudio + TorchScript**：在实时约束下部署PyTorch模型
- **ONNX Runtime**：通用神经推理，适用于音频模型
- **Faust**：函数式音频编程语言；可集成神经网络推理

### 4.6 现场音乐的延迟考量

| 应用场景 | 最大可接受延迟 | 备注 |
|-------------|----------------------|-------|
| 吉他放大器建模 | 1-5 ms | 演奏者能感知5-10 ms以上的延迟 |
| 人声处理 | 5-10 ms | 比吉他稍宽容 |
| 实时监听 | < 10 ms | 一般准则 |
| 现场电子音乐 | 10-50 ms | 通常更宽容 |
| 非交互式（流传输） | 100+ ms | 适用于广播/电信 |

**延迟来源：**
1. **算法延迟**：模型固有延迟（来自因果填充、下采样）
2. **计算延迟**：处理一帧音频的时间（必须小于帧时长）
3. **缓冲延迟**：音频接口缓冲区大小（通常64-256个采样点 = 44.1 kHz下1.5-6 ms）
4. **总往返延迟**：A/D转换 + 缓冲 + 处理 + 缓冲 + D/A转换

### 4.7 关键论文

- van den Oord et al. (2016) -- WaveNet: A Generative Model for Raw Audio
- Defossez et al. (2022) -- EnCodec (causal/non-causal streaming architecture)
- "Streamable Neural Audio Synthesis with Non-Causal Convolutions" -- bridging non-causal models and real-time
- Chowdhury -- RTNeural: Real-time neural inference for audio
- "Fast Temporal Convolutions for Real-Time Audio Signal Processing" (DAFx 2020)

### 4.8 当前挑战

- **质量-延迟权衡**：更高质量的模型（更多层、非因果）具有更高延迟
- **模型大小**：大型模型（EnCodec、MusicGen）无法在嵌入式硬件上以音频速率运行
- **ARM/嵌入式部署**：移动和边缘设备上的计算能力有限
- **动态模型**：具有可变计算量的模型（如注意力机制）可能导致缓冲区欠载
- **训练-推理不匹配**：流式模式的性能可能与离线模式不同

---

## 5. 音频标记化与离散表示

### 5.1 为什么要离散化音频？

连续音频波形是高维的（CD品质下44,100个采样点/秒）。对于语言模型方法（AudioLM、MusicLM、MusicGen、VALL-E），音频必须转换为类似于文本token的**离散token序列**：

```
Audio waveform -> [Neural Codec] -> Token sequence -> [Language Model] -> Token sequence -> [Codec Decoder] -> Audio waveform
```

标记化的特性决定了：
- **序列长度**：更少的token = 更高效的语言建模
- **信息保持**：token必须捕获足够的信息以实现高保真重建
- **语义内容**：理想情况下，token应捕获有意义的结构（音素、音符）
- **语言模型兼容性**：单流token比多流token更易于标准LLM处理

### 5.2 向量量化（VQ）

基础构建块。给定连续向量 `z` 和码本 `C = {e_1, ..., e_K}`：

```
VQ(z) = argmin_k ||z - e_k||^2
```

量化向量是最近的码本条目。这将连续空间映射到K个离散编码。

**训练**：码本条目与编码器/解码器联合学习。两种主要更新策略：
1. **基于梯度**：通过标准梯度下降更新码本嵌入（使用STE处理不可微的argmin）
2. **EMA（指数移动平均）**：每个码本条目跟踪分配给它的编码器输出的移动平均值：`e_k = decay * e_k + (1 - decay) * mean(z_i where VQ(z_i) = k)`

**码本坍缩**：一种常见故障模式，仅使用码本中的一小部分条目。模型"忽略"大部分码本，降低了有效容量。

**坍缩的解决方案：**
- **码本重置**：重新初始化未使用的条目（EnCodec）
- **熵正则化**：添加鼓励均匀使用码本的损失项
- **LFQ**：完全避免可学习码本（WavTokenizer）
- **乘积量化**：拆分向量并独立量化子向量

### 5.3 残差向量量化（RVQ）

当前编解码器中的主要量化方法。堆叠多个VQ层，每层量化前一层的残差（误差）：

```
Step 1: q_1 = VQ_1(z)           -> residual r_1 = z - q_1
Step 2: q_2 = VQ_2(r_1)         -> residual r_2 = r_1 - q_2
Step 3: q_3 = VQ_3(r_2)         -> residual r_3 = r_2 - q_3
...
Step N: q_N = VQ_N(r_{N-1})
```

**重建**：`z_hat = q_1 + q_2 + ... + q_N`

**可变比特率**：仅使用前K层（K <= N）以获得较低比特率；使用更多层则质量提高。

**数学解释**：RVQ执行迭代细化。每层捕获从粗到细的细节级别。前几层捕获大部分能量/信息；后面的层添加精细细节。

**在DAC中**（Kumar et al., 2023）：关键改进是在训练期间在RVQ层之间应用**停止梯度**。否则，后期量化器层的梯度会破坏早期层的稳定性。DAC还在训练期间使用量化器丢弃。

**实践中的参数：**
- **EnCodec**：N = 8-32层，K = 1024编码/层，10比特/层
- **DAC**：N = 9层，K = 1024编码/层
- **SoundStream**：N = 4-12层

### 5.4 无查找量化（LFQ）

在WavTokenizer中使用（Pan et al., 2024）。不学习码本嵌入，LFQ将每个维度独立量化为二值：

```
LFQ(z_i) = sign(z_i)   (per dimension i)
```

对于d维向量，这产生2^d个可能的编码。当d=12时，即4,096个编码。

**优势：**
- **无码本坍缩**：所有2^d个编码都是可达的；没有可学习的参数可以坍缩
- **简单性**：无需码本存储，无需EMA更新
- **效率**：二值量化速度极快

**训练损失：**
- **承诺损失**：`||z - sg(z_q)||^2`——鼓励编码器产生接近量化二值的值
- **熵损失**：`H(z_q)`——鼓励所有编码的均匀分布（最大化码本利用率）
- **直通估计器**：前向传播使用量化值；反向传播以恒等方式传递梯度

### 5.5 分组和结构化变体

**分组RVQ（GRVQ）**——HiFi-Codec (Yang et al., 2024)：
- 将潜在维度划分为组，每组使用独立的RVQ进行量化
- 减少所需的总码本数

**跨尺度RVQ（CS-RVQ）**——ESC (EMNLP 2024)：
- 跨尺度组合不同的量化粒度

**网格量化**——TQCodec (2025/2026)：
- 以网格结构组织码本条目，用于高效序列编码

### 5.6 语义token与声学token的分离

来自AudioLM（Borsos et al., 2022）和MusicLM（Agostinelli et al., 2023）的关键洞察：

**语义token**（高级内容）：
- 从自监督模型提取（w2v-BERT、HuBERT、MERT）
- 捕获音素、音符、节奏模式
- 时间分辨率较低
- 对语言建模更相关（"是什么"）

**声学token**（精细细节）：
- 来自神经编解码器码本（SoundStream、EnCodec）
- 捕获音色、房间声学、说话人身份
- 时间分辨率较高
- 对重建更相关（"怎么做"）

**层次化生成**（AudioLM/MusicLM）：
1. 自回归生成语义token（建模高级结构）
2. 以语义token为条件生成粗粒度声学token（添加韵律、音色）
3. 以粗粒度token为条件生成精细声学token（添加波形细节）

这种层次结构产生比直接建模所有编解码器token更好的结果，因为语言模型可以在填充细节之前专注于语义层面的结构。

### 5.7 从多码本到单流

2024-2025年的一个主要趋势是减少token流数量：

| 方法 | token流 | token数/秒 | LLM复杂度 |
|----------|--------------|------------|-----------------|
| EnCodec RVQ (8层) | 8并行 | 600 | 延迟模式交错 |
| EnCodec RVQ (32层) | 32并行 | 2,400 | 复杂展平 |
| MusicGen | 4并行（带延迟模式） | 300 | 交错自回归 |
| WavTokenizer (LFQ) | 1 | ~75 | 标准下一token预测 |

**MusicGen的方法**：使用EnCodec的4层码本，配合**延迟模式**，其中每个码本流偏移一个位置，然后展平为单个序列。Transformer并行预测每个时间步的所有码本层。

**单流编解码器**（WavTokenizer、SemantiCodec）：完全消除了多流问题，使得可以直接使用标准LLM架构（LLaMA风格的Transformer），无需特殊的交错或延迟模式。

### 5.8 关键论文

- van den Oord et al. (2017) -- VQ-VAE: Neural Discrete Representation Learning
- Razavi et al. (2019) -- Generating Diverse High-Fidelity Images with VQ-VAE-2
- Zeghidour et al. (2021) -- SoundStream
- Defossez et al. (2022) -- EnCodec
- Borsos et al. (2022) -- AudioLM: A Language Modeling Approach to Audio Generation
- Agostinelli et al. (2023) -- MusicLM: Generating Music From Text
- Kumar et al. (2023) -- DAC (Improved RVQGAN)
- Copet et al. (2023) -- MusicGen: Simple and Controllable Music Generation
- Pan et al. (2024) -- WavTokenizer
- Liu et al. (2024) -- SemantiCodec
- Yang et al. (2024) -- HiFi-Codec

### 5.9 当前挑战

- **token效率**：平衡重建质量与token数量。更少的token使语言建模更容易，但可能损失音频保真度。
- **语义对齐**：编解码器token捕获声学细节，但可能与语义概念（音符、音素）对齐不佳。SemantiCodec和相关工作解决了这一问题。
- **token不一致性**：音频中的微小扰动可能导致离散token的大幅变化（ACL 2025论文分析了这一问题）。
- **领域通用性**：在语音上训练的编解码器可能无法推广到音乐，反之亦然。SUNAC和SDCodec解决了多领域标记化问题。
- **重建上限**：生成音频的质量从根本上受编解码器重建质量的限制。即使完美的语言模型预测，如果编解码器无法忠实重建，效果也会很差。
- **立体声/空间音频**：大多数编解码器以单声道运行。空间音频标记化的研究仍然不足。

---

## 附录：工具与库

| 工具 | 用途 | 链接 |
|------|---------|------|
| **librosa** | 音频分析、特征提取 | https://librosa.org |
| **torchaudio** | PyTorch音频，可微分变换 | https://pytorch.org/audio |
| **nnAudio** | GPU加速，可微分音频变换 | https://github.com/KinWaiCheuk/nnAudio |
| **demucs** | 源分离 (Meta) | https://github.com/facebookresearch/demucs |
| **audiocraft** | MusicGen、AudioGen、EnCodec (Meta) | https://github.com/facebookresearch/audiocraft |
| **descript-audio-codec** | DAC推理/训练 | https://github.com/descriptinc/descript-audio-codec |
| **stable-audio-tools** | Stable Audio训练/推理 | https://github.com/Stability-AI/stable-audio-tools |
| **diffusers** | HuggingFace扩散流水线（包含AudioLDM 2） | https://github.com/huggingface/diffusers |
| **RTNeural** | 音频的实时神经推理 | https://github.com/jatinchowdhury18/RTNeural |
| **WavTokenizer** | 基于LFQ的音频标记器 | https://github.com/novelfm/WavTokenizer |
| **FunCodec** | 神经编解码器研究工具包 | https://github.com/modelscope/FunCodec |
| **Faust** | 函数式音频编程语言 | https://faust.grame.fr |

---

*最后更新：2026年5月*
