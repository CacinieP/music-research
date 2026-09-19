# 音乐 AI 的音频工程：技术笔记

> English version: [audio-engineering.md](audio-engineering.md)

涵盖特征表示、神经编解码器、扩散、流式处理与音频标记化。内容于 **2026-09-19** 对照所链接的论文与实现核对。下文模型结果对应具体公开配置，不构成完整的最新排行榜。

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

梅尔频谱图通过梅尔尺度滤波器组聚合 STFT 频点。**功率**梅尔频谱图为：

```text
M(t, m) = sum_k H_m(k) * |X(t, k)|^2
```

`X` 为复数 STFT，`H_m` 为滤波器权重。将 `|X|^2` 换成 `|X|` 得到幅度梅尔频谱图。对数压缩需要正数下限，例如 `log(max(M, epsilon))`。复现时应明确幂次、归一化、对数定义与参考电平。

HTK 梅尔公式是 `2595 * log10(1 + f / 700)`，但这只是其中一种约定；librosa 默认采用 Slaney 约定，除非设置 `htk=True`。`n_fft` 控制 FFT 长度，`win_length` 控制实际窗长，`hop_length` 控制帧间隔；应满足 `fmax <= sample_rate / 2`。[librosa 梅尔频谱图文档](https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html)

梅尔特征丢弃相位并合并频率细节。Griffin–Lim 或神经声码器能由估计频谱合成波形，但都不保证还原原始波形。梅尔特征常用于标签任务和 AudioLDM；MusicGen 建模编解码器 token，Jukebox 使用波形 VQ-VAE 编码。采用频谱损失不意味着模型直接生成梅尔频谱图。

### 1.2 色度特征

色度特征将跨八度的频谱能量折叠到音级，十二平均律音乐通常使用 12 维。加权表达式为：

```text
C(t, c) = sum_k W(c, k) * |X(t, k)|^p
```

映射 `W` 取决于调音、频率表示与归一化；`p` 常取 1 或 2。STFT 色度、CQT 色度，以及平滑、量化后的 CENS 各有不同的不变性目标。色度适合和声、调性、对齐和版本识别，但损失八度与音色细节。十二维色度并不能普遍表示所有调律体系。

### 1.3 恒定 Q 变换（CQT）

CQT 使用几何间隔的中心频率，并保持近似恒定的中心频率与带宽之比：

```text
f_k = f_min * 2^(k / B),  k = 0, ..., K-1
Q = f_k / bandwidth_k
X_CQT(t, k) = sum_n x[t*H + n] * w_k[n] * exp(-j*2*pi*f_k*n/f_s)
```

`B` 为每八度频点数，`H` 为帧移。窗长随频率变化，低频使用更长的窗。如果包含两端中心频率，频点数为 `K = 1 + floor(B * log2(f_max / f_min))`，还需满足奈奎斯特频率和滤波器支持范围限制。省略 `+1` 的常见近似描述的是区间跨度，而非包含端点的精确频点数。

**复数 CQT 系数保留相位**，幅度 CQT 才丢弃相位。能否逆变换取决于具体变换与采样方案，幅度表示并不自动可逆。Brown 1991 年的论文是基础工作；实现与逆变换见 [librosa CQT 文档](https://librosa.org/doc/0.11.0/generated/librosa.cqt.html)。

### 1.4 可变 Q 变换（VQT）

librosa 的滤波器带宽定义为：

```text
bandwidth_k = alpha * f_k + gamma
Q_k = f_k / bandwidth_k
```

`alpha` 为相对带宽系数，`gamma` 为以 Hz 为单位的带宽偏移量。`gamma=0` 时为恒定 Q；正的 `gamma` 相对 CQT 拓宽低频滤波器，改善其时间分辨率。它**不是** 0 到 1 的插值参数：`gamma=1` 不会使 VQT 变成 STFT，改变 gamma 也不会把几何间隔的中心频率变为线性间隔。变换返回复数系数。[librosa VQT 文档](https://librosa.org/doc/0.11.0/generated/librosa.vqt.html)

### 1.5 MFCC（梅尔频率倒谱系数）

MFCC 对对数梅尔能量应用离散余弦变换：

```text
MFCC(t, d) = DCT_m(log(max(M(t, m), epsilon)))[d]
```

保留少量系数可获得紧凑的频谱包络描述，适合传统分类器、聚类与有限算力场景。系数数目、是否包含第 0 项、DCT 归一化和对数约定都是定义的一部分。MFCC 并非无损音频表示。

### 1.6 原始波形

波形前端在学习压缩前保留采样信号。例如，SincNet 使用参数化带通卷积，SoundStream、EnCodec 和 DAC 使用卷积编码器；WaveNet 则逐采样点自回归预测。算力开销取决于下采样、架构与序列长度，波形输入不等于逐采样点自回归推理。[SincNet](https://arxiv.org/abs/1808.00158)、[WaveNet](https://arxiv.org/abs/1609.03499)

### 1.7 学习得到的表示

| 模型 | 训练表示 | 常见用途 |
|---|---|---|
| MERT（Li 等；2023 预印本，ICLR 2024） | 掩码预测，使用 RVQ-VAE 声学目标和 CQT 音乐目标；原始版本有 95M/330M 参数 | 音乐理解特征 |
| MusicFM（Won、Hung、Le；2023 预印本） | 音频自监督表示学习 | 帧级与片段级 MIR |
| LAION-CLAP（Wu 等；2022 预印本） | 音频与文本对比对齐 | 检索、提示驱动的零样本分类 |
| Jukebox 表示 | 音乐生成模型的隐藏状态 | 迁移特征；需选择合适层 |

训练目标应区分：音频文本对比模型使用配对文本监督，纯音频自监督模型则需要额外机制才能完成文本检索。参数量和帧率依检查点而定。[MERT](https://arxiv.org/abs/2306.00107)、[MusicFM](https://arxiv.org/abs/2311.03318)、[CLAP](https://arxiv.org/abs/2211.06687)

| 表示 | 形状，省略批次或声道轴 | 相位 | 主要限制 |
|---|---|---|---|
| 功率梅尔 | `n_mels × T` | 丢弃 | 合并频率细节 |
| 色度 | 通常 `12 × T` | 丢弃 | 丢失八度与音色信息 |
| 复数 CQT/VQT | `n_bins × T`，复数 | 保留 | 分辨率和逆变换依实现而定 |
| MFCC | `n_coeffs × T` | 丢弃 | 截断移除频谱细节 |
| 波形 | `channels × n_samples` | 包含于信号中 | 压缩前采样率高 |

可微性取决于实现。NumPy/librosa 特征提取不属于 PyTorch 自动微分计算图；nnAudio 等张量实现可提供可微变换。

---

## 2. 神经音频编解码器

### 2.1 架构与码率计算

```text
waveform -> encoder -> latent vectors -> quantizer -> indices
indices -> codebook vectors -> decoder -> reconstructed waveform
```

对 `N` 个大小为 `K` 的码本，帧率为 `F` 时，定长编码的名义有效载荷为：

```text
indices_per_second = F * N
bits_per_second = F * N * ceil(log2(K))
```

这不包含包头、缩放元数据、填充或熵编码影响。帧率、索引数和 Transformer 自回归步数是不同量。PCM 名义码率为 `sample_rate * bits_per_sample * channels`。

### 2.2 SoundStream

Zeghidour 等，[SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312)，2021 年预印本 / TASLP 第 30 卷（2022）。结合卷积编码器/解码器、残差向量量化（RVQ）、对抗训练和重建目标。量化器层丢弃使一个训练模型支持多档码率。论文评估了 24 kHz 音频、3–18 kbps 配置，并包含流式处理以及联合压缩与增强实验。没有历史比较依据时，不应称其为首个支持实时压缩的神经编解码器。

### 2.3 EnCodec

Défossez 等，[High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)，2022 年预印本 / TMLR 2023。使用 SEANet 风格卷积、LSTM 和 RVQ 瓶颈。训练要点包括多尺度 STFT 判别器、损失平衡器、波形/频谱重建与对抗特征匹配。发布的 RVQ 使用 EMA 更新码本并替换低使用率编码，不应默认它另有通过梯度更新的码本损失。

[官方模型定义](https://github.com/facebookresearch/encodec/blob/main/encodec/model.py) 区分如下：

| 发布模型 | 因果性 | 编码器帧移 | 帧率 | 名义码率 |
|---|---|---|---|---|
| 24 kHz 单声道 | 因果 | 320 采样点 | 75 Hz | 1.5、3、6、12、24 kbps |
| 48 kHz 立体声 | 非因果 | 320 采样点 | 150 Hz | 3、6、12、24 kbps |

每个索引从 1,024 个编码中选择，即 10 比特。24 kHz、6 kbps 配置使用八个码本：`75 * 8 * 10 = 6000` 比特/秒。48 kHz 发布模型采用归一化与重叠分段，并非因果流式模型。

### 2.4 DAC — Descript Audio Codec

Kumar 等，[High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546)，NeurIPS 2023。关键设计包括低维分解式码本查找、查找向量 L2 归一化、周期性 Snake 激活、量化器层丢弃，以及多尺度频谱/对抗目标。码本条目通过码本损失学习，与 EnCodec 的 EMA 更新不同。1,024 个条目并不比 EnCodec 的码本更大。

[44.1 kHz 默认架构](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/dac.py) 的帧移为 512，使用九个 1,024 项码本。因此约为 `86.13` 帧/秒、`775.20` 个索引/秒、不含额外开销的 `7.75` kbps。相对 **单声道** 44.1 kHz/16-bit PCM，名义压缩比约为 91 倍。这些算术结果不代表听觉透明或无损。[量化器实现](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/quantize.py)

### 2.5 FunCodec

[FunCodec](https://github.com/modelscope/FunCodec) 是神经语音编解码研究工具包，提供模块化训练和推理组件。工具包能力、发布检查点与某个特定编解码器的属性应分别描述。面向语音的工具包本身不能证明高保真音乐性能。

### 2.6 SemantiCodec

Liu 等，[SemantiCodec](https://arxiv.org/abs/2405.00233)，2024。语义编码器使用经 k-means 离散化的 AudioMAE 特征，声学编码器表示剩余细节，再由扩散解码器结合两者重建音频。论文介绍了总计 25/50/100 token 每秒、约 0.31–1.40 kbps 的配置。语义与声学编码仍是不同组件，不能简单当作与 WavTokenizer 等价的单码本标记器。

### 2.7 WavTokenizer

Ji 等，[WavTokenizer](https://arxiv.org/abs/2408.16532)，2024 年预印本 / ICLR 2025。它使用 **可学习的向量量化码本**，大小为 4,096，在 24 kHz 音频上生成 40 或 75 token 每秒。扩大 VQ 空间、码本利用率机制、注意力与傅里叶解码器支持其单量化器设计。它**不使用**二值无查找量化。4,096 选一的索引需要 12 比特，并不意味着学习到的潜在向量是 12 维二值向量。[官方实现](https://github.com/jishengpeng/WavTokenizer)

### 2.8 其他已核实的编解码方向

- **HiFi-Codec**（Yang 等，2023）：分组残差向量量化，论文系统使用四个码本，评估重点为语音/TTS 数据集。[论文](https://arxiv.org/abs/2305.02765)
- **TQCodec**（He 等，2026 年 3 月预印本）：面向 44.1 kHz、32–128 kbps 音乐，使用 SEANet、SimVQ、相位感知损失和感知驱动的分频带比特分配。论文并未提出网格量化（trellis quantization）。[论文](https://arxiv.org/abs/2603.01592)
- **SUNAC**（Aihara 等，2025 年预印本 / ICASSP 2026）：通过声源类别提示，从混合音频中直接选择并编码声源，也可处理同一类型的多个声源。其目标超出了普通领域感知。[论文](https://arxiv.org/abs/2511.16126)

### 2.9 如何比较质量

不存在通用的“达到某码率就听觉透明”阈值；听觉透明也不等于无损。应同时比较采样率、声道数、数据集、码率计算方法、重建协议和延迟。

- **MUSHRA / 听音测试**：适合评价中间质量的音频，需报告受试者、锚点、置信区间和测试材料。
- **ViSQOL**：客观感知质量估计，应明确语音/音频模式与版本，不能替代听音测试。[实现](https://github.com/google/visqol)
- **PESQ、STOI**：面向语音质量/可懂度，不能当作通用的音乐保真度指标。
- **频谱误差**：可诊断重建变化，但不能完整描述听觉伪影或立体声声像。

使用未参与训练的音乐，例如正确隔离的 MUSDB18-HQ 测试曲目。AudioSet、语音数据集和 MusicCaps 用途不同，带有描述文本并不自动使数据集成为标准编解码基准。

---

## 3. 基于扩散的音频模型

### 3.1 DDPM 与基于分数的模型

DDPM 中设 `0 < beta_t < 1`，`alpha_t = 1 - beta_t`，`alpha_bar_t = product_{s=1..t}(alpha_s)`：

```text
q(x_t | x_(t-1)) = Normal(sqrt(alpha_t)*x_(t-1), beta_t*I)
x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon
p_theta(x_(t-1) | x_t) = Normal(mu_theta(x_t,t), sigma_t^2*I)
```

常见网络预测的是 `epsilon_theta`，反向均值 `mu_theta` 由该预测与调度计算，两者不是同一个输出。预测 `x_0` 或 velocity 是其他参数化方法，对应的损失权重也需要明确。[Ho 等，DDPM](https://arxiv.org/abs/2006.11239)

当扩散系数 `g(t)` 为与状态无关的标量时，分数 SDE 为：

```text
forward:      dx = f(x,t) dt + g(t) dw
reverse:      dx = [f(x,t) - g(t)^2 * grad_x log p_t(x)] dt + g(t) dw_bar
probability flow ODE:
              dx = [f(x,t) - 0.5*g(t)^2 * grad_x log p_t(x)] dt
```

反向 SDE 从较大的 `t` 积分到较小的 `t`，即 `dt < 0`。使用精确分数函数时，概率流 ODE 具有相同的单时刻边缘分布。学习的分数与数值求解器引入近似误差；计算似然还需要积分散度。VE、VP、sub-VP 是该框架内不同的噪声过程。[Song 等，ICLR 2021](https://arxiv.org/abs/2011.13456)

### 3.2 音频模型系列

| 模型 | 表示与条件 | 来源 |
|---|---|---|
| AudioLDM（Liu 等，2023） | 在梅尔频谱 VAE 潜在空间中扩散；训练扩散模型时用 CLAP 音频嵌入，推理时换成文本嵌入；由声码器重建波形 | [论文](https://arxiv.org/abs/2301.12503) |
| AudioLDM 2（Liu 等，2023/2024） | AudioMAE 派生的 **Language of Audio（LOA）**；语言模型预测中间表示，用于潜在扩散条件 | [论文](https://arxiv.org/abs/2308.05734) |
| 初代 Stable Audio（Evans 等，2024） | 波形自编码器潜在表示、卷积扩散架构、CLAP 文本特征、时间条件 | [Fast Timing-Conditioned Latent Audio Diffusion](https://arxiv.org/abs/2402.04825) |
| Stable Audio Open（Evans 等，2024） | 波形 VAE、T5 文本条件、扩散 Transformer；44.1 kHz 立体声，最长 47 秒 | [论文](https://arxiv.org/abs/2407.14358) |
| DiffWave（Kong 等，ICLR 2021） | 波形扩散，包含频谱条件的音频合成 | [论文](https://arxiv.org/abs/2009.09761) |

不同 Stable Audio 版本的架构与文本编码器不同，不能把 CLAP 和 T5 合并成未公开的配置。开放权重也有模型许可，与代码许可分别适用。

### 3.3 无分类器引导（CFG）

训练时对一部分样本丢弃条件。采用常见的 **scale `s`** 约定：

```text
epsilon_guided = epsilon_uncond + s * (epsilon_cond - epsilon_uncond)
```

- `s=0`：无条件预测。
- `s=1`：普通条件预测。
- `s>1`：向条件方向外推，可能改善条件遵循度，同时减少多样性或引入伪影。

原论文也使用 `(1+w)*epsilon_cond - w*epsilon_uncond`，对应关系是 **`s = 1 + w`**。在该约定下 `w=0` 才是普通条件预测，`w=1` 已经添加引导。具体 scale 应针对检查点与采样器选择。[Ho 与 Salimans](https://arxiv.org/abs/2207.12598)

### 3.4 采样与权衡

频谱扩散需要波形重建阶段，波形扩散直接建模采样信号，波形潜在扩散则通过自编码器缩短序列。瓶颈在表示能力与算力之间做权衡。

DDIM 在随机性参数为零时可确定性采样；DPM-Solver 类方法可减少模型求值次数；一致性模型可通过训练或蒸馏实现少步生成。速度取决于序列长度、模型大小、采样器与硬件，扩散并非必然比自回归生成慢。长时音乐结构、精细事件控制与立体声一致性仍需专门评估。

---

## 4. 实时音频处理

### 4.1 处理期限与因果性

对采样率 `f_s`、每块 `B` 个采样点的系统，每块处理期限是 `B/f_s` 秒。平均实时因子小于 1 只是必要条件，长尾延时仍可造成可听见的缓冲欠载。现场监听常要求总延迟只有几毫秒，可接受值依乐器、信号链与演奏者而定。

音频回调应避免阻塞 I/O、无界等待的锁、内存分配及其他耗时不可预测的操作。预分配缓冲，在回调外准备模型，并在目标机器上测量最坏情况。

因果系统只使用当前与过去输入；可流式系统也可以使用有限前瞻并延后输出。流式、严格因果和低延迟是不同属性。

### 4.2 因果卷积

步长为 1 的因果卷积可写为：

```text
y[t] = sum_(k=0..K-1) w[k] * x[t - d*k]
receptive_field = 1 + sum_l (K_l - 1)*d_l
```

感受野公式假设所有层步长为 1。核大小为 `K`，扩张率为 `1,2,...,2^(L-1)` 时，得到 `1 + (K-1)*(2^L-1)`。有步进下采样的网络必须计入步长乘积。

过去上下文意味着缓存需求，不自动引入 `(K-1)` 个采样点延迟。因果 FIR 在 `x[t]` 到达后即可计算 `y[t]`。前瞻、分帧、重采样和调度决定算法延迟；线性相位滤波器的群延迟又是另一概念，不能单凭因果性推断。

### 4.3 编解码器流式处理

EnCodec 24 kHz 模型是因果的，但公开的整文件接口本身并不是有状态的流式实现。部署必须保留卷积与循环网络状态，并匹配离线填充和边界行为。

24 kHz 下 320 个采样点帧移，意味着 **每个潜在帧 13.33 ms**，即 **75 帧/秒**。这既不是实测端到端延迟，也不是整个感受野。应分别测量编解码缓冲、重采样、推理、封包和设备延迟。[EnCodec 实现](https://github.com/facebookresearch/encodec)

### 4.4 非因果流式处理

Caillon 与 Esling 的 [Streamable Neural Audio Synthesis With Non-Causal Convolutions](https://arxiv.org/abs/2204.07064)（DAFx 2022）在训练后通过缓存运算与插入延迟，将非因果卷积模型转为流式，同时保持计算图及并行分支对齐。其方法比普通 overlap-add 更具体。所需未来上下文会转化为延迟；48 kHz 下 20 ms 对应 960 个采样点。

### 4.5 推理框架

| 工具 | 范围与部署要点 |
|---|---|
| [RTNeural](https://github.com/jatinchowdhury18/RTNeural) | 为支持的层类型提供 C++ 推理；文档流程是将 TensorFlow/PyTorch 权重导出 JSON、准备模型、调用 forward，不是通用 ONNX 导入器。 |
| [anira](https://arxiv.org/abs/2506.12665) | Ackva 与 Schulz；IS2 2024 论文，2025 arXiv 版本。使用静态线程池将推理移出音频回调并管理延迟；论文评估了 ONNX Runtime、LibTorch、TensorFlow Lite。 |
| 通用张量/ONNX 运行时 | 可支持多种模型，但不会自动保证回调耗时有界；预热、分配行为、调度与线程数均影响结果。 |
| [Faust](https://faust.grame.fr/) | DSP 语言与工具链，神经推理集成依所选后端或外部组件而定。 |

### 4.6 延迟预算与验证

```text
round_trip = ADC + input_buffers + algorithmic_delay
             + processing/scheduling + output_buffers + DAC
```

不要重复计算已被缓冲隐藏的计算耗时。使用脉冲或回环测量真实输入到输出路径，并报告采样率、块大小、硬件、后端与负载。检查流式/离线结果一致性、状态重置、启动与尾部处理，以及计算超期时的行为。离线跑得快不足以证明可以稳定用于现场。

---

## 5. 音频标记化与离散表示

### 5.1 VQ 与 RVQ

对码本 `C={e_1,...,e_K}`：

```text
index(z) = argmin_k ||z - e_k||^2
VQ(z) = e_index(z)
```

索引是离散值，量化向量是对应码本嵌入。直通估计器向编码器提供替代梯度。码本可以采用独立梯度损失，或对分配计数与向量和做 EMA 更新；简单平均每批均值通常不等价于该 EMA 算法。[VQ-VAE](https://arxiv.org/abs/1711.00937)

RVQ 逐层细化残差：

```text
r_0 = z
q_i = VQ_i(r_(i-1))
r_i = r_(i-1) - q_i
z_hat = sum_i q_i
```

截断经过可变码率训练的模型可减少载荷。更多层通常增加表示容量，但不能保证每个信号的听觉质量都单调上升。前几层也不保证对应显式音符或音素。低维归一化查找（DAC）、低使用率编码替换（EnCodec）、初始化与利用率正则化分别采用不同方式处理码本利用不足。

### 5.2 无查找量化与分组量化

二值 LFQ 将 `d` 个分量分别映射到符号值，产生最多 `2^d` 种离散组合。不使用可学习码本并不保证 token 均匀分布，熵目标与训练设计仍然重要。LFQ 与 WavTokenizer 的可学习 VQ 是不同方法。[LFQ 参考：Language Model Beats Diffusion — Tokenizer is Key to Visual Generation](https://arxiv.org/abs/2310.05737)

分组 RVQ 把通道划分为组，在各组内做残差量化，HiFi-Codec 即为一例。分组数与层数共同决定编码流总数，优势应在匹配的质量和码率下检验。

### 5.3 语义与声学 Token

AudioLM/MusicLM 组合自监督音频特征与神经编解码器生成的多层级 token。较粗的语义表示帮助建模长程内容，声学编码支持波形细节。这是学习得到的分工，并非把“音符”和“音色”严格符号化拆开；语义特征可能保留声学信息，编解码器编码也可包含语义。[AudioLM](https://arxiv.org/abs/2209.03143)、[MusicLM](https://arxiv.org/abs/2301.11325)

### 5.4 Token 速率与 MusicGen 延迟模式

| 配置 | 帧率 | 编码流数 | 每秒索引总数 |
|---|---|---|---|
| EnCodec 24 kHz、6 kbps | 75 Hz | 8 | 600 |
| EnCodec 24 kHz、24 kbps | 75 Hz | 32 | 2,400 |
| DAC 44.1 kHz、九个码本 | 约 86.13 Hz | 9 | 约 775.20 |
| MusicGen 的 32 kHz 编解码器 | 50 Hz | 4 | 200 |
| WavTokenizer | 40 或 75 Hz | 1 | 40 或 75 |

MusicGen 使用 32 kHz EnCodec 配置、四个码本、50 Hz 帧率。它错开各码本流，并通过不同输出头在每个 Transformer 步骤预测多个索引，因此每秒约需 **50 个自回归步骤**，另有边界开销；并非展平成 200 步序列。这些数字对应原始单声道配置。[Copet 等，MusicGen](https://arxiv.org/abs/2306.05284)

### 5.5 实用评估

应分别评估重建质量、语言模型可预测性、下游语义、领域覆盖和立体声表现。减少 token 可能降低建模开销，同时增加解码器负担。编解码重建保真度是重要瓶颈，但不是所有生成样本听觉质量的严格上界：生成潜在向量可能不同于测试音频编码得到的向量。单一编解码指标不足以证明完整系统的音乐实用性。

---

## 附录：工具与库

| 工具 | 用途 |
|---|---|
| [librosa](https://librosa.org/) | 音频分析与特征提取 |
| [nnAudio](https://github.com/KinWaiCheuk/nnAudio) | 基于张量的音频变换 |
| [AudioCraft](https://github.com/facebookresearch/audiocraft) | MusicGen 与相关音频模型 |
| [DAC](https://github.com/descriptinc/descript-audio-codec) | 编解码器训练与推理 |
| [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) | Stable Audio 模型工具 |
| [RTNeural](https://github.com/jatinchowdhury18/RTNeural) | 面向音频的 C++ 神经推理 |
| [anira](https://github.com/anira-project/anira) | 音频应用的推理调度 |
| [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) | 单码本音频标记器 |
| [FunCodec](https://github.com/modelscope/FunCodec) | 神经语音编解码工具包 |

数据集、MIR 指标及模型评测背景见[音乐理解 / MIR](music-understanding-mir-zh.md)。
