# 音乐生成：技术研究笔记

截至 2025--2026 年音乐生成技术综合调研。涵盖符号生成、音频级生成、歌声合成、可控生成与评估方法。

> English version: [music-generation.md](music-generation.md)

---

## 1. 符号音乐生成

### 1.1 表示格式

符号音乐操作离散表示（音符、时序、力度）而非原始音频。表示格式的选择从根本上决定了模型能学到什么。

#### MIDI (Musical Instrument Digital Interface)
- 基于事件的协议：`Note On`、`Note Off`、`Velocity`、`Time Shift`、`Program Change`
- 复调通过带有 delta-time 偏移的重叠 note-on/note-off 事件处理
- 标准 "MIDI-Like" 分词 (Oore et al., 2018)：按起始时间排序的事件流
- 局限：原始 MIDI 事件缺乏显式节拍结构（小节线、拍位）；时间纯粹是相对的

#### 钢琴卷帘 (Piano Roll)
- 二维矩阵表示：音高（行）x 时间帧（列），值表示力度/激活
- 与 CNN 架构直接兼容（可视为图像）
- 丢失显式音符级事件结构；边界检测是隐式的
- 分辨率权衡：精细时间分辨率 = 非常大的矩阵

#### ABC 记谱法
- 民间/传统音乐中常见的文本音乐记谱
- 紧凑、人类可读，天然适合语言模型分词
- 仅限于单声部或简单复调音乐

#### 基于 Token 的表示（当前 Transformer 标准）

| 格式 | 核心思想 | 参考文献 |
|------|----------|----------|
| **MIDI-Like** | 将原始 MIDI 事件作为 token (Note On/Off, Time Shift) | Oore et al., 2018 |
| **REMI** | 增强的 MIDI 事件，添加显式 Bar、Position、Tempo、Chord token | Huang & Yang, 2020 (Pop Music Transformer) |
| **Compound Word (CP)** | 将每个时间步的多个属性组合为单一复合 token | Hsiao et al., 2021 |
| **REMI+ / REMI-z** | REMI 的多轨扩展，支持轨道感知分词 | NeurIPS 2025 |
| **Pianoroll-Event** | 混合空间 + 序列编码，桥接钢琴卷帘和 token 视图 | arXiv 2601.19951 (2025) |
| **MIDI-Token** | MIDITok 包中比较的各种分词策略 | Natole et al., ISMIR 2021 |

REMI (Huang & Yang, 2020) 影响最大：它添加了显式的 `Bar`、`Position`、`Tempo` 和 `Chord` token，使模型可以直接学习节拍结构，而非从 delta-time 推断。REMI-z (2025) 将此扩展到多轨音乐，每轨独立的 token 流。

### 1.2 关键模型

#### Music Transformer (Huang, Vaswani et al., 2018/2019)
- **论文**: "Music Transformer: Generating Music with Long-Term Structure" (arXiv:1809.04281, ICLR 2019)
- **作者**: Cheng-Zhi Anna Huang, Ashish Vaswani, et al. (Google Magenta)
- **架构**: 仅解码器 Transformer，带有改进的**相对注意力**机制
- **核心贡献**: 自注意力中的相对位置表示使模型能学习音乐中的相对音程和循环模式，而非绝对位置。这一点至关重要，因为音乐本质上是关系性的（音程、和弦进行、节奏循环）
- **表示**: MIDI-Like 分词（来自 MAESTRO 数据集的演奏级钢琴表演）
- **结果**: 生成具有长期结构重复（A-B-A 形式、反复主题）的连贯分钟级钢琴作品
- **局限**: 单乐器（仅钢琴）；MIDI-Like 分词缺乏显式节拍结构

#### Pop Music Transformer (Huang & Yang, 2020)
- **架构**: 使用 REMI 分词的 Transformer
- **核心贡献**: 引入带有显式 Bar/Position token 的 REMI 表示
- **训练数据**: 流行钢琴编曲
- **相对 Music Transformer 的改进**: 由于显式节拍 token，节奏结构和拍级建模更好

#### MuPT (2024--2025)
- 用于符号音乐生成的预训练 Transformer
- 旨在提高较长作品的结构一致性
- 聚焦符号音乐的预训练规模

#### Transformer-GAN 用于符号音乐 (AAAI)
- 结合自回归 Transformer 生成器和 GAN 判别器
- 通过对抗训练进行质量控制，生成分钟级作品

### 1.3 符号音乐的自回归 vs 扩散方法

**自回归 (AR) 方法** (Music Transformer, 基于 REMI 的模型):
- 将音乐建模为 token 序列，根据之前的 token 预测下一个
- 优势：对序列化音乐的天然适配；处理可变长度输出；强局部连贯性
- 劣势：长序列的误差累积；顺序生成速度慢；可能难以处理全局结构

**扩散方法**用于符号音乐:
- 将扩散（渐进去噪）应用于钢琴卷帘或潜在表示
- 优势：并行生成；可直接建模全局结构；较少误差累积
- 劣势：固定输出长度；对基于事件的音乐适配不够自然；可能遗漏细粒度序列依赖

**混合方法**正在兴起：结合 AR 处理序列精细结构，用扩散进行全局规划。

2025 年的一项研究 (arXiv:2506.08570) 提供了 AR vs flow-matching 用于文本到音乐生成的首次控制性比较，发现扩散/flow-matching 在多样性和可控性方面更优，而 AR 在训练稳定性和局部结构连贯性方面更好。

### 1.4 关键数据集

| 数据集 | 内容 | 规模 |
|--------|------|------|
| **MAESTRO** | 对齐的 MIDI-音频钢琴演奏 | ~200 小时 |
| **Lakh MIDI** | 多乐器 MIDI 文件 | 180K 文件 |
| **POP909** | 含旋律/伴奏的流行钢琴编曲 | 909 首歌 |
| **AD Pianoforte** | 古典钢琴演奏 | ~100 小时 |
| **MetaMIDI** | 大规模 MIDI 集合 | 436K 文件 |

### 1.5 开放性问题
- **长篇结构**: 在多分钟作品中维持全局曲式（主歌-副歌-桥段）、主题发展和和声方向
- **多轨生成**: 跨多种乐器的协调生成，包括适当的声部进行、节奏组协调和编配
- **可控性**: 超越文本条件的精确和声、曲式、风格和力度控制
- **表示鸿沟**: 没有单一表示能同等良好地捕获所有音乐维度（音高、节奏、力度、音色、曲式）

---

## 2. 音频级音乐生成

### 2.1 神经音频编解码器（音频生成的基础）

音频级模型需要将高维波形（44,100 样本/秒）压缩为可操作的表示。

#### EnCodec (Defossez et al., Meta, 2022)
- **论文**: "High Fidelity Neural Audio Compression" (arXiv:2210.13438)
- **架构**: 带有**残差向量量化 (RVQ)** 瓶颈的卷积编码器-解码器
  - 编码器：跨步卷积网络将音频压缩为潜在帧
  - RVQ：多层层次化码本将连续潜在表示量化为离散 token。残差量化：第一个码本捕获粗结构，后续码本捕获逐渐更精细的细节
  - 解码器：转置卷积从量化的潜在表示重建波形
  - 判别器：多尺度 STFT + 多周期判别器用于对抗训练
- **比特率**: 1.5 到 24 kbps（24 kHz 模型）；3 到 24 kbps（48 kHz 立体声模型）
- **帧率**: ~50 Hz（48 kHz 模型为 75 Hz），4--8 个码本层
- **训练**: 重建损失 + 对抗损失 + 量化承诺损失
- **角色**: 作为 MusicGen 及许多其他模型的音频分词器

#### SoundStream (Zeghidour et al., Google, 2021)
- EnCodec 的前身；引入了用于神经音频编解码器的 RVQ 方法
- 用于 AudioLM 和 MusicLM 流水线
- 24 kHz, 3 kbps 流式音频编解码器

#### DAC (Descript Audio Codec, Kumar et al., 2023)
- 改进的神经编解码器，使用蛇形激活函数和更大的码本
- 44.1 kHz，用于部分 Stable Audio 变体

### 2.2 Jukebox (OpenAI, 2020)

- **论文**: "Jukebox: A Generative Model for Music" (Dhariwal, Jun, Payne, Kim, Radford, Sutskever; arXiv:2005.00341)
- **架构**: 层次化 VQ-VAE + 自回归 Transformer
  - **3 层层次化 VQ-VAE**: 在三个时间分辨率上压缩原始音频
    - 顶层: ~8 Hz -- 捕获长程音乐结构（旋律、和声、曲式）
    - 中层: ~34 Hz -- 捕获音色、人声特征
    - 底层: ~65 Hz 或原始 -- 精细音频细节
  - 每层有自己的离散嵌入码本
  - **自回归 Transformer**（仅解码器，类似 GPT-2/GPT-3）建模每层的 token 分布
  - **上采样**: 顶层 token 有条件地上采样到中层，再到底层
  - **歌词条件**: 歌词通过强制对齐与音频对齐，为顶层 Transformer 提供条件
  - **艺术家/流派元数据**: 通过艺术家和流派嵌入提供额外条件
- **训练数据**: 120 万首歌曲（60 万首英文），带歌词和元数据，来自网络
- **输出**: 44.1 kHz 立体声音频，最长数分钟
- **质量**: 可辨认的歌声和流派适当的器乐编配，但有明显的伪影，音质低于人类水平
- **局限**: 生成速度极慢（单首歌曲需数小时）；与人类创作音乐的质量差距大；层次化上采样可能在层间引入不一致
- **意义**: 证明了将 VQ-VAE + 自回归模型扩展到原始音频是可行的，为后续工作确立了范式

### 2.3 MusicLM (Google Brain, 2023)

- **论文**: "MusicLM: Generating Music From Text" (Agostinelli, Denk, Borsos, et al.; arXiv:2301.11325)
- **架构**: 三阶段层次化序列到序列
  - **阶段 1 -- MuLan 文本编码**: 通过 MuLan（对比式音频-文本嵌入模型，类似于音频版 CLIP）编码文本描述。将文本映射到共享的音频-文本嵌入空间
  - **阶段 2 -- 语义建模**: 自回归 Transformer 在 MuLan 文本嵌入条件下生成**语义 token**（高层音乐结构）。使用 SoundStream 编解码器的顶层码本
  - **阶段 3 -- 声学建模**: 自回归 Transformer 使用 SoundStream 的完整 RVQ 堆栈将语义 token 上采样为细粒度**声学 token**（完整音频细节）
  - 语义和声学建模的分离显著减少了顶层的有效序列长度
- **关键组件**:
  - **MuLan**: 对齐音乐和文本嵌入的对比式音频-文本模型
  - **SoundStream**: 通过 RVQ 提供离散 token 的神经音频编解码器
  - **Transformer**: 在语义和声学两层均有使用
- **能力**: 从丰富描述的文本到音乐；旋律条件（哼唱旋律生成完整编配）；长期结构连贯性
- **输出**: 24 kHz，最长数分钟
- **评估**: 引入 MusicCaps 基准（来自 AudioSet/YouTube 的 5,521 个带有人工撰写字幕的 10 秒片段）。使用 FAD、KL 散度、MuLan 相似度和人工并排比较评估
- **因训练数据版权问题未公开发布**
- **意义**: 确立了层次化语义-声学范式和 MusicCaps 评估基准

### 2.4 MusicGen (Meta, 2023)

- **论文**: "Simple and Controllable Music Generation" (Copet, Kreuk, Gat, Remez, Kant, Synnaeve, Adi, Defossez; arXiv:2306.05284)
- **架构**: **单阶段自回归 Transformer**（无层次化语义/声学分离）
  - 操作 EnCodec 离散音频 token（32 kHz, 4 个码本, 50 Hz）
  - 单个 Transformer 使用延迟模式交错预测每个时间步的所有 4 个码本 token
  - 文本条件通过**T5 编码器**（冻结）和/或**基于色度的旋律条件**
  - **旋律条件**: 从音频提示中提取色度特征作为额外条件，实现"生成跟随此旋律的音乐"
- **关键简化**: 通过使用单个 Transformer 处理 EnCodec 的多码本 token，消除了 MusicLM/Jukebox 的层次化多阶段方法
- **训练数据**:
  - 2 万小时授权音乐（Meta 内部数据集）
  - Shutterstock 和 Pond5 音乐库
  - MusicCaps 用于文本对齐评估
- **模型规模**: 300M（小）、1.5B（中）、3.3B（大）
- **输出**: 32 kHz，标准最长 30 秒（可通过滑动窗口扩展）
- **框架**: 作为 Meta **AudioCraft** 库的一部分发布（开源）
- **评估**: 发布时在 MusicCaps 基准上达到 SOTA FAD 分数；人工评估显示优于基线
- **可控性**: 文本提示 + 可选的旋律/音频条件
- **伴随模型 -- MAGNeT**: 非自回归（掩码）变体，使用相同 EnCodec 基础实现更快的并行生成

### 2.5 AudioLDM 和 AudioLDM 2 (2023--2024)

#### AudioLDM (Liu et al., 2023)
- **论文**: "AudioLDM: Text-to-Audio Generation with Latent Diffusion Models" (Haohe Liu et al.)
- **架构**: 从 Stable Diffusion 适配到音频域的潜在扩散模型 (LDM)
  - **VAE 编码器-解码器**: 将 mel-spectrogram 压缩到低维潜在空间（基于 AudioMAE/HiFi-GAN 组件）
  - **U-Net 扩散主干**: 在潜在空间中操作，以文本嵌入为条件
  - **文本条件**: CLAP (Contrastive Language-Audio Pretraining) 嵌入桥接文本-音频模态差距
  - **推理**: 在潜在空间进行扩散去噪，然后 VAE 解码为 mel-spectrogram，再经声码器转为波形
- **输出**: 音效、环境音、短音乐片段（约 10 秒）
- **优势**: 计算效率高（在压缩的潜在空间而非完整频谱图空间中扩散）；支持文本到音频、音频到音频和修复

#### AudioLDM 2 (Liu et al., 2023--2024)
- **论文**: "AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining" (arXiv:2308.05734)
- **相对 AudioLDM 1 的关键改进**:
  - **统一框架**: 单一架构处理语音、音乐和音效
  - **GPT-2 集成**: GPT-2 语言模型与潜在扩散的联合微调，增强文本理解
  - **自监督预训练**: 提高质量和泛化能力
  - **优化架构**: 16 kHz 改进模型，更多训练数据
  - **性能**: 同时在文本到音频、文本到音乐和文本到语音基准上匹配 SOTA
- **三种变体**: 分别针对音频、音乐和语音优化

### 2.6 Stable Audio (Stability AI, 2023--2024)

#### Stable Audio 1.0 (2023)
- **架构**: 三组件潜在扩散模型
  1. **自编码器**: 基于 VAE 的编码器将 44.1 kHz 立体声音频压缩为紧凑的潜在表示
  2. **文本编码器**: 冻结的文本编码器（T5 或 CLAP）用于提示条件
  3. **扩散模型**: 基于 U-Net 的去噪器，在潜在表示上操作，带文本交叉注意力
- **输出**: 可变长度立体声音频，最长约 47 秒，44.1 kHz
- **训练数据**: 授权音乐和音频

#### Stable Audio 2.0 (2024 年 4 月)
- **架构升级**: 将 U-Net 主干替换为 **Diffusion Transformer (DiT)**
  - **高压缩自编码器**: 将原始音频波形投影到连续潜在表示，潜在帧率为 **21.5 Hz**（显著压缩）
  - **DiT 主干**: 基于 Transformer 的去噪器替换 U-Net，跟随 DiT 架构趋势（如 Stable Diffusion 3, Sora）
  - **T5 文本编码器**: 用于文本条件
  - **潜在扩散**: 整个扩散过程在学习的潜在空间中进行
- **输出**: 长达 **3 分钟**的完整曲目，具有连贯的音乐结构（引子-发展-尾声）
- **意义**: 首个生成 3 分钟结构化音乐曲目并开放权重的模型
- **发布**: Hugging Face 上开放权重（Stable Audio Open 1.0 变体）

#### Stable Audio Open (2024 年 6 月)
- 开放权重发布（约 48.6 万免版税音频录音 + 约 7 万音乐曲目）
- 训练数据：Free Music Archive (FMA)、Freesound、Creative Commons 授权音频
- 明确未使用受版权保护的音乐训练
- 支持在自定义数据集上微调
- 生成最长约 47 秒，44.1 kHz

#### Stable Audio 3 (2025, 开发中)
- **语义-声学自编码器**: 新型自编码器将音频投影到紧凑的潜在空间，同时保留语义和声学信息
- Diffusion Transformer 在改进的潜在表示上操作
- 以文本和期望输出特征为条件

### 2.7 YuE (Multimodal Art Projection, 2025)

- **论文**: "YuE: Scaling Open Foundation Models for Long-Form Music Generation" (Yuan et al.; arXiv:2503.08638, ICLR 2025)
- **作者**: Ruibin Yuan, Hanfeng Lin, Haohe Liu 及来自多个机构的约 50 位合作者
- **架构**: 基于 **LLaMA2** 架构，适配音乐
  - **轨道解耦下一 token 预测**: 生成期间分离人声和伴奏轨，克服原始音频 token 中密集混合信号的挑战
  - **结构化渐进条件**: 通过渐进条件化歌词片段实现长上下文歌词对齐，保持分钟级连贯性
  - **多任务多阶段预训练方案**: 多阶段训练确保收敛和跨多样音乐任务的泛化
  - **重新设计的上下文学习**: 实现多功能的风格迁移（如将日本 City Pop 转换为英文说唱，同时保留伴奏）和双向生成
- **规模**: 在**万亿级 token** 上训练
- **输出**: 长达 **5 分钟**的人声 + 伴奏音乐
- **能力**:
  - 歌词到歌曲生成（主要任务）
  - 跨流派和语言的风格迁移
  - 双向生成（从某点向前和向后）
  - 音乐理解：学习的表示在 MARBLE 音乐理解基准上匹配或超越 SOTA
- **微调**: 支持额外控制和对"长尾"（代表性不足的）语言的增强支持
- **硬件**: 可在消费级 GPU 上运行（10 GB 显存生成约 1 分钟音乐）
- **意义**: 首个在全曲生成的音乐性和人声灵活性方面匹配或超越专有系统（Suno, Udio）的开源模型
- **局限**: 生成速度（实时倍率）；偶尔的歌词错位；质量因流派而异

### 2.8 Suno（专有，2023--2026）

- **公司**: Suno AI (Cambridge, MA)
- **架构**: 未公开；据信结合了扩散模型和自回归模型，分别处理歌词、人声和伴奏
- **演进**:
  - V1--V3 (2023--2024): 快速迭代，从简单片段提升到结构化歌曲
  - **V4** (2024 年末): 质量重大飞跃；专业水准的输出，具有连贯的主歌-副歌结构
  - **V4.5** (2025): 进一步质量提升；被认为是带人声的完整歌曲类的最佳模型
  - **V5** (预计 2025): 开发中；预计在音频保真度、人声真实感和歌曲结构方面带来进一步飞跃
- **能力**:
  - 文本到歌曲：从文本描述生成完整歌曲
  - 歌词输入：自定义或 AI 生成的歌词
  - 通过提示进行流派/风格控制
  - 最长曲目：4 分钟
  - 多流派支持
- **优势**: 带人声的完整歌曲整体质量最佳；对初学者友好的界面；强的流派遵循能力
- **局限**: 专有（无开放权重）；超越提示的细粒度控制有限；较长生成中偶尔出现结构性伪影

### 2.9 Udio（专有，2024--2026）

- **公司**: Udio（前身为 Uncharted Labs）
- **架构**: 未公开
- **能力**:
  - 文本到歌曲生成
  - 最长曲目：2 分钟
  - 以更多音乐变化著称：速度变化、切分、旋律变化
  - 输出比竞争对手更容易被误认为非 AI 作品
- **演进**: 正转向成为"超级粉丝的 AI 游乐场"而非纯粹的生成工具
- **优势**: 更多音乐惊喜和变化；质量被认为更"人性化"
- **局限**: 最大长度短于 Suno；对初学者不如 Suno 易用

### 2.10 其他值得关注的模型

#### TangoFlux (2024)
- 515M 参数；生成长达 30 秒的 44.1 kHz 音频
- 使用 **flow matching**（整流流）代替扩散——一种有前景的替代架构
- 推理速度比同等扩散模型更快

#### TVC-MusicGen (INTERSPEECH 2025)
- MusicGen 的时变结构控制
- 用于背景音乐生成的新方法，具有动态时间结构控制
- 解决 MusicGen 中静态条件的局限

#### MusicLDM
- 将 Stable Diffusion + AudioLDM 适配于音乐生成
- 聚焦降低生成输出的抄袭风险

#### ACE-Step (ACE Studio + 阶跃星辰/StepFun, 2025)
- **论文**: "ACE-Step: A Step Towards Music Generation Foundation Model" (Junmin Gong, Sean Zhao, et al.; arXiv:2506.00045)
- **开源**音乐生成基础模型（3.5B 参数 v1，Apache 2.0 许可）
- **架构**: 集成**扩散生成与 Sana 的深度压缩自编码器 (DCAE)** 和轻量级**线性 Transformer**
- 使用 **REPA（表示对齐）**训练：利用 MERT 和 m-hubert 对齐语义表示，实现快速收敛
- **性能**: 在 A100 GPU 上约 20 秒合成长达 4 分钟的音乐——比 LLM 方法快 15 倍
- 强大的**中文（普通话）歌词与人声**支持
- 支持声音克隆、歌词编辑、混音、歌词到人声、歌唱到伴奏
- 目标：建立音乐 AI 基础模型（类似 Stable Diffusion 对图像的影响）

#### MusicFlow (ICML 2024)
- **论文**: "MusicFlow: Cascaded Flow Matching for Text Guided Music Generation" (K R Prajwal, Bowen Shi, et al.; arXiv:2410.20478, ICML 2024 Poster)
- **级联流匹配**框架：两个流匹配网络分别建模语义和声学特征的条件分布，基于自监督表示
- 使用**掩码预测**作为训练目标，支持零样本泛化到音乐填充和续写任务
- 在 MusicCaps 上实现更优质量和文本一致性，尽管模型体积 2--5 倍更小、迭代步数仅需 1/5

#### SongCreator (NeurIPS 2024)
- **论文**: "SongCreator: Lyrics-based Universal Song Generation" (Shun Lei et al.; NeurIPS 2024 Poster)
- **架构**: **双序列语言模型 (DSLM)** 在单一框架内将人声和伴奏作为并行序列处理
- **核心创新**: 可配置的**注意力掩码策略**将同一基础模型路由到不同任务（歌曲生成、人声生成、伴奏生成、歌曲编辑、歌曲理解），无需更改架构
- 支持通过不同音频提示独立控制人声和伴奏的声学条件
- 在八个歌曲相关任务上达到 SOTA 或竞争力性能

#### MusicFX (Google, 2023--2025)
- 基于 MusicLM 技术的消费级音乐生成工具
- 作为 Google AI Test Kitchen 的一部分发布
- 从文本提示生成短音乐片段
- 对生成音频应用 **SynthID 水印**以识别
- 禁止生成模仿特定艺术家的音乐（版权保障）
- 定期更新以提升质量和多样性

#### SkyMusic (昆仑万维, 2024--2025)
- 昆仑万维的中文文本到音乐生成平台
- 支持多语言输出（中文、英文及其他）
- 集成于昆仑万维的 AI 生态（Skywork 模型）
- 新兴中国 AI 音乐生成领域的组成部分

---

## 3. 歌声合成 (Singing Voice Synthesis, SVS)

### 3.1 概述

SVS 从乐谱输入（歌词音素序列 + 音高/时长标注）生成歌声。与文本到语音 (TTS) 不同，SVS 需要处理持续元音、颤音、音高滑音、呼吸控制和表现性的歌唱时序。

### 3.2 关键模型

#### DiffSinger (Liu, Ren et al., 2021/2022)
- **论文**: "DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism" (arXiv:2105.02446, AAAI 2022)
- **作者**: Jinglin Liu, Zhi Ren, Yi Ren, Chen Zhang, Zhou Zhao (浙江大学)
- **架构**:
  - **扩散概率模型**作为声学模型：参数化马尔可夫链迭代地将噪声转换为以乐谱为条件的 **mel-spectrogram**
  - **浅层扩散机制**: 核心创新。DiffSinger 不是运行从纯高斯噪声到 mel-spectrogram 的完整扩散链，而是从简单基线模型（如前馈 Transformer 预测）提供的中间表示开始扩散。这：
    - 减少了所需的扩散步数
    - 提高了训练稳定性
    - 产生比全链扩散更高质量的输出
  - **输入**: 乐谱（音素 + F0 音高轮廓 + 音符时长）
  - **输出**: Mel-spectrogram，通过神经声码器（HiFi-GAN 等）转换为波形
- **训练**: Mel-spectrogram 上的重建损失 + 扩散去噪目标
- **评估**: MOS 分数与之前的神经 SVS 系统持平或超越
- **意义**: 确立了扩散作为 SVS 的强大方法；"浅层扩散"技巧被广泛采用

#### VISinger / VISinger 2 (Xia et al., NWPU, 2022--2023)

**VISinger** (ICASSP 2022):
- **论文**: "VISinger: Variational Inference with Adversarial Learning for End-to-End Singing Voice Synthesis"
- **作者**: Yiwei Xia et al. (西北工业大学)
- **架构**:
  - **完全端到端**: 直接从乐谱生成波形，消除传统多阶段流水线（独立的时长模型、声学模型、声码器）
  - **变分推断 (VI)**: 建模自然歌声所需的复杂声学分布
  - **对抗训练**: 判别器确保真实的音频输出
  - **优势**: 参数少于多阶段系统；更简单的训练流水线
- **局限**: 相位预测问题导致有声段出现故障和抖动

**VISinger 2** (INTERSPEECH 2023):
- **论文**: "VISinger 2: High-Fidelity End-to-End Singing Voice Synthesis"
- **关键改进**: 将 **数字信号处理 (DSP) 合成器**集成到端到端框架中
  - DSP 合成器处理相位相关成分，避免了 VISinger 1 中导致伪影的问题性直接文本到相位映射
  - 在更高采样率下产生更高保真度的输出
- **性能**: 以更少参数实现比两阶段模型更好的质量

#### XiaoiceSing (Microsoft, 2021)
- **论文**: "XiaoiceSing: A High-Quality and Integrated Singing Voice Synthesis System" (Microsoft Xiaoice 团队)
- **架构**: 从 TTS 适配的序列到序列 (seq2seq) 模型
  - **编码器**: 将音素/音符序列编码为隐藏表示
  - **注意力机制**: 对齐编码器输出与解码器帧
  - **解码器**: 自回归地生成声学特征 (mel-spectrogram)
  - **方差适配器**: 预测时长、音高 (F0) 和能量，用于表现性歌唱控制
  - **声码器**: 神经声码器 (HiFi-GAN/WaveNet) 将 mel-spectrogram 转换为波形
- **歌声特定特征**:
  - 显式 F0 预测实现精确音高跟踪
  - 时长控制确保正确的音乐时序
  - 呼吸和颤音建模实现表现性歌唱
- **训练数据**: 数小时高质量歌唱录音，带有精确的音素级音高和时长标注
- **意义**: 最早的高质量神经 SVS 系统之一，建立在 Microsoft 的 TTS 技术基础上

#### ExpressiveSinger (2024)
- 用于多语言和多风格 SVS 的级联扩散模型
- 支持多种歌唱风格和语言

#### RDSinger (2024)
- 基于参考的扩散网络，用于高保真 SVS
- 使用参考音频条件控制音色和风格

#### DITSinger (2025)
- 研究 SVS 质量的缩放效应
- 解决 SVS 中不明确的缩放规律和系统方法论

#### OpenDiffSinger (社区, 2022--2025)
- DiffSinger 框架的开源社区分支和扩展
- 主要持续贡献：多说话人/多语言支持扩展、改进的音素字典和时长建模
- 新增数据集准备和训练流水线简化的 GUI 工具
- 声码器改进集成 NSF (神经源滤波器) 和 HiFi-GAN 变体
- OpenVPI 生态系统提供 SVS 数据集创建、音素对齐和训练的配套工具

#### ACE Studio / ACE Singer (2024--2025)
- ACE Studio（中国 AI 音乐科技公司）的商业 SVS 系统
- 从乐谱/歌词输入生成真实歌声——真正的 SVS，非语音转换
- 支持多语言（中文、英文、日文）、表现力参数控制（呼吸、颤音、力度）、多个声音库和 DAW 集成（VST 插件）
- 在商业 SVS 市场与 Synthesizer V (Dreamtonics) 和 XiaoiceSing 竞争

#### DiffSinger 加速 (2025)
- 多项持续努力加速 DiffSinger 推理，包括一致性蒸馏、渐进蒸馏和少步 ODE 求解器
- 在保持质量的同时减少扩散采样步数，实现接近实时推理
- 无单一规范 "Lite" 论文；加速技术来自更广泛的扩散模型加速文献

### 3.3 SVS 评估

- **MOS (Mean Opinion Score)**: 金标准；人工评分者在 1--5 分制上评价自然度
- **音高准确度**: 以生成与目标音高轮廓之间的 F0 RMSE 衡量
- **时长准确度**: 生成的音素时长与乐谱的对齐程度
- **频谱质量**: 生成与参考频谱图之间的对数频谱距离等指标
- **主观听音测试**: 系统间的对比偏好测试
- **挑战**: 没有单一自动化指标能捕获歌唱的完整感知质量（音准、表现力、音色自然度、呼吸控制）

### 3.4 开放性问题
- **表现性控制**: 对颤音、力度、分句和情感表达的细粒度控制
- **零样本说话人适配**: 从短参考片段生成任意声音的歌唱
- **多语言支持**: 大多数 SVS 系统是语言特定的（中文、英文、日文）
- **实时 SVS**: 用于交互式应用的低延迟生成
- **歌唱技巧建模**: 嘶吼、假声、怒音及其他声乐技巧

---

## 4. 可控生成

### 4.1 条件模态

可控音乐生成允许用户指定超越自由文本的音乐属性。

#### 基于文本的条件
- **自由文本**: 所需音乐的自然语言描述（用于 MusicGen, Stable Audio, Suno, Udio）
- **文本编码器**: T5 (MusicGen, Stable Audio), CLAP (AudioLDM), MuLan (MusicLM)
- **局限**: 文本对音乐属性本质上不精确；"upbeat jazz" 对不同模型意味着不同的东西

#### 音乐特定属性控制
- **Mustango** (Melechovsky, Guo, Ghosal, Majumder, Herremans, Poria; NAACL 2024)
  - **架构**: 带有结构化文本条件的潜在扩散模型
  - **核心创新**: 从文本提示中解析音乐特定属性——流派、调性、速度、和弦、乐器
  - 使用语言模型从文本中提取结构化音乐参数，然后以这些显式属性为条件进行生成
  - 在西方音乐上达到 SOTA 可控性
  - **局限**: 可控性限于西方音乐理论概念

#### 旋律条件
- MusicGen：从音频提示中基于色度的旋律提取；生成跟随所提供旋律的音乐
- MusicLM：哼唱到编配的能力
- YuE：风格迁移，在改变人声风格时保留伴奏

#### 情感条件
- **LARA-Gen**: 实现音乐生成的连续情感控制
- **EBS (Emotion-Based Sampling)**: 使用情感标签控制生成过程的算法 (IEEE TMM)
- **带连续值情感的符号音乐**: 在符号生成中控制织体和情感弧线
- 方法：效价-唤醒度空间映射；注入到条件中的情感嵌入向量

#### 乐器和音色控制
- 通过文本或音频参考指定特定乐器或音色特征
- AudioLDM：音频到音频的风格迁移
- MusicGen：可通过建立音色参考的音频提示进行条件生成

### 4.2 细粒度控制方法

#### SegTune (2025)
- **论文**: "SegTune: Structured and Fine-Grained Control for Song Generation" (arXiv:2510.18416)
- 实现对音乐输出不同方面的段落级控制
- 支持结构化生成，其中不同段落（前奏、主歌、副歌）可以具有不同属性
- 用于可控长篇文本到音频生成的细粒度条件

#### TVC-MusicGen (INTERSPEECH 2025)
- MusicGen 的时变结构控制
- 在生成过程中变化的动态结构控制
- 解决标准 MusicGen 始终应用统一条件的局限

#### 上下文学习 (YuE)
- YuE 为音乐生成重新设计上下文学习
- 实现风格迁移、双向生成和少样本适配
- 可在保留结构元素的同时在流派/语言间转换

### 4.3 控制方法分类

| 方法 | 粒度 | 示例系统 |
|------|------|----------|
| 自由文本 | 粗 | MusicGen, Stable Audio, Suno |
| 结构化文本属性 | 中 | Mustango |
| 旋律条件 | 中 | MusicGen, MusicLM |
| 音频参考 | 中 | AudioLDM, YuE |
| 情感标签/连续值 | 中 | LARA-Gen, EBS |
| 时变/段落 | 细 | SegTune, TVC-MusicGen |
| 乐谱级 | 细 | 符号模型（基于 REMI） |

### 4.4 开放性问题
- **精确和声控制**: 指定精确的和弦进行、转调和声部进行
- **曲式级控制**: 控制主歌-副歌-桥段结构、歌曲长度和过渡
- **实时交互控制**: 在播放期间调整生成参数
- **多属性控制**: 同时控制多个属性而不相互干扰
- **非西方音乐**: 大多数可控系统假设西方调性和声

---

## 5. 视频到音乐生成

跨模态音乐生成——从视觉输入（视频、图像）生成音乐——是连接计算机视觉和音频生成的新兴研究方向。

### 5.1 关键模型

#### MuVi (arXiv:2410.07840, 2024)
- **论文**: "MuVi: Video-to-Music Generation with Rhythmic Alignment"
- 从视频输入生成音乐，实现视觉运动和音乐节拍结构之间的**节奏对齐**
- **视觉节奏提取器**: 从视频提取节奏线索（运动强度、场景转换）形成"视觉节奏"表示
- **音乐生成模块**: 使用提取的视觉节奏作为条件，生成与视频动态时间对齐的音乐
- 通过客观节奏对齐分数和主观人工评估验证

#### CMT (对比多模态 Transformer)
- 使用对比学习对齐视频和音乐表示的跨模态生成框架
- Transformer 架构处理视频帧并通过跨模态注意力生成对应音乐
- 注意：多篇论文使用类似命名；引用时需确认具体 arXiv ID

#### M2UGen (多模态音乐理解与生成)
- 统一框架桥接音乐理解（字幕、问答、分析）和生成（文本到音乐、图像到音乐）
- 基于 LLM 的架构，集成专用音频和视觉编码器与音乐生成解码器
- 处理文本、图像和音频输入进行跨模态音乐创作

#### Video2Music
- 基于视频语义和运动条件生成背景音乐
- 从视频帧提取多模态特征（视觉、运动、语义）作为条件信号
- 解决时间对齐：确保生成音乐的节奏和情绪匹配场景转换

#### MuVi
- 视觉到音乐生成，关注视频运动和音乐节拍结构之间的节奏对齐

### 5.2 评估挑战
- **CMMD (对比音乐-视频度量)**: 评估生成音乐与视频内容对齐度的指标
- **时间同步**: 音乐节拍与视觉场景转换的对齐程度
- **情感一致性**: 生成音乐的情绪与视觉内容的匹配程度
- 视频到音乐评估缺乏标准化基准

### 5.3 新兴趋势
- **基于扩散的 V2M**: 使用潜在扩散从视频生成更高保真度的音频
- **LLM 条件 V2M**: 以大语言模型作为音乐生成骨干，视觉特征作为条件
- **情感驱动生成**: 将视觉情感内容（颜色、运动、面部表情）映射到音乐参数

---

## 6. 人类偏好对齐

将音乐生成模型与人类审美偏好对齐，类似于语言模型中的 RLHF，是快速兴起的研究领域。

### 6.1 音乐偏好基准

- **音乐生成模型与指标的人类偏好基准研究** (ICASSP 2025)
  - 系统的人类偏好研究，基准测试多个音乐生成模型和评估指标
  - 评估现有自动化指标（FAD、KL 散度等）与人类感知质量的相关性
  - 关键发现：当前指标与人类偏好相关性有限，推动偏好对齐方法的发展
  - 为未来音乐生成中的 RLHF 式对齐工作提供实证基础
- **音乐偏好对齐**是新兴方向：将 RLHF/DPO 方法（在 LLM 中已验证）应用于音乐生成

### 6.2 DPO 与偏好对齐方向

- 将 DPO 框架（最初为 LLM 开发）应用于音乐生成是活跃研究方向
- 直接从偏好配对优化，无需单独的奖励模型，理论上更稳定和高效
- 多篇 2025 年论文探索此方向，但尚无统一公认的框架名称

### 6.3 评估基准

- **SongBench** (Make-It-Music, 2025, arXiv:2502.19324): 南开大学提出的带监督音乐质量标签的策划数据集，用于歌曲生成。Make-It-Music 框架利用这些质量标签改进训练
- **FakeMusicCaps**: 用于检测和归因任务的 AI 生成音乐数据集

### 6.4 开放性问题
- **奖励建模**: 音乐的多属性特性（和声、节奏、旋律、音色）和高主观性使奖励建模独特困难
- **偏好数据收集**: 可扩展可靠的音乐配对人类比较收集
- **多维对齐**: 同时对齐多个音乐维度而不产生权衡
- **文化敏感性**: 人类偏好在不同文化和音乐传统之间差异显著

---

## 7. 评估

### 7.1 自动化指标

#### Frechet Audio Distance (FAD)
- **来源**: Kilgour et al., INTERSPEECH 2019；从图像域的 Frechet Inception Distance (FID) 适配而来
- **方法**: 计算参考集和生成集的音频嵌入高斯分布之间的 Frechet 距离
- **嵌入模型**: 通常使用 VGGish、PANNs (Pre-trained Audio Neural Networks) 或 CLAP 音频编码器
- **特性**:
  - **分布级无需参考**: 不需要成对比较；比较分布统计量
  - 同时衡量质量和多样性
  - 越低越好
- **逐曲 FAD** (Microsoft, 2023): 扩展版本为单个样本计算 FAD，与人类感知质量 (MOS) 呈中到强相关。可在 `microsoft/fadtk` 获取
- **局限**: 依赖嵌入模型质量；可能无法捕获所有感知相关维度；对数据集偏差敏感

#### 标签分布上的 KL 散度
- 计算分类器在参考音频 vs 生成音频上预测的标签分布（如流派、乐器标签）之间的 KL 散度
- 衡量生成音频是否具有与参考分布相似的高层属性
- **局限**: 不如 FAD 稳健；在退化输出（如静默音频）上可能行为不一致。在 AudioLDM 评估工具包中被标记为不可靠

#### MuLan 相似度 / CLAP 分数
- 使用对比模型 (MuLan, CLAP) 计算文本和音频嵌入之间的余弦相似度
- 衡量文本-音频对齐度：生成的音频与文本提示匹配的程度
- 越高越好
- 用于 MusicLM 及后续模型

#### FrEchet Music Distance (FMD) (2024)
- **论文**: arXiv:2412.07948 (2024 年 12 月)
- 专门为符号音乐评估适配的 FAD
- 操作符号表示而非音频
- 解决生成式符号音乐模型评估指标的缺失

#### 其他自动化指标
- **IS (Inception Score)**: 使用分类器预测衡量生成样本的质量和多样性
- **FID (Frechet Inception Distance)**: 图像域指标，偶尔适配于频谱图表示
- **对数频谱距离**: 衡量生成与参考音频之间的频谱相似度

### 7.2 人工评估

#### Mean Opinion Score (MOS)
- 感知质量评估的金标准
- 人工评分者在 1--5 分制上为音频打分（差到优）
- 通常衡量整体质量、自然度或保真度
- 用作验证自动化指标的真值

#### 对比偏好测试
- 不同模型输出的并排比较
- 评分者在特定维度（质量、与提示的相关性、音乐性）上选择哪个样本更好
- 用于 MusicLM、MusicGen 及大多数主要模型评估

#### MusicCaps 基准
- **数据集**: 5,521 个带人工撰写字幕的 10 秒音乐片段（Google，随 MusicLM 发布）
- **来源**: 通过 AudioSet 来自 YouTube 视频；多样化流派
- **用途**: 文本到音乐模型的标准基准
- **评估协议**: 从 MusicCaps 字幕生成音频，计算与参考片段的 FAD 和其他指标
- **局限**: 仅 10 秒片段；尽管覆盖广泛，文化/流派多样性有限

### 7.3 综合评估综述 (2025)

- **论文**: "A Survey on Evaluation Metrics for Music Generation" (arXiv:2509.00051, 2025)
- 整合该领域的评估方法
- 关键发现：
  - FAD 和 KL 散度是最常用的自动化指标
  - 逐曲 FAD 与人类感知质量相关性最佳
  - 没有单一指标能捕获音乐质量的所有维度
  - 人工评估对可靠评估仍然是必要的
  - 基准数据集（MusicCaps, AudioSet）引入了自身的偏差

### 7.4 评估生成音乐的挑战

1. **多维性**: 音乐质量涵盖旋律、和声、节奏、音色、曲式、力度和表现力。没有单一指标能捕获所有维度。

2. **主观性**: 音乐质量本质上是主观的；不同的听众、文化和流派有不同的标准。

3. **长篇评估**: 大多数指标在短片段（10--30 秒）上操作。评估分钟级的结构连贯性缺乏现有指标的支持。

4. **文本-音频对齐**: 衡量生成音频与文本提示的匹配程度需要理解两种模态，这本身就是一个开放研究问题。

5. **原创性 vs 质量的权衡**: 模型应优先生成新颖音乐还是可能非常接近训练数据的高质量音乐？指标往往奖励后者。

6. **分布级 vs 样本级**: 大多数指标（FAD, KL）衡量分布属性，而非单个样本质量。逐曲指标正在兴起但尚未标准化。

7. **文化偏差**: 基准和评估方法以西方音乐为中心。

8. **缺乏标准化基准**: 不同论文使用不同的数据集、划分和协议，使跨论文比较困难。

### 7.5 新兴评估方法

- **LLM 作为评审**: 使用大语言模型从描述和音频特征评估音乐质量
- **音乐理解基准**: MARBLE 基准（YuE 使用）在音乐理解任务上评估学习的表示
- **人类偏好对齐**: AAAI 2025 论文 (arXiv:2511.15038) 关于将生成式音乐 AI 与人类偏好对齐
- **FakeMusicCaps**: 从 MusicCaps 派生的 AI 生成音乐数据集，用于检测和归属任务
- **SongBench** (Make-It-Music, 2025, arXiv:2502.19324): 南开大学提出的带监督音乐质量标签的策划数据集

---

## 8. 架构对比总结

| 模型 | 年份 | 方法 | 分词器/编解码器 | 最大长度 | 文本条件 | 开源 |
|------|------|------|-----------------|----------|----------|------|
| Jukebox | 2020 | 层次化 AR (VQ-VAE) | VQ-VAE (3 层) | 数分钟 | 歌词 + 元数据 | 是 |
| MusicLM | 2023 | 层次化 AR (语义 + 声学) | SoundStream | 数分钟 | MuLan 文本嵌入 | 否 |
| MusicGen | 2023 | 单阶段 AR Transformer | EnCodec (32 kHz, 4 CB) | ~30 秒 | T5 + 旋律 | 是 |
| AudioLDM | 2023 | 潜在扩散 (U-Net) | VAE (mel-spectrogram) | ~10 秒 | CLAP | 是 |
| AudioLDM 2 | 2024 | 潜在扩散 + GPT-2 | VAE + 改进 | ~30 秒 | CLAP + GPT-2 | 是 |
| Stable Audio 2.0 | 2024 | 潜在扩散 (DiT) | 压缩自编码器 (21.5 Hz) | 3 分钟 | T5 | 部分 |
| Stable Audio Open | 2024 | 潜在扩散 (DiT) | 自编码器 | ~47 秒 | T5 | 是 |
| YuE | 2025 | 基于 LLaMA2 的 AR Transformer | 学习的音乐 token | 5 分钟 | 歌词 + 风格 | 是 |
| Suno V4.5 | 2025 | 未公开 | 未公开 | 4 分钟 | 文本 + 歌词 | 否 |
| Suno V5 | 预计 | 未公开 | 未公开 | TBD | 文本 + 歌词 | 否 |
| Udio | 2024--25 | 未公开 | 未公开 | 2 分钟 | 文本 + 歌词 | 否 |
| TangoFlux | 2024 | Flow Matching | 学习的 | 30 秒 | 文本 | 是 |
| ACE-Step | 2025 | DCAE + 线性 Transformer + 扩散 (REPA) | DCAE (Sana) + MERT/m-hubert | 4 分钟 | 文本 + 歌词 | 是 (ACE Studio + StepFun) |
| MusicFlow | 2024 | 级联流匹配 | 自监督语义/声学 | ~30 秒 | 文本 | 是 |
| SongCreator | 2024 | 双序列 LM (DSLM) + 注意力掩码 | EnCodec 风格 | 全曲 | 歌词 + 音频提示 | 是 |
| MusicFX | 2023--25 | 基于 MusicLM (Google) | SoundStream | 短片段 | 文本 | 否 |
| SkyMusic | 2024--25 | 未公开 | 未公开 | 全曲 | 文本 (多语言) | 否 |

---

## 9. 关键开放性问题与未来方向

1. **长篇连贯性**: 在多分钟作品中维持音乐结构、主题发展和全局曲式仍然是主要挑战。YuE 的 5 分钟输出是一个里程碑，但质量仍然不一致。

2. **评估**: 缺乏全面、标准化的评估框架。FAD 是事实标准但存在已知弱点。人工评估昂贵且主观。

3. **可控性**: 用户能指定的（文本提示）与他们想控制的（特定和声、结构、音色）之间的差距仍然很大。结构化控制（Mustango, SegTune）有前景但有限。

4. **版权与伦理**: 在受版权保护的音乐上训练引发法律和伦理问题。明确避免版权数据的开放模型（Stable Audio Open）指明了方向，但可能牺牲质量。

5. **实时生成**: 当前模型离高质量的实时输出还很远。游戏、现场表演和创意工具的交互式音乐生成需要低延迟推理。

6. **文化多样性**: 大多数模型偏向西方流行音乐。对非西方音阶、乐器、曲式和歌唱风格的支持有限。

7. **多模态集成**: 音乐生成与视频、舞蹈等其他模态的结合尚处于起步阶段但正在增长（ISMIR 2025 关于视觉到音乐生成的综述）。

8. **数据规模与质量**: 专有系统（Suno, Udio）可能在远大于学术/开放模型的数据集上训练，这是其质量优势的原因之一。

9. **AR 与扩散的融合**: 结合自回归（序列连贯性、可变长度）和扩散/flow-matching（并行生成、全局结构、多样性）优势的混合架构是一个主要趋势。

10. **偏好对齐**: 将生成模型与人类音乐偏好大规模对齐是一个新兴研究领域 (AAAI 2025)，类似于语言模型中的 RLHF。

---

## 参考文献（关键论文）

- Huang & Vaswani: "Music Transformer" (arXiv:1809.04281, ICLR 2019)
- Huang & Yang: "Pop Music Transformer / REMI" (arXiv:2002.00212, 2020)
- Dhariwal et al.: "Jukebox: A Generative Model for Music" (arXiv:2005.00341, 2020)
- Zeghidour et al.: "SoundStream" (arXiv:2107.03312, 2021)
- Defossez et al.: "EnCodec" (arXiv:2210.13438, 2022)
- Liu et al.: "DiffSinger" (arXiv:2105.02446, AAAI 2022)
- Xia et al.: "VISinger" (ICASSP 2022) and "VISinger 2" (INTERSPEECH 2023)
- Agostinelli et al.: "MusicLM" (arXiv:2301.11325, 2023)
- Liu et al.: "AudioLDM" (2023) and "AudioLDM 2" (arXiv:2308.05734, 2023--2024)
- Copet et al.: "MusicGen" (arXiv:2306.05284, 2023)
- Stability AI: "Stable Audio 2.0" (April 2024)
- Melechovsky et al.: "Mustango" (NAACL 2024, arXiv:2311.08355)
- Yuan et al.: "YuE" (arXiv:2503.08638, ICLR 2025)
- "A Survey on Evaluation Metrics for Music Generation" (arXiv:2509.00051, 2025)
- "Auto-Regressive vs Flow-Matching: A Comparative Study" (arXiv:2506.08570, 2025)
- "Aligning Generative Music AI with Human Preferences" (arXiv:2511.15038, AAAI 2025)
- "Pianoroll-Event" (arXiv:2601.19951, 2025)
- "REMI-z: Track-Aware Tokenization" (NeurIPS 2025)
- "SegTune: Structured and Fine-Grained Control" (arXiv:2510.18416, 2025)
- ACE-Step: "A Step Towards Music Generation Foundation Model" (arXiv:2506.00045, ACE Studio + StepFun, 2025)
- MusicFlow: "Cascaded Flow Matching for Text Guided Music Generation" (arXiv:2410.20478, ICML 2024)
- SongCreator: "Lyrics-based Universal Song Generation" (NeurIPS 2024 Poster, Shun Lei et al.)
- "Benchmarking Music Generation Models and Metrics via Human Preference Studies" (ICASSP 2025)
- CMT: "Contrastive Multimodal Transformer for Video-to-Music" (需确认具体 arXiv ID)
- M2UGen: "Multi-Modal Music Understanding and Generation" (需确认具体 arXiv ID)
- "Make-It-Music / SongBench" (arXiv:2502.19324, 2025, 南开大学)
- OpenDiffSinger community: GitHub (持续更新, 2022--2025)
- "Discrete Audio Tokens: More Than a Survey" (arXiv:2506.10274, 2025)
