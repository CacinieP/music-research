# 音乐生成：技术研究笔记

精选符号生成、音频生成、歌声、控制、视频条件和评测研究。内容核对日期：2026-09-19。下述参数对应注明的论文/版本，不代表整个产品系列的所有后续版本。

> English version: [music-generation.md](music-generation.md)

---

## 1. 符号音乐生成

### 1.1 表示

符号模型生成音符、时长、时序、配器或记谱；还需要合成器或渲染器才能转成音频。

| 表示 | 编码内容 | 权衡 |
|------|----------|------|
| MIDI / MIDI-like 事件 | 音符开关、音高、力度、时序、乐器及可选控制 | 演奏事件紧凑，但分词须定义排序和时间分辨率 |
| 钢琴卷帘 | 音高 × 时间的激活/力度，可按轨道分开 | 网格方便；分辨率越高越大，相邻同音重复需要起音处理 |
| ABC 记谱 | 拍号、音高、时长、声部等文本记谱 | 紧凑可读，支持特性取决于解析器及语料 |
| REMI | 显式小节/拍位、音高、时长、力度及选定元数据 | 直接表示节拍位置，以量化损失换取结构 |
| Compound Word | 将属性组合，使用各自的属性嵌入/预测 | 缩短序列，需明确复合事件的结构 |

**MIDI 与分词不同：** `Time Shift` 是模型 token，不是原生 MIDI 通道消息类型；力度是音符消息的属性。标准 MIDI 文件存储 delta time，可包含速度/拍号元数据，并能据此推导小节。可能省略节拍结构的是选定的事件词表，而非 MIDI 整体。

ABC 支持多声部和复调，并非只能表达单旋律。[MuPT 论文](https://arxiv.org/abs/2404.06393) 使用同步多轨 ABC（SMT-ABC）处理不同轨道的小节对齐。

已核验的分词来源：

- [Pop Music Transformer / REMI](https://arxiv.org/abs/2002.00212)，Yu-Siang Huang、Yi-Hsuan Yang，2020。
- [Compound Word Transformer](https://arxiv.org/abs/2101.02402)，Hsiao 等，2021。
- [REMI-z 实现及论文引用](https://github.com/Sonata165/REMI-z)，*Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization*，NeurIPS 2025。REMI-z 在小节结构中按轨道组织音符；REMI+ 与 REMI-z 是不同方案，不是同一个 2025 格式的两个名称。
- [Pianoroll-Event](https://arxiv.org/abs/2601.19951)，Qian 等，**2026 年 1 月**：用帧、间隙、模式和音乐结构事件结合网格结构与紧凑编码。
- [MidiTok](https://miditok.readthedocs.io/en/latest/) 是支持多种方案的分词库，不是名为“MIDI-Token”的单一表示。可复现研究需固定版本与分词配置。

### 1.2 代表模型

**Music Transformer**（Huang 等，ICLR 2019）提出面向长音乐序列的高效相对注意力实现。原实验使用 **JSB Chorales 和 Piano-e-Competition**，不是 MAESTRO。相对序列位置有助于重复与时序建模，但本身并不编码移调不变的音程。效率改进针对中间相对位置张量，而非消除注意力的全部二次复杂度。[论文](https://arxiv.org/abs/1809.04281)。

**Pop Music Transformer** 将 Transformer-XL 式模型与 REMI 结合，进行节拍感知的流行钢琴生成。论文在其评测下显示节奏收益，不代表所有流派都一定更好。[论文](https://arxiv.org/abs/2002.00212)。

**MuPT**（2024）研究符号音乐 Transformer 预训练、SMT-ABC、上下文与规模扩展。结论依赖语料和表示，“更大符号模型总能保持曲式”并非一般结论。[论文](https://arxiv.org/abs/2404.06393)。

### 1.3 自回归、扩散与流匹配

自回归模型将 token 序列分解为：

$$
p(x\mid c)=\prod_{t=1}^{T}p(x_t\mid x_{<t},c).
$$

它天然支持顺序续写，但需付出逐步采样成本。上下文限制和训练/推理差异可能损害长程连贯性。

扩散与流匹配模型迭代地将噪声转为钢琴卷帘、潜在序列或其他表示。每一步内部的位置可以并行处理，但仍有多个采样步骤。固定输出长度是部分实现的常见设置，不是数学要求；时长条件、掩码与分块可以支持可变长度。

流匹配学习选定概率路径的向量场；扩散学习去噪/分数相关模型。它们是相关的连续生成方法，但不是同义词，也都不能保证在所有设置下多样性或结构更好。

[Auto-Regressive vs Flow-Matching](https://arxiv.org/abs/2506.08570)（Tal、Kreuk、Adi，2025）在控制数据与训练设置下比较**文本条件的音频音乐**。其中的质量、控制、编辑和采样结论不能用来证明所有符号音乐模型的优劣。

### 1.4 数据集

| 数据集 | 内容 | 对应版本规模 |
|--------|------|--------------|
| [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) | 紧密对齐的钢琴音频/MIDI 演奏 | 约 200 小时；使用官方版本划分 |
| [Lakh MIDI v0.1](https://colinraffel.com/projects/lmd/) | 去重 MIDI 集合 | 176,581 个文件；LMD-matched 是较小子集 |
| [POP909](https://arxiv.org/abs/2008.07142) | 含 melody、bridge、piano 轨的流行钢琴编曲 | 909 首歌曲 |

文件数、不同作品数、演奏次数与片段数不同。报告训练/测试集时说明清洗、去重及歌曲/艺术家隔离。

### 1.5 开放问题

长曲式、主题再现、多轨协调、表现性演奏时序与和声控制仍是重要问题。符号结构不能直接指定录音音色和制作，也没有单一表示能同等覆盖所有音乐维度。

## 2. 音频级音乐生成

### 2.1 编解码器与连续自编码器

离散编解码器将波形转为语言模型使用的 token 序列；连续自编码器为扩散/流模型提供潜在序列。因此“音频生成必须有离散 codec”不正确。

| 编解码器 | 已核验的设计与设置 |
|----------|--------------------|
| [SoundStream](https://arxiv.org/abs/2107.03312)（2021） | 卷积编解码器、残差向量量化（RVQ）、对抗训练及量化器 dropout；支持多种码率 |
| [EnCodec](https://github.com/facebookresearch/encodec)（2022） | 卷积/循环编解码器、RVQ、多尺度 STFT 对抗训练；原始发布含 24 kHz 单声道与 48 kHz 立体声 |
| [DAC](https://arxiv.org/abs/2306.06546)（2023） | 改进 RVQGAN，使用周期激活与改进的码本学习；包括高保真 44.1 kHz 压缩 |

原始 EnCodec 配置中，24 kHz 模型为 **75 Hz** 帧率，支持 1.5/3/6/12/24 kbps；48 kHz 模型为 **150 Hz**，支持 3/6/12/24 kbps。激活码本数随码率变化。**MusicGen 使用另外训练的 32 kHz、50 Hz EnCodec 配置**，并非直接套用上述两种配置。

RVQ 逐级量化残差。前后码本的信息可能不同，但没有保证“这一层是和声、那一层是音色”的标签。帧率表示时间位置数；每秒总 token 数还取决于码本数和声道组织方式。

### 2.2 Jukebox（OpenAI，2020）

[Jukebox](https://cdn.openai.com/papers/jukebox.pdf) 使用三层 VQ-VAE 和自回归先验/上采样器，以艺术家/流派元数据为条件；歌词条件模型还接收文本。论文训练数据为 120 万首歌曲，并描述了 60 万首英文歌子集。

44.1 kHz **单声道**波形按 **8、32、128 倍**降采样，对应底层、中层、顶层约 **5,512.5、1,378.1、344.5 token/s**。这些是降采样因子，而非 8/34/65 Hz 的 token 速率。每层码本有 2,048 项。

生成由顶层逐级条件上采样。滑动窗口支持多分钟输出，但顶层上下文约 24 秒；能生成几分钟不代表能组织重复副歌或全曲曲式。采样计算成本高。代码与 checkpoint 见[官方仓库](https://github.com/openai/jukebox)，用途应核对相应许可。

### 2.3 MusicLM（Google，2023）

[MusicLM](https://arxiv.org/html/2301.11325v1) 结合三种独立预训练的表示：

1. **MuLan** 提供音乐-文本条件。生成器训练时使用量化的 MuLan **音频**嵌入，推理时换为 **文本**嵌入。
2. **w2v-BERT 特征经 k-means 聚类**得到 25 Hz 的语义 token。
3. **SoundStream** 提供声学 token：本系统中为 24 kHz 单声道、50 Hz 帧率、12 层 RVQ。

自回归阶段依次预测语义、粗声学和细声学 token。语义 token 不是 SoundStream 的第一层码本。系统展示了文本/旋律条件和多分钟样例。原论文未发布模型权重；后续消费端访问是另一个问题。

MusicLM 引入 **MusicCaps**：5,521 个带人工描述的十秒样本。它是评测资源，不是生成器的 280,000 小时训练集合。参见[原论文](https://arxiv.org/abs/2301.11325)。

### 2.4 MusicGen（Meta，2023）

[MusicGen](https://arxiv.org/abs/2306.05284) 用单个自回归 Transformer 建模 codec 流。发布模型以冻结的 T5 文本编码器为条件；旋律版本增加 chroma 特征。

单声道设置使用 32 kHz 音频、4 个 50 Hz 码本及延迟交错。一个自回归步骤中，不同码本对应**错开的 codec 时刻**，不是都对应同一个音频帧。相比展开所有码本 token，这减少了顺序模型步数。

[官方文档](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md) 列出 300M、1.5B、3.3B 模型规模，20,000 小时授权训练音乐（含内部集合及 Shutterstock/Pond5），以及独立的立体声版本。立体声使用左右声道的独立码流。标准训练上下文为 30 秒，支持的实现可进行续写/扩展。

Chroma 合并八度信息，也可能包含和声内容；它是旋律引导，不是精确乐谱或歌词控制。代码与权重条款不同：AudioCraft 代码为 MIT，发布的 MusicGen 权重在[模型卡](https://github.com/facebookresearch/audiocraft/blob/main/model_cards/MUSICGEN_MODEL_CARD.md) 中使用 CC-BY-NC 4.0。“开放权重”不等于不限用途的商业授权。

[MAGNeT](https://arxiv.org/abs/2401.04577) 使用掩码式非自回归音频 token 建模，是相关生成方法，不是对不变的 MusicGen checkpoint 换一个更快采样器。

### 2.5 AudioLDM 与 AudioLDM 2

**AudioLDM**（2023）在 VAE 的 mel 频谱潜在空间中训练扩散模型，再由 VAE 解码和 HiFi-GAN 声码器生成波形。训练时使用 CLAP 音频嵌入，采样时可换为 CLAP 文本嵌入。AudioMAE 并不是 AudioLDM 1 的定义性 VAE 组件。原设置面向音效、音乐等短通用音频。[论文](https://arxiv.org/abs/2301.12503)。

**AudioLDM 2**（2023 年预印；2024 年期刊版）引入基于 AudioMAE 的 **language of audio（LOA）**。GPT-2 式模型从条件模态预测 LOA，潜在扩散再以 LOA 为条件生成音频。这比“联合微调 GPT-2 改善文本理解”更具体。应区分语音、音乐和通用音频 checkpoint；采样率与时长依赖 checkpoint 和实现。[论文](https://arxiv.org/abs/2308.05734)。

### 2.6 Stable Audio 版本

| 版本 | 官方记录的行为 | 区别 |
|------|----------------|------|
| Stable Audio 1.0（2023 年 9 月） | 发布时提供免费版 45 秒、Pro 90 秒生成 | 47 秒不是其通用上限（[发布说明](https://stability.ai/news-updates/stable-audio-using-ai-to-generate-music)） |
| Stable Audio 2.0（2024 年 4 月） | 最长 3 分钟、44.1 kHz 立体声，支持文本/音频条件 | 商业发布，使用授权 AudioSparx 数据；不是 Open 1.0 checkpoint（[公告](https://stability.ai/news/stable-audio-2-0)） |
| Stable Audio Open 1.0（2024 年 6 月） | 最长 47 秒、44.1 kHz 立体声；连续自编码器、T5、DiT | 独立开放权重模型，有各自许可和能力边界（[模型卡](https://huggingface.co/stabilityai/stable-audio-open-1.0)） |

[Stable Audio Open 论文](https://arxiv.org/html/2407.14358v1) 报告**总计 486,492 条录音**：Freesound 472,618 条加 FMA 13,874 条，约 7,300 小时，源素材许可包括 CC0/CC-BY/CC-Sampling+。不是“48.6 万录音再加 7 万音乐”。自编码器约为每秒 21.5 潜在帧。Creative Commons 授权作品仍可能受版权保护，不能描述为“没有版权数据”。

截至核对日期，官方文档已经介绍 **Stable Audio 3.0**。旧稿“Stable Audio 3 在 2025 年开发中”的预测已过时；这个独立产品系列应查询[官方版本指南](https://stability.ai/guides/stable-audio-3-prompt-guide)。不能将上述历史参数直接用于后续版本。

### 2.7 YuE（2025）

[YuE](https://arxiv.org/abs/2503.08638) 由 Ruibin Yuan 等提出，是基于 LLaMA2 的长篇歌词到歌曲模型系列。论文描述了轨道解耦下一 token 预测、渐进结构条件、多任务/多阶段万亿 token 规模训练，并展示最长五分钟样例。

它还研究音频参考/上下文条件，以及 MARBLE 上的表示评测。作者报告相对选定专有系统有竞争力的结果；这不是独立确认的普遍排名，也不能证明“首个超越 Suno/Udio”。

Checkpoint、阶段、上下文和硬件要求应查[官方仓库](https://github.com/multimodal-art-projection/YuE)。10 GB 显存不是标准管线的通用要求；内存和延迟取决于实现、卸载、精度及输出时长。

### 2.8 商业歌曲生成器

**Suno 与 Udio** 提供文本/歌词条件歌曲生成，但本文所引来源没有公开其完整架构与训练方案。不能从听感推断 AR/扩散组件、训练数据规模或通用质量排名。

Suno 官方记录为 **v5 于 2025-09-23 发布**、**v5.5 于 2026-03-26 发布**、**v6 于 2026-09-09 发布**。因此“v5 预计发布”已过时。参见 [v5 发布](https://suno.com/release-notes/introducing-v5-the-world-s-best-music-model) 与[版本记录](https://suno.com/release-notes)。

对于 [Udio](https://www.udio.com/)，应区分新生成片段与续写后完整歌曲的时长。产品限制、访问、编辑和导出能力需同时注明版本/套餐及访问日期。本文不采用无依据的通用两分钟上限，也不声称其输出天然更像人类。

### 2.9 其他研究系统

| 系统 | 已核验的贡献 |
|------|--------------|
| [TangoFlux](https://arxiv.org/abs/2412.21037)（2024 年预印） | 流匹配文本到音频和 CLAP 排序偏好优化；音频任务范围比音乐更广 |
| [MusicLDM](https://arxiv.org/abs/2308.01546)（2023 年预印，ICASSP 2024） | 音乐适配潜在扩散、节拍同步音频/潜在 mixup；新颖性改善不保证绝无复现 |
| [MusicFlow](https://proceedings.mlr.press/v235/prajwal24a.html)（ICML 2024） | 语义和声学特征的级联流匹配，掩码条件支持填充和续写 |
| [SongCreator](https://arxiv.org/abs/2409.06029)（2024） | 双序列语言模型，通过可配置注意力掩码处理人声/伴奏任务 |
| [ACE-Step](https://arxiv.org/abs/2506.00045)（2025 年报告） | 扩散、音乐适配 DCAE、线性 Transformer，以及训练时的 MERT/m-HuBERT 表示对齐 |

ACE-Step 报告在其设置下用 A100 约 20 秒生成最长四分钟音乐。这是作者报告的测试结果，不是脱离硬件的延迟承诺。MERT/m-HuBERT 是表示对齐教师，不是音频 codec。比较能力时应将该[项目](https://ace-step.github.io/)及报告与后续 ACE-Step 版本区分。

## 3. 歌声合成

乐谱条件 SVS 根据歌词、音符与时序预测歌声，需要处理长元音、音符过渡、发音对齐和表现性音高。它不同于改变录音中的歌手声音，也不同于根据文字描述生成整首歌。

| 系统 | 正确描述 |
|------|----------|
| [XiaoiceSing](https://arxiv.org/abs/2006.06261)（2020） | FastSpeech 式非自回归频谱/F0/时长预测，原系统使用 WORLD |
| [DiffSinger](https://arxiv.org/abs/2105.02446)（2021 年预印，AAAI 2022） | Liu、Li、Ren、Chen、Zhao；以浅起点进行乐谱条件 mel 扩散，再经声码器 |
| [VISinger](https://arxiv.org/abs/2110.08813)（2021 年预印，ICASSP 2022） | Yongmao Zhang 等；含音高/时长建模的变分/流/对抗端到端合成 |
| [VISinger 2](https://arxiv.org/abs/2211.02903)（2022 年预印，INTERSPEECH 2023） | DSP 谐波/噪声合成引导波形解码，改善相位处理 |
| [DiTSinger](https://arxiv.org/abs/2510.09016)（2025） | 扩散 Transformer 扩展与字级时间范围约束的隐式对齐 |
| [OpenVPI DiffSinger](https://github.com/openvpi/DiffSinger) | 社区框架，能力取决于版本和声音库 |

应分别评估对齐有声帧的音高、相对乐谱/音素标注的时序、歌词可懂度、自然度、表现力和歌手身份。更正后的数据集及指标定义见[SVS 笔记](music-singing-synthesis-zh.md)。

## 4. 可控生成

### 4.1 条件模态

| 输入 | 预期控制 | 局限 |
|------|----------|------|
| 自由文本 | 流派、配器、情绪、制作 | 有歧义，通常不足以表达音符级要求 |
| 旋律/chroma | 音高类变化和旋律引导 | Chroma 丢失八度，不能唯一确定音符/和弦排列 |
| 乐谱/和弦/节拍序列 | 显式音乐内容或时间目标 | 依赖标注与支持词表 |
| 参考音频 | 风格、音色、续写或编辑上下文 | 可能混合身份、风格与内容 |
| 段落描述和边界 | 随时间变化的属性 | 边界遵循与过渡质量要单独测试 |
| 情绪标签/连续坐标 | 感知效价/唤醒度等情绪 | 标注和解释随听者/文化变化 |

**Mustango** 预测或条件化速度、节拍位置、调性、和弦等音乐信息，并使用音乐信息感知的扩散去噪器。其 **MusicBench** 是音乐-文本训练资源，不是人类偏好排行榜。[论文](https://arxiv.org/abs/2311.08355)。

### 4.2 时变控制

[TVC-MusicGen](https://www.isca-archive.org/interspeech_2025/yang25f_interspeech.html)（INTERSPEECH 2025）用自监督结构信息，使生成受段落边界与描述控制。研究在语言模型和扩散模型上测试此方法；名称并不表示只涉及 Meta 原始 MusicGen checkpoint。

[SegTune](https://arxiv.org/abs/2510.18416)（2025 年预印；后修订为 ACL 2026 论文）是非自回归歌曲生成框架，以局部提示对应时间段，以全局提示控制整体风格。时长预测器生成句级时间戳歌词。局部控制仍需检验歌词对齐、过渡及属性相互影响。

### 4.3 剩余挑战

精确和弦排列、转调、曲式、多重控制、实时编辑及非西方音乐系统需要超越更丰富的文字提示。应区分“接受某控制输入”和“可靠遵循控制”，只评估实际指定的音乐约束。

## 5. 视频到音乐生成

### 5.1 代表模型

| 模型 | 模态和贡献 |
|------|------------|
| [CMT](https://arxiv.org/abs/2111.08380)（2021） | **Controllable Music Transformer**：将视频时序/运动连接到符号音乐节奏、密度与强度，不是“Contrastive Multimodal Transformer” |
| [Video2Music](https://arxiv.org/abs/2311.00968)（2023） | 情感多模态 Transformer，使用语义、场景、运动和情绪特征，生成符号/和弦内容后动态渲染 |
| [M²UGen](https://arxiv.org/abs/2311.11255)（2023 年预印） | 多模态编码器和 LLM 连接音乐生成器，支持理解、生成/编辑 |
| [MuVi](https://arxiv.org/abs/2410.12957)（2024） | 视觉适配、音乐-视觉对比预训练和流匹配生成，处理语义与节奏对齐 |

MuVi 的完整标题为 *Video-to-Music Generation with Semantic Alignment and Rhythmic Synchronization*。

### 5.2 评测

分别评价语义匹配、情绪一致、时间同步和音频/音乐质量。注明视觉事件定义、节拍提取器、时间容差与负样本对。好配乐不一定在每个剪辑点都落一拍，同步目标取决于任务。

视觉/音频嵌入分数不能单独证明精细时序或叙事适配。指标和基准名称需对应具体论文与实现，不能默认存在通用“CMMD = 对比音乐-视频度量”标准。

## 6. 人类偏好对齐

RLHF 类方法从听者偏好学习奖励，再按奖励优化生成。DPO 类方法从偏好/非偏好样本对、相对参考策略进行优化；用于连续扩散或流模型时需要相应目标函数，不能直接照搬语言模型损失。

[人类偏好基准研究](https://arxiv.org/abs/2506.19085) 提供模型比较和指标分析，但本身不能证明训练出的奖励能泛化。[Aligning Generative Music AI with Human Preferences: Methods and Challenges](https://arxiv.org/abs/2511.15038) 是 2025 年预印本，录用于 **AAAI 2026 Senior Member Track**。

区分真实听者比较与 CLAP 排序等代理标签。奖励优化可能改善一个属性，却减少多样性或利用评审弱点。保留独立听测、未见提示/风格与源素材重叠检查。

## 7. 评测

| 指标 / 协议 | 衡量内容 | 重要边界 |
|-------------|----------|----------|
| FAD | 参考/生成音频嵌入拟合分布的距离，越低越接近 | 需要参考集合但不需成对录音；不测提示对齐 |
| CLAP / MuLan 相似度 | 学到的提示-音频关联 | 粗语义，不是精确音符或歌词 |
| 分类器 KL | 已定义标签分布的差异 | 必须说明成对还是总体协议 |
| FMD | 符号音乐嵌入分布差异 | 不是波形保真度 |
| 音高 / 和弦 / 节拍遵循 | 与音乐条件的一致程度 | 需要目标控制和可靠提取 |
| MOS / 成对偏好 | 听者评分或选择 | 需要明确人群、任务、设计及不确定性 |

[MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps) 是短片段文本到音乐评测资源；[AIME](https://huggingface.co/datasets/disco-eth/AIME) 包含偏好数据。MARBLE 衡量音乐理解表示，不直接评价歌曲质量。

没有单一指标能证明质量、原创性、结构或文化适切性。FAD 依赖编码器、参考集、样本量和预处理；逐曲 FAD 变体需单独验证，不是普遍最好的听感代理。公式、协议与来源见[评测笔记](music-evaluation-zh.md)。

## 8. 架构对比

以下时长是论文/版本设置或展示样例，不是统一的架构极限。

| 系统 | 生成方式 | 表示 / 条件 | 文档记录的范围 |
|------|----------|-------------|----------------|
| Jukebox（2020） | 层次 AR | 三层 VQ-VAE；元数据/歌词 | 44.1 kHz 单声道，窗口式多分钟生成 |
| MusicLM（2023） | 语义/粗/细 AR | w2v-BERT + SoundStream；MuLan | 24 kHz 单声道，多分钟样例 |
| MusicGen（2023） | 单阶段 AR | 32 kHz EnCodec；T5/chroma | 30 秒上下文；单声道及独立立体声版本 |
| AudioLDM（2023） | 潜在扩散 | Mel VAE；CLAP | 通用短音频 |
| AudioLDM 2（2023–2024） | LOA 预测 + 潜在扩散 | AudioMAE/GPT-2 及声学解码 | 语音/音乐/通用音频变体 |
| Stable Audio 2.0（2024） | 潜在扩散 | 商业模型；文本/音频输入 | 最长 3 分钟，44.1 kHz 立体声 |
| Stable Audio Open 1.0（2024） | DiT 潜在扩散 | 连续自编码器；T5 | 最长 47 秒，44.1 kHz 立体声 |
| YuE（2025） | 轨道解耦 AR | 音乐 token；歌词/风格/参考 | 最长五分钟样例 |
| ACE-Step（2025 报告） | 线性 Transformer 扩散 | 音乐 DCAE；歌词/文本 | 报告中最长四分钟 |
| MusicFlow（2024） | 级联流匹配 | 语义与声学特征 | 文本条件、填充、续写 |
| SongCreator（2024） | 双序列 LM | 人声/伴奏流和注意力掩码 | 多种歌曲生成/编辑任务 |
| Suno / Udio | 此处未明确 | 服务提供文本/歌词控制 | 随版本和套餐变化 |

代码可用、权重可用、训练数据可用及许可是不同维度，简单“开源：是/否”会遮蔽这些区别。

## 9. 开放问题与研究实践

- **长篇音乐：** 在完整输出中评价动机发展、重复段落、过渡与结尾，而非只看标称时长。
- **精确控制：** 测量旋律、和声、曲式、配器与表现力的遵循及相互影响。
- **效率：** 区分首段音频延迟、吞吐量及总生成时间。部分系统在合适硬件上快于音频时长，但这不等于交互式流式生成。
- **数据与原创性：** 独立检查训练/测试重叠及近邻复现。授权数据和开放权重不自动证明输出新颖。
- **文化覆盖：** 由合适听者和标注评测支持的语言、律制、乐器及传统。
- **证据：** 区分论文发现、官方产品规格、作者报告的测试与假设，避免未验证的排行榜、会场、硬件和架构断言。

> 相关：[歌声合成](music-singing-synthesis-zh.md)、[评测](music-evaluation-zh.md)、[音乐理解](music-understanding-mir-zh.md)、[音频工程](audio-engineering-zh.md)。
