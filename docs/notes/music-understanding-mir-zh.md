# 音乐理解 / 音乐信息检索（MIR）

> English version: [music-understanding-mir.md](music-understanding-mir.md)

涵盖方法、数据集、评测定义与开放问题。内容于 **2026-09-19** 对照一手论文、数据集发布说明和评测文档核对。公开结果均应结合具体实验条件理解，本文不声称提供完整的最新 SOTA 排名。

---

## 1. 自动标签与分类

### 问题定义

为录音分配流派、配器、情绪或声学属性标签。标签任务通常是多标签分类，固定流派基准也可以是多类单标签分类。标签词表由数据集与任务决定，并不存在通用的 50–100 标签限制。

### 代表架构

- **CNN**：在时频特征上提取局部模式并聚合为片段级预测。卷积和循环网络仍是有用基线，扩大感受野也可以捕获长程上下文。
- **Music Tagging Transformer**（Won、Choi、Serra，ISMIR 2021）：卷积前端后接时间自注意力，使用 noisy-student 半监督训练；作者还提出了按艺术家隔离的 Million Song Dataset 划分。它并非简单的纯图像 patch Transformer。[论文](https://arxiv.org/abs/2111.13457)、[官方实现](https://github.com/minzwon/semi-supervised-music-tagging-transformer)
- **LAION-CLAP**（Wu 等，2022 年预印本 / ICASSP 2023）：音频与文本双编码器通过对比学习对齐，支持检索与提示驱动的零样本分类。LAION-Audio-630K 是通用音频集合，并非全是音乐。[论文](https://arxiv.org/abs/2211.06687)
- **CLaMP 3**（Wu 等，Findings of ACL 2025）：对齐符号乐谱、演奏表示、音频与多语言文本，以文本连接未对齐的模态。不能据此扩展成任意图像理解能力。[论文](https://aclanthology.org/2025.findings-acl.133/)

### 数据集与协议

| 数据集 | 范围 | 评测注意事项 |
|---|---|---|
| MagnaTagATune | 约 2.59 万个音乐短片段；常用前 50 标签子集 | 原始标签词表更大，需明确过滤和划分 |
| MTG-Jamendo | 清洗后的基础集合为 55,609 曲目、195 标签 | 公开划分保留 55,525 曲目、183 标签；流派/乐器/情绪和 top-50 子集不同 |
| GTZAN | 1,000 个片段、十种流派标签 | 重复片段、艺术家重叠和标签问题影响评测 |
| FMA | 完整发布包含 106,574 曲目 | small/medium/large/full 的时长与标签不同，不能混用 |
| NSynth | 305,979 个独立音符、11 个乐器家族 | 乐器家族、具体乐器身份和声源类型分类是不同任务 |

来源：[MagnaTagATune](https://mirg.city.ac.uk/datasets/magnatagatune/index1.html)、[MTG-Jamendo 发布说明](https://github.com/MTG/mtg-jamendo-dataset)、[FMA 发布说明](https://github.com/mdeff/fma)、[NSynth 发布说明](https://magenta.tensorflow.org/datasets/nsynth)。

报告 ROC-AUC 和 PR-AUC/average precision 时应明确平均方式。正标签稀少时，只看 ROC-AUC 可能掩盖检索精确率不足。比较系统前应固定标签子集、艺术家划分、片段长度和预训练数据规则。单个很高的 GTZAN 准确率不足以证明流派识别已经解决。[GTZAN 数据审计](https://arxiv.org/abs/1306.1461)

### 开放问题

噪声或缺失标签、文化特定的流派分类、长录音、稀有乐器与分布偏移。语言提示敏感性、未见流派表现应与监督标签任务分别评估。

---

## 2. 音乐转录

### 问题定义

把音频转换为符号事件，包括音高、起音、止音，以及可选的力度、乐器和踏板事件。音高轮廓、钢琴卷帘和完整 MIDI 转录是不同输出。

### 钢琴与多乐器模型

| 模型 | 核心思路 | 来源 |
|---|---|---|
| Onsets and Frames（Hawthorne 等，ISMIR 2018） | 联合预测起音与帧级活动，起音检测限制音符何时开始 | [论文](https://arxiv.org/abs/1710.11153) |
| 高分辨率钢琴转录（Kong 等，2020 年预印本 / TASLP 2021） | 回归起音/止音时间，解码精细音符与踏板事件 | [论文](https://arxiv.org/abs/2010.01815) |
| MT3（Gardner 等，ICLR 2022） | T5 风格编码器/解码器将频谱片段映射为事件 token 序列，跨转录数据集联合训练 | [论文](https://arxiv.org/abs/2111.03017)、[代码](https://github.com/magenta/mt3) |

MT3 使用不同 token 类型表示时间、音高、乐器和音符状态，并不是一个 token 打包音符的全部属性。其多乐器配置不会仅因事件词表有力度相关状态，就能恢复完整的表现性力度。

**2025 AMT Challenge** 报告被 **NeurIPS 2025 的 AI for Music Workshop** 接收，2026 年 3 月上传 arXiv。报告共有八支有效参赛队，其中两支超过 MT3 基线，复调和音色变化仍有困难。这不是 NeurIPS 主会基准论文。[挑战报告](https://arxiv.org/abs/2603.27528)

鼓转录预测击打时刻和鼓件类别。ENST-Drums 等打击乐数据集的标签映射与录音条件不同，比较时需统一鼓件词表和起音容差。

### 代表数据集

| 数据集 | 内容 | 重要区别 |
|---|---|---|
| MAPS | 带对齐符号标签的钢琴录音与合成钢琴 | 按录音条件与音乐内容划分 |
| MAESTRO | 约 200 小时钢琴演奏，MIDI 紧密对齐 | 版本不同；v3 从 v2 中移除了六段含弦乐伴奏的录音 |
| MusicNet | 330 个古典合奏录音 | 乐器/音符标签存在对齐不确定性 |
| Slakh2100 | 2,100 个合成多轨混音 | 合成音频不等于真实合奏录音 |
| URMP | 44 段小型合奏表演 | 分离及合成演奏，附乐器标注 |

来源：[MAESTRO 发布说明](https://magenta.tensorflow.org/datasets/maestro)、[MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html)、[Slakh](http://www.slakh.com/)、[URMP](https://labsites.rochester.edu/air/projects/URMP.html)。

### 评测：区分 F1 定义

- **帧级 F1**：比较每帧活动音高。
- **不含止音的音符 F1**：匹配音高与起音。
- **含止音的音符 F1**：进一步要求结束时间符合容差。
- **力度/乐器感知指标**：增加匹配条件，应明确标明。

`mir_eval.transcription` 的默认音符匹配使用 50 ms 起音容差、50 音分音高容差。启用止音评分时，止音容差为 `max(50 ms, 参考音符时长的 20%)`。设置 `offset_ratio=None` 可忽略止音。在相同预测和匹配协议下，增加止音条件不可能提高 F1。[评测文档](https://mir-eval.readthedocs.io/latest/api/transcription.html)

作为**历史实例**，Kong 等报告其 MAESTRO 评测中 **起音 F1 为 96.72%**。这不是含止音音符 F1 96.72%，也不是当前所有钢琴转录系统的通用分数。数据集版本和比较协议见[原论文](https://arxiv.org/abs/2010.01815)。

### 开放问题

真实合奏混音、重叠谐波、歌声、表现性力度与奏法、踏板、标签对齐和跨录音条件泛化。钢琴基准高分不代表同等的多乐器性能。

---

## 3. 音源分离

### 问题定义

从混合音频估计声源波形。MUSDB18 标准四轨为人声、鼓、贝斯和 **other（其他）**。other 可以包含多种乐器，四轨分离不等于完整的单乐器隔离。

### 代表模型

- **Spleeter**（Hennequin、Khlif、Voituret、Moussallam；JOSS 2020，软件于 2019 年发布）：在 **线性频率 STFT 幅度** 上做 U-Net 风格分离，预测掩码并使用混合音频相位重建波形。它不以梅尔频谱图为输入。吞吐量取决于硬件与配置。[论文](https://joss.theoj.org/papers/10.21105/joss.02154)、[代码](https://github.com/deezer/spleeter)
- **Demucs / Hybrid Demucs**：波形编码器/解码器，后续结合频谱分支。[官方仓库](https://github.com/facebookresearch/demucs)
- **HT Demucs**（Rouard、Massa、Défossez；ICASSP 2023）：时域和频谱分支通过自注意力、交叉注意力交互。[论文](https://arxiv.org/abs/2211.08553)
- **BSRNN**（Luo、Yu；2022 年预印本 / TASLP 2023）：频谱划分为子带，并交替进行序列级和频带级循环建模，不只是各频带独立 RNN 后最终拼接。[论文](https://arxiv.org/abs/2209.15174)
- **BS-RoFormer**（Lu、Wang、Kong、Hung；2023 年预印本）：频带划分、层级注意力、旋转位置编码与复数掩码估计。论文报告的是 **SDX23** 音乐分离赛道获胜，不能写成原始 BSRNN 论文赢得 URGENT 2025。[论文](https://arxiv.org/abs/2309.02612)

### 数据集

| 数据集 | 发布信息 | 用途 |
|---|---|---|
| MUSDB18 | 150 首：**100 训练、50 测试**；立体声、44.1 kHz | 四轨基准，原版为有损压缩发布 |
| MUSDB18-HQ | 相同曲目与划分，未压缩 WAV 声源 | 不应混淆 HQ 训练/评测和压缩版预处理 |
| MoisesDB | 多轨录音，乐器层级更细 | 比固定四轨更精细的声源定义 |
| Slakh2100 | 合成多轨混音 | 可控混音与联合符号任务 |

训练集可继续划分训练/验证；验证曲目不能代替独立的 50 首测试曲目。[MUSDB18 官方说明](https://sigsep.github.io/datasets/musdb.html)、[MoisesDB](https://github.com/moises-ai/moises-db)

### 指标与公开实例

BSS Eval 分解估计误差，计算 SDR（信号失真比）、SIR（信号干扰比）、SAR（信号伪影比），对声源声像还可计算 ISR（声像空间失真比）。SI-SDR、整曲 SDR、逐窗 BSS Eval SDR 是不同指标。需明确实现/版本、允许的滤波、窗口、静音参考处理、跨曲目聚合和跨声源平均方式。[museval](https://github.com/sigsep/sigsep-mus-eval)

| 公开系统 | 报告结果 | 条件 |
|---|---|---|
| HT Demucs | 9.20 dB SDR | 论文中的稀疏注意力、按声源微调配置，使用额外 800 首训练音乐 |
| 较小的 BS-RoFormer | 平均 SDR 9.80 dB | 论文中的 MUSDB18-HQ 基准，不使用额外训练数据 |

这些是文献实例，不是相同数据条件下的排名。跨曲目均值、中位数和跨声源平均值不能互换，比较必须保留原文的聚合协议。[HT Demucs 论文](https://arxiv.org/abs/2211.08553)、[BS-RoFormer 论文](https://arxiv.org/abs/2309.02612)

### 开放问题

分离相似乐器、处理未见编曲、保留立体声相位与瞬态、减少伪影和满足流式延迟约束。SDR 不能完整描述感知质量；最难分离的声源取决于模型和数据，并不存在“贝斯总最难”的规则。“Open-unmix”是具体基线项目名，不是任意声源分离场景的术语。

---

## 4. 音乐情感识别（MER）

### 任务与方法

区分**从音乐中感知的情绪**和**音乐在听众身上诱发的情绪**。目标可以是离散类别，也可以是效价（愉悦程度）与唤醒度（激活程度）的连续数值，按整曲或随时间标注。听众背景与标注指令决定模型学习的对象。

CNN、循环网络、Transformer 和预训练嵌入均可用于分类或回归。歌词和元数据可能增加信息，但多模态收益必须在相同划分上比较，不能预先保证。分类准确率与连续维度相关系数衡量不同任务，不能直接排序。

### 代表数据集

| 数据集 | 内容 | 标注范围 |
|---|---|---|
| DEAM | 1,802 个片段/完整歌曲 | 静态和连续效价/唤醒度 |
| PMEmo | 794 首歌曲，含选定副歌片段 | 静态/动态效价-唤醒度和 **皮电活动（EDA）**，并非通用 EEG/ECG 数据集 |
| Emotify | 四种流派、400 个一分钟片段 | 九种 GEMS 情感类别，标注诱发情绪 |

来源：[DEAM — Aljanaki、Yang、Soleymani，PLOS ONE 2017](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0173392)、[PMEmo 官方介绍](https://www.next.zju.edu.cn/cn/archive/pmemo/)、[Emotify 官方标注](https://www2.projects.science.uu.nl/memotion/emotifydata/)。PMEmo 数据集论文发表于 **ICMR 2018**，不是 ICASSP 2018。

### 评测与限制

连续任务可报告 MSE/MAE、Pearson 相关系数或一致性相关系数；分类任务使用考虑类别分布的指标。需统一标签尺度、时间对齐、标注者聚合和划分规则。不要把同一歌曲的相邻片段分散到训练集与测试集。

标注分歧可能反映真实体验差异，并非全是错误。效价-唤醒度并不穷尽音乐情绪，GEMS 提供音乐特定的另一框架。Hevner 的历史情绪形容词圆环不能写成“13 种情绪模型”。本文不声称存在跨数据集通用的相关系数范围。

---

## 5. 音乐基础模型

### 模型类别

| 模型 | 核实后的方法/范围 | 来源 |
|---|---|---|
| MERT | Li 等，ICLR **2024**；RVQ-VAE 声学教师和 CQT 音乐教师监督掩码预训练；原始版本为 95M/330M | [论文](https://arxiv.org/abs/2306.00107) |
| MusicFM | Won、Hung、Le；*A Foundation Model for Music Informatics*，2023 年预印本；音乐音频自监督表示学习 | [论文](https://arxiv.org/abs/2311.03318) |
| JukeMIR | Castellon、Donahue、Liang，ISMIR 2021；研究从 Jukebox 提取的表示 | [Codified Audio Language Modeling Learns Useful Representations for MIR](https://arxiv.org/abs/2107.05677) |
| CLAP | 音频/文本对比对齐；训练并非仅含音乐 | [论文](https://arxiv.org/abs/2211.06687) |
| CLaMP 3 | 跨模态、跨语言的音乐共享表示，以文本连接模态 | [论文](https://aclanthology.org/2025.findings-acl.133/) |

纯音频预训练本身不能实现文本驱动的零样本分类。MERT 所选教师并非简单的 HuBERT 蒸馏；未指定多模态扩展时，不能把 MusicFM 写成“音频加可选文本”。

### 评测模式

- **线性探测**：冻结编码器，训练线性输出头。
- **冻结特征的下游模型**：训练可能非线性的任务头，不一定是线性探测。
- **微调**：更新部分或全部预训练参数，并明确数据/算力预算。
- **零样本**：不做任务训练，使用指定判定规则、提示与候选标签；提示选择过程仍可能泄漏测试信息。

**MARBLE** 是 Yuan 等的 *Music Audio Representation Benchmark for Universal Evaluation*，发表于 **NeurIPS 2023**。会议版本包含 **12 个数据集、18 个任务**及多种评测设置。早期预印本的数量不同，应明确版本。它同时覆盖序列任务和片段级理解。[会议论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/7cbeec46f979618beafb4f46d8f39f36-Paper-Datasets_and_Benchmarks.pdf)

### 开放问题

预训练与评测录音重叠、领域迁移、提示/语言覆盖、计算成本和时间分辨率。帧率和强项随检查点及特征层变化。某基准上的调性检测弱项不能证明所有基础模型都不能表示调性；小数据微调也可能过拟合，不保证胜过冻结特征。

---

## 6. 节拍/速度追踪与和弦/调性识别

### 节拍、小节首拍与速度

节拍追踪预测脉冲时刻，小节首拍（downbeat）追踪识别每小节第一拍，速度估计预测节奏速率。它们相关但不同：全局 BPM 正确不代表节拍相位正确，也不代表能追踪变速。

经典起音强度/动态规划方法与 CNN、TCN、RNN、Transformer 激活模型并存，后者常接动态贝叶斯网络（DBN）。**Beat This!**（Foscarin、Schlüter、Widmer；ISMIR 2024）结合卷积和 Transformer，不使用 DBN 后处理；论文在部分比较中 F1 更高但连续性指标较弱，说明一个分数不足以描述表现。[论文](https://arxiv.org/abs/2407.21658)、[代码与数据配置](https://github.com/CPJKU/beat_this)

常用评测集合有 Ballroom、Hainsworth、SMC、GTZAN 节奏标注、Beatles/Isophonics 和 GiantSteps tempo。需核实具体标注版本与划分，不能假设所有录音都具备相同节奏标注。

`mir_eval` 节拍 F1 默认容差为 **70 ms**。连续性指标考察更长的正确节拍序列；节拍层级指标可能允许半速/倍速。速度评测应说明是否接受八度速度误差，以及相对容差。[节拍评测](https://mir-eval.readthedocs.io/latest/api/beat.html)

### 自动和弦估计

预测和弦标签和时间区间，例如 `C:maj`、`G:min7` 或无和弦。色度/NNLS 特征加时间解码构成传统基线，神经方法学习频谱特征和和弦序列上下文。大三/小三和弦描述和弦性质，不是全曲的大调/小调调性。

**McGill Billboard** 作者描述的发布包含 **890 个榜单位置、740 首不同歌曲**的标注，而非约 200 首。构造划分时应明确处理重复榜单条目。Isophonics 也是常用标注集合。[McGill Billboard 项目](https://ddmal.ca/research/The_McGill_Billboard_Project_%28Chord_Analysis_Dataset%29/)

需报告和弦词表简化方式、按时长加权的重叠分数、转位与无和弦片段的处理，以及划分。`majmin`、`triads`、`tetrads` 和仅根音评分衡量不同条件；没有词表和协议的“80% 和弦准确率”信息不足。[和弦评测](https://mir-eval.readthedocs.io/latest/api/chord.html)

### 调性检测

估计全局或局部主音与调式。调性轮廓匹配和学习分类器是不同路线。GiantSteps Key 是已知数据集，`meters.tsv` 这样的文件名本身不足以构成可核实的调性基准。

区分精确准确率与 MIREX 风格加权分数：后者对五度关系、关系大小调和平行大小调等关联调性给部分分。无调性、转调、调律变化与非西方调式体系可能不符合固定 24 调任务的假设。[调性评测](https://mir-eval.readthedocs.io/latest/api/key.html)

### 开放问题

弹性速度、不规则拍号、拍号变化、模糊和声、扩展/转位和弦与合适的调性体系。应直接评测这些场景，而非从固定拍号流行音乐基准外推。

---

## 7. 音乐推荐

### 方法

基于内容的检索使用手工特征或学习的音频嵌入；协同过滤从交互历史学习；混合推荐结合内容、交互和上下文。音频相似性可帮助新曲目冷启动，但不等价于听众偏好。

余弦相似度是有用的嵌入基线。FAISS、ScaNN 等近似最近邻系统可提升大规模检索效率，但索引和相似度函数需针对实际嵌入/任务验证。元数据和曲库使用权是与音频分析分开的要求。

不能假设新应用普遍可访问 Spotify 的 audio-features、audio-analysis 或 recommendations 端点。Spotify 于 2024 年 11 月宣布对新的 Web API 使用场景施加限制，当前权限需按实际应用核实。[官方变更公告](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api)

### 评测

| 指标 | 含义 |
|---|---|
| Precision@K | 前 K 个返回项目中相关项目的比例 |
| Recall@K | 所有相关项目中被前 K 个结果检索到的比例 |
| NDCG@K | 按理想排序归一化的折损相关性增益 |
| 覆盖率 | 指定策略下得到曝光/推荐的曲库比例 |
| 惊喜度 | 相对预期基线，有用且意外的发现；不只是新颖性 |

使用按时间划分，记录候选集合与负采样规则，并区分离线排序和线上参与度。流行度偏差、反馈循环、新听众冷启动、长尾覆盖与文化/上下文差异仍需关注。

---

## 8. 翻唱检测与版本识别

### 任务与方法

在调性、速度、乐器或编曲变化下，识别同一底层作品的不同演奏版本。这与识别重复或近乎相同的录音不同。

**Chromaprint/AcoustID** 面向 **近乎相同的音频识别**，不是任意移调、变速下的通用翻唱识别。版本识别可以使用色度/和声序列及显式对齐、移调处理，也可以通过同作品/不同作品监督学习嵌入。孪生网络或 triplet 训练可以学习有用的不变性，但不保证自动获得。[Chromaprint 声明的范围](https://github.com/acoustid/chromaprint)

### 数据集与评测

- **Covers80**：80 对、160 段录音，是规模较小的历史基准。[数据集](https://labrosa.ee.columbia.edu/projects/coversongs/covers80/)
- **Da-TACOS**：15,000 曲目的基准子集，以及独立的 10,000 曲目翻唱分析子集。发布内容为特征与元数据，**不包含音频文件**。[官方发布](https://github.com/MTG/da-tacos)
- **SecondHandSongs**：作品/表演版本元数据，可支持标签构建；作品数、表演版本数与可下载音频是不同概念。

报告 mAP、MRR 或 recall/hit rate@K 时，应明确查询/候选库划分、相关版本和自身匹配排除规则。测试对未见作品的泛化时，应按作品划分。高相似度只能提供候选匹配，不能判定所有权或授权状态。

结构变化、串烧、现场版本和大幅重编仍然困难。哼唱检索是相关任务，但输入分布不同。

---

## 9. 主旋律提取

### 任务与方法

估计主旋律随时间变化的基频，并判断各时刻是否存在旋律。它不会直接输出分离的人声波形或卡拉 OK 伴奏，后两者属于音源分离。

- **MELODIA / Salamon 与 Gómez（2012）**：通过谐波显著性与音高轮廓选择提取复调旋律。[作者方法与资源](https://www.justinsalamon.com/melody-extraction.html)
- **神经显著性或序列模型**：从时频特征预测音高/旋律存在性，可结合时间解码或音源分离。
- **pYIN（Mauch 与 Dixon，ICASSP 2014）**：面向单音输入的概率基频估计，本身不是完整的复调主旋律提取器。[论文](https://webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2014/MauchDixon-PYIN-ICASSP2014.pdf)
- **CREPE（Kim 等，ICASSP 2018）**：神经单音音高追踪器；在分离轨上使用它不能消除音源分离误差。[官方实现](https://github.com/marl/crepe)

[SALAMI](https://ddmal.ca/research/salami/annotation/) 提供音乐结构标注，并非音源分离或旋律提取算法。

### 评测

| 指标 | 二值旋律存在性标注下的定义 |
|---|---|
| 原始音高准确率（RPA） | 参考旋律存在帧中，估计音高落入容差内的比例，常用 50 音分 |
| 原始色度准确率（RCA） | 允许八度等价后的音高准确率 |
| 整体准确率（OA） | 旋律存在且音高正确的帧，加上正确判断无旋律的帧，再除以全部帧 |
| 有声召回率（VR） | 正确检测的有声帧数，除以参考有声帧数 |
| 有声误报率（VFA） | 被误判为有声的帧数，除以参考无声帧数 |

OA 不是“RPA 加一个有声分数”，分母和正确条件均不同。需说明音高容差、时间网格对齐、有声约定和数据集。该评测术语中的“有声”也适用于乐器，表示旋律存在，不一定是人声。[旋律评测文档](https://mir-eval.readthedocs.io/latest/api/melody.html)

### 开放问题

密集混音中的弱旋律、多条同时存在的旋律线、八度错误、器乐音色多样性、表现性装饰音和实时约束。不存在一个单一“MIREX 数据集”的 RPA 能概括所有这些场景。

---

## 10. 跨领域 MIR 挑战

1. **代表性**：常用数据集只覆盖有限的音乐传统、乐器、录音条件与听众。应分领域测量迁移能力，而非声称每个模型对所有非西方音乐都会失败。
2. **数据泄漏**：按任务需要隔离艺术家、作品、录音和衍生片段。检查与预训练数据的重叠，模型选择过程不使用测试集。
3. **可比较的评测**：公开数据版本、划分、指标实现、阈值、聚合方式与额外数据。相似的百分比或分贝值可能对应不同任务。
4. **部署**：区分批处理吞吐量、流式支持与实测端到端延迟，测试真实编解码格式、麦克风、噪声和录音长度。
5. **标注与不确定性**：保留有意义的分歧。模型输出是估计，尤其在流派、情绪、和声及模糊旋律任务中。

## 11. MIR 任务全景

| 任务 | 输出 | 评测重点 | 主要挑战 |
|---|---|---|---|
| 自动标签 | 标签/分数 | 分标签 ROC/PR 指标及划分 | 稀有标签与领域迁移 |
| 转录 | 符号事件 | 区分起音、止音、帧、乐器、力度条件 | 真实多乐器混音 |
| 音源分离 | 声源波形 | 明确 SDR 协议并结合听音测试 | 相似声源与伪影 |
| 情绪识别 | 标签或轨迹 | 标注目标及听众/数据划分 | 主观性与上下文 |
| 节拍追踪 | 节拍/小节首拍时刻 | F1、连续性与节拍层级约定 | 弹性速度与不规则拍号 |
| 和弦识别 | 带标签时间区间 | 词表与时长权重 | 模糊/扩展和声 |
| 调性检测 | 主音/调式 | 精确分数与加权分数的区别 | 转调与调性体系假设 |
| 推荐 | 排序项目 | 候选集、相关性、用户/时间划分 | 冷启动与流行度偏差 |
| 版本识别 | 匹配录音排序 | 按作品隔离的检索协议 | 编曲与结构变化 |
| 旋律提取 | 基频/旋律存在性轮廓 | 分别报告音高与有声指标 | 密集或多条旋律 |

相关笔记：[音乐生成](music-generation-zh.md)、[音频工程](audio-engineering-zh.md)、[音乐理论基础](music-theory-fundamentals-zh.md)、[音乐风格](music-styles-zh.md)。
