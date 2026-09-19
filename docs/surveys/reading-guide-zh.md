# AI 音乐综述阅读指南：已核对文献

[English](reading-guide.md)

本指南收录可明确识别的综述与教程，于 2026-09-19 对照作者、arXiv、出版方或项目页面核对。除明确注明其他出版版本外，年份表示所链接预印本的首次提交年份。范围摘要描述原文；阅读用途是本仓库的编辑建议。列表为精选，不声称穷尽最新论文。

## 1. 深度学习生成：概念基线

**Jean-Pierre Briot、Gaëtan Hadjeres、François-David Pachet，*Deep Learning Techniques for Music Generation — A Survey*（2017 预印本，后有修订）。** [arXiv:1709.01620](https://arxiv.org/abs/1709.01620)

该综述按目标、表示、架构、挑战与生成策略组织系统。示例有助于区分乐谱、演奏与音频生成，解释为何表示和控制与网络架构同样值得研究。

**阅读用途**：建立比较方法的框架。历史模型不能被称为 2026 年最先进水平；所链接文献并非无法识别的“A Survey on Deep Learning for Music Generation (2023)”。

**相关笔记**：[音乐生成](../notes/music-generation-zh.md)、[乐理](../notes/music-theory-fundamentals-zh.md)、[评测](../notes/music-evaluation-zh.md)。

## 2. 不同表示层级的生成

**Shulei Ji、Jing Luo、Xinyu Yang，*A Comprehensive Survey on Deep Music Generation: Multi-level Representations, Algorithms, Evaluations, and Future Directions*（2020）。** [arXiv:2011.06801](https://arxiv.org/abs/2011.06801)

这篇草稿综述乐谱、演奏和音频生成，并讨论各层级的任务、表示、数据集、方法与评测。

**阅读用途**：将“写音符”“演奏乐谱”“合成声音”区分为不同研究目标。编解码器语言模型、扩散、流匹配与完整歌曲生成应补读后续原始论文。出版时间和具体结论须查原文，不能从综述标题推断。

**相关笔记**：[音乐生成](../notes/music-generation-zh.md)、[音频工程](../notes/audio-engineering-zh.md)。

## 3. 音频 tokenization

**Pooneh Mousavi 等，*Discrete Audio Tokens: More Than a Survey!*（2025）。** [arXiv:2506.10274](https://arxiv.org/abs/2506.10274)

该工作结合 tokenizer 综述与语音、音乐、通用音频基准，按架构、量化、训练、流式能力和用途分类，并评测重建、下游任务与声学语言建模。

**阅读用途**：在重建质量之外比较 tokenizer。将结论用于音乐前，应检查采样率、码率、领域和模型版本。离散编解码 token 是 MusicGen 等系统的重要环节；AudioLDM 使用连续潜在扩散，并非所有音频生成器都需要离散化。

**相关笔记**：[音频工程](../notes/audio-engineering-zh.md)、[音乐生成](../notes/music-generation-zh.md)。

## 4. 音乐基础模型

**Yinghao Ma 等，*Foundation Models for Music: A Survey*（2024）。** [arXiv:2408.14340](https://arxiv.org/abs/2408.14340)

该综述涵盖表示学习、生成与多模态学习，讨论预训练、架构、tokenization、微调、可控性、音乐 agent、数据、评测与伦理，其范围明显超出音乐理解编码器。

**阅读用途**：比较基础模型的角色和训练范式。把后续模型归入综述前，应检查实际版本及模型原始出版物；不能从泛泛的“后续更新”推导出“MERT v2”“MusicFM v2”等名称。

**相关笔记**：[音乐理解](../notes/music-understanding-mir-zh.md)、[生成](../notes/music-generation-zh.md)、[风格](../notes/music-styles-zh.md)。

## 5. 歌声合成

**Yin-Ping Cho、Fu-Rong Yang、Yung-Chuan Chang、Ching-Ting Cheng、Xiao-Han Wang、Yi-Wen Liu，*A Survey on Recent Deep Learning-driven Singing Voice Synthesis Systems*（2021）。** [arXiv:2110.02511](https://arxiv.org/abs/2110.02511)

该综述比较从乐谱与歌词生成歌声的神经系统架构、优势和限制。

**阅读用途**：可明确识别的历史 SVS 入门。它不是 2024 年综述，不能证明当前扩散模型的占比，也不覆盖全部后续系统。可配合 [DiffSinger](https://arxiv.org/abs/2105.02446)及更新模型的原始论文阅读。应区分乐谱歌词驱动 SVS、歌声转换、歌词到歌曲与歌声条件伴奏生成。

**相关笔记**：[歌声合成](../notes/music-singing-synthesis-zh.md)。

## 6. 评测

**Faria Binte Kader、Santu Karmaker，*A Survey on Evaluation Metrics for Music Generation*（2025）。** [arXiv:2509.00051](https://arxiv.org/abs/2509.00051)

该综述为符号与音频音乐评测建立分类，并讨论人工感知、跨文化偏差与协议不一致带来的限制。

**阅读用途**：组织多维评测方案。综述不会使某项指标自动适用于所有任务；定义、参考信号要求与实现细节还需查指标原始论文，并和自动分数一起报告听评协议与不确定性。

**相关笔记**：[音乐评测](../notes/music-evaluation-zh.md)。

## 7. 音源分离：实践教程

**作者维护的 *Open-Source Tools & Data for Music Source Separation* 教程。** [教程入口](https://source-separation.github.io/tutorial/intro/src_sep_101.html)

这是教程，不是已核实的“Music Source Separation: A Brief Overview (2023)”论文。它介绍任务及开源数据与工具生态。

**阅读用途**：理解混合音频／音源定义并开始实践。架构比较和特定数据集结果须查原始模型论文。单个 SDR 数值不能证明通用性能上限，波形模型也不天然优于所有频谱模型。应注明分轨集合、评测实现、聚合方法和训练数据。

**相关笔记**：[音乐理解](../notes/music-understanding-mir-zh.md)、[音频工程](../notes/audio-engineering-zh.md)。

## 8. 风格、数据集与文化覆盖

以下为原始研究与研究项目，不用编造的泛化综述标题替代：

- **Joan Serrà 等（2012），[*Measuring the evolution of contemporary western popular music*](https://arxiv.org/abs/1205.5651)**：音高、音色与响度的语料分析示例；结论应在数据和测量范围内理解。
- **Bob L. Sturm（2013），[*The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use*](https://arxiv.org/abs/1306.1461)**：具体说明数据集缺陷怎样影响流派分类评测。
- **[CompMusic](https://compmusic.upf.edu/)**：强调文化特殊性分析和专门语料的项目，可用于检查表示与标签中的假设。

**相关笔记**：[音乐风格](../notes/music-styles-zh.md)、[音乐理解](../notes/music-understanding-mir-zh.md)。

## 9. 建议阅读路径

| 目标 | 建议顺序 |
|---|---|
| 音乐 AI 入门 | 乐理笔记 → Briot 综述 → Ji 综述 → 一篇原始系统论文 |
| 音频生成 | Ji 综述 → tokenizer 综述 → 生成论文 → 评测综述 |
| MIR／迁移学习 | 基础模型综述 → 模型论文 → 数据集与协议文档 → GTZAN 审计 |
| 歌声合成 | Cho 综述 → DiffSinger → 新任务论文 → 评测笔记 |
| 风格／跨文化 | 风格笔记 → CompMusic → 特定语料的分析与评测 |

以上是编辑性学习路径，不是引用量或完整度排名。

## 10. 引用与维护边界

原指南曾列入缺乏充分识别信息的条目，例如归于 Brée 的“AI and Music: A Comprehensive Survey”以及“McKinney MIR Survey (2009)”。现不将它们保留为已确立参考文献。这不证明不存在相关出版物，而是所提供的标题／作者／年份组合未获核实。

新增文献应记录精确标题、作者、年份／版本、稳定来源与由原文支持的范围摘要。模型论文、产品页、教程、基准与综述应按实际类型标注。本仓库提供双语笔记和关联链接，不能替代原始方法与实验限制。

原始模型论文见[已核对参考列表](../../references/README-zh.md)。主题笔记包括[音频工程](../notes/audio-engineering-zh.md)、[音乐生成](../notes/music-generation-zh.md)、[音乐理解](../notes/music-understanding-mir-zh.md)、[评测](../notes/music-evaluation-zh.md)、[歌声合成](../notes/music-singing-synthesis-zh.md)、[风格](../notes/music-styles-zh.md)与[乐理](../notes/music-theory-fundamentals-zh.md)。
