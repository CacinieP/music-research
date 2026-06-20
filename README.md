# Music Research

[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey?style=flat-square)](LICENSE)
[![Notes](https://img.shields.io/badge/research-notes%20%2B%20surveys-39c5bb?style=flat-square)](docs/)

Music understanding & generation research repository.

This is a curated research workspace for audio engineering, MIR, generative music, music foundation models, edge deployment, and copyright/compliance. It is meant to be useful both as a reading map and as a starting point for reproducible experiments.

## Scope

- **Audio Engineering** — DSP, feature extraction, audio processing pipelines
- **Music Information Retrieval (MIR)** — tag prediction, source separation, transcription, similarity
- **Generative Models** — symbolic music generation, audio synthesis, singing voice synthesis
- **Foundation Models for Music** — large-scale pre-training, multi-modal music understanding
- **Edge Deployment** — model optimization, real-time inference, embedded hardware
- **Copyright & Compliance** — training data rights, generation infringement, legislation
- **Music Theory Fundamentals** — pitch, harmony, rhythm, melody, form — bridging theory to AI practice
- **Music Styles & Genre** — genre taxonomy, style dimensions, style-conditioned generation, cross-cultural style
- **Music Evaluation** — human evaluation protocols, automatic metrics (FAD, CLAP Score, perceptual), evaluation gaps
- **Singing Voice Synthesis** — acoustic modeling, controllable singing, voice conversion, datasets

## Directory Structure

```
docs/
  notes/                — technical research notes (EN + ZH)
    music-understanding-mir.md      — MIR survey
    music-generation.md             — generation survey
    audio-engineering.md            — audio engineering notes
    music-copyright-compliance.md   — copyright & compliance
    edge-deployment.md              — edge deployment & real-time inference
    model-reproduction-guide.md     — reproduction framework
    music-theory-fundamentals.md    — music theory fundamentals
    music-styles.md                  — music styles: genre, aesthetics, style-aware generation
    music-evaluation.md             — evaluation metrics, human protocols, benchmark gaps
    music-singing-synthesis.md      — singing voice synthesis: models, datasets, controllable singing
  surveys/              — literature reviews
    reading-guide.md    — survey reading guide: key AI music surveys annotated
datasets/               — public dataset catalog
src/                    — code structure guide & tool recommendations
references/             — curated paper lists and resources
```

## Research Notes

| Topic | EN | ZH |
|-------|----|----|
| Music Understanding / MIR | [music-understanding-mir.md](docs/notes/music-understanding-mir.md) | [music-understanding-mir-zh.md](docs/notes/music-understanding-mir-zh.md) |
| Music Generation | [music-generation.md](docs/notes/music-generation.md) | [music-generation-zh.md](docs/notes/music-generation-zh.md) |
| Audio Engineering | [audio-engineering.md](docs/notes/audio-engineering.md) | [audio-engineering-zh.md](docs/notes/audio-engineering-zh.md) |
| Copyright & Compliance | [music-copyright-compliance.md](docs/notes/music-copyright-compliance.md) | — (merged bilingual) |
| Edge Deployment | [edge-deployment.md](docs/notes/edge-deployment.md) | — (merged bilingual) |
| Model Reproduction Guide | [model-reproduction-guide.md](docs/notes/model-reproduction-guide.md) | — (merged bilingual) |
| Music Theory Fundamentals | [music-theory-fundamentals.md](docs/notes/music-theory-fundamentals.md) | [music-theory-fundamentals-zh.md](docs/notes/music-theory-fundamentals-zh.md) |
| Music Styles | [music-styles.md](docs/notes/music-styles.md) | [music-styles-zh.md](docs/notes/music-styles-zh.md) |
| Music Evaluation | [music-evaluation.md](docs/notes/music-evaluation.md) | [music-evaluation-zh.md](docs/notes/music-evaluation-zh.md) |
| Singing Voice Synthesis | [music-singing-synthesis.md](docs/notes/music-singing-synthesis.md) | [music-singing-synthesis-zh.md](docs/notes/music-singing-synthesis-zh.md) |

## Other Resources

- [Code Structure & Compute Guide](src/README.md) — directory structure, GPU/VRAM requirements, environment recommendations
- [Public Dataset Catalog](datasets/README.md) — download links, sizes, formats, licensing
- [References & Reading List](references/README.md) — 50+ key papers organized by topic

## License

[Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE)

---

# 音乐研究

[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey?style=flat-square)](LICENSE)
[![Notes](https://img.shields.io/badge/research-notes%20%2B%20surveys-39c5bb?style=flat-square)](docs/)

音乐理解与生成研究仓库。

这是一个精选的研究工作空间，涵盖音频工程、音乐信息检索（MIR）、生成式音乐、音乐基础模型、边缘部署、版权与合规。既可作为阅读导航，也可作为可复现实验的起点。

## 研究范围

- **音频工程** — DSP、特征提取、音频处理管线
- **音乐信息检索（MIR）** — 标签预测、音源分离、转录、相似度
- **生成模型** — 符号音乐生成、音频合成、歌声合成
- **音乐基础模型** — 大规模预训练、多模态音乐理解
- **边缘部署** — 模型优化、实时推理、嵌入式硬件
- **版权与合规** — 训练数据权利、生成侵权、立法动态
- **乐理基础** — 音高、和声、节奏、旋律、曲式 — 乐理到 AI 的桥梁
- **音乐风格与流派** — 流派分类、风格计算维度、风格条件生成、跨文化风格
- **音乐评测** — 人类评测协议、自动指标（FAD、CLAP Score、感知指标）、评测鸿沟
- **歌声合成** — 声学建模、可控歌声生成、歌声转换、数据集

## 目录结构

```
docs/
  notes/                — 技术研究笔记（英文 + 中文）
    music-understanding-mir.md      — MIR 综述
    music-generation.md             — 生成综述
    audio-engineering.md            — 音频工程笔记
    music-copyright-compliance.md   — 版权与合规
    edge-deployment.md              — 边缘部署与实时推理
    model-reproduction-guide.md     — 复现框架
    music-theory-fundamentals.md    — 乐理基础
    music-styles.md                  — 音乐风格
    music-evaluation.md             — 评测方法与协议
    music-singing-synthesis.md      — 歌声合成
  surveys/              — 文献综述
    reading-guide.md    — 综述论文导读：8 篇关键综述的解读与阅读路径
datasets/               — 公开数据集目录
src/                    — 代码结构指南与工具推荐
references/             — 精选论文列表与资源
```

## 研究笔记

| 主题 | 英文 | 中文 |
|------|------|------|
| 音乐理解 / MIR | [music-understanding-mir.md](docs/notes/music-understanding-mir.md) | [music-understanding-mir-zh.md](docs/notes/music-understanding-mir-zh.md) |
| 音乐生成 | [music-generation.md](docs/notes/music-generation.md) | [music-generation-zh.md](docs/notes/music-generation-zh.md) |
| 音频工程 | [audio-engineering.md](docs/notes/audio-engineering.md) | [audio-engineering-zh.md](docs/notes/audio-engineering-zh.md) |
| 版权与合规 | [music-copyright-compliance.md](docs/notes/music-copyright-compliance.md) | —（双语合并版） |
| 边缘部署 | [edge-deployment.md](docs/notes/edge-deployment.md) | —（双语合并版） |
| 模型复现指南 | [model-reproduction-guide.md](docs/notes/model-reproduction-guide.md) | —（双语合并版） |
| 乐理基础 | [music-theory-fundamentals.md](docs/notes/music-theory-fundamentals.md) | [music-theory-fundamentals-zh.md](docs/notes/music-theory-fundamentals-zh.md) |
| 音乐风格 | [music-styles.md](docs/notes/music-styles.md) | [music-styles-zh.md](docs/notes/music-styles-zh.md) |
| 音乐评测 | [music-evaluation.md](docs/notes/music-evaluation.md) | [music-evaluation-zh.md](docs/notes/music-evaluation-zh.md) |
| 歌声合成 | [music-singing-synthesis.md](docs/notes/music-singing-synthesis.md) | [music-singing-synthesis-zh.md](docs/notes/music-singing-synthesis-zh.md) |

## 其他资源

- [代码结构与算力指南](src/README.md) — 目录结构、GPU/VRAM 需求、环境推荐
- [公开数据集目录](datasets/README.md) — 下载链接、大小、格式、许可
- [参考文献与阅读清单](references/README.md) — 按主题整理的 50+ 关键论文

## 许可

[知识共享署名 4.0 国际 (CC BY 4.0)](LICENSE)
