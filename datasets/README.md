# 公开数据集目录

音乐 AI 研究中常用的公开数据集，按任务分类整理。包含规模、格式、许可证和下载方式。

> English version: [README.md](README.md)（本文件为中英双语）

---

## 自动标注 / 分类

| 数据集 | 规模 | 标注 | 格式 | 许可证 | 下载 |
|--------|------|------|------|--------|------|
| **MagnaTagATune** | ~25K 片段 (30s) | 50 标签 | MP3 | 非商业 | [mtg.upf.edu](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) |
| **MTG-Jamendo** | ~55K 曲目 | 700+ 标签 | MP3 + JSON | CC BY | [mtg.upf.edu](https://mtg.github.io/mtg-jamendo-dataset/) |
| **GTZAN** | 1,000 曲目 | 10 流派 | WAV | 非商业 | [kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) |
| **FMA** | ~106K 曲目 | 流派层级 | MP3 | CC BY / 混合 | [github.com/mdeff/fma](https://github.com/mdeff/fma) |
| **NSynth** | ~305K 单音 | 乐器/音高/力度 | WAV | CC BY 4.0 | [magenta.tensorflow.org](https://magenta.tensorflow.org/datasets/nsynth) |
| **AudioSet** | ~2M 片段 (10s) | 527 事件类 | YouTube 链接 | CC BY 4.0 | [research.google.com/audioset](https://research.google.com/audioset/) |

## 音乐转录

| 数据集 | 规模 | 内容 | 格式 | 许可证 | 下载 |
|--------|------|------|------|--------|------|
| **MAESTRO** | ~200h / 1282 场 | 钢琴 + 对齐 MIDI | WAV + MIDI | CC BY-NC-SA 4.0 | [magenta.tensorflow.org](https://magenta.tensorflow.org/datasets/maestro) |
| **MAPS** | ~240 首 | 钢琴（合成+录制）+ MIDI | WAV + MIDI | 研究 | [tik.ee.ethz.ch](https://www.tik.ee.ethz.ch/~kbal/disclaimer.php?dataset=maps) |
| **MusicNet** | 330 录音 | 多乐器合奏 | WAV + MIDI | CC0 | [homes.cs.washington.edu](https://homes.cs.washington.edu/~thickstn/musicnet.html) |
| **Slakh2100** | 2,100 混合 | 多轨 MIDI 合成 | WAV + MIDI | CC BY 4.0 | [github.com/sigsep/sigsep-mus-db](https://github.com/slakh/slakh2100) |
| **POP909** | 909 首 | 流行钢琴 + 旋律/伴奏 | WAV + MIDI | CC BY 4.0 | [github.com/music-x-lab/POP909-Dataset](https://github.com/music-x-lab/POP909-Dataset) |

## 源分离

| 数据集 | 规模 | 音轨 | 格式 | 许可证 | 下载 |
|--------|------|------|------|--------|------|
| **MUSDB18-HQ** | 150 曲目 | 4 轨 (人声/鼓/贝斯/其他) | WAV 44.1kHz | CC BY-NC 4.0 | [zenodo.org](https://zenodo.org/records/3338373) |
| **MoisesDB** | 240 曲目 | 可变轨数 | WAV | CC BY-NC 4.0 | [github.com/moises_ai/moisesdb](https://github.com/moises_ai/moisesdb) |
| **MedleyDB** | 122 曲目 | 多轨 | WAV | CC BY 4.0 | [medleydb.weebly.com](https://medleydb.weebly.com) |
| **URMP** | 44 演奏 | 多乐器合奏 | WAV + MIDI | 研究 | [github.com/jhuang448/URMP](https://github.com/jhuang448/URMP) |

## 音乐生成

| 数据集 | 规模 | 内容 | 格式 | 许可证 | 下载 |
|--------|------|------|------|--------|------|
| **MusicCaps** | 5,521 片段 | 10s 音乐 + 文字描述 | WAV + JSON | CC BY 4.0 | [huggingface.co](https://huggingface.co/datasets/google/MusicCaps) |
| **Lakh MIDI** | 180K 文件 | 多乐器 MIDI | MIDI | 研究 | [colinraffel.com](https://colinraffel.com/projects/lmd/) |
| **MetaMIDI** | 436K 文件 | 大规模 MIDI | MIDI | CC0 | [github.com/AI-Guru/MetaMIDI](https://github.com/AI-Guru/MetaMIDI) |
| **AD Pianoforte** | ~100h | 古典钢琴 | MIDI | 研究 | 联系作者 |
| **FMA** | ~106K 曲目 | 完整曲目 | MP3 | CC BY / 混合 | [github.com/mdeff/fma](https://github.com/mdeff/fma) |

## 情感识别

| 数据集 | 规模 | 标注 | 格式 | 许可证 | 下载 |
|--------|------|------|------|--------|------|
| **DEAM** | ~1,800 曲目 | 逐秒 V-A 值 | MP3 + CSV | CC BY-SA | [cvml.unige.ch/databases/DEAM](http://cvml.unige.ch/databases/DEAM/) |
| **PMEmo** | ~794 曲目 | 静态+动态 V-A | MP3 + CSV | 研究 | [dl.acm.org](https://dl.acm.org/doi/10.1145/3240508.3240520) |
| **Emotify** | 400 曲目 | 8 类情感 | MP3 | 研究 | [github.com/ashokfernandes/Emotify-Dataset](https://github.com/ashokfernandes/Emotify-Dataset) |

## 节拍 / 和弦 / 调性

| 数据集 | 规模 | 标注 | 格式 | 下载 |
|--------|------|------|------|------|
| **Ballroom** | 698 片段 | 节拍 + 速度 | WAV | [mtg.upf.edu](http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html) |
| **Billboard** | ~200 曲目 | 和弦 + 节拍 | WAV + lab | [ddmal.music.mcgill.ca](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)) |
| **Isophonics** | Beatles/Queen 专辑 | 和弦 + 节拍 + 段落 | WAV + lab | [isophonics.net](http://isophonics.net/content/reference-annotations) |
| **GiantSteps Key** | 660 曲目 | 调性标注 | WAV + CSV | [github.com/GiantSteps/giantsteps-key-dataset](https://github.com/GiantSteps/giantsteps-key-dataset) |

## 基础模型预训练

| 数据集 | 规模 | 内容 | 格式 | 备注 |
|--------|------|------|------|------|
| **AudioSet** | ~2M 片段 | 通用音频事件 | YouTube 链接 | MERT/CLAP 预训练 |
| **Music4All** | ~100K 曲目 | 元数据丰富 | MP3 | 需申请 |
| **Free Music Archive** | ~106K 曲目 | 多流派 | MP3 | CC 协议，可商用 |
| **Freesound** | ~500K 片段 | 通用音频片段 | WAV | Stable Audio Open 训练数据 |

---

## 下载建议

- **国内用户**：HuggingFace 使用 `hf-mirror.com` 镜像；Zenodo 通常可直连
- **大规模数据**：优先使用 `huggingface_datasets` 或官方脚本，避免手动下载
- **存储预算**：MAESTRO ~120GB；MUSDB18-HQ ~15GB；完整 AudioSet ~1.5TB
- **预处理**：大多数数据集提供预处理脚本；注意采样率统一（通常重采样到 16/22.05/44.1 kHz）
