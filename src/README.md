# 代码结构与工具指南

本目录用于存放实验代码、推理脚本和评估工具。当前仅提供结构建议和工具推荐，不包含源代码。

> English version: [README.md](README.md)（本文件为中英双语）

---

## 推荐目录结构

```
src/
├── inference/          # 模型推理脚本
│   ├── encodec/        # EnCodec 编解码
│   ├── musicgen/       # MusicGen 生成
│   ├── stable_audio/   # Stable Audio 生成
│   ├── demucs/         # 源分离
│   └── mert/           # MERT 特征提取
├── evaluation/         # 评估工具
│   ├── fad/            # FAD 计算
│   └── metrics/        # SDR, F1, MOS 等
├── features/           # 特征提取 pipeline
│   ├── spectrogram/    # mel/CQT/VQT 提取与可视化
│   └── tokens/         # 音频离散化（RVQ/LFQ）
├── data/               # 数据预处理
│   ├── download/       # 数据集下载脚本
│   └── preprocess/     # 格式转换、切分、标注
└── notebooks/          # Jupyter 实验笔记
```

---

## 各模型算力需求

### 推理

| 模型 | 任务 | 最低显存 | 推荐显存 | 精度 | 推理框架 | 备注 |
|------|------|----------|----------|------|----------|------|
| **EnCodec** (24kHz) | 编解码 | ~0.5 GB | 1 GB | FP32/FP16 | audiocraft | 流式支持 |
| **DAC** (44.1kHz) | 编解码 | ~1 GB | 2 GB | FP32 | descript-audio-codec | |
| **MusicGen Small** (300M) | 文本到音乐 | ~2 GB | 4 GB | FP16 | audiocraft | 30 秒生成 |
| **MusicGen Medium** (1.5B) | 文本到音乐 | ~4 GB | 8 GB | FP16 | audiocraft | |
| **MusicGen Large** (3.3B) | 文本到音乐 | ~8 GB | 16 GB | FP16 | audiocraft | |
| **Demucs HT** | 源分离 | ~2 GB | 4 GB | FP32 | demucs | 4 轨分离 |
| **BS-RoFormer** | 源分离 (SOTA) | ~4 GB | 8 GB | FP32 | 自定义 | L=12 配置 |
| **MERT** (330M) | 特征提取 | ~1 GB | 2 GB | FP32 | transformers | |
| **Stable Audio Open** | 文本到音乐 | ~6 GB | 12 GB | FP16 | stable-audio-tools | 47 秒生成 |
| **YuE** (7B) | 全曲生成 | ~10 GB | 24 GB | FP16/INT8 | 自定义 LLaMA2 | 5 分钟生成 |
| **DiffSinger** | 歌声合成 | ~2 GB | 4 GB | FP32 | | |

### 训练 / 微调

| 任务 | 最低 GPU | 推荐 GPU | 训练时间（估） | 备注 |
|------|----------|----------|---------------|------|
| EnCodec 微调 | 1x RTX 3090 (24GB) | 4x A100 (40GB) | 1--3 天 | 需大量音频数据 |
| MusicGen LoRA 微调 | 1x A100 (40GB) | 2--4x A100 | 数小时 | audiocraft 支持 |
| 源分离训练 | 1x RTX 3090 | 2--4x A100 | 2--5 天 | MUSDB18-HQ |
| MERT 预训练 | 8x A100 (80GB) | 32x A100 | 1--2 周 | 160K 曲目 |
| 基础模型预训练 | 32x A100 | 128+ GPU | 数周--数月 | 大规模音乐数据 |

---

## 环境配置建议

### 消费级方案（个人研究）

| 配置 | 适用场景 | 预算参考 |
|------|----------|----------|
| RTX 4060 (8GB) + 16GB RAM | EnCodec, Demucs, MERT 推理 | ~6000 元 |
| RTX 4090 (24GB) + 32GB RAM | MusicGen Medium, Stable Audio 推理；小规模微调 | ~18000 元 |
| 2x RTX 4090 (48GB) + 64GB RAM | MusicGen Large, 源分离训练, LoRA 微调 | ~35000 元 |

### 云端方案

| 平台 | GPU | 优势 | 适用场景 |
|------|-----|------|----------|
| **Google Colab Pro+** | A100 (40GB) | 低门槛，按月付费 | 实验、原型 |
| **RunPod** | A100, H100 | 按时计费，社区模板 | 推理、微调 |
| **AutoDL / 矩池云** | A100, RTX 4090 | 国内访问快，价格低 | 日常实验 |
| **Lambda Cloud** | H100 8x | 大规模训练 | 预训练 |

### 推荐软件栈

```
Python 3.10+
PyTorch 2.1+（CUDA 12.x）
├── audiocraft        # MusicGen, EnCodec
├── demucs            # 源分离
├── transformers      # MERT, CLAP
├── stable-audio-tools # Stable Audio
├── torchaudio        # 特征提取
├── librosa           # 音频分析
├── nnAudio           # GPU 可微音频变换
└── fadtk             # FAD 评估
```

---

## 关键工具链接

| 工具 | 用途 | 地址 |
|------|------|------|
| AudioCraft | MusicGen/EnCodec 推理与训练 | github.com/facebookresearch/audiocraft |
| Demucs | 源分离 | github.com/facebookresearch/demucs |
| MERT | 音乐基础模型 | github.com/yizhilll/MERT |
| CLAP | 音频-文本对比学习 | github.com/LAION-AI/CLAP |
| Stable Audio Tools | Stable Audio 训练/推理 | github.com/Stability-AI/stable-audio-tools |
| DAC | Descript 音频编解码器 | github.com/descriptinc/descript-audio-codec |
| WavTokenizer | LFQ 音频分词器 | github.com/novelfm/WavTokenizer |
| FunCodec | 神理编解码器工具包 | github.com/modelscope/FunCodec |
| FADtk | FAD 评估工具 | github.com/microsoft/fadtk |
| museval | 源分离评估 | github.com/sigsep/sigsep-mus-eval |
| Beat This | 节拍追踪 | github.com/CPJKU/beat_this |
| RTNeural | 实时神经推理 | github.com/jatinchowdhury18/RTNeural |
