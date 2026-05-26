# 端侧部署与实时推理：音乐 AI 模型 (2025--2026)

模型优化、硬件平台、延迟预算、部署框架、实际应用案例及各模型算力需求。

> English version: [edge-deployment.md](edge-deployment.md)（本文件为中英双语）

---

## 1. 模型优化技术

### 量化 (Quantization)

**INT8 量化**：最成熟、最广泛使用的音频模型优化技术。权重和激活值用 8 位整数表示，模型体积缩小约 4x（相比 FP32），质量下降通常极小。

**INT4 量化**：更激进，内存从 FP16 减少约 75%。GPTQ 和 AWQ（Activation-aware Weight Quantization）保护重要权重，比简单舍入效果更好。Microsoft 实测表明 AWQ 在 4-bit 精度下通常优于 GPTQ。

**音频模型的关键区分**：
- **基于 Transformer 的音频模型** (MusicGen, 音频-语言模型)：量化方式类似 LLM。INT8 安全；INT4 需要 GPTQ/AWQ 方法
- **基于扩散的音频模型** (Stable Audio)：步数蒸馏比纯权重量化影响更大，因为去噪步数主导推理成本

Intel **OpenVINO** 与 MusicGen 的集成展示了量化音频 Transformer 在端侧硬件上的具体部署流程。

### 剪枝 (Pruning)

**结构化剪枝**（移除整个注意力头、层或通道）比非结构化剪枝提供更好的硬件加速和实时性能。

2025 年 notable：**Slimmable NAM** 引入可在运行时调整大小和计算成本的模型，让用户动态权衡质量和 CPU 效率。

### 知识蒸馏 (Knowledge Distillation)

**Presto! (ICLR 2025)**：通过**步数蒸馏**（减少采样步数）和**层蒸馏**（减少每步 Transformer 层数）双重策略加速文本到音乐生成。直接针对扩散模型的主导成本因素。

---

## 2. 硬件平台

| 平台 | 计算能力 | 适用音乐 AI 任务 | 备注 |
|------|----------|-----------------|------|
| **NVIDIA Jetson Orin** | 最高 275 TOPS (INT8) | 全部任务：源分离、音箱建模、音乐生成 | 通用端侧 AI 平台。`jetson-voice` 支持 ASR/TTS。TensorRT 加速 |
| **Apple Neural Engine (ANE)** | ~15--38 TOPS (M2--M4) | 声音分类、轻量源分离、音箱建模 | Core ML 自动调度到 ANE，但并非所有操作兼容。实时音频 DSP 在 ANE 上有时序不可预测问题。Apple Silicon GPU 优秀：Demucs 在 M 系列 GPU 上 12s 分离 7 分钟歌曲 |
| **Qualcomm Hexagon NPU** | 最高 73 TOPS INT8 (Snapdragon 8 Gen 3+) | 实时源分离、声音分类、轻量生成 | Music.ai 在 SXSW 2025 展示了完全在 Snapdragon NPU 上运行的实时音频分离——无云端依赖 |
| **Raspberry Pi 5** | 四核 ARM Cortex-A76 @ 2.4GHz | 神理音箱建模 (NAM, AIDA-X)、轻量效果 | Zynthian V5 支持 AIDA-X。RTNeural 在 Pi 5 上实现"超低延迟+大量剩余 CPU 周期"。NAM 还可在 Daisy Seed (ARM Cortex-M7) 上运行 |
| **嵌入式 DSP** (SHARC, CMSIS) | 定点/浮点 DSP 操作 | 吉他效果、音箱建模（逐样本） | ANIRA 论文对比实时音频推理架构 |

---

## 3. 延迟预算

| 应用场景 | 最大可接受往返延迟 | 目标 | 备注 |
|----------|-------------------|------|------|
| **吉他效果 / 音箱建模**（实时监听） | 5--10 ms | < 5 ms | 专业音箱建模器 (Kemper, ToneX, Helix) 目标 ~3 ms 或更低 |
| **舞台耳返** | 2--5 ms | < 2 ms | 最严苛延迟要求。48 kHz 下 64 样本 = 1.33 ms |
| **人声处理**（实时） | 5--10 ms | < 5 ms | 修音、混响、压缩 |
| **实时源分离** | 10--20 ms | < 10 ms | 处理缓冲块；通常离线但向实时演进 |
| **交互式生成**（实时循环、伴奏） | 20--50 ms | < 20 ms | 要求较监听宽松 |
| **后期制作 / 离线** | 无实时约束 | N/A | 源分离、轨提取、批量生成 |

**关键阈值**：
- **< 5 ms**：基本不可感知；专业监听目标
- **5--10 ms**：训练过的音乐人可察觉但一般可接受
- **> 10 ms**：越来越有问题；干扰节奏和手感
- **> 20 ms**：明显可闻延迟；实时监听不可接受

---

## 4. 部署框架

| 框架 | 适用场景 | 音频特定说明 |
|------|----------|-------------|
| **RTNeural** | 实时逐样本神经推理（VST/AU 插件） | C++ 引擎 (Jatin Chowdhury, CCRMA/Stanford)。硬实时、无锁设计。驱动 AIDA-X 和 NAM 插件。集成 SuperCollider, Pure Data, Max/MSP |
| **ONNX Runtime** | 跨平台部署 (CPU/GPU/NPU) | 支持 CoreML, TensorRT, CUDA 执行提供者 |
| **TensorRT** | NVIDIA 硬件最大吞吐 | 最适合批量/离线推理。不适合逐样本实时音频 |
| **Core ML** | Apple 生态 (iOS/macOS) | 自动调度到 CPU/GPU/ANE。声音分类支持好。实时音频在 ANE 上有时序不可预测性。转换路径：PyTorch → ONNX → Core ML |
| **TFLite / LiteRT** | 移动/Android, edge TPU | Qualcomm Hexagon NPU 加速通过 QNN delegate。在 Snapdragon NPU 上可达 100x CPU 加速 |
| **OpenVINO** | Intel CPU/GPU/VPU | MusicGen 集成已展示优化推理 |
| **Neutone SDK** | 神理音频 VST/AU 插件开发 | 开源框架 (2025 论文)。Neutone FX 是免费 VST/AU 主插件，将研究模型桥接到音乐人友好的插件 |
| **ANIRA** | 实时音频 NN 推理 | 支持 LibTorch 和 ONNX Runtime 后端 |

**VST/AU 插件开发栈**：标准路径 = **JUCE**（跨平台 C++ 音频插件框架）+ **RTNeural**（神经推理）或 **ONNX Runtime / LibTorch**（大型模型）

---

## 5. 实际部署案例

### 神理音箱建模器

| 产品 | 类型 | 架构 | 部署 |
|------|------|------|------|
| **NAM** | 开源 | WaveNet/LSTM via RTNeural | VST/AU/LV2 插件；Raspberry Pi 5 (Zynthian)；Daisy Seed (ARM Cortex-M7)；MOD Dwarf |
| **AIDA-X** | 开源 | RTNeural | AU/CLAP/LV2/VST 插件；Zynthian (RPi 5)；Android app |
| **IK ToneX** | 商业 | 专有神经捕获 | ToneX Pedal 硬件 + VST/AU 插件。延迟 ~3 ms 或更低 |
| **Neural DSP Quad Cortex** | 商业硬件 | 专有 | 独立硬件单元。Nano Cortex 2025 发布 |
| **Kemper Profiler** | 商业硬件 | Profile 放大器 | 业界标准硬件建模器。延迟 ~3 ms 或更低 |

### 实时源分离

- **Music.ai @ SXSW 2025**：在 Qualcomm Snapdragon NPU 上完全端侧运行实时音频分离——无云端依赖
- **Demucs on Apple Silicon**：M 系列 GPU 上 12 秒分离 7 分钟歌曲
- **DAW 集成**：源分离正直接嵌入 DAW（Logic Pro, Ableton, FL Studio）

---

## 6. 各模型算力需求

| 模型 | 参数量 | 显存 FP32 | 显存 FP16/INT8 | 推理时间（约） | 备注 |
|------|--------|-----------|-----------------|---------------|------|
| **EnCodec** (24kHz) | ~15M | < 1 GB | < 0.5 GB | ~1--5 ms/帧（实时可行） | 音频编解码器；轻量卷积 + Transformer |
| **Demucs** (htdemucs) | ~230M+ | 7--8 GB | 2--4 GB (FP16) | M 系列 GPU 12s/7min 歌曲 | 源分离。4 分钟以上长曲目在 8GB GPU 上可能 OOM |
| **MusicGen Small** | 300M | 4--6 GB | 2--3 GB (FP16) | ~10--30s / 30s 音频 (RTX 3090) | 适配 12GB 消费级 GPU |
| **MusicGen Medium** | 1.5B | 15--16 GB | 8 GB (FP16) | ~30--60s / 30s 音频 | 量化后适配 12GB GPU |
| **MusicGen Large** | 3.3B | 24+ GB | 12--16 GB (FP16) | ~60--120s / 30s 音频 | 需 RTX 3090/4090 或 A100 |
| **Stable Audio Open** | ~1.2B+ | 8--12 GB | 4--6 GB (FP16) | 类似 MusicGen Medium | 潜在扩散模型 |
| **YuE** | 7B | 16+ GB | 8 GB (量化后) | 分钟级/完整歌曲 | 量化后最低 8GB；适配 12GB 消费级 GPU |
| **ACE-Step** | 3.5B | ~12 GB | 6--8 GB (FP16) | 秒级/片段 | 更高效的文本到音乐架构 |
| **NAM (WaveNet)** | 100K--2M | N/A (CPU) | N/A | < 5 ms 往返 | RTNeural 实时运行；文件大小数百 KB 到数 MB |
| **NAM (LSTM)** | ~10K--100K | N/A (CPU) | N/A | < 2 ms 往返 | 文件小至 ~7 KB；可在 Raspberry Pi 和 ARM Cortex-M7 上运行 |

**端侧部署标准路径**：PyTorch → ONNX → INT8/INT4 量化 → TensorRT (Jetson) / Core ML (Apple) / QNN/LiteRT (Qualcomm) / RTNeural (嵌入式/实时)

---

## 7. 关键结论

1. **实时音箱建模在端侧已是解决的问题**：NAM 和 AIDA-X 在 Raspberry Pi 5 甚至 ARM Cortex-M7 上实现 < 5ms 延迟

2. **源分离正在走向端侧**：Music.ai 在 SXSW 2025 展示了 Snapdragon NPU 上完全端侧实时分离

3. **音乐生成仍主要依赖云端/GPU**：MusicGen Small (300M) 在半精度下需 2--4GB 显存。INT4 量化和蒸馏正在缩小差距，但端侧高质量交互生成尚未可行

4. **部署栈正在成熟**：PyTorch → ONNX → TensorRT/Core ML/QNN 是标准流水线。RTNeural 和 Neutone SDK 填补了通用 ML 运行时无法服务的实时音频插件空白

5. **结构化剪枝 + INT8 量化**是音频 Transformer 最实用的组合。对扩散模型，**步数蒸馏**比权重量化收益更大

---

## 参考文献

- Lightweight Transformer Architectures for Edge Devices ([arXiv 2025](https://arxiv.org/html/2601.03290v1))
- INT4 Quantization Guide ([Microsoft](https://medium.com/data-science-at-microsoft/a-practical-guide-to-int4-quantization-for-slms-gptq-vs-awq-olive-and-real-world-results-2f63d6963d1d))
- Presto! Step+Layer Distillation ([ICLR 2025](https://arxiv.org/abs/2410.05167))
- MusicGen ([Hugging Face](https://huggingface.co/facebook/musicgen-large))
- Demucs ([GitHub](https://github.com/facebookresearch/demucs))
- RTNeural ([GitHub](https://github.com/jatinchowdhury18/RTNeural))
- NAM ([neuralampmodeler.com](https://www.neuralampmodeler.com/))
- AIDA-X ([GitHub](https://github.com/AidaDSP/AIDA-X))
- Neutone SDK ([arXiv 2025](https://arxiv.org/html/2508.09126v1))
- Music.ai SXSW 2025 ([music.ai](https://music.ai/blog/press/music-ai-sxsw-2025))
- LiteRT on Qualcomm NPU ([Google Blog](https://developers.googleblog.com/unlocking-peak-performance-on-qualcomm-npu-with-litert/))
- MusicGen + OpenVINO ([OpenVINO](https://docs.openvino.ai/2024/notebooks/music-generation-with-output.html))
- Cloud-Edge Hybrid for Music ([Springer 2026](https://link.springer.com/article/10.1186/s13677-026-00854-0))
- ANIRA ([ResearchGate](https://www.researchgate.net/publication/384712488_ANIRA_An_Architecture_for_Neural_Network_Inference_in_Real-Time_Audio_Applications))
- Slimmable NAM ([arXiv 2025](https://arxiv.org/html/2511.07470v1))
- Audio Latency JND ([ACM](https://dl.acm.org/doi/fullHtml/10.1145/3678299.3678331))