# 代码与算力指南

本目录包含可在 CPU 上验证的 [audio_baselines.py](audio_baselines.py)：音级熵、调性与和弦模板、对齐 SNR、音频预处理。模型调用示例见 [Cookbook](../docs/cookbook/recipes.md)。核对日期：2026-09-19。

## 资源估算

仅权重的存储量为参数个数 × 每参数位数 / 8。1B 参数的 FP16 权重约 2 GB；这不是模型运行的峰值显存。激活、KV cache、文本编码器、音频编解码器、算子工作区、batch 和音频时长都影响资源需求。

| 模型/任务 | 资源判断依据 |
|---|---|
| MusicGen | small/medium/large 分别为 300M/1.5B/3.3B；官方 AudioCraft 指南对 medium 给出至少 16 GB GPU 内存的推理要求 |
| MERT-v1-330M | 330M 参数，24 kHz 音频；长音频应制定分块与聚合策略。仅 FP32 权重已约 1.32 GB，原稿“1 GB FP32 最低显存”不成立 |
| Demucs | checkpoint、segment、overlap、shifts、精度与设备共同决定内存/耗时；不以整首歌时长直接断言 OOM |
| Stable Audio / YuE | 明确版本和官方实现；不同代模型、offloading 和量化配置不能合并成一个最低显存数字 |
| 训练/微调 | 实测 micro-batch、优化器状态、精度、梯度检查点、累积步数和吞吐，不能从推理显存推断训练预算 |

来源：[MusicGen 指南](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)、[MERT 模型卡](https://huggingface.co/m-a-p/MERT-v1-330M)、[Demucs](https://github.com/facebookresearch/demucs)。

两张 24 GB GPU 不会自动成为一张 48 GB GPU：数据并行通常各卡复制模型；模型并行或分片需要实现与通信支持。云平台套餐也不保证每次分配同一型号。原稿无实测支撑的购机报价、训练天数与型号“最低配置”已删除。

## 环境与复现

- CPU 示例采用 [requirements-examples.txt](../requirements-examples.txt)，从仓库根目录运行测试。
- 不同模型使用单独环境；遵循官方 Python、Torch、TorchAudio、CUDA 和扩展编译版本。
- 固定代码 commit、权重 revision、依赖锁、输入与随机种子；记录首轮和预热后耗时、峰值内存及失败配置。
- “可加载”“可推理”和“已复现论文指标”是不同完成程度，实验记录应写清楚。

参见[模型复现指南](../docs/notes/model-reproduction-guide.md)与[端侧部署](../docs/notes/edge-deployment.md)。

## 工具入口

| 工具 | 用途 |
|---|---|
| [AudioCraft](https://github.com/facebookresearch/audiocraft) | MusicGen 与 EnCodec |
| [Demucs](https://github.com/facebookresearch/demucs) | 音源分离 |
| [MERT](https://github.com/yizhilll/MERT) | 自监督音乐表征 |
| [LAION CLAP](https://github.com/LAION-AI/CLAP) | 音频/文本嵌入 |
| [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools) | Stable Audio 训练与推理 |
| [DAC](https://github.com/descriptinc/descript-audio-codec) | 音频编解码 |
| [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) | 单码本低 token 率神经音频编解码；不是 LFQ |
| [FADtk](https://github.com/microsoft/fadtk) | FAD 分布指标 |
| [museval](https://github.com/sigsep/sigsep-mus-eval) | 源分离评测 |
| [MidiTok](https://github.com/Natooz/MidiTok) | 符号音乐 tokenization |
| [RTNeural](https://github.com/jatinchowdhury18/RTNeural) | 小型实时神经网络推理 |
