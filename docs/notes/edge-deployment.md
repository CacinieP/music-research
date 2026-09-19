# 端侧部署与实时推理：音乐 AI 模型

核对日期：2026-09-19。本文讨论估算方法、部署约束和可复现实测。原稿中缺少设备型号、版本和测试条件的固定显存、TOPS 与速度结论已移除。

## 1. 先区分吞吐、延迟与流式处理

实时因子定义为

$$\mathrm{RTF}=\frac{t_{\mathrm{compute}}}{t_{\mathrm{audio}}}.$$

RTF 小于 1 表示平均处理快于音频时长，但不保证低延迟或硬实时：模型可能仍需整段输入、未来上下文或很长的初始缓冲。

音频回调包含 $B$ 个样本、采样率为 $f_s$ 时，单块时间为 $B/f_s$。48 kHz 下 64 样本约为 1.33 ms；这只是一个缓冲块，不能当作完整往返延迟。完整链路还包括输入/输出缓冲、转换器、驱动、模型 look-ahead、计算与调度。

| 场景 | 应测量什么 |
|---|---|
| 音箱建模、监听效果 | 物理输入到输出的往返延迟，回调最坏耗时，丢帧/爆音 |
| 流式分离与伴奏 | 算法 look-ahead、首块延迟、稳态 RTF、长时间漂移 |
| 离线分离与生成 | 端到端耗时、音频时长、峰值内存、输出质量 |

不同乐器、演奏方式和监听环境的延迟容忍度不同，不宜把 5/10/20 ms 写成普适的人耳阈值。

## 2. 参数量不等于峰值显存

仅权重的理想存储量为

$$M_{\mathrm{weights}}=\frac{P b}{8}\ \text{bytes},$$

其中 $P$ 是参数个数，$b$ 是每参数位数。1B 参数的 FP16 权重约 2 GB（十进制），不包括激活、KV cache、文本编码器、音频编解码器、工作区和运行时开销。INT8 相对 FP32 的权重容量理想缩至四分之一；INT4 相对 FP16 同理。量化尺度、零点和未量化层会增加实际占用。

- 自回归模型的 KV cache 随生成长度、层数和 batch 增长；具体大小也取决于注意力头布局与缓存精度。
- 全注意力的计算复杂度随序列长度平方增长。FlashAttention 等实现能减少注意力中间存储，但不能把所有模型的内存/时间都概括为线性。
- 扩散模型重复去噪；降低采样步数可减耗时，但会改变质量，需要单独验证。
- 训练还需要梯度、优化器状态及中间激活；推理权重下界不是训练资源预算。

[AudioCraft MusicGen 文档](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)对 1.5B 模型给出至少 16 GB GPU 内存的推理要求；这与“FP16 权重约 3 GB”并不矛盾。特定优化版本可更省内存，必须同时给出版本和测量条件。

## 3. 优化方法及适用范围

| 方法 | 潜在收益 | 必须验证的条件 |
|---|---|---|
| FP16/BF16、INT8/INT4 | 权重、带宽与算子加速 | 目标内核是否支持；量化范围与校准数据；听感和任务指标 |
| 结构化剪枝 | 减少实际张量尺寸 | 需模型恢复训练与目标设备实测；稀疏率不是加速比 |
| 步数/层数蒸馏 | 减少扩散求解步骤和每步开销 | 蒸馏成本、条件覆盖、长音频和人声质量 |
| 分块/缓存 | 降低峰值内存，支持持续输入 | 边界伪影、状态传递、look-ahead 与重叠开销 |

GPTQ/AWQ 等权重量化方法不能仅因音频模型也用了 Transformer 就保证同等质量；“INT8 安全”不是无需测试的结论。[Presto!](https://arxiv.org/abs/2410.05167)是结合步数与层数蒸馏的音乐生成研究实例，其收益应引用论文配置而非推广到所有扩散模型。

## 4. 硬件与运行时

| 路线 | 适合的评估方向 | 限制 |
|---|---|---|
| NVIDIA GPU / Jetson + TensorRT | 分块分离、批量推理、生成 | 显存、算子支持、热功耗；吞吐优化不自动保证音频回调截止时间 |
| Apple Silicon + Core ML / MPS | macOS/iOS 端侧模型 | CPU/GPU/ANE 分配、算子回退与可预测性必须实测 |
| Qualcomm + QNN / LiteRT | Android NPU 模型 | 转换、量化格式、目标 SoC 和 delegate 版本 |
| CPU / ARM + RTNeural | 小型神经音频效果 | 模型结构、线程行为和缓冲长度须匹配硬实时要求 |
| ONNX Runtime / OpenVINO | 跨平台或 Intel 硬件 | 执行提供者支持范围；不能假定所有算子都在加速器执行 |

不同厂商 TOPS 在精度、稀疏性与算子条件上可能不同，不能直接推断音乐模型速度。多张 GPU 的显存也不会自动成为连续的大显存池。

Apple 官方转换路线是 PyTorch 图捕获后使用 `coremltools.convert`；ONNX 不是现代 Core ML 的必经中间格式。[Core ML 工作流](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html)

[RTNeural](https://github.com/jatinchowdhury18/RTNeural)是适用于小型实时神经网络的 C++ 推理库。[AIDA-X](https://github.com/AidaDSP/AIDA-X)使用它；不能因此把所有 Neural Amp Modeler 实现都说成“经 RTNeural”。[NAM 核心](https://github.com/sdatkinson/NeuralAmpModelerCore)有自己的实现和支持结构。

## 5. 复现实测协议

1. 记录设备完整型号、OS、驱动、运行时与模型 commit/checkpoint、采样率、声道和精度。
2. 固定输入时长、batch、分块、重叠、生成 token/采样步数及随机种子。
3. 区分加载/编译、冷启动和预热后的推理；GPU 计时时同步设备。
4. 报告多次测量的中位数与尾部延迟、峰值内存、RTF，保留失败/OOM 配置。
5. 在真实目标设备上对比浮点与优化版本的任务指标和盲听结果。
6. 硬实时音频回调避免分配、文件 I/O、阻塞锁和不可控任务调度，验证长时间运行的截止时间违约情况。

English summary: RTF measures throughput, not round-trip latency. Weight bytes are only a lower bound on runtime memory. Quantization, pruning and distillation need task-specific quality checks on the target device; record the complete benchmark configuration.
