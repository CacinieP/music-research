# 模型复现指南框架

本文档提供模型复现的思路框架和记录模板，不包含源代码。重点在于环境选型、算力评估和记录规范。

> English version: [model-reproduction-guide.md](model-reproduction-guide.md)（本文件为中英双语）

---

## 复现前的准备

### 1. 明确目标

- **推理验证**：确认论文声称的效果能否复现
- **对比实验**：在相同条件下对比多个模型
- **微调适配**：在自己的数据上微调已有模型
- **完整复现**：从零训练，验证论文结论

### 2. 算力评估清单

| 评估项 | 说明 |
|--------|------|
| 模型参数量 | 决定最低显存需求（FP16 下 1B ≈ 2GB 显存） |
| 训练数据规模 | 决定训练时长和存储需求 |
| 是否需要预训练 | 基础模型预训练 vs 仅微调，差距 10--100x 算力 |
| 输入输出长度 | 音频长度直接影响显存（长音频线性增长） |
| 批大小 | 受显存限制；可用梯度累积模拟大 batch |

### 3. 环境选型决策树

```
推理（< 4GB 显存）→ Colab Free / 本地 RTX 3060
推理（4--16GB）   → Colab Pro / 本地 RTX 4090 / AutoDL
微调（单卡）       → AutoDL A100 (40GB) / RunPod A100
训练（多卡）       → 云端 4--8x A100
预训练            → 32x A100+ / 集群
```

---

## 复现记录模板

每次复现应记录以下内容：

### 基本信息模板

```markdown
## [模型名称] 复现记录

### 环境信息
- 日期：YYYY-MM-DD
- 硬件：GPU 型号 / 数量 / 显存
- 系统：OS / CUDA 版本 / 驱动版本
- 框架：PyTorch 版本 / 其他依赖版本
- 代码来源：官方 repo commit hash

### 数据
- 训练数据集及规模
- 预处理步骤
- 数据加载配置（batch size, num_workers）

### 训练配置
- 超参数（learning rate, scheduler, epochs）
- 精度（FP32 / FP16 / BF16 / 混合精度）
- 分布式策略（DDP / FSDP / DeepSpeed）
- 训练时长（实际 wall-clock time）

### 结果
- 论文报告的指标 vs 实际复现的指标
- 与论文的差距分析
- 训练曲线（loss, metric）截图

### 遇到的问题
- 环境问题
- 训练不稳定（loss spike, NaN）
- 与论文不符的结果及排查过程

### 结论
- 是否成功复现
- 关键发现和经验
```

---

## 推荐的对比实验框架

### 源分离对比

| 维度 | 记录内容 |
|------|----------|
| 模型 | Spleeter → Demucs → HT Demucs → BSRNN → BS-RoFormer |
| 数据 | MUSDB18-HQ test set |
| 指标 | SDR/SIR/SAR（per stem + overall） |
| 效率 | 推理时间、显存占用、RTF（real-time factor） |
| 感知 | 人工听感评价（可选） |

### 文本到音乐对比

| 维度 | 记录内容 |
|------|----------|
| 模型 | MusicGen → AudioLDM 2 → Stable Audio → YuE |
| Prompt | 使用 MusicCaps 的统一 prompt 集 |
| 指标 | FAD, CLAP Score, 人工偏好 |
| 效率 | 生成 30 秒音频所需时间 |
| 条件 | 文本质量对结果的影响 |

### 编解码器对比

| 维度 | 记录内容 |
|------|----------|
| 模型 | EnCodec → DAC → WavTokenizer → HiFi-Codec |
| 比特率 | 1.5 / 3 / 6 / 12 / 24 kbps |
| 指标 | ViSQOL, PESQ, 重建波形对比 |
| 音乐类型 | 人声 / 古典 / 电子 / 摇滚 |

---

## 常见坑与注意事项

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 显存不足 | 音频序列太长或 batch 过大 | 梯度累积 / chunk 处理 / FP16 |
| 训练 NaN | 混合精度 + 小学习率不稳定 | 使用 BF16 / 梯度裁剪 / 预热 |
| 论文无法复现 | 随机种子、数据划分、超参未完全公开 | 联系作者 / 多次实验取平均 |
| 依赖冲突 | 不同模型要求不同 PyTorch/CUDA 版本 | 使用 conda 环境隔离 / Docker |
| 下载慢 | HuggingFace / GitHub 国内访问受限 | 镜像站 / 代理 / 离线下载 |
| 生成质量不稳定 | 随机种子 + 采样策略 | 固定种子；记录多组采样参数 |

---

## Jupyter Notebook 组织建议

```
notebooks/
├── 01_feature_visualization.ipynb    # mel/CQT/chroma 可视化对比
├── 02_codec_comparison.ipynb         # EnCodec vs DAC vs WavTokenizer 重建对比
├── 03_source_separation.ipynb        # Demucs/BSRNN 分离效果试听
├── 04_generation_comparison.ipynb    # MusicGen/Stable Audio/YuE 生成对比
├── 05_mir_benchmark.ipynb            # MERT/CLAP 下游任务评估
└── utils.py                          # 共享工具函数
```

每个 Notebook 应包含：
1. 环境安装（`!pip install ...`）
2. 模型加载
3. 输入示例 + 可视化
4. 推理 + 输出可视化
5. 指标计算
6. 结论与观察
