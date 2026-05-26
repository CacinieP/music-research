# 音乐版权与 AI 合规 (2025--2026)

音乐 AI 生成面临的版权法律问题、生成侵权风险、各国立法差异、行业实践及合规指南。

> English version: [music-copyright-compliance.md](music-copyright-compliance.md)（本文件为中英双语）

---

## 1. 训练数据版权

### 关键诉讼

**RIAA v. Suno 和 Udio（2024 年 6 月）**
- RIAA 代表 Sony/Universal/Warner 在联邦法院起诉 Suno（波士顿）和 Udio（纽约）
- 核心指控：大规模版权侵权——在未经授权的情况下使用版权录音训练模型
- 索赔：每件侵权作品 $150,000

**和解（2025 年 10--11 月）**
- UMG 于 2025 年 10 月与 Udio 达成和解，计划 2026 年推出授权 AI 音乐平台
- Warner Music 于 2025 年 11 月与 Suno 达成约 $5 亿和解，被称为"里程碑"交易。旧版 Suno 模型正在被淘汰
- Forbes（2025 年 12 月）将此概括为"Launch, Train, Settle"模式

**GEMA v. OpenAI（德国，2025 年）**
- GEMA 对 OpenAI 提起诉讼，指控其在 AI 训练数据中未经授权使用受版权保护的歌词和音乐作品

**独立艺术家 v. Google**
- 独立艺术家起诉 Google 的 Lyria 音乐 AI 模型，指控其基于数千万未授权版权作品训练

### 合理使用 vs 授权要求

**Anthropic 里程碑判决（2025 年 6 月）**
- 法院：美国加州北区联邦法院（Judge William Alsup）
- 关键裁定：
  - 在**合法获取**的版权作品上训练 AI 构成合理使用
  - 为训练目的创建**盗版**作品的集中库不构成合理使用，属于侵权
- 这是美国首个重大判决，确立了 AI 模型在版权作品上训练可构成合理使用，但关键区别在于训练数据的获取方式

**美国版权局报告**
- Part 2（2025 年 1 月）：确认现有版权法足以处理 AI 生成作品的版权性；完全由 AI 生成的作品因缺乏人类作者性而不可获得版权注册
- Part 3（2025 年 5 月）：108 页报告，分析未经授权使用版权作品训练 AI 是否构成合理使用；结论：合理使用不会总是保护 AI 开发者免于授权义务，需逐案分析

### 合理使用四要素在音乐 AI 中的适用

| 要素 | 分析 |
|------|------|
| 使用目的与性质 | 训练是否具有转换性？ |
| 版权作品的性质 | 音乐等创意作品享有更强保护 |
| 使用量 | 模型摄取了多少原始作品？ |
| 市场影响 | AI 输出是否与原作品竞争或替代？ |

---

## 2. 生成侵权风险

### 实质性相似性

- 实质性相似性是判断 AI 生成音乐是否侵权的核心法律测试
- AI 的"黑箱"特性使判断困难：不清楚模型接触了哪些训练数据、输出是否与特定版权作品实质性相似
- 如果 AI 工具重现了特定旋律或歌词，很可能构成侵权，但证明此类复制很困难

### 水印与检测方法

| 方法 | 机构/年份 | 特点 |
|------|-----------|------|
| **AudioSeal** | Meta / ICML 2024 | 首个专为 AI 生成音频设计的局部检测水印；开源 |
| **Digimarc** | 商业 / 2025 年 7 月 | 新一代商业音频水印，确保创作者准确补偿 |
| **XAttnMark** | ICML 2025 | 基于交叉注意力的鲁棒音频水印 |
| **Timbru** | 学术 2025 | 多比特音频水印，达到 SOTA 鲁棒性 |
| **RAW-Bench** | INTERSPEECH 2025 | 评估音频水印算法的综合现实世界基准 |

- 关键挑战：水印对移除/覆写攻击的鲁棒性仍是开放问题（2025 年 SoK 综评）

---

## 3. 各国立法

### 美国
- 版权局 Part 2+3 报告确立：AI 生成作品不可注册版权（缺乏人类作者性）
- Anthropic 判决：合法获取的训练数据 = 合理使用；盗版 = 侵权
- 法律格局仍在演变，Suno/Udio 音乐相关案件持续发展

### 欧盟
- **AI 法案**：2024 年 8 月 1 日生效
  - GPAI 义务（第 50--55 条）：2025 年 8 月 2 日可执行
  - 完全适用（高风险系统、透明度/第 50 条）：2026 年 8 月 2 日
- **第 53(1)(c) 条**：GPAI 提供者必须实施遵守 EU 版权法的政策，包括 TDM（文本与数据挖掘）退出机制
- **第 53(1)(d) 条**：必须发布训练数据内容的"足够详细摘要"
- 即使开源 GPAI 提供者也须遵守版权和训练数据摘要义务

### 中国
- **CNIPA 指导（2025 年 4 月）**：AI 生成作品能否获得版权取决于是否体现人类创造性
- **北京互联网法院 Li v. Liu（2023/2024）**：里程碑判决——AI 生成图像可获得版权保护，前提是反映人类智力/创造性投入。与美国立场显著不同
- **常熟法院（2025 年 3 月）**：确认 AI 生成图像可获得版权保护
- AI 平台可为生成的侵权内容承担辅助侵权责任
- 生成式 AI 监管规定要求内容尊重知识产权

### 日本
- **著作权法第 30-4 条**：允许为"信息分析"（包括 AI 训练）使用版权作品，无需许可或支付
- 日本政府重申不会对 AI 训练数据执行版权
- 被视为全球最 AI 友好的版权制度之一
- 与美国（合理使用争议）和 EU（TDM 退出机制）形成鲜明对比

### 英国
- **2026 年 3 月 18 日**：英国政府确认不会引入 AI 训练的广泛版权例外
- 最初曾提议类似 EU 的 TDM 版权例外（含权利人退出机制），但在创意产业强烈反对后放弃
- 采用**市场导向授权模式**：AI 开发者必须与权利人协商许可

---

## 4. 行业实践

| 公司 | 训练数据策略 | 合规特点 |
|------|------------|----------|
| **Suno** | 含版权录音（据称包括 Mariah Carey 到 Chuck Berry） | 2025 年 11 月与 Warner 和解 (~$5 亿)；旧模型正在淘汰 |
| **Udio** | 类似 Suno | 2025 年 10 月与 UMG 和解；参与新授权 AI 平台 |
| **Stability AI (Open)** | 仅使用 CC/免版税数据（Freesound 47 万 + FMA 1.4 万） | 明确避开版权材料；可微调 |
| **Stability AI (商业)** | AudioSparx 授权数据集 (>80 万音频) | 遵守退出请求，确保公平补偿 |
| **Google** | MusicLM 使用合成/虚假数据集 | 减少对版权材料的依赖 |
| **Meta** | AudioCraft 代码 MIT 许可；模型权重许可更严格 | 开发 AudioSeal 水印 |

**Merlin Network 政策（2024 年 12 月）**：要求对其目录的任何 AI 训练须获得"事先明确的特定授权"

---

## 5. 新兴标准

### C2PA（内容来源与真实性联盟）
- 将防篡改来源元数据嵌入数字媒体（包括音频文件）的标准
- 2025 年：美国国会图书馆启动 C2PA 工作组；EBU 为广播公司举办 C2PA 活动
- Content Authenticity Initiative (CAI) 推动全球采用

### 透明度与合规工具
- **TransparentMeta**：专为 AI 生成音频合规设计，符合 EU AI 法案透明度要求
- **MassiveMusic**：提供授权的、专家策划的音乐数据集用于合规 AI 训练

---

## 6. 合规实践指南

1. **使用合法获取的数据**：Anthropic 判决（2025 年 6 月）确立了合法获取 = 合理使用，盗版 = 侵权。这是最关键的合规区分。

2. **优先使用明确授权或 CC 许可的数据**：Stable Audio Open 的方法（Freesound + FMA，全部 Creative Commons）是合规的金标准。

3. **详尽记录训练数据**：EU AI 法案第 53(1)(d) 条要求发布"足够详细摘要"。即使不在 EU，文档记录也能防范未来法律挑战。

4. **实施 TDM 退出合规机制**：如果模型将在 EU 提供，必须尊重 EU 版权指令下的 TDM 退出声明。

5. **考虑合成数据**：Google 为 MusicLM 使用合成/虚假数据集训练的方法展示了完全避免版权风险的可行路径。

6. **小规模精选数据集**：2025 年 ACM 研究表明小数据集对音乐 GenAI 也可有效，降低法律风险。

7. **实施水印**：使用 AudioSeal 或类似工具在生成音频中嵌入水印，辅助检测并展示合规诚意。

8. **测试实质性相似性**：部署前测试生成输出与已知版权作品的相似性。

9. **嵌入 C2PA Content Credentials**：在生成音频文件中采用 C2PA 标准嵌入来源元数据。

10. **获取明确授权**：遵循 Merlin Network 2024 年 12 月政策，对任何训练须获得"事先明确的特定授权"。

---

## 7. 关键事件时间线

| 时间 | 事件 |
|------|------|
| 2023 年 2 月 | 美国版权局：AI 生成作品无人类作者性不可注册 |
| 2023 年 11 月 | 北京互联网法院 Li v. Liu：AI 生成作品可获版权（含人类创造性） |
| 2024 年 8 月 | EU AI 法案生效 |
| 2024 年 6 月 | RIAA 起诉 Suno/Udio |
| 2025 年 1 月 | 美国版权局 Part 2 报告发布 |
| 2025 年 5 月 | 美国版权局 Part 3 报告发布 |
| 2025 年 6 月 | Anthropic 合理使用判决 |
| 2025 年 7 月 | Digimarc 发布新一代音频水印 |
| 2025 年 8 月 | EU AI 法案 GPAI 义务可执行 |
| 2025 年 10 月 | UMG 与 Udio 和解 |
| 2025 年 11 月 | Warner 与 Suno 和解 (~$5 亿) |
| 2025 年 12 月 | Forbes "Launch, Train, Settle" 分析 |
| 2026 年 3 月 | 英国放弃 TDM 版权例外计划 |
| 2026 年 8 月 | EU AI 法案完全适用 |

---

## 参考文献

- [RIAA Suno/Udio 起诉公告](https://www.riaa.com/record-companies-bring-landmark-cases-for-responsible-ai-againstsuno-and-udio-in-boston-and-new-york-federal-courts-respectively/)
- [美国版权局 AI 报告](https://www.copyright.gov/ai/)
- [EU AI 法案](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
- [Anthropic 判决分析 (Authors Alliance)](https://www.authorsalliance.org/2025/06/24/anthropic-wins-on-fair-use-for-training-its-llms-loses-on-building-a-central-library-of-pirated-books/)
- [Forbes: Launch, Train, Settle](https://www.forbes.com/sites/virginieberger/2025/12/18/launch-train-settle-how-suno-and-udios-licensing-deals-made-copyright-infringement-profitable/)
- [Warner-Suno 和解 (The Guardian)](https://www.theguardian.com/business/2025/nov/26/warner-music-signs-deal-with-ai-song-generator-suno-after-settling-lawsuit)
- [AudioSeal (GitHub)](https://github.com/facebookresearch/audioseal)
- [C2PA](https://c2pa.org/)
- [Digimarc 音频水印](https://www.digimarc.com/press-releases/2025/07/16/digimarc-revolutionizes-audio-content-authentication-protection-next)
- [Li v. Liu 判决](https://legalblogs.wolterskluwer.com/copyright-blog/beijing-internet-court-grants-copyright-to-ai-generated-image-for-the-first-time/)
- [CNIPA 2025 年 4 月指导](https://english.cnipa.gov.cn/art/2025/4/25/art_3090_199316.html)
- [日本著作权法第 30-4 条分析](https://cepa.org/article/ai-boom-or-copyright-doom-lessons-from-asia/)
- [英国放弃 TDM 例外](https://www.hoganlovells.com/en/publications/ai-and-copyright-uk-government-backs-away-from-exceptions-for-ai-training-proposes-maintaining-the)
- [Stable Audio Open (Hugging Face)](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [音频水印鲁棒性综述 (arXiv)](https://arxiv.org/html/2503.19176v2)