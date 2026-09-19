# 音乐版权与 AI 合规

资料核对日期：2026-09-19。本页区分法规、机构解释、当事方公告和工程建议；适用于研究导航，具体项目仍需按法域、用途和授权合同判断。

## 1. 先区分不同权利和行为

音乐作品（词、曲）、录音制品、表演，以及声音/人格权益可能分别受保护。取得录音文件或模型权重，不等于取得所有这些权利。应分别核对数据获取、训练复制、模型分发、声音克隆和生成内容发布。

“公开可下载”“免版税”“Creative Commons”和“公有领域”不是同义词。CC 作品通常仍有版权；BY、NC、ND、SA 等条件必须分别核对。数据集元数据的许可也不自动覆盖链接指向的录音。参见 [Creative Commons 许可说明](https://creativecommons.org/share-your-work/cclicenses/) 和[数据集目录](../../datasets/README.md)。

## 2. 美国：版权性与训练合理使用是两个问题

[美国版权局 AI 报告](https://www.copyright.gov/ai/) Part 2（2025-01-29）讨论生成内容的版权性：纯机器生成且缺少人类作者性的部分不受保护；人类原创的选择、编排、修改等贡献仍可能受保护。不能把它概括成“使用 AI 的作品一律没有版权”。

Part 3 于 2025-05-09 发布预出版版，讨论生成式 AI 训练。合理使用须结合使用目的、作品性质、使用数量及市场影响等因素作个案分析；报告本身不是法院判决。合法购得副本也不意味着一切后续训练必然构成合理使用。[17 U.S.C. §107](https://www.copyright.gov/title17/92chap1.html#107)

原笔记把 Anthropic 个案概括成“合法获取 = 合理使用；盗版 = 侵权”的普遍公式，已删除。数据来源、建库行为和具体训练用途需要分别判断，不能把书籍案件中的结论直接推广到音乐模型。

### 音乐行业案件与协议

2024 年 6 月，多家唱片公司分别起诉 Suno 和 Udio，指控未经授权复制录音；这是原告的指控，不能写成法院已认定的全部事实。每件作品最高 150,000 美元是故意侵权情形下法定赔偿的上限之一，并非每件作品自动获赔。[RIAA 起诉公告](https://www.riaa.com/record-companies-bring-landmark-cases-for-responsible-ai-againstsuno-and-udio-in-boston-and-new-york-federal-courts-respectively/)、[17 U.S.C. §504](https://www.copyright.gov/title17/92chap5.html#504)

- UMG 于 2025-10-29 宣布与 Udio 和解及授权合作，并在公告中计划于 2026 年推出新平台。该公告证明协议和当时的计划，不能单独证明平台已上线。[UMG 公告](https://www.universalmusic.com/universal-music-group-and-udio-announce-udios-first-strategic-agreements-for-new-licensed-ai-music-creation-platform/)
- WMG 于 2025-11-25 宣布与 Suno 合作并解决双方此前诉讼。官方公告未披露“约 5 亿美元和解”数字，原笔记中该金额已移除。与某一权利人的协议不代表取得其他权利人的许可。[WMG 公告](https://www.wmg.com/news/warner-music-group-and-suno-forge-groundbreaking-partnership)

## 3. 欧盟：分阶段义务，版权与透明度分开

AI 法案于 2024-08-01 生效。GPAI 模型相关义务与第 50 条透明度规则属于不同部分，不能将“第 50–55 条”全部归为同一日期的 GPAI 义务。

- GPAI 第 53(1)(c) 条要求版权合规政策，包括识别并遵守 DSM 指令第 4(3) 条规定的权利保留；第 53(1)(d) 条要求公开足够详细的训练内容摘要。开源豁免不免除这两项义务。
- GPAI 规则自 2025-08-02 起分阶段适用；较早已投放市场的模型有过渡安排。具体日期须结合模型投放时间与修订条文。
- 透明度规则自 2026-08-02 起适用；这不表示全部高风险系统规则同日生效。欧委会现行说明列有 2027、2028 年的高风险系统适用日期。

来源：[AI 法案文本](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng)、[欧委会现行实施时间线](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)。TDM 例外及权利保留与训练摘要披露是不同义务，公开摘要本身不授予训练许可。

## 4. 中国、日本与英国

### 中国

《生成式人工智能服务管理暂行办法》第七条要求使用具有合法来源的数据和基础模型，涉及知识产权时不得侵害他人依法享有的知识产权；其适用范围应按第二条判断。不能将图像案件中对特定人类创作投入的判断，推广为所有 AI 音乐自动取得版权。声音克隆也不能仅凭曲库许可推断已获本人同意。[网信办原文](https://www.cac.gov.cn/2023-07/13/c_1690898327029107.htm)

### 日本

著作权法第 30-4 条为非享受作品所表达思想或感情的使用等情形提供例外，但有目的与不得不合理损害权利人利益等限制。训练与输出的法律判断应分开；“日本不对 AI 训练执行版权”是不准确的概括。[文化厅 AI 与著作权说明](https://www.bunka.go.jp/english/policy/copyright/pdf/94055801_01.pdf)

### 英国

2026 年 3 月的政府报告表示，带权利人退出机制的广泛例外已不再是政府的首选方案，且仍需进一步工作。这不同于已经立法永久禁止一切新例外，也不意味着现有研究例外消失。[政府报告](https://www.gov.uk/government/publications/report-and-impact-assessment-on-copyright-and-artificial-intelligence/report-on-copyright-and-artificial-intelligence)

## 5. 训练数据与模型许可

| 项目 | 可核实的边界 |
|---|---|
| Stable Audio Open 1.0 | 模型卡列出 486,492 条录音，来自 Freesound 与 FMA 的选定许可子集；并非所有 FMA/Freesound 内容都可按同一条件使用。模型本身另有许可 |
| MusicGen / AudioCraft | 代码与权重分开：AudioCraft 代码 MIT，发布的 MusicGen 权重 CC BY-NC 4.0；不可据代码许可推断权重可商用 |
| MusicLM | 论文报告大规模音乐训练，不能说模型只用“合成/虚假数据”。MusicCaps 是评测相关描述数据集，也不能据此推断训练语料全部公开 |

来源：[Stable Audio Open 模型卡](https://huggingface.co/stabilityai/stable-audio-open-1.0)、[AudioCraft](https://github.com/facebookresearch/audiocraft)、[MusicLM 论文](https://arxiv.org/abs/2301.11325)。

## 6. 水印、来源记录与相似性检测

[AudioSeal](https://github.com/facebookresearch/audioseal)提供音频水印与局部检测机制。检测效果依赖训练域、处理链、压缩和攻击条件；带水印不等于合法，未检出也不等于内容由人类创作。

[C2PA](https://c2pa.org/specifications/specifications/)为媒体来源声明提供签名和可验证的绑定机制。它可帮助核对来源声明是否遭修改，但不替代版权许可或证明声明内容必然真实。

音频指纹、旋律匹配、歌词重复检测可以筛查潜在复制；CLAP 相似度和 FAD 都不是侵权判定器。应记录输入、生成配置、检索库、阈值及人工复核结果。

## 7. 研究项目记录清单

1. 分开记录词曲、录音、表演和声音使用的授权范围，保存数据来源、版本及许可原文。
2. 明确是否允许训练、微调、商用、再分发及生成结果发布；落实适用的署名与退出要求。
3. 将版权性、训练合规、输出相似性和人格权益作为不同问题检查。
4. 保留数据划分、去重和输出筛查记录；水印与来源凭证作为辅助证据。
5. 对具体商业部署，核对服务地区、最新法条、合同和案件状态。

English summary: Access, training, distribution and output use require separate analysis. Copyright in human contributions, fair use, dataset licences and model licences are distinct questions. Watermarks and provenance records help trace content but do not grant rights.
