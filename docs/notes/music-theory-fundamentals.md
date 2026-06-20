# Music Theory Fundamentals for AI Music Research

Essential music theory for researchers working in MIR, generation, and music foundation models. Bridges abstract theory to practical concerns: tokenization design, representation choices, evaluation metrics, and controllable generation.

---

## 1. Pitch 音高

### 1.1 Frequency and the 12-TET System

人类感知的音高（pitch）由声波基频（fundamental frequency）决定。标准音 A4 = 440 Hz（ISO 16），国际标准音高（concert pitch）。

西方音乐采用 **12 平均律（12-TET, Twelve-Tone Equal Temperament）**：一个八度（octave）被等分为 12 个半音（semitones），每个半音的频率比为 2^(1/12) ≈ 1.0595。

$$
f_n = f_0 \cdot 2^{n/12}
$$

12-TET 的优势在于所有调性（keys）的等距性，使转调（transposition）和和声（harmony）计算简单。代价是纯律（just intonation）的协和性被轻微妥协。

**对 AI 的意义**：12-TET 的音高离散性（12 个音高类 pitch classes）是大多数表示设计的基础。MIDI 编号直接用整数映射 12-TET 半音。

### 1.2 Pitch Class and Octave

- **Pitch class (音高类)**：一个八度内的 12 个半音之一（C, C#, D, ..., B），不区分八度。
- **Octave (八度)**：频率翻倍的区间，国际标准从 C1 到 C10（MIDI 0–127 覆盖 C-1 到 G9）。
- **Scientific pitch notation**：如 C4 = middle C = MIDI 60。

Pitch class 在音级分析（pitch-class set analysis）和音高类向量（pitch-class vector, 12 维直方图）中被大量使用。

### 1.3 Enharmonic Equivalence

等音（enharmonic equivalents）：不同记谱但 12-TET 下相同频率的音高，如 C# = Db, D# = Eb, F# = Gb, G# = Ab, A# = Bb。

等音在 AI 表示中通常被合并处理（如 pitch-class vector），但在音乐理解任务中区分等音是必要的（调性分析、和弦识别）。

---

## 2. Intervals and Scales 音程与调式

### 2.1 Intervals (音程)

音程是两个音高之间的距离，以半音数度量：

| 半音数 | 音程名称 | 缩写 |
|--------|----------|------|
| 0 | 纯一度 | P1 |
| 1 | 小二度 | m2 |
| 2 | 大二度 | M2 |
| 3 | 小三度 | m3 |
| 4 | 大三度 | M3 |
| 5 | 纯四度 | P4 |
| 6 | 增四度 / 减五度 | A4 / d5 (tritone) |
| 7 | 纯五度 | P5 |
| 8 | 小六度 | m6 |
| 9 | 大六度 | M6 |
| 10 | 小七度 | m7 |
| 11 | 大七度 | M7 |
| 12 | 纯八度 | P8 |

**三全音（tritone, A4/d5）**：6 个半音，中世纪被称为"diabolus in musica"（音乐中的魔鬼），因其高度不协和性。在七和弦中分隔三音与七音，是爵士和声的核心张力来源。

**对 AI 的意义**：音程是旋律相似性（melodic similarity）和音程类向量（interval-class vector）分析的基础。许多 MIR 特征提取器直接输出音程分布。

### 2.2 Diatonic Scales (自然音阶)

**大调（Major）音阶**：全-全-半-全-全-全-半 模式（W-W-H-W-W-W-H），从主音（tonic）开始的音程序列为 [2,2,1,2,2,2,1]。

**小调（Minor）音阶**有三种形式：
- **自然小调（Natural minor）**：[2,1,2,2,1,2,2] — 无升降的自然形态
- **和声小调（Harmonic minor）**：自然小调升高七音 — 小七度变大七度，形成独特的 ii° 减七和弦
- **旋律小调（Melodic minor）**：上行升高六、七音，下行还原

调式（modes）是大调音阶从不同级开始的七种变体：Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian。每种模式的特征音（characteristic note）决定其色彩：Lydian 的 #4 明亮梦幻，Dorian 的 b3+b7 温暖忧郁，Phrygian 的 b2 西班牙式暗色。

### 2.3 Pentatonic and Non-Diatonic Scales

- **五声音阶（Pentatonic）**：5 个音，去除大调音阶的四音与七音。跨文化普遍存在（中国、日本、苏格兰、非洲），无半音使其容易即兴且听感安全。
- **全音阶（Whole-tone）**：6 个全音，无半音，德彪西印象派核心音阶，声部解决方向模糊。
- **减音阶（Diminished）**：交替全半模式，八度内 8 个音，高度对称，可循环上行/下行。
- **布鲁斯音阶（Blues）**：五声音阶 + 减五音（blue note），折中半音，布鲁斯表达核心。

**对 AI 的意义**：音阶选择决定了生成音乐的"调性感"和风格。许多 controllable generation 系统通过 scale degree 条件控制色彩。五声音阶的普遍性是跨文化生成的重要考虑。

---

## 3. Harmony 和声

### 3.1 Triads (三和弦)

三和弦由三个音构成：根音（root）、三音（third）、五音（fifth）。四个品质（qualities）：

| 品质 | 三音 | 五音 | 缩写 | 色彩 |
|------|------|------|------|------|
| 大三（Major） | M3 | P5 | △ 或 Maj | 明亮、开放 |
| 小三（Minor） | m3 | P5 | m 或 - | 柔和、暗淡 |
| 减三（Diminished） | m3 | d5 | ° | 紧张、悬停 |
| 增三（Augmented） | M3 | A5 | + | 飘渺、无归属 |

### 3.2 Seventh Chords (七和弦)

在三和弦基础上加入七音，四个核心类型：

| 类型 | 三音 | 五音 | 七音 | 符号 | 常见功能 |
|------|------|------|------|------|----------|
| 大七（Maj7） | M3 | P5 | M7 | △7 | I（主和弦） |
| 小七（m7） | m3 | P5 | m7 | m7 | ii, vi（下属/关系小调） |
| 属七（Dominant 7） | M3 | P5 | m7 | 7 | V（属和弦），最具张力 |
| 半减七（m7b5） | m3 | d5 | m7 | ø7 | vii°（导七） |

爵士扩展七和弦：9th (M2), 11th (P4), 13th (M6) 作为附加音（tensions），仅当半音关系正确时使用（如 9th 仅在大七/属七上可用，不在小七上）。

### 3.3 Diatonic Functions 和声功能

**调性中心（tonic center）**：大调的 I / 小调的 i。功能和声（functional harmony）将每个和弦映射到调性中的角色：

```
T  S  D  T
|  |  |  |
I→IV→V→I
i→iv→V→i
```

- **T（Tonic, 主功能）**：I, vi, iii — 稳定、归宿
- **S（Subdominant, 下属功能）**：IV, ii — 过渡、向前推进
- **D（Dominant, 属功能）**：V, vii° — 最大张力、必须解决到 T

属七和弦（V7）必须解决到主和弦（I）是古典和声的"铁律"，其内部三全音（tritone）的张力驱动解决。

**对 AI 的意义**：和弦功能是 controllable generation 的核心控制维度。V→I 的解决概率是评估生成和声合理性的首要指标。许多 generation 模型的 chord condition 直接基于罗马数字标记。

### 3.4 Cadences 终止式

终止式（cadence）是乐句的结束和声配置：

| 终止式 | 进行 | 特征 |
|--------|------|------|
| 完美正格终止（PAC） | V→I | 声部最稳定的终止，所有声部解决到位 |
| 不完满正格终止（IAC） | V→I | 声部未完全到位 |
| 变格终止（Plagal） | IV→I | "阿门"终止，柔和 |
| 半终止（HC） | 结束于 V | 乐句停在属和弦，开放 |
| 伪终止（Deceptive） | V→vi | 属到关系小调，"欺骗"预期 |

**对 AI 的意义**：终止式识别是音乐结构分析的关键任务。V→vi 的 deceptive cadence 是流行音乐的标志性"意外"手法。

### 3.5 Voice Leading 声部进行

声部进行（voice leading）关注各声部的独立运动线（melodic lines），而非和弦的纵向堆叠。黄金规则：

- **避免平行五度/八度**：两个声部同时纯五度/纯八度运动（尤其在古典对位中）
- **共同音保留（common tone retention）**：V→I 中，三音和五音上行小二度解决
- **引导音（leading tone）**：B（大调的七级）的半音上行解决到 C（主音）
- **三全音解决**：V7 中的三全音（F→B）反向半音运动（F↑E, B↓C）

**对 AI 的意义**：声部进行是评分模型生成对位/和声质量的核心标准。常见的和弦级标注系统（Roman numeral）忽略了声部细节，而 voice-leading -aware 的评估更严格。

---

## 4. Rhythm and Meter 节奏与节拍

### 4.1 Note Values and Duration

国际标准时值体系（基于全音符 = 1 为基准）：

| 记谱 | 相对全音符 | 与上一级关系 |
|------|-----------|--------------|
| Whole (全音符) | 1 | 基准 |
| Half (二分) | 1/2 | ÷2 |
| Quarter (四分) | 1/4 | ÷2 |
| Eighth (八分) | 1/8 | ÷2 |
| Sixteenth (十六分) | 1/16 | ÷2 |
| Thirty-second (三十二分) | 1/32 | ÷2 |

时值的指数衰减（每次÷2）使其天然适合用二进制对数或对数时间量化表示。

### 4.2 Time Signatures 拍号

拍号（time signature）定义每小节的节拍组织：

- **简单拍号**：X/4 形式（2/4, 3/4, 4/4）。上方数字=每小节拍数，下方=每拍的时值（四分=4, 八分=8）
- **复合拍号**：X/8 形式（6/8, 9/12, 12/8）。上方数字=每小节"三连音组"数，每组三拍
- **混合拍号**：如 5/4（2+3 或 3+2）、7/8（2+2+3 等不对称组合）

**4/4 拍**（common time）是最普遍的流行音乐拍号，其强拍弱拍模式为 [S, w, S, w]（strong-weak）。**3/4 拍**（waltz time）是 [S, w, w]。

### 4.3 Tempo 速度

速度以 BPM（beats per minute，每分钟拍数）度量：

| 术语 | BPM 范围 | 特征 |
|------|----------|------|
| Grave | 20–40 | 庄重、极慢 |
| Largo | 40–60 | 广板 |
| Adagio | 60–72 | 慢板 |
| Andante | 72–88 | 行板（步行速度） |
| Moderato | 88–108 | 中板 |
| Allegro | 108–132 | 快板 |
| Vivace | 132–168 | 活泼 |
| Presto | 168–200 | 急板 |
| Prestissimo | 200+ | 极急 |

意大利术语的模糊性是特点而非缺陷——BPM 只是一个参考锚点。

### 4.4 Tuplets and Rhythmic Displacement

**三连音（Tuplet）**：在某一时值内安排非标准数量的音符：

- **三连音（triplet）**：3 个音占 2 个的标准时值（"玩世不恭的规则"）
- **五连音（quintuplet）**：5 个音占 4 个时值
- **六连音（sextuplet）**：6 个音占 4 个时值 = 两组三连音，或 6 个音占 2 个时值

三连音的"不协和节奏感"（metric dissonance）是推动音乐前进的关键手段。

**切分（Syncopation）**：强拍位置的声部移到弱拍/弱位，打破预期节奏模式。切分是放克（funk）、雷鬼（reggae）、嘻哈节奏的核心。

**Swing（摇摆）**：将平均划分的时值转化为长短组合（如八分音符变为"长-短"≈ 2:1 或 3:1 比例），爵士 swing 的比例约 7:3 到 2:1，由"swing feel"而非精确标记决定。

**对 AI 的意义**：节奏表示是 generation 中最容易出错的维度——"大致对但不精确"的节奏感在人类听感中很刺耳。Swing 比例和 tuplet 的正确性是两个高频失败点。BPM 估计是 MIR 的基础任务（beat tracking, downbeat detection）。

---

## 5. Melody and Contour 旋律

### 5.1 Melodic Contour (旋律轮廓)

旋律轮廓（contour）描述音高的上升/下降/保持模式，忽略具体音高和时值：

| 类型 | 描述 | 常见于 |
|------|------|--------|
| Ascending (上行) | 总体音高上升 | 疑问句、向上动力 |
| Descending (下行) | 总体音高下降 | 答案句、乐句终止 |
| Arch (拱形) | 上行后下行 | 完整乐句，最常见 |
| Undulating (波动) | 交替上行下行 | 流畅旋律 |
| Static (静止) | 音高变化小 | 抒情、冥想 |

**音程偏倚（interval bias）**：自然语言中的词频分布类似，旋律中小音程（≤3 半音）出现频率远高于大音程（≥7 半音）——Zipf-like distribution。

### 5.2 Motif and Phrase (动机与乐句)

**动机（Motif）**：最短的音乐"词汇"，2–8 个音符的辨识模式。如贝多芬第五交响曲的 [G-G-G-Eb]（da-da-da-dum）。

动机通过以下手法发展：
- **Sequencing（模进）**：动机在不同音高层级重复
- **Inversion（倒影）**：音程方向反转（上行变下行）
- **Retrograde（逆行）**：音符顺序反转
- **Augmentation（扩大）**：时值加倍
- **Diminution（缩小）**：时值减半

**乐句（Phrase）**：音乐的"句子"，通常 4–8 小节，以终止式结束。类比语法：乐句 = 句子，乐段 = 段落。

### 5.3 Tension and Release 张力与释放

旋律张力的来源：
- **音高张弛**：不协和音程（三全音、小九度）vs 协和音程（纯五度、纯八度）
- **节奏张弛**：切分、切分解决
- **动态张弛**：强→弱（decrescendo），强拍→弱拍
- **密度张弛**：音符密度变化

**Meyer's expectation theory**：音乐张力来自"假设-确认/否认"循环。听者对下一步的预测越确定，预测被违背时的张力越强。

**对 AI 的意义**：旋律生成中的"无聊"问题本质上是张力曲线太平滑（no surprise = no interest）。Evaluation 中常用的 surprise metric（pitch/interval entropy）直接量化这一点。

---

## 6. Texture and Arrangement 织体与编曲

### 6.1 Musical Textures

| 织体 | 描述 | 例子 |
|------|------|------|
| Monophonic (单声) | 单一旋律线，无伴奏 | 格里高利圣咏、独唱 |
| Homophonic (主声) | 旋律 + 和声伴奏 | 流行歌曲、赞美诗 |
| Polyphonic (复调) | 多条独立旋律线 | 赋格、卡农、弦乐四重奏 |
| Heterophonic (支声) | 同一旋律的变体同时 | 中国民间音乐、爵士即兴合奏 |

### 6.2 Instrument Ranges and Transposition

各乐器的音域（range）和移调（transposition）：

| 乐器 | 音域（近似） | 移调乐器 |
|------|-------------|----------|
| Piano | A0–C8 (A1–C9 实际) | No |
| Violin | G3–E7 | No |
| Cello | C2–C6 | No |
| Flute | C4–C7 | No |
| Bb Trumpet | F#3–D6 | Yes (Bb, 下行大二度) |
| Bb Clarinet | E3–C7 | Yes (Bb, 下行大二度) |
| Horn in F | B1–F5 | Yes (F, 上行纯五度) |

移调乐器（transposing instruments）的记谱音高 ≠ 实际演奏音高，是生成多轨 MIDI 时常见的"错位"来源。

### 6.3 Frequency Bands and Musical Content

不同频率带对应不同的音乐元素：

| 频率带 | 对应元素 | 重要性 |
|--------|----------|--------|
| Sub-bass (20–60 Hz) | 极低音、触觉感 | 氛围 |
| Bass (60–250 Hz) | 根音、低音线条 | 和声基础 |
| Low-mid (250–500 Hz) | 中低音、人声温暖感 | 临场感 |
| Mid (500 Hz–2 kHz) | 人声可懂度、旋律清晰 | **最关键** |
| Upper-mid (2–4 kHz) | 谐波清晰度、齿音 | 临场感 |
| Presence (4–6 kHz) | 空气感、可懂度 | 清晰度 |
| Brilliance (6–20 kHz) | 泛音细节、空间感 | 光泽 |

**对 AI 的意义**：频率带是 MIR 频谱特征（MFCC, log-mel spectrogram）的设计依据，也是 source separation 的目标带（vocals, bass, drums, other）。Edge deployment 中移除 20 kHz 以上的频段可无损节省计算。

---

## 7. Timbre and Orchestration 音色与配器

### 7.1 Timbre 音色

音色（timbre）是区分相同音高/响度下不同音源的声音特征。物理上由谐波结构（harmonic content）、起音（attack）特征和噪声成分决定。

**Attack-Decay-Sustain-Release (ADSR) 包络**：
- **Attack（起音）**：从静音到峰值的时间。短 attack = 打击感（percussive），长 attack = 弦乐/管乐（sustained）
- **Decay（衰减）**：从峰值到 sustain 水平的时间
- **Sustain（持续）**：长音的稳定阶段
- **Release（释音）**：按键释放后的衰减时间

ADSR 包络是区分音色的核心维度，也是声音设计的基本工具。

### 7.2 Orchestration Principles

配器基本原则：
- **音域分配**：避免所有声部挤在同一频率带（midrange congestion）
- **运动方向**：上行旋律配下行低音（opposite motion）创造张弛感
- **音色对比**：在乐句之间切换音色组合保持听感新鲜
- **起音对齐**：不同声部的 attack 时间对齐产生"coincident"的打击感

### 7.3 Psychoacoustics 心理声学

**临界频带（Critical bands）**：人耳约 24 个 Bark 带，频率分辨率在低频（<500 Hz）约 100 Hz，高频（>4 kHz）约 1/3 octave。

**掩蔽效应（Masking）**：
- **同时掩蔽**：强音附近的弱音不可闻
- **预掩蔽/后掩蔽**：强音开始前约 2–5 ms、结束后约 50–150 ms 内弱音被掩蔽

掩蔽效应是 MP3 等有损压缩和 source separation 算法的心理声学基础。

**对 AI 的意义**：心理声学模型是 MIR 特征提取（如 Mel-scale filter bank）和 evaluation（PEMO-Q, ViSQOL 等感知指标）的理论基础。

---

## 8. Musical Form and Structure 曲式与结构

### 8.1 Common Forms

| 曲式 | 结构 | 常见于 |
|------|------|--------|
| Binary (二段) | A-B | 巴洛克舞曲 |
| Ternary (三段) | A-B-A | 咏叹调、器乐慢板 |
| Verse-Chorus (主副歌) | V-C-V-C-B-C | 流行歌曲 |
| AABA (32-bar) | A-A-B-A | 大乐队时代、爵士标准曲 |
| Rondo (回旋) | A-B-A-C-A | 古典奏鸣曲终曲 |
| Sonata (奏鸣) | 呈示-展开-再现 | 古典第一乐章 |

### 8.2 Phrase and Cadence Patterns

流行/摇滚音乐的 8/16-bar 乐句结构：
```
| Intro | V1  | V2  | Chorus  | V3  | V4  | Chorus  | Bridge  | Chorus  | Outro |
```
每个 section 的终止式类型指示其"完成度"：Chorus 以 PAC 结束（完整），Verse 以 HC 结束（开放等待 Chorus），Bridge 以 deceptive cadence 结尾（制造转合部的对比）。

**对 AI 的意义**：Section 标注（labeling）是音乐结构分析（structure analysis）的标准任务。生成模型如果能隐含学习 section-level 结构，其输出的"完整感"会大幅提升。Chord-level 标注 → section-level 标注的层级化是当前 research 方向。

---

## 9. Notation and Encoding in AI Systems AI 系统中的记谱与编码

### 9.1 Common Pitfalls

将乐理映射到 tokenization 时的常见问题：
- **等音处理不一致**：C# 和 Db 在不同系统中可能被编码为不同 token
- **移调乐器轨道的调性偏移**：多轨 MIDI 中移调乐器需要调性适配
- **节奏精确度**：人工演奏的微小时序偏移（swing, groove）在量化时丢失
- **多轨同步**：不同轨道的 tempo/调性变化需要全局一致性

### 9.2 Relevant Representations Revisited

结合乐理视角重新审视常用表示：

| 表示 | 乐理映射 | 优势 | 局限 |
|------|----------|------|------|
| MIDI-Like | Note On/Off + Time Shift | 直接映射事件模型 | 无节拍结构 |
| REMI | + Bar, Position, Tempo, Chord | 显式节拍结构 | Chord token 需要预计算 |
| Compound Word | 多属性复合 token | 并行属性，表达力强 | 需要属性对齐策略 |
| Piano Roll | pitch × time 矩阵 | 空间化，适合 CNN | 丢失事件结构 |
| ABC Notation | 文本记谱 | 人类可读，适合 LM | 仅限简单织体 |

**Chord token 的乐理挑战**：REM 系统的 chord token 基于预计算和弦标注（如 madmom, librosa 的 chord estimation），标注质量直接影响训练数据质量。和弦估计（chord estimation）本身是 MIR 中误差率较高的任务（尤其复杂织体），形成"垃圾进垃圾出"循环。

### 9.3 Key Detection 调性检测

调性检测（key detection）的目标是推断音乐的调（key, 如 C major, A minor）。方法：
- **Krumhansl-Schmuckler key-finding algorithm**（1982）：基于音高类和音程的相关系数匹配调性模板
- **Binary template profiles**：24 个调性模板（12 大调 + 12 小调）与输入的音高类直方图做相关性
- **Machine learning approaches**：CNN/CRNN 端到端调性分类，利用时频上下文

调性检测准确率在简单织体（钢琴独奏）约 90%+，在复杂织体（管弦乐、多乐器混音）下降到 60–70%。

---

## 10. Summary and Further Reading 总结与延伸

### 核心乐理 → AI 研究映射

| 乐理概念 | AI 应用场景 |
|----------|-------------|
| 12-TET / Pitch class | Tokenization 设计、音高类向量 |
| 音程 / 三全音 | 旋律相似性、MIR 特征、张力建模 |
| 调式 / 调性 | Controllable generation、风格控制 |
| 和弦功能 (T-S-D) | Chord condition、和声评估 |
| 终止式 | Section 标注、结构评估 |
| 节奏 / Swing | 节奏表示、groove modeling |
| ADSR / 音色 | Source separation、音色迁移 |
| 曲式 | Section-level 结构生成 |

### 延伸阅读

**系统教材**：
- Piston, W. *Harmony* (revised 1987) — 功能和声经典教材
- Kostka, S. & Payne, D. *Tonal Harmony* (8th ed.) — 现代和声学标准教材
- Aldwell, E. & Cadwallader, A. *Harmony and Voice Leading* (4th ed.) — 强调声部进行的和声学

**面向计算/音乐信息检索**：
- Downie, J.S. *Music Information Retrieval* (Annual Review of Information Science and Technology, 2004) — MIR 领域综述
- Müller, M. *Fundamentals of Music Processing* (2nd ed., Springer 2021) — 计算音乐学教材，含 MIR 算法
- McKinney, M.F. *Music Information Retrieval: A Survey of the Field* (2009) — 早期综述

**认知/心理声学**：
- Zbikowski, L.M. *Conceptualizing Music* (Oxford 2002) — 音乐认知与计算模型的桥梁
- Bregman, A.S. *Auditory Scene Analysis* (MIT 1990) — 听觉场景分析，source separation 的认知基础
- Krumhansl, C.L. *Cognitive Foundations of Musical Pitch* (Oxford 1990) — 音高感知的认知科学

---

> 本文档为 AI 音乐研究者编写，侧重乐理概念在计算系统中的可操作映射。乐理体系本身博大精深，本文选取与 MIR、generation、foundation models 直接相关的部分。
