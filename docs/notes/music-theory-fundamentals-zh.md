# AI 音乐研究乐理基础

面向 MIR、音乐生成和音乐基础模型研究者的乐理核心知识。聚焦将抽象乐理映射到计算系统的可操作视角：tokenization 设计、表示选择、评估指标、可控生成。

---

## 1. 音高（Pitch）

### 1.1 频率与十二平均律

人类感知的音高由声波基频（fundamental frequency）决定。国际标准音高 A4 = 440 Hz（ISO 16）。

西方音乐采用 **十二平均律（12-TET, Twelve-Tone Equal Temperament）**：一个八度（octave）被等分为 12 个半音（semitones），每个半音的频率比为 2^(1/12) ≈ 1.0595。

$$
f_n = f_0 \cdot 2^{n/12}
$$

12-TET 的优势在于所有调性（keys）的等距性，使转调（transposition）和和声计算简单。代价是纯律（just intonation）的协和性被轻微妥协。

**对 AI 的意义**：12-TET 的音高离散性（12 个音高类 pitch classes）是大多数表示设计的基础。MIDI 编号直接用整数映射 12-TET 半音。

### 1.2 音高类与八度

- **音高类（Pitch class）**：一个八度内的 12 个半音之一（C, C#, D, ..., B），不区分八度。
- **八度（Octave）**：频率翻倍的区间，国际标准从 C1 到 C10。MIDI 0–127 覆盖 C-1 到 G9。
- **科学记谱（Scientific pitch notation）**：如 C4 = middle C = MIDI 60。

音高类在音级分析（pitch-class set analysis）和音高类向量（pitch-class vector, 12 维直方图）中被大量使用。

### 1.3 等音（Enharmonic Equivalence）

等音：不同记谱但在 12-TET 下相同频率的音高，如 C# = Db, D# = Eb, F# = Gb, G# = Ab, A# = Bb。

等音在 AI 表示中通常被合并处理（如 pitch-class vector），但在音乐理解任务中区分等音是必要的（调性分析、和弦识别）。

---

## 2. 音程与调式（Intervals and Scales）

### 2.1 音程（Intervals）

音程是两个音高之间的距离，以半音数度量：

| 半音数 | 音程名称 | 缩写 |
|--------|----------|------|
| 0 | 纯一度 | P1 |
| 1 | 小二度 | m2 |
| 2 | 大二度 | M2 |
| 3 | 小三度 | m3 |
| 4 | 大三度 | M3 |
| 5 | 纯四度 | P4 |
| 6 | 增四度 / 减五度 | A4 / d5（三全音） |
| 7 | 纯五度 | P5 |
| 8 | 小六度 | m6 |
| 9 | 大六度 | M6 |
| 10 | 小七度 | m7 |
| 11 | 大七度 | M7 |
| 12 | 纯八度 | P8 |

**三全音（tritone, A4/d5）**：6 个半音，中世纪被称为"音乐中的魔鬼"（diabolus in musica），因其高度不协和性。在七和弦中分隔三音与七音，是爵士和声的核心张力来源。

**对 AI 的意义**：音程是旋律相似性（melodic similarity）和音程类向量（interval-class vector）分析的基础。许多 MIR 特征提取器直接输出音程分布。

### 2.2 自然音阶（Diatonic Scales）

**大调（Major）音阶**：全-全-半-全-全-全-半 模式（W-W-H-W-W-W-H），从主音（tonic）开始的音程序列为 [2,2,1,2,2,2,1]。

**小调（Minor）音阶**有三种形式：
- **自然小调（Natural minor）**：[2,1,2,2,1,2,2]
- **和声小调（Harmonic minor）**：自然小调升高七音
- **旋律小调（Melodic minor）**：上行升高六、七音，下行还原

**调式（modes）**是大调音阶从不同级开始的七种变体：Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian。特征音决定色彩：Lydian 的 #4 明亮梦幻，Dorian 的 b3+b7 温暖忧郁，Phrygian 的 b2 西班牙式暗色。

### 2.3 五声与特殊音阶

- **五声音阶（Pentatonic）**：5 个音，去除大调音阶的四音与七音。跨文化普遍存在（中国、日本、苏格兰、非洲），无半音使其容易即兴。
- **全音阶（Whole-tone）**：6 个全音，德彪西印象派核心，声部解决方向模糊。
- **减音阶（Diminished）**：交替全半模式，高度对称。
- **布鲁斯音阶（Blues）**：五声音阶 + 减五音（blue note），布鲁斯表达核心。

---

## 3. 和声（Harmony）

### 3.1 三和弦（Triads）

| 品质 | 三音 | 五音 | 缩写 | 色彩 |
|------|------|------|------|------|
| 大三（Major） | M3 | P5 | △ | 明亮、开放 |
| 小三（Minor） | m3 | P5 | m | 柔和、暗淡 |
| 减三（Diminished） | m3 | d5 | ° | 紧张、悬停 |
| 增三（Augmented） | M3 | A5 | + | 飘渺、无归属 |

### 3.2 七和弦（Seventh Chords）

| 类型 | 三音 | 五音 | 七音 | 符号 | 常见功能 |
|------|------|------|------|------|----------|
| 大七（Maj7） | M3 | P5 | M7 | △7 | I（主和弦） |
| 小七（m7） | m3 | P5 | m7 | m7 | ii, vi |
| 属七（Dominant 7） | M3 | P5 | m7 | 7 | V，最具张力 |
| 半减七（m7b5） | m3 | d5 | m7 | ø7 | vii°（导七） |

爵士扩展七和弦加入 9th (M2), 11th (P4), 13th (M6) 作为附加音（tensions）。

### 3.3 功能和声（Functional Harmony）

功能和声将每个和弦映射到调性中的角色：

```
T    S    D    T
|    |    |    |
I → IV → V → I
i → iv → V → i
```

- **T（Tonic, 主功能）**：I, vi, iii — 稳定
- **S（Subdominant, 下属功能）**：IV, ii — 过渡
- **D（Dominant, 属功能）**：V, vii° — 最大张力，必须解决到 T

属七和弦（V7）必须解决到主和弦（I）是古典和声的"铁律"，其内部三全音的张力驱动解决。

### 3.4 终止式（Cadences）

| 终止式 | 进行 | 特征 |
|--------|------|------|
| 完美正格终止（PAC） | V→I | 声部最稳定的终止 |
| 变格终止（Plagal） | IV→I | "阿门"终止，柔和 |
| 半终止（HC） | 结束于 V | 开放 |
| 伪终止（Deceptive） | V→vi | "欺骗"预期，流行标志性手法 |

### 3.5 声部进行（Voice Leading）

黄金规则：
- 避免平行五度/八度
- 共同音保留（common tone retention）
- 引导音（leading tone）半音上行解决到主音
- 三全音反向半音运动

---

## 4. 节奏与节拍（Rhythm and Meter）

### 4.1 时值体系

| 音符 | 相对全音符 | 与上一级 |
|------|-----------|----------|
| 全音符 | 1 | 基准 |
| 二分 | 1/2 | ÷2 |
| 四分 | 1/4 | ÷2 |
| 八分 | 1/8 | ÷2 |
| 十六分 | 1/16 | ÷2 |

时值的指数衰减天然适合对数时间量化。

### 4.2 拍号（Time Signatures）

- **简单拍号**：X/4（2/4, 3/4, 4/4）。上方=每小节拍数，下方=每拍时值
- **复合拍号**：X/8（6/8, 9/8, 12/8）。上方=三连音组数
- **混合拍号**：5/4（2+3 或 3+2）、7/8 等不对称组合

4/4 拍强拍弱拍模式 [S, w, S, w]；3/4 拍 [S, w, w]。

### 4.3 速度（Tempo, BPM）

| 术语 | BPM | 特征 |
|------|-----|------|
| Largo | 40–60 | 广板 |
| Adagio | 60–72 | 慢板 |
| Andante | 72–88 | 行板 |
| Moderato | 88–108 | 中板 |
| Allegro | 108–132 | 快板 |
| Vivace | 132–168 | 活泼 |
| Presto | 168–200 | 急板 |

### 4.4 三连音与切分

**三连音（Tuplet）**：在标准时值内安排非标准数量的音符，如三连音是 3 个音占 2 个标准时值。三连音的"不协和节奏感"（metric dissonance）是推动音乐前进的关键。

**切分（Syncopation）**：强拍声部移到弱拍，打破预期。放克、雷鬼、嘻哈节奏的核心。

**Swing（摇摆）**：八分音符变为"长-短"组合（≈ 2:1 到 7:3），由 swing feel 而非精确标记决定。

---

## 5. 旋律（Melody）

### 5.1 旋律轮廓（Contour）

| 类型 | 描述 |
|------|------|
| 上行（Ascending） | 总体音高上升 |
| 下行（Descending） | 总体音高下降 |
| 拱形（Arch） | 上行后下行，最常见 |
| 波动（Undulating） | 交替上下 |
| 静止（Static） | 音高变化小 |

**音程偏倚（interval bias）**：小音程（≤3 半音）出现频率远高于大音程——Zipf-like distribution。

### 5.2 动机与乐句

**动机（Motif）**：最短的"音乐词汇"，2–8 个音符的辨识模式。通过模进、倒影、逆行、扩大、缩小发展。

**乐句（Phrase）**：音乐的"句子"，通常 4–8 小节，以终止式结束。

### 5.3 张力与释放

Meyer 的期望理论：张力来自"假设-确认/否认"循环。预测越确定，被违背时张力越强。

旋律熵（melodic entropy）直接量化生成旋律的"无聊度"——熵太低=可预测=无聊，熵太高=无规律=噪声。

---

## 6. 织体与配器（Texture and Arrangement）

### 6.1 织体类型

| 织体 | 描述 | 例子 |
|------|------|------|
| 单声（Monophonic） | 单一旋律 | 格里高利圣咏 |
| 主声（Homophonic） | 旋律+伴奏 | 流行歌曲 |
| 复调（Polyphonic） | 多条独立旋律 | 赋格、卡农 |
| 支声（Heterophonic） | 同一旋律变体同时 | 中国民间音乐 |

### 6.2 频率带与音乐元素

| 频率带 | 对应元素 |
|--------|----------|
| Sub-bass (20–60 Hz) | 极低音、触觉感 |
| Bass (60–250 Hz) | 根音、低音线 |
| Low-mid (250–500 Hz) | 中低音、温暖感 |
| Mid (500 Hz–2 kHz) | 人声/旋律，**最关键** |
| Upper-mid (2–4 kHz) | 谐波清晰度 |
| Presence (4–6 kHz) | 空气感、可懂度 |
| Brilliance (6–20 kHz) | 泛音细节、空间感 |

### 6.3 音色与 ADSR

ADSR 包络（Attack-Decay-Sustain-Release）是区分音色的核心维度。短 attack = 打击感（打击乐），长 attack = 弦乐/管乐（持续音）。

---

## 7. 曲式（Musical Form）

| 曲式 | 结构 | 常见于 |
|------|------|--------|
| 二段（Binary） | A-B | 巴洛克舞曲 |
| 三段（Ternary） | A-B-A | 咏叹调 |
| 主副歌（V-C） | V-C-V-C-B-C | 流行歌曲 |
| AABA（32-bar） | A-A-B-A | 爵士标准曲 |

流行/摇滚 8/16-bar 乐句结构：Intro → Verse → Chorus → Bridge → Chorus → Outro。Chorus 以 PAC 结束（完整），Verse 以 HC 结束（开放等待 Chorus）。

---

## 8. 乐理 → AI 研究映射速查

| 乐理概念 | AI 应用场景 |
|----------|-------------|
| 12-TET / 音高类 | Tokenization 设计、音高类向量 |
| 音程 / 三全音 | 旋律相似性、MIR 特征、张力建模 |
| 调式 / 调性 | 可控生成、风格控制 |
| 和弦功能 (T-S-D) | Chord condition、和声评估 |
| 终止式 | Section 标注、结构评估 |
| 节奏 / Swing | 节奏表示、groove modeling |
| ADSR / 音色 | Source separation、音色迁移 |
| 曲式 | Section-level 结构生成 |

---

> 本文档为 AI 音乐研究者编写，侧重乐理概念在计算系统中的可操作映射。乐理体系本身博大精深，本文选取与 MIR、generation、foundation models 直接相关的部分。
