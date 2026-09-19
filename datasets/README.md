# 公开数据集目录

核对日期：2026-09-19。规模均指特定公开版本或论文口径，不代表当前可下载量。**标注/元数据许可不自动覆盖音频、作品或表演权利；公开下载不等于可商用。** 每项链接指向项目或正式发布记录。

## 自动标注、分类与音色

| 数据集 | 规模与内容 | 许可/使用边界 |
|---|---|---|
| [MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) | 约 2.6 万个约 29 秒片段；top-50 是常用评测标签子集，不是全部原始标注 | 核对发布方与 Magnatune 音频条款，不能从托管网页推断统一商用许可 |
| [MTG-Jamendo](https://mtg.github.io/mtg-jamendo-dataset/) | 清洗版本 55,609 首、195 标签；常用划分含 183 标签（流派/乐器/情绪主题） | 元数据 CC BY-NC-SA 4.0；官网限定非商业研究/学术用途，其他用途须 Jamendo 事先书面授权；仍须核对音频逐曲条款 |
| [FMA](https://github.com/mdeff/fma) | 106,574 首；small/medium/large/full 子集规模不同 | 元数据 CC BY 4.0；音频为逐曲不同 CC 条款，部分含 NC/ND/SA |
| [NSynth](https://magenta.tensorflow.org/datasets/nsynth) | 305,979 个四秒单音；16 kHz、1,006 个乐器来源 | CC BY 4.0；单音集合，不是完整多声部歌曲 |
| [AudioSet](https://research.google.com/audioset/download.html) | 约 200 万个十秒 YouTube 片段的事件标注，527 类 | 公开 CSV/特征与视频音频权利分别核对；链接存在性会变化 |

GTZAN 是常用的 1,000 个片段、10 类流派基准，但有重复、标签和划分问题。第三方镜像的许可字段不能代表原始音频权利；使用时记录副本来源并检查艺人/录音泄漏。参见[故障分析论文](https://arxiv.org/abs/1306.1461)。

## 转录与符号音乐

| 数据集 | 规模与文件 | 许可/使用边界 |
|---|---|---|
| [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) | 1,276 场演奏，约 199 小时，WAV + 对齐 MIDI；1,282 是 v2 口径 | CC BY-NC-SA 4.0；保持官方按作品划分 |
| [MusicNet](https://homes.cs.washington.edu/~thickstn/musicnet.html) | 330 个古典录音及音符/乐器时间标注；不是默认提供完整 MIDI 的音频包 | 使用发布版本的许可与标注说明 |
| [Slakh2100](https://zenodo.org/records/4599666) | 2,100 个由 MIDI 合成的多轨混音与音轨/MIDI | CC BY 4.0，注明版本及生成来源；合成数据不等于真实录音域 |
| [POP909](https://github.com/music-x-lab/POP909-Dataset) | 909 首流行钢琴编配的 MIDI 与节拍、和弦等标注 | 公开包以 MIDI/标注为主，不应写成可自由下载原唱 WAV；核对仓库许可和底层作品权利 |
| [Lakh MIDI v0.1](https://colinraffel.com/projects/lmd/) | 176,581 个去重 MIDI，其中 45,129 个匹配 Million Song Dataset | 发布数据 CC BY 4.0；匹配不等于取得商业录音的音频分发权 |

## 源分离

| 数据集 | 规模与内容 | 许可/使用边界 |
|---|---|---|
| [MUSDB18-HQ](https://zenodo.org/records/3338373) | 150 首：100 train / 50 test；44.1 kHz 立体声混音和 vocals/drums/bass/other | 教育用途，商用需权利人明确许可；各来源条款不同，不能统一标为 CC BY-NC 4.0 |
| [MoisesDB](https://github.com/moises-ai/moises-db) | 240 首，细分层级与可变数量音轨 | CC BY-NC-SA 4.0，按正式下载记录核对版本 |
| [MedleyDB](https://medleydb.weebly.com/downloads.html) | v1 为 122 首多轨录音；另有扩展版本 | 核对申请条款，原稿 CC BY 4.0 不正确；常用许可含 NC/SA |
| [URMP](https://labsites.rochester.edu/air/projects/URMP.html) | 44 个室内乐合奏片段；独立声部音视频与乐谱/音高标注 | 研究用途与访问条款以项目页为准，不假定全部标注是 MIDI |

## 描述、情感与结构

| 数据集 | 内容与注意事项 |
|---|---|
| [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) | 5,521 个片段的描述、aspect 列表、YouTube ID 和起止时间；CC BY-SA 4.0 针对所发布数据，原始 YouTube 音频另行核对，不是直接打包 WAV |
| [DEAM](https://cvml.unige.ch/databases/DEAM/) | 情绪的 valence/arousal 标注；按正式版本区分片段与整曲、静态与动态标注，使用前核对申请和许可（本次页面抓取失败） |
| [Emotify](https://www.projects.science.uu.nl/memotion/emotifydata/) | 400 个音乐片段，GEMS 的九种情感标签；不是八类通用情感数据 |
| [Isophonics](https://isophonics.net/content/reference-annotations) | 和弦、节拍、调性、结构等参考标注；对应商业录音通常不随标注提供 |
| [GiantSteps Key](https://github.com/GiantSteps/giantsteps-key-dataset) | 电子舞曲调性标注及音频下载脚本；下载脚本不意味着取得音频再分发许可 |

原稿的 MetaMIDI、AD Pianoforte、SongBench、FakeMusicCaps 等条目缺少能支持其规模、格式、许可的匹配来源，本次未作为已验证下载资源保留。尤其原 SongBench 链接 `2502.19324` 对应其他领域论文，不能作为音乐数据集引用。

## 数据使用与复现

1. 下载前保存项目版本、许可、条款和校验和；区分标注、音频与代码许可。
2. 记录官方 train/validation/test 划分。按作品、艺人、录音版本去重，避免同曲不同切片跨集合。
3. 检查实际可用文件数、坏文件和失效链接；YouTube 型数据集的重建结果通常不会完全相同。
4. 按任务决定采样率与声道：模型输入采样率不等于数据集的原始采样率。
5. 存储预算以下载页为准。例如 MUSDB18-HQ 压缩包约 22.7 GB，不能沿用原稿的约 15 GB。
6. 报告数据用途与授权边界；FMA/Freesound 不是可无条件商用的统一数据池。MERT 与 CLAP 的训练数据也不能都笼统写成 AudioSet。

English summary: Version every dataset and separate metadata licences from audio rights. Availability is not permission. Preserve official splits, document missing files, and avoid claims about unverified mirrors or download packages.
