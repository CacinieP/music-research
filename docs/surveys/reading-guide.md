# Survey Reading Guide: Verified Literature in AI Music

[中文版](reading-guide-zh.md)

This guide links identifiable surveys and tutorials, checked against author, arXiv, publisher, or project pages on 2026-09-19. Years denote the linked preprint's first submission unless another publication is explicitly named. Scope summaries describe the cited source; suggested uses are this repository's reading advice. The list is selective and is not a claim to cover every recent paper.

## 1. Deep-learning generation: a conceptual baseline

**Jean-Pierre Briot, Gaëtan Hadjeres, and François-David Pachet, *Deep Learning Techniques for Music Generation — A Survey* (2017 preprint, subsequently revised).** [arXiv:1709.01620](https://arxiv.org/abs/1709.01620)

The survey organizes systems by objective, representation, architecture, challenge, and generation strategy. Its examples help distinguish generating scores, performances, and audio, and explain why representation and control matter alongside network architecture.

**Read for:** a framework for comparing approaches. Its historical models should not be presented as the 2026 state of the art. The linked work is not an unidentified “A Survey on Deep Learning for Music Generation (2023).”

**Related notes:** [generation](../notes/music-generation.md), [theory](../notes/music-theory-fundamentals.md), [evaluation](../notes/music-evaluation.md).

## 2. Generation across representation levels

**Shulei Ji, Jing Luo, and Xinyu Yang, *A Comprehensive Survey on Deep Music Generation: Multi-level Representations, Algorithms, Evaluations, and Future Directions* (2020).** [arXiv:2011.06801](https://arxiv.org/abs/2011.06801)

This draft surveys score, performance, and audio generation and discusses tasks, representations, datasets, methods, and evaluation across those levels.

**Read for:** separating “compose notes,” “perform a score,” and “synthesize sound” as research objectives. Supplement it with later original papers for codec language models, diffusion, flow matching, and full-song generation. Publication dates and specific claims should be checked in each original source rather than inferred from a survey title.

**Related notes:** [generation](../notes/music-generation.md), [audio engineering](../notes/audio-engineering.md).

## 3. Audio tokenization

**Pooneh Mousavi et al., *Discrete Audio Tokens: More Than a Survey!* (2025).** [arXiv:2506.10274](https://arxiv.org/abs/2506.10274)

This work combines a tokenizer survey with benchmarking across speech, music, and general audio. It classifies approaches by architecture, quantization, training, streamability, and application, and evaluates reconstruction, downstream tasks, and acoustic language modeling.

**Read for:** comparing tokenizers beyond reconstruction alone. Check each benchmark's sample rate, bitrate, domain, and model version before transferring conclusions to music. Discrete codec tokens are central to systems such as MusicGen, while AudioLDM uses continuous latent diffusion; not all audio generators require discrete tokenization.

**Related notes:** [audio engineering](../notes/audio-engineering.md), [generation](../notes/music-generation.md).

## 4. Music foundation models

**Yinghao Ma et al., *Foundation Models for Music: A Survey* (2024).** [arXiv:2408.14340](https://arxiv.org/abs/2408.14340)

The survey covers representation learning, generation, and multimodal learning. It discusses pretraining, architectures, tokenization, fine-tuning, controllability, music agents, data, evaluation, and ethical questions. Its scope therefore extends well beyond music-understanding encoders.

**Read for:** comparing foundation-model roles and training paradigms. Check the paper's actual version and original model publications before attributing later systems to it. Names such as “MERT v2” or “MusicFM v2” should not be inferred from a generic statement about future updates.

**Related notes:** [MIR](../notes/music-understanding-mir.md), [generation](../notes/music-generation.md), [style](../notes/music-styles.md).

## 5. Singing voice synthesis

**Yin-Ping Cho, Fu-Rong Yang, Yung-Chuan Chang, Ching-Ting Cheng, Xiao-Han Wang, and Yi-Wen Liu, *A Survey on Recent Deep Learning-driven Singing Voice Synthesis Systems* (2021).** [arXiv:2110.02511](https://arxiv.org/abs/2110.02511)

This survey reviews neural systems that synthesize singing from scores and lyrics, comparing their architectures, strengths, and limitations.

**Read for:** an identifiable historical SVS introduction. It is not a 2024 survey, and it does not establish the current prevalence of diffusion or cover every later system. Pair it with [DiffSinger](https://arxiv.org/abs/2105.02446) and the original papers for newer systems. Distinguish score-and-lyrics SVS, singing voice conversion, lyrics-to-song generation, and singing-conditioned accompaniment.

**Related note:** [singing synthesis](../notes/music-singing-synthesis.md).

## 6. Evaluation

**Faria Binte Kader and Santu Karmaker, *A Survey on Evaluation Metrics for Music Generation* (2025).** [arXiv:2509.00051](https://arxiv.org/abs/2509.00051)

The survey develops a taxonomy for symbolic and audio music evaluation and discusses limitations involving human perception, cross-cultural bias, and inconsistent protocols.

**Read for:** organizing an evaluation plan across different quality dimensions. An overview does not make an individual metric valid for every task. Read the original metric papers for definitions, reference requirements, and implementation details; report listener protocol and uncertainty alongside automatic scores.

**Related note:** [evaluation](../notes/music-evaluation.md).

## 7. Source separation: a practical tutorial

**The author-maintained *Open-Source Tools & Data for Music Source Separation* tutorial.** [Tutorial introduction](https://source-separation.github.io/tutorial/intro/src_sep_101.html)

This is a tutorial, not a verified paper called “Music Source Separation: A Brief Overview (2023).” It introduces the task and the open-source data/tool ecosystem.

**Read for:** mixture/source definitions and practical orientation. Follow original model papers for architecture comparisons and dataset-specific results. A single SDR figure cannot establish a universal performance ceiling, and a waveform model is not inherently better than every spectrogram model. Specify stem set, evaluation implementation, aggregation, and training data.

**Related notes:** [MIR](../notes/music-understanding-mir.md), [audio engineering](../notes/audio-engineering.md).

## 8. Style, datasets, and cultural coverage

These are original studies and a research project, rather than invented generic survey titles:

- **Joan Serrà et al. (2012), [*Measuring the evolution of contemporary western popular music*](https://arxiv.org/abs/1205.5651).** An example of corpus-based pitch, timbre, and loudness analysis; interpret its findings within its data and measurement scope.
- **Bob L. Sturm (2013), [*The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use*](https://arxiv.org/abs/1306.1461).** A concrete demonstration of how dataset faults affect genre-classification evaluation.
- **[CompMusic](https://compmusic.upf.edu/).** A project emphasizing culturally specific analysis and dedicated corpora, useful for examining assumptions in representations and labels.

**Related notes:** [style](../notes/music-styles.md), [MIR](../notes/music-understanding-mir.md).

## 9. Suggested reading paths

| Goal | Suggested sequence |
|---|---|
| New to music AI | Theory note → Briot survey → Ji survey → one original system paper |
| Audio generation | Ji survey → tokenizer survey → a generation paper → evaluation survey |
| MIR / transfer learning | Foundation-model survey → model paper → dataset/protocol documentation → GTZAN audit |
| Singing synthesis | Cho survey → DiffSinger → newer task-specific papers → evaluation note |
| Style / cross-cultural work | Style note → CompMusic → corpus-specific analysis and evaluation |

These are editorial learning paths, not rankings by citation count or completeness.

## 10. Citation and maintenance boundaries

The previous guide included several entries without enough bibliographic information to identify them, including “AI and Music: A Comprehensive Survey” attributed to Brée and a “McKinney MIR Survey (2009).” They are not retained as established references. This does not prove no related publication exists; it means the supplied title/author/year combination was not verified.

For additions, record an exact title, authors, year/version, stable source link, and a short scope summary supported by the source. A model paper, product page, tutorial, benchmark, and survey should be labeled according to what each actually is. This repository offers bilingual notes and cross-links; it does not replace reading the original methods and experimental limitations.

See the [verified reference list](../../references/README.md) for original model papers and [audio engineering](../notes/audio-engineering.md), [generation](../notes/music-generation.md), [MIR](../notes/music-understanding-mir.md), [evaluation](../notes/music-evaluation.md), [singing synthesis](../notes/music-singing-synthesis.md), [style](../notes/music-styles.md), and [theory](../notes/music-theory-fundamentals.md) for topic notes.
