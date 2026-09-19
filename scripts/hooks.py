"""Use Unicode section IDs and a bilingual topic navigation."""
from pathlib import Path
from pymdownx.slugs import slugify

TOPICS = [('music-theory-fundamentals','乐理基础'),('audio-engineering','音频工程'),
          ('music-understanding-mir','音乐理解 · MIR'),('music-generation','音乐生成'),
          ('music-evaluation','音乐评测'),('music-singing-synthesis','歌声合成'),
          ('music-styles','风格与流派')]

def on_config(config):
    config.mdx_configs.setdefault('toc',{})['slugify'] = slugify(case='lower')
    notes=[{'概览':'docs/notes/index.md'}]
    for slug,label in TOPICS:
        notes.append({label:[{'中文':f'docs/notes/{slug}-zh.md'},{'English':f'docs/notes/{slug}.md'}]})
    for slug,label in [('edge-deployment','端侧部署'),('model-reproduction-guide','复现指南'),('music-copyright-compliance','版权与合规')]:
        notes.append({label:f'docs/notes/{slug}.md'})
    config.nav=[{'开始阅读':'index.md'},{'笔记目录':'docs/index.md'},{'研究笔记':notes},
                {'文献导读':[{'概览':'docs/surveys/index.md'},{'中文':'docs/surveys/reading-guide-zh.md'},{'English':'docs/surveys/reading-guide.md'}]},
                {'参考文献':[{'中文':'references/README-zh.md'},{'English':'references/index.md'}]},
                {'实践':[{'代码示例':'docs/cookbook/recipes.md'},{'示例目录':'docs/cookbook/index.md'},{'代码与算力':'src/index.md'},{'数据集':'datasets/index.md'}]},
                {'研究随笔':[{'概览':'docs/thoughts/index.md'},{'随笔':'docs/thoughts/personal-thoughts.md'}]}]
    config.nav.append({'构建与发布':'PUBLISHING.md'})
    if (Path(config.docs_dir)/'docs/review-2026-09-19.md').exists():
        config.nav.append({'审校记录':'docs/review-2026-09-19.md'})
    return config
