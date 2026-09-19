"""Export the validated music reading site into GitHub Wiki Markdown."""
from __future__ import annotations
import argparse
import hashlib
import json
import posixpath
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
REPOS = ('music-research',)


def anchor_name(heading):
    """ASCII anchors survive both the GFM API and the live Wiki sanitizer."""
    return 'section-' + hashlib.sha256(heading.encode('utf-8')).hexdigest()[:20]


def page_name(source):
    if source == Path('index.md'):
        return 'Home'
    parts = list(source.with_suffix('').parts)
    if parts[-1] == 'index':
        parts.pop()
    name = '--'.join(parts)
    if re.search(r'[\\/:*?"<>|]', name):
        raise ValueError(f'Unsupported wiki page name: {name}')
    return name


def catalog(root):
    result = {}
    for source in sorted((root / '.site-docs').rglob('*.md')):
        rel = source.relative_to(root / '.site-docs')
        output = rel.parent / 'index.html' if rel.name == 'index.md' else rel.with_suffix('') / 'index.html'
        rendered = root / 'site' / output
        if not rendered.exists():
            raise ValueError(f'Build the reading site first: {rendered}')
        soup = BeautifulSoup(rendered.read_text(), 'html.parser')
        article = soup.select_one('article')
        headings = [str(h['id']) for h in article.select('h1[id],h2[id],h3[id],h4[id],h5[id],h6[id]')]
        anchors = [anchor_name(heading) for heading in headings]
        if len(anchors) != len(set(anchors)):
            raise ValueError(f'Wiki anchor collision: {rel}')
        result[rel] = {'name': page_name(rel), 'headings': headings, 'source': source}
    names = [item['name'].casefold() for item in result.values()]
    if len(names) != len(set(names)):
        raise ValueError('Wiki filename collision')
    if not result:
        raise ValueError('Prepare and build the reading site first')
    return result


def wiki_url(repo, item, fragment=''):
    suffix = ''
    if fragment:
        fragment = unquote(fragment)
        if fragment not in item['headings']:
            raise ValueError(f'Missing wiki anchor: {item["name"]}#{fragment}')
        suffix = '#' + anchor_name(fragment)
    return f'https://github.com/CacinieP/{repo}/wiki/{quote(item["name"], safe="-")}' + suffix


def rewrite_url(url, source, repo, catalogs):
    parsed = urlsplit(url)
    target_repo = repo
    if parsed.netloc == 'caciniep.github.io':
        parts = unquote(parsed.path).strip('/').split('/')
        if not parts or parts[0] not in REPOS:
            return url
        target_repo = parts[0]
        path = '/'.join(parts[1:])
        target = Path(path) / 'index.md' if not path or parsed.path.endswith('/') else Path(path)
        # Generated chapter pages end in index.md; lesson routes do not.
        if target not in catalogs[target_repo] and target.name == 'index.md':
            target = target.parent.with_suffix('.md')
    elif parsed.scheme or parsed.netloc:
        return url
    elif not parsed.path:
        target = source
    else:
        target = Path(posixpath.normpath(posixpath.join(source.parent.as_posix(), unquote(parsed.path))))
    if target in catalogs[target_repo]:
        return wiki_url(target_repo, catalogs[target_repo][target], parsed.fragment)
    if target.suffix == '.md' or parsed.fragment:
        raise ValueError(f'Unresolved wiki link: {source}: {url}')
    return f'https://github.com/CacinieP/{repo}/blob/master/{quote(target.as_posix(), safe="/")}'


def protect_inline_math(match):
    # Backtick math protects TeX from GFM; macros protect table delimiters.
    tex = match[1].replace(r'\|', r'\Vert{}')
    tex = re.sub(r'(?<!\\)\|', lambda m: r'\vert{}', tex)
    return '$`' + tex.strip() + '`$'


def protect_wiki_brackets(text):
    """Gollum treats literal [[...]] as a Wiki link even when GFM does not."""
    # Inline formulas have already become $`...`$; preserve their TeX just as
    # ordinary inline code. Display formulas and fenced code return earlier.
    chunks = re.split(r'(`+[^`]*`+)', text)
    for i in range(0, len(chunks), 2):
        chunks[i] = chunks[i].replace('[[', '&#91;&#91;').replace(']]', '&#93;&#93;')
    return ''.join(chunks)


def convert(text, source, repo, catalogs):
    headings = iter(catalogs[repo][source]['headings'])
    result = []; fence = None; display_math = False
    for line in text.splitlines(keepends=True):
        if fence is None and line.strip() == '$$':
            result.append('```\n\n' if display_math else '\n```math\n')
            display_math = not display_math
            continue
        if display_math:
            # GFM adds a Markdown hard-break escape to a bare trailing \\.
            # A separating space preserves the TeX row separator verbatim.
            if line.rstrip('\r\n').endswith('\\\\'):
                line = line.rstrip('\r\n') + ' \n'
            result.append(line)
            continue
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            mark = match[1]
            if fence is None: fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence): fence = None
            result.append(line); continue
        if fence:
            result.append(line); continue
        if re.match(r'^#{1,6}\s', line):
            anchor = next(headings, None)
            if anchor is None: raise ValueError(f'Heading mismatch: {source}')
            result.append(f'<a name="{anchor_name(anchor)}"></a>\n\n')
        chunks = re.split(r'(`+[^`]*`+)', line)
        for i in range(0, len(chunks), 2):
            chunks[i] = re.sub(r'(\]\()([^\s)]+)(\))', lambda m: m[1] + rewrite_url(m[2], source, repo, catalogs) + m[3], chunks[i])
            chunks[i] = re.sub(r'(?<!\\)\$([^$\n]+)(?<!\\)\$', protect_inline_math, chunks[i])
            chunks[i] = protect_wiki_brackets(chunks[i])
        result.append(''.join(chunks))
    if display_math: raise ValueError(f'Unclosed display math: {source}')
    if next(headings, None) is not None:
        raise ValueError(f'Unmatched rendered heading: {source}')
    return ''.join(result)


def export(root):
    import yaml
    repo = yaml.safe_load((root / 'mkdocs.yml').read_text())['repo_name'].split('/')[-1]
    catalogs = {repo: catalog(root)}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    destination = root / '.wiki-docs'
    if destination.exists(): shutil.rmtree(destination)
    destination.mkdir()
    manifest = {'repository': repo, 'source_revision': revision, 'pages': {}}
    for source, item in catalogs[repo].items():
        body = convert(item['source'].read_text(), source, repo, catalogs)
        reading = f'https://caciniep.github.io/{repo}/'
        intro = f'> [在线阅读版]({reading}) · [源仓库](https://github.com/CacinieP/{repo}) · [Wiki 首页](https://github.com/CacinieP/{repo}/wiki/Home)\n\n'
        target = destination / (item['name'] + '.md')
        target.write_text(intro + body)
        manifest['pages'][target.name] = {'source': source.as_posix(), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'anchors': [anchor_name(x) for x in item['headings']]}
    links = [f'- [首页](https://github.com/CacinieP/{repo}/wiki/Home)']
    if Path('concept-index.md') in catalogs[repo]:
        links.append('- [全局索引](' + wiki_url(repo, catalogs[repo][Path('concept-index.md')]) + ')')
    for source, item in catalogs[repo].items():
        if source.name != 'index.md':
            links.append(f'- [{source.stem}]({wiki_url(repo, item)})')
    links += [f'- [在线阅读版](https://caciniep.github.io/{repo}/)']
    (destination / '_Sidebar.md').write_text('## 阅读导航\n\n' + '\n'.join(links) + '\n')
    (destination / '_Footer.md').write_text(f'内容按 [CC BY 4.0](https://github.com/CacinieP/{repo}/blob/master/LICENSE) 共享。源版本：[{revision[:7]}](https://github.com/CacinieP/{repo}/commit/{revision})。请在源仓库修改正文，再重新生成 Wiki。\n')
    (destination / '.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'Prepared {len(manifest["pages"])} Wiki pages plus sidebar and footer for {repo}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    export(ROOT)
