"""Fail on broken local HTML links, fragments, assets, or unconverted wiki links."""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
from bs4 import BeautifulSoup


def check(root, peer=None):
    errors = []
    pages = {p.resolve(): BeautifulSoup(p.read_text(), 'html.parser') for p in root.rglob('*.html')}
    anchors = {p: {str(el['id']) for el in soup.select('[id]')} for p, soup in pages.items()}
    home = pages.get((root / 'index.html').resolve())
    canonical = home.select_one('link[rel=canonical]') if home else None
    project = urlsplit(canonical['href']).path.strip('/').split('/')[0] if canonical else ''
    manifest = {}
    formulas = []
    for path, soup in pages.items():
        relative = path.relative_to(root.resolve()).as_posix()
        manifest[relative] = sorted(anchors[path])
        for element in soup.select('.arithmatex'):
            tex = element.get_text()
            if tex.startswith(('\\(', '\\[')):
                tex = tex[2:-2]
            formulas.append({'source': relative, 'tex': tex, 'display': element.name == 'div'})
        article = soup.select_one('article')
        if article:
            for el in article.select('code, pre'):
                el.decompose()
            # Compilation success is insufficient: malformed delimiters can
            # leave formulas as plain text instead of creating math elements.
            for el in article.select('.arithmatex'):
                el.decompose()
            if re.search(r'(?<!\\)\$', article.get_text()):
                errors.append(f'{relative}: unrendered math delimiter in article')
            if re.search(r'\[\[[^\[\]\n]+\]\]', article.get_text()):
                errors.append(f'{relative}: unconverted wiki link in article')
        # Use original HTML because code samples may contain actual linked assets.
        soup = BeautifulSoup(path.read_text(), 'html.parser')
        for tag, attribute in [('a', 'href'), ('img', 'src'), ('script', 'src'), ('link', 'href')]:
            for element in soup.find_all(tag):
                href = element.get(attribute)
                if not href:
                    continue
                url = urlsplit(href)
                if url.scheme or url.netloc or href.startswith('//'):
                    if peer and url.netloc == 'caciniep.github.io':
                        parts = unquote(url.path).strip('/').split('/')
                        if parts and parts[0] == peer.parent.name:
                            target = peer.joinpath(*parts[1:])
                            if url.path.endswith('/'):
                                target /= 'index.html'
                            if not target.is_file():
                                errors.append(f'{relative}: missing peer page {href}')
                            elif url.fragment:
                                peer_ids = {str(e['id']) for e in BeautifulSoup(target.read_text(), 'html.parser').select('[id]')}
                                if unquote(url.fragment) not in peer_ids:
                                    errors.append(f'{relative}: missing peer fragment {href}')
                    continue
                if not url.path:
                    target = path
                elif url.path.startswith('/'):
                    absolute_path = unquote(url.path).lstrip('/')
                    if project and absolute_path.startswith(project + '/'):
                        absolute_path = absolute_path[len(project) + 1:]
                    target = root / absolute_path
                else:
                    target = path.parent / unquote(url.path)
                if target.is_dir():
                    target /= 'index.html'
                target = target.resolve()
                if not target.exists():
                    errors.append(f'{relative}: missing {href}')
                elif url.fragment and target.suffix == '.html' and unquote(url.fragment) not in anchors.get(target, set()):
                    errors.append(f'{relative}: missing fragment {href}')
    (root / 'math-formulas.json').write_text(json.dumps(formulas, ensure_ascii=False))
    (root / 'site-map.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit('\n'.join(sorted(set(errors))))
    print(f'Validated links, anchors and assets in {len(pages)} HTML pages')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site', nargs='?', type=Path, default=Path('site'))
    parser.add_argument('--peer-site', type=Path)
    args = parser.parse_args()
    check(args.site, args.peer_site)
