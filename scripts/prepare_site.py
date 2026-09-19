"""Build a disposable Markdown tree; source documents stay readable on GitHub."""
from pathlib import Path
from urllib.parse import quote,urlsplit,unquote
import posixpath,re,shutil

ROOT=Path(__file__).resolve().parents[1]

def sources():
    return [Path('README.md'), Path('PUBLISHING.md')] + sorted(p.relative_to(ROOT) for folder in ('docs','datasets','references','src') for p in (ROOT/folder).rglob('*.md'))

def output_path(path):
    return path.with_name('index.md') if path.name=='README.md' else path

def transform(text, source):
    result=[];fence=None
    for line in text.splitlines(keepends=True):
        mark=re.match(r'^\s*(`{3,}|~{3,})',line)
        if mark:
            m=mark[1]
            if fence is None:fence=m
            elif m[0]==fence[0] and len(m)>=len(fence):fence=None
            result.append(line);continue
        if fence:result.append(line);continue
        def link(m):
            u=urlsplit(m[2])
            if u.scheme or u.netloc or not u.path:return m[0]
            target=Path(posixpath.normpath(posixpath.join(source.parent.as_posix(),unquote(u.path))))
            if u.path.endswith('/'):target=target/'README.md'
            target=output_path(target)
            rel=posixpath.relpath(target.as_posix(),output_path(source).parent.as_posix())
            return m[1]+quote(rel,safe='/.-')+('?' + u.query if u.query else '')+('#'+u.fragment if u.fragment else '')+m[3]
        chunks=re.split(r'(`+[^`]*`+)',line)
        for i in range(0,len(chunks),2):
            chunks[i]=re.sub(r'(\]\()([^\s)]+)(\))',link,chunks[i])
            if line.lstrip().startswith('|'):
                chunks[i]=re.sub(r'(?<!\\)\$[^$\n]+(?<!\\)\$',lambda m:re.sub(r'(?<!\\)\|',r'\\vert{}',m[0]),chunks[i])
        line=''.join(chunks)
        line=re.sub(r'^\s*\$\$(.+)\$\$\s*$',lambda m:'\n$$\n'+m[1]+'\n$$\n\n',line)
        result.append(line)
    if fence:raise ValueError(f'Unclosed fence: {source}')
    return ''.join(result)

def main():
    dest=ROOT/'.site-docs'
    if dest.exists():shutil.rmtree(dest)
    for source in sources():
        target=dest/output_path(source);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(transform((ROOT/source).read_text(),source))
    for directory in sorted([d for d in dest.rglob('*') if d.is_dir()],key=lambda d:len(d.parts),reverse=True):
        if not (directory/'index.md').exists():
            links=[]
            for child in sorted(directory.iterdir()):
                if child.is_dir():links.append(f'- [{child.name}]({quote(child.name)}/index.md)')
                elif child.suffix=='.md':links.append(f'- [{child.stem}]({quote(child.name)})')
            (directory/'index.md').write_text('# '+directory.name+'\n\n'+'\n'.join(links)+'\n')
    for name in ('LICENSE','requirements-examples.txt','requirements-examples-lock.txt','src/audio_baselines.py'):
        shutil.copy2(ROOT/name,dest/name)
    shutil.copytree(ROOT/'web',dest/'assets')
    shutil.copytree(ROOT/'node_modules/mathjax/es5',dest/'assets/mathjax')
    print(f'Prepared {len(sources())} source Markdown documents')

if __name__=='__main__':main()
