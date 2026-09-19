"""Compare source TeX against the built HTML so missing formulas cannot pass."""
import collections,json,re
from pathlib import Path
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]

def formulas(text):
    lines=[];fence=None
    for line in text.splitlines(keepends=True):
        mark=re.match(r'^\s*(`{3,}|~{3,})',line)
        if mark:
            m=mark[1]
            if fence is None:fence=m
            elif m[0]==fence[0] and len(m)>=len(fence):fence=None
            lines.append('\n');continue
        lines.append('\n' if fence else line)
    if fence:raise ValueError('Unclosed fenced block')
    text=re.sub(r'(`+)[^`]*?\1','', ''.join(lines))
    result=[]
    for m in re.finditer(r'(?<!\\)(\${1,2})(.+?)(?<!\\)\1(?!\$)',text,re.S):
        result.append((m[1]=='$$',re.sub(r'\s+',' ',m[2]).strip()))
    return collections.Counter(result)

def main():
    errors=[];total=0;documents=0
    for source in sorted((ROOT/'.site-docs').rglob('*.md')):
        relative=source.relative_to(ROOT/'.site-docs')
        html=ROOT/'site'/(relative.parent/'index.html' if relative.name=='index.md' else relative.with_suffix('')/'index.html')
        expected=formulas(source.read_text());total+=expected.total();documents+=1
        soup=BeautifulSoup(html.read_text(),'html.parser')
        actual=collections.Counter((el.name=='div',re.sub(r'\s+',' ',el.get_text()[2:-2]).strip()) for el in soup.select('.arithmatex'))
        if expected!=actual:
            errors.append({'source':str(relative),'missing':list((expected-actual).elements()),'extra':list((actual-expected).elements())})
    if errors:raise SystemExit(json.dumps(errors,ensure_ascii=False,indent=2))
    print(f'Compared {total} source formulas across {documents} documents: no missing or changed TeX')

if __name__=='__main__':main()
