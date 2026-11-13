"""Inline the generated tables into paper_src.tex -> paper.tex, then compile with pdflatex (run from this folder)."""
import re, subprocess
s = open('paper_src.tex').read()
s = re.sub(r'\\input\{tab/(\w+)\}', lambda m: open(f'tab/{m.group(1)}.tex').read().strip(), s)
open('paper.tex', 'w').write(s)
for _ in range(2): subprocess.run(['pdflatex', '-interaction=nonstopmode', 'paper.tex'], stdout=subprocess.DEVNULL)
print('paper.pdf built')
