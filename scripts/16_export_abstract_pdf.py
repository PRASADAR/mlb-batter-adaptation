#!/usr/bin/env python3
"""Deterministic one-page Letter PDF from exact abstract Markdown source."""
from pathlib import Path
import re,json
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab import rl_config
import matplotlib
import fitz

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'paper/abstract.md'
text=source.read_text().strip()
lines=text.splitlines(); title=lines[0].removeprefix('# ')
fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
for name,fn in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Title','DejaVuSerif.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fontdir/fn)))
rl_config.invariant=1
out=ROOT/'paper/abstract.pdf'
doc=SimpleDocTemplate(str(out),pagesize=(612,792),rightMargin=53,leftMargin=53,topMargin=44,bottomMargin=42,title=title,author='',subject='Research abstract',creator='Deja Swing reproducible PDF exporter')
styles={
    'title':ParagraphStyle('title',fontName='Title',fontSize=20,leading=24,textColor=HexColor('#142D3A'),spaceAfter=13),
    'author':ParagraphStyle('author',fontName='Body',fontSize=9,leading=12,textColor=HexColor('#566776'),spaceAfter=15),
    'heading':ParagraphStyle('heading',fontName='Bold',fontSize=10.5,leading=14,textColor=HexColor('#126B70'),spaceBefore=10,spaceAfter=4),
    'body':ParagraphStyle('body',fontName='Body',fontSize=10,leading=14,textColor=HexColor('#172B36'),spaceAfter=3),
}
story=[]
for block in text.split('\n\n'):
    if block.startswith('# '): kind='title';block=block[2:]
    elif block.startswith('## '):kind='heading';block=block[3:]
    elif block.startswith('Author:'):kind='author'
    else:kind='body'
    story.append(Paragraph(escape(block.replace('\n',' ')),styles[kind]))
doc.build(story)
pdf=fitz.open(out);extracted=' '.join(p.get_text() for p in pdf)
normalize=lambda x:re.sub(r'\s+',' ',re.sub(r'^#+\s*','',x,flags=re.MULTILINE)).strip()
assert normalize(text)==normalize(extracted),'PDF text differs from Markdown'
assert len(pdf)==1,f'Expected one page, got {len(pdf)}'
assert all(h in extracted for h in ['Introduction','Methods','Results','Conclusion'])
assert all(abs(p.rect.width-612)<1 and abs(p.rect.height-792)<1 for p in pdf)
(ROOT/'tmp/pdfs').mkdir(parents=True,exist_ok=True)
pdf[0].get_pixmap(matrix=fitz.Matrix(1.8,1.8)).save(ROOT/'tmp/pdfs/abstract.png')
word_count=len(normalize(text).split());assert word_count<500
info={'pages':len(pdf),'word_count_including_title_headings_author':word_count,'all_sections_present':True,'exact_text_match':True,'page_size':'US Letter','render':'tmp/pdfs/abstract.png'}
(ROOT/'results/logs/pdf_validation.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2))
