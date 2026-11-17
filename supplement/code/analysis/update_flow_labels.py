"""Update three implementation claims in the editable flow SVG and PDF.

The original vector inputs are preserved under code/history/figure2_original. The PDF is
edited as vector text so the remaining original figure rendering is preserved;
the SVG remains editable and uses the same revised labels.
"""
from pathlib import Path
import re
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[2]
BACKUP=ROOT/'code/history/figure2_original'
TARGET=ROOT/'paper/figures'
SVG_TARGET=ROOT/'paper/temp/figure_sources'
FONT=Path(r'C:\Windows\Fonts\cambria.ttc')


def main():
    svg=(BACKUP/'fig2_flow.svg').read_text(encoding='utf8')
    replacements={
        'width="1120" height="960"':'width="183mm" height="156.8571429mm"',
        'Separate minimal conflict K':'Separate certified conflict K',
        '≥14-job window: stop without cut (unknown)':'Unresolved window: no cut',
        '≤ 6 lazy rounds':'Max 16 solves',
        'font-size="10.5"':'font-size="10.8"',
        'EDF · res-EDF · density':'EDF / res-EDF / density',
        '<text x="700" y="566" font-size="10.8"':'<text x="700" y="560" font-size="10.8"',
        '<text x="838" y="558" font-size="13"':'<text x="828" y="558" font-size="13"',
    }
    for index,(old,new) in enumerate(replacements.items()):
        assert svg.count(old)>0 and (index==0 or svg.count(old)==1),(old,svg.count(old))
        svg=svg.replace(old,new,1)
    svg=re.sub(r'<defs>\s*<filter id="soft".*?</filter>\s*</defs>','',svg,flags=re.S)
    svg=svg.replace(' filter="url(#soft)"','')
    SVG_TARGET.mkdir(parents=True,exist_ok=True)
    (SVG_TARGET/'fig2_flow.svg').write_text(svg,encoding='utf8',newline='\n')

    doc=fitz.open(BACKUP/'fig2_flow.pdf');page=doc[0]
    spans=[s for b in page.get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans']]
    def locate(key):
        matches=[s for s in spans if key in s['text']]
        assert len(matches)==1,(key,len(matches))
        return matches[0]
    # Each tuple contains original span, replacement, background, ink,
    # centering, optional font size, and x/y movement.
    specs=[
        (locate('minimal conflict'),'Separate certified conflict K',None,(1,1,1),True,None,0,0),
        (locate('job window'),'Unresolved window: no cut',None,(85/255,)*3,True,None,0,0),
        (locate('lazy rounds'),'Max 16 solves',None,(122/255,0,25/255),False,None,0,0),
        (locate('res-EDF'),'EDF / res-EDF / density',None,(122/255,0,25/255),True,8.1,0,-4.5),
        (locate('Yes'),'Yes',None,(122/255,0,25/255),False,None,-7.5,0),
    ]
    font=fitz.Font(fontfile=str(FONT))
    for span,_,background,_,_,_,_,_ in specs:
        page.add_redact_annot(fitz.Rect(span['bbox'])+(-.6,-.4,.6,.4),fill=background)
    page.apply_redactions(images=0,graphics=0,text=0)
    for span,text,_,ink,center,override,dx,dy in specs:
        size=override or span['size']
        x=(525-font.text_length(text,fontsize=size)/2 if center else span['origin'][0])+dx
        page.insert_text((x,span['origin'][1]+dy),text,fontname='CambriaFix',fontfile=str(FONT),
                         fontsize=size,color=ink,overlay=True)
    doc.subset_fonts()
    doc.save(str(TARGET/'fig2_flow.pdf'),garbage=4,deflate=True)
    print('Revised editable SVG and vector PDF:',TARGET/'fig2_flow.pdf')


if __name__=='__main__': main()
