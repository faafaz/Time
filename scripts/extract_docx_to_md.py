from docx import Document
import re
import sys
from pathlib import Path

def extract_docx_to_md(input_path: str, output_path: str) -> None:
    doc = Document(input_path)
    out = []
    last_blank = False
    for para in doc.paragraphs:
        text = (para.text or '').strip()
        style = para.style.name if getattr(para, 'style', None) else ''
        if not text:
            if not last_blank:
                out.append('')
            last_blank = True
            continue
        last_blank = False
        level = None
        if style:
            m = re.search(r"Heading\s*(\d+)", style, re.I)
            if m:
                level = int(m.group(1))
            else:
                m2 = re.search(r"标题\s*(\d+)", style)
                if m2:
                    level = int(m2.group(1))
                elif 'Heading' in style or '标题' in style:
                    level = 1
        if level:
            level = max(1, min(level, 6))
            out.append('#' * level + ' ' + text)
        else:
            out.append(text)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text('\n'.join(out), encoding='utf-8')

if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'docs/基于自适应频带分解与多尺度时序建模的风电超短期功率预测_二稿.docx'
    outp = sys.argv[2] if len(sys.argv) > 2 else 'docs/paper_extracted.md'
    extract_docx_to_md(inp, outp)
    print(f"Wrote: {outp}")

