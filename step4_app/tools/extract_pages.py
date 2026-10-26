"""从真实年报 PDF 抽取精选页，合成多模态演示语料。

页集选自《盐湖股份 2024 年年度报告》（177 页，135 页含表格、41 页含图片），
覆盖三种模态的代表性形态：
  · 纯文本页      —— 重要提示、钾肥/锂盐行业讨论
  · 表格页        —— 释义大表 23 行（大表格独立块）、主要会计数据 15 行、
                     非经常性损益 13 行、小表格若干
  · 图片密集页    —— 行情图表页（VLM 描述的素材）、logo 页（小图过滤的素材）

用法:
    python -X utf8 step4_app/tools/extract_pages.py [源PDF路径]
默认源 = D:/workspace/财务报表/盐湖股份/盐湖股份_2024年年度报告.pdf，
输出 = step4_app/sample_reports/盐湖股份2024年报（精选页）.pdf
"""

import sys
from pathlib import Path

import fitz  # PyMuPDF

SRC_DEFAULT = Path("D:/workspace/财务报表/盐湖股份/盐湖股份_2024年年度报告.pdf")
OUT = Path(__file__).resolve().parent.parent / "sample_reports" / "盐湖股份2024年报（精选页）.pdf"

# 1-based 页码：重要提示 / 释义大表 / 公司信息+主要会计数据 / 注册变更 /
# 非经常性损益 / 行业讨论+图表 / 图片密集页
PAGES = [4, 5, 7, 8, 10, 12, 13, 14, 16, 17]


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC_DEFAULT
    doc = fitz.open(src)
    out = fitz.open()
    for pno in PAGES:
        page = doc[pno - 1]                                # 转 0-based
        out.insert_pdf(doc, from_page=pno - 1, to_page=pno - 1)
        tables = page.find_tables().tables
        imgs = [r for x in page.get_images(full=True) for r in page.get_image_rects(x)]
        tinfo = ", ".join(f"{len(t.rows)}行" for t in tables) or "-"
        print(f"  p{pno:02d}  表格[{tinfo}]  图片{len(imgs)}张  {page.get_text().strip()[:30]}")
    out.save(OUT)
    print(f"\n[OK] {len(PAGES)} 页 → {OUT.name}（{OUT.stat().st_size // 1024} KB）")
    doc.close(); out.close()


if __name__ == "__main__":
    main()
