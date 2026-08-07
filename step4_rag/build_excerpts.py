"""把盐湖股份 2023-2025 年报转成 RAG demo 语料（一次性转换工具）。

产出：sample_reports/盐湖股份<年>年年报（节选）.md ——
- 第二节的「主要会计数据」摘要表（营收/归母净利润/ROE 等）
- 第三节的「主要业务 + 主营业务分析 + 未来发展的展望」

关键处理：**表格行自包含化**。PDF 表格提取成"表头行 + 数据行"，递归切分按行切块后，
表头（年份）和数据（数值）会落在不同 chunk，LLM 对不上年份。所以把年份列头注入数据行：
    "营业收入（元）" → "营业收入（元）：2024年 = 15,134,119,500.89；2023年 = 21,578,504,361.18（同比 -29.86%）"
每行自带年份上下文，切到哪个 chunk 语义都完整。没有年份表头的表格原样保留。

运行: PYTHONUTF8=1 python step4_rag/build_excerpts.py
"""

import os
import re

import fitz

SRC = r'D:/workspace/财务报表/盐湖股份'
DST = os.path.join(os.path.dirname(__file__), 'sample_reports')

# 提取的小节：第二节摘要表 + 第三节三段叙述（风险内容在 2023/2024 的展望里，2025 无独立风险节）
SEC2_HEAD = '主要会计数据和财务指标'
SEC3_BLOCKS = [
    (r'报告期内公司从事的主要业务', '主要业务'),
    (r'主营业务分析', '主营业务分析'),
    (r'公司未来发展的展望', '未来展望'),
]


def _slice_between(lines, start_pat, end_pat):
    """切出 [start, end) 区间；start 用 $ 锚定（跳过带点号的目录行），end 从 start 之后找。"""
    s = next(i for i, l in enumerate(lines) if re.match(start_pat + r'\s*$', l.strip()))
    e = next(i for i, l in enumerate(lines) if re.match(end_pat, l.strip()) and i > s)
    return lines[s + 1:e]


def _headings(sec):
    """节内小节标题：`一、xxx` 形式的行。"""
    return [(i, l.strip()) for i, l in enumerate(sec) if re.match(r'^[一二三四五六七八九十]+、', l.strip())]


def _block(sec, head_pat, heads):
    """取某小节标题到下一个小节标题之间的文本。"""
    hit = next(((i, l.strip()) for i, l in heads if re.search(head_pat, l)), None)
    if not hit:
        return None
    hi = hit[0]
    nxt = next((i for i, l in heads if i > hi), len(sec))
    return hit[1], '\n'.join(l.strip() for l in sec[hi + 1:nxt]).strip()


def _table_self_contained(block: str) -> str:
    """把「年份表头 + 数据行」的表格改写成自包含行。

    表头行形如 "2024年 2023年 本年比上年增减 2022年"（余额类列年份带"末"）；数据行 = 指标文本 + 尾部
    N 个值（数字或百分比），值顺序与表头列一致。每行重组为 "指标文本：2024年 = X；2023年 = Y（同比 Z%）"。
    没有年份表头（或行不规整）的文本原样保留。
    """
    out = []
    cols = None  # 当前表头列标签，如 ["2024年", "2023年", "本年比上年增减", "2022年"]
    for raw in block.split('\n'):
        s = raw.strip()
        if not s:
            continue
        # 表头行：可选前缀文本 + 至少两个年份列 + 其余列（如 "行业分类 项目 单位 2024年 2023年 同比增减"）
        m = re.match(r'^(.*?)((?:\d{4}年(?:末)?\s+){2,})(\S.*)$', s)
        if m:
            cols = m.group(2).split() + [c for c in re.split(r'\s{2,}', m.group(3).strip()) if c]
            continue
        if not cols:
            out.append(s)
            continue
        toks = s.split()
        if len(toks) < len(cols) + 1:
            continue
        head, vals = ' '.join(toks[:-len(cols)]), toks[-len(cols):]
        if not all(re.fullmatch(r'[\d,]+(?:\.\d+)?|-?[\d.]+%|--|不适用', v) for v in vals):
            out.append(s)            # 行不规整（如标签被 PDF 列序打断），原样保留
            continue
        pairs = [f'{c} = {v}' for c, v in zip(cols, vals)]
        # 三列以上且倒数第二列是百分比（增减列）→ 并进前一列：2023年 = Y（同比 -41.07%）
        if len(vals) >= 3 and '%' in vals[-2] and '同比' not in cols:
            pairs[-3] += f'（同比 {vals[-2]}）'
            pairs.pop(-2)
        out.append(f'{head}：' + '；'.join(pairs) + '。')
    return '\n'.join(out)


def _clean_lines(text: str) -> str:
    """清掉 PDF 提取混进来的碎片行 + 压缩列对齐空格。
    丢：纯数字行（页码）、勾选框残行、无中文内容的页眉/页码残留行（如 "5   E"）。
    留：去前导/尾部空白、行内连续空格压成单个（杀 PDF 多列对齐空格，如资质证书表）。
    空行保留作段落分隔（splitter 靠空行识别段落）。"""
    out = []
    for raw in text.split('\n'):
        s = raw.strip()
        if not s:
            out.append('')                               # 保留空行作段落分隔
            continue
        if re.fullmatch(r'\d+', s):                      # 页码
            continue
        if re.fullmatch(r'[□√]\s*(是|适用)?\s*[□√]?\s*(否|不适用)?\s*', s):   # 勾选框残行
            continue
        core = re.sub(r'[\s\dA-Za-z.%,（）()\-]', '', s)  # 去掉数字/字母/符号后的有效内容
        if not core:                                     # 无中文内容 = 页眉/页码残留（如 "5   E"、"2024"）
            continue
        s = re.sub(r'(?:[A-Za-z]{1,3}[.,，。、\s]+){4,}[A-Za-z]{1,3}', '', s)  # 删 PDF 打碎的英文残片（连续短字母 token）
        s = s.strip()
        if not s:
            continue
        out.append(re.sub(r' {2,}', ' ', s))             # 写回清洗后的行：压行内多空格、去前后空白
    return '\n'.join(out)


def _fix_product_orphans(block: str) -> str:
    """产销量表：产品名（氯化钾/碳酸锂制造业）被 PDF 列序挤到本组的「生产量」行，
    同组的「销售量/库存量」行成了孤儿（且 PDF 行序是 销量→产量→库存 交错，
    "最近出现"会张冠李戴，须向后找本组的产品名）。"""
    lines = block.split('\n')
    out = []
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            out.append(s)
            continue
        if re.match(r'^((?:氯化钾|碳酸锂)制造业)\s+(销售量|生产量|库存量)\s', s):
            out.append(s)
            continue
        if re.match(r'^(销售量|生产量|库存量)\s', s):
            nxt = next((l for l in lines[i + 1:] if re.match(r'^((?:氯化钾|碳酸锂)制造业)\s', l.strip())), '')
            m = re.match(r'^((?:氯化钾|碳酸锂)制造业)', nxt.strip())
            if m:
                out.append(f'{m.group(1)} {s}')
                continue
        out.append(s)
    return '\n'.join(out)


def build_one(year: str) -> None:
    doc = fitz.open(os.path.join(SRC, f'盐湖股份_{year}年年度报告.pdf'))
    pages = [re.sub(r'\n{3,}', '\n\n', pg.get_text(sort=True)).strip() for pg in doc]
    doc.close()
    lines = '\n\n'.join(pages).split('\n')

    out = [f'# 盐湖股份 {year} 年年报（节选）', '']

    sec2 = _slice_between(lines, r'第二节 公司简介和主要财务指标', r'第三节 管理层讨论与分析')
    r = _block(sec2, SEC2_HEAD, _headings(sec2))
    if r:
        out += ['## 主要会计数据', _table_self_contained(r[1]), '']

    sec3 = _slice_between(lines, r'第三节 管理层讨论与分析', r'第四节')
    h3 = _headings(sec3)
    for pat, label in SEC3_BLOCKS:
        r = _block(sec3, pat, h3)
        if r:
            body = _table_self_contained(r[1])
            if label == '主营业务分析':
                body = _fix_product_orphans(body)
            out += [f'## {r[0]}', body, '']

    fn = os.path.join(DST, f'盐湖股份{year}年年报（节选）.md')
    md = _clean_lines('\n'.join(out))
    open(fn, 'w', encoding='utf-8').write(md)
    print(f'{year}: {len(md)} 字 ≈ {len(md.encode("utf-8")) / 1024:.0f} KB → {os.path.basename(fn)}')


if __name__ == '__main__':
    os.makedirs(DST, exist_ok=True)
    for y in ('2023', '2024', '2025'):
        build_one(y)
