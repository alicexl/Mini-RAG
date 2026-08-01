"""Step 2: 文档分块 —— 理解 Chunking

学习目标:
- 为什么要把长文档切成块？（整篇压一个向量有什么问题）
- chunk_size 切多大才合适？
- 递归切分：为什么比"固定字符数硬切"好？
- overlap（重叠）：怎么避免在切分边界丢信息

运行:
    PYTHONUTF8=1 python step2_chunking/demo.py

本 demo 纯标准库，不依赖 Ollama/向量模型——切分本身不需要向量化。
"""


# ===== RecursiveTextSplitter：递归文本切分器 =====
# 思路同 langchain 的 RecursiveCharacterTextSplitter（教学简化版）：
# 按分隔符优先级切——先用最粗的边界（段落 \n\n），切出的块还太大就换更细的
# （句子 "。"），再不行换更细的（逗号），最后兜底按字符硬切。
# 目标：让每个 chunk 尽量落在语义边界上，而不是把一句话劈成两半。
class RecursiveTextSplitter:
    def __init__(self, chunk_size=200, chunk_overlap=0,
                 separators=("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators)

    def split_text(self, text: str) -> list[str]:
        # 两步：先把文本递归切成"语义小块"，再把相邻小块合并成 ~chunk_size 的 chunk
        pieces = self._split(text, self.separators)
        return self._merge(pieces)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """递归切：返回尽可能落在语义边界的小块（每块 <= chunk_size，或已无可细分）。"""
        # 在 separators 里找优先级最高、且出现在文本中的分隔符；都没有则用空串（按字符）
        chosen, rest = "", []
        for i, sep in enumerate(separators):
            if sep == "":
                chosen, rest = "", []
                break
            if sep in text:
                chosen, rest = sep, separators[i + 1:]
                break

        # 空串兜底：直接按 chunk_size 硬切字符（此时每块必 <= chunk_size）
        if chosen == "":
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]

        # 按 chosen 切，并把分隔符加回每块末尾（保留边界，合并时能还原原文）。
        # 跳过纯空白 part：递归到句子层时，上层遗留的 \n\n 会被 split 暴露成空白
        # 残片，不跳过就会包成 "\n。" 这种垃圾块混进来。
        pieces = []
        for part in text.split(chosen):
            if not part.strip():
                continue
            piece = part + chosen
            if len(piece) <= self.chunk_size:
                pieces.append(piece)                       # 够小，直接收
            elif rest:
                pieces.extend(self._split(piece, rest))    # 太大，换更细的分隔符继续切
            else:
                pieces.append(piece)                       # 没有更细的分隔符了，整块收
        return pieces

    def _merge(self, pieces: list[str]) -> list[str]:
        """把小块合并成接近 chunk_size 的 chunk；相邻 chunk 之间留 chunk_overlap 的重叠。"""
        chunks, cur, cur_len = [], [], 0
        for p in pieces:
            if cur and cur_len + len(p) > self.chunk_size:
                # 累积要超了 → 先把当前这批封成一个 chunk
                chunks.append("".join(cur))
                # overlap：从开头弹出若干块，直到剩余长度 <= overlap
                # （下一个 chunk 会接着这段开始 → 相邻块共享边界内容，避免漏检）
                while cur and cur_len > self.chunk_overlap:
                    cur_len -= len(cur[0])
                    cur.pop(0)
            cur.append(p)
            cur_len += len(p)
        if cur:
            chunks.append("".join(cur))
        return chunks


# ===== 固定字符切分（反面教材）=====
def split_fixed(text: str, chunk_size: int) -> list[str]:
    """最朴素的切分：硬按 chunk_size 个字符切，不管句子边界。"""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# ===== Demo =====
SAMPLE_TEXT = """\
盐湖股份2024年实现营业收入150亿元，同比增长12%。公司主营业务为钾肥和碳酸锂，其中钾肥业务是主要收入来源。钾肥产能位居全国前列，2024年产量保持稳定。

碳酸锂价格在2024年持续下跌，从年初的10万元/吨跌至年末的7万元/吨。尽管如此，盐湖股份的碳酸锂产能仍位居全国前列，2024年产量达到3.5万吨。公司依托察尔汗盐湖的资源优势，具备较低的生产成本。

展望未来，公司计划继续扩大碳酸锂产能，并推进盐湖提锂技术的研发。同时，钾肥业务有望受益于国际农产品价格上涨带来的需求提升。管理层对2025年的业绩持谨慎乐观态度。"""


def _print_chunks(chunks: list[str], title: str):
    print(f"\n{'─' * 64}")
    print(f"{title}  →  共 {len(chunks)} 块")
    print('─' * 64)
    for i, c in enumerate(chunks, 1):
        preview = c.replace("\n", "↵ ")   # 把换行显示出来，方便看清块边界
        print(f"  [{i}] ({len(c):>3}字) {preview}")


def main():
    print("=" * 64)
    print("Step 2: 文档分块 —— 理解 Chunking")
    print("=" * 64)

    print(f"\n示例文本（{len(SAMPLE_TEXT)} 字，3 个段落）：")
    print("  " + SAMPLE_TEXT[:42].replace("\n", "↵ ") + " …")

    # 演示 1：固定字符切分（反面教材）
    print("\n" + "=" * 64)
    print("演示 1：固定字符切分（chunk_size=40，硬切，不管句子）")
    print("=" * 64)
    _print_chunks(split_fixed(SAMPLE_TEXT, 40), "固定字符 chunk_size=40")
    print("\n⚠ 句子被从中间切断——'10万元/吨'、'3.5万吨' 这类关键信息可能被劈成两半。")

    # 演示 2：递归切分（正解）
    print("\n" + "=" * 64)
    print("演示 2：递归切分（chunk_size=80, overlap=0，按语义边界）")
    print("=" * 64)
    splitter = RecursiveTextSplitter(chunk_size=80, chunk_overlap=0)
    _print_chunks(splitter.split_text(SAMPLE_TEXT), "递归 chunk_size=80")
    print("\n✓ 每块都落在句号/换行边界上，语义完整。")

    # 演示 3：chunk_size 太小 vs 太大
    print("\n" + "=" * 64)
    print("演示 3：chunk_size 的权衡（太小 vs 太大）")
    print("=" * 64)
    for size in (40, 200):
        s = RecursiveTextSplitter(chunk_size=size, chunk_overlap=0)
        _print_chunks(s.split_text(SAMPLE_TEXT), f"递归 chunk_size={size}")
    print("\n  chunk_size=40 太小：一句话被拆成多块，单块语义不全；块数多，检索碎。")
    print("  chunk_size=200 太大：多段塞一块，检索定位不到具体段落，信息被稀释。")

    # 演示 4：overlap
    print("\n" + "=" * 64)
    print("演示 4：overlap 重叠（chunk_size=100, overlap=0 vs 50）")
    print("=" * 64)
    for ov in (0, 50):
        s = RecursiveTextSplitter(chunk_size=100, chunk_overlap=ov)
        _print_chunks(s.split_text(SAMPLE_TEXT), f"overlap={ov}")
    print("\n  overlap=0：相邻块各管各，若关键句恰好在切分边界，可能两边都不完整。")
    print("  overlap=50：相邻块共享约 50 字，边界附近的信息两边都能检索到——代价是存储/计算略增。")


if __name__ == "__main__":
    main()
