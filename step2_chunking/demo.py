"""Step 2: 文档分块 —— 理解 Chunking

学习目标:
- 为什么要把长文档切成块？（整篇压一个向量有什么问题）
- chunk_size 切多大才合适？
- 递归切分：为什么比"固定字符数硬切"好？
- overlap（重叠）：怎么避免在切分边界丢信息

运行:
    PYTHONUTF8=1 python step2_chunking/demo.py
"""


# ===== RecursiveTextSplitter：递归文本切分器 =====
# 思路同 langchain 的 RecursiveCharacterTextSplitter（教学简化版）：
# 按分隔符优先级切——先用最粗的边界（段落 \n\n），切出的块还太大就换更细的
# （句子 "。"），再不行换更细的（逗号），最后兜底按字符硬切。
# 目标：让每个 chunk 尽量落在语义边界上，而不是把一句话劈成两半。
class RecursiveTextSplitter:
    # 按优先级从高到低排列的分隔符：段落 → 换行 → 句号 → … → 空格 → 空串（字符级兜底，必须放末尾）
    SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")

    def __init__(self, chunk_size=200, chunk_overlap=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        # 两步：先递归切成落在语义边界的小块（每块自带后接分隔符），再把相邻小块累积成 ~chunk_size 的 chunk
        pieces = self._split(text, self.SEPARATORS)
        return self._merge(pieces)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """递归切：返回落在语义边界的小块（每块 <= chunk_size，自带后接的分隔符）。

        命中 sep 后用它切 text，给**非末片焊上本层 sep、末片不焊**。末片不焊是因为：它后面在
        原文接的是上层分隔符——那个上层 sep 已被外层焊在 part 尾部、跟着递归下行到这里的 text
        里（sep 物理地在文本中，不靠参数传）。这样本层 sep 只焊在非末片，不会和残留在末片尾部
        的上层 sep 堆出 "。，" 标点粘连。
        """
        # 按优先级找第一个出现在 text 里的分隔符；一路找到空串还没命中，就字符级硬切
        for i, sep in enumerate(separators):
            if sep == "":          # 遍历到空串 = 没有语义边界可用了，字符级硬切兜底
                return [text[j:j + self.chunk_size] for j in range(0, len(text), self.chunk_size)]
            if sep in text:        # 命中第一个出现的分隔符
                rest = separators[i + 1:]
                break

        # 按 sep 切出片段，跳过空白残片，避免 "\n" 垃圾片混进来
        parts = [p for p in text.split(sep) if p.strip()]
        pieces = []
        for k, p in enumerate(parts):
            piece = p if k == len(parts) - 1 else p + sep    # 末片不焊（接的是上层带来的 sep），其余焊本层 sep
            if len(piece) <= self.chunk_size:
                pieces.append(piece)                          # 够小，直接收（自带后接分隔符）
            else:
                pieces.extend(self._split(piece, rest))       # 太大，带着焊好的 sep 换更细的分隔符切
        return pieces

    def _merge(self, pieces: list[str]) -> list[str]:
        """把小块累积成接近 chunk_size 的 chunk；相邻 chunk 之间留 chunk_overlap 的重叠。"""
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
盐湖股份2024年实现营业收入150亿元，同比增长12%；归母净利润45亿元，同比增长20%。公司业绩增长主要得益于钾肥价格回暖与碳酸锂产销两旺。年报显示，公司整体毛利率较上年提升3个百分点，盈利能力持续改善。

钾肥是公司的传统主业，产能位居全国前列。2024年氯化钾产量约500万吨，销量保持稳定，国内市场份额持续领先。钾肥业务贡献了公司过半的营业收入和利润，是业绩的压舱石。报告期内，公司推进百万吨钾肥扩建项目，投产后将进一步巩固产能优势。

碳酸锂是公司的第二增长曲线。2024年碳酸锂产量达到3.5万吨，产能位居全国前列。公司依托察尔汗盐湖丰富的卤水资源，采用盐湖提锂工艺，生产成本显著低于矿石提锂企业，具备较强的成本优势。盐湖提锂的吨成本约为矿石法的六成，在价格下行周期中仍能保持盈利。

碳酸锂价格在2024年持续下跌，从年初的10万元/吨跌至年末的7万元/吨，行业整体承压。尽管价格走低，盐湖股份凭借低成本优势，碳酸锂业务仍保持盈利，成为少数逆周期盈利的锂盐企业。

公司持续加大研发投入，重点推进盐湖提锂技术的迭代升级与提锂吸附剂的国产化替代。2024年研发投入同比增长15%，多项技术成果实现产业化应用。公司还与高校联合攻关高镁锂比卤水提锂难题，进一步降低生产成本。

展望未来，公司计划继续扩大碳酸锂产能，目标三年内将产能提升至5万吨。钾肥业务有望受益于国际农产品价格上涨带来的需求提升。管理层对2025年的业绩持谨慎乐观态度，同时提示碳酸锂价格波动与下游需求不及预期的风险。公司表示将持续优化产品结构，提升抗周期能力。"""


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

    print(f"\n示例文本（{len(SAMPLE_TEXT)} 字，6 个段落）：")
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

    # 演示 3：chunk_size 太小 vs 合适 vs 太大
    print("\n" + "=" * 64)
    print("演示 3：chunk_size 的权衡（太小 vs 合适 vs 太大）")
    print("=" * 64)
    for size in (40, 200, 800):
        s = RecursiveTextSplitter(chunk_size=size, chunk_overlap=0)
        _print_chunks(s.split_text(SAMPLE_TEXT), f"递归 chunk_size={size}")
    print("\n  chunk_size=40 太小：一句话被拆成多块，单块语义不全；块数爆炸，检索碎。")
    print("  chunk_size=200 合适：每块聚焦一个主题，落在甜区（中文经验 200~500 字）。")
    print("  chunk_size=800 太大：整篇塞一块，回到'整篇压一个向量'的信息稀释。")

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
