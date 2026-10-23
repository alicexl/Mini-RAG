"""Step 1: 基础件 —— Embedding（向量化）与 Chunking（分块）

RAG 的两道地基工序，一个 demo 讲完：
- 前半 Embedding：文本变向量，余弦相似度找语义接近（需要本地 Ollama）
- 后半 Chunking：长文档切 chunk，递归切分保语义边界 + overlap（纯标准库，零依赖）

运行:
    python -X utf8 step1_basics/demo.py

前半需要 Ollama 已启动 + qwen3-embedding:0.6b 已下载；没起 Ollama 也能跑——
前半会提示失败原因，然后继续跑后半（分块不依赖任何服务）。
"""

import os

import requests

# ===== Embedder：把文本变成向量 =====
# 设计要点：
# - model / base_url 从环境变量读，默认连本地 Ollama + qwen3-embedding:0.6b
# - embed_batch 一次请求嵌入多段文本（Ollama /api/embed 支持 input 为列表）
# - cosine_similarity 是纯数学，不依赖任何第三方库
_DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")


class Embedder:
    """把文本喂给 Ollama 的 embedding 模型，拿回向量。"""

    def __init__(self, base_url: str = _DEFAULT_BASE_URL, model: str = _DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入：一次请求把多段文本都变成向量。"""
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": list(texts)},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """两个向量的余弦相似度，范围 [-1, 1]，越接近 1 越相似。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0  # 零向量防御，避免除零
        return dot / (norm_a * norm_b)


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

        命中 sep 后用它切 text，给**非末片焊上本层 sep、末片不焊**。末片不焊是因为：本层切出的
        末片，要么后面在原文里什么都没有（文档真末尾），要么尾巴上已物理带着上层分隔符（如
        "，"-层切出的末片 "…较强的成本优势。"，句号是外层焊下来跟着文本进来的）——这两种情况
        焊本层 sep 都会错位（"。，" 粘连、或标点落空）。所以本层 sep 只焊在非末片。

        关键：parts 保留 `split` 的全部结果（含空串、"\n\n" 空白残片）——末片位由空白残片占着，
        真句子落在非末片位置、正常焊回自己的分隔符。
        """
        # 按优先级找第一个出现在 text 里的分隔符；一路找到空串还没命中，就字符级硬切兜底
        for i, sep in enumerate(separators):
            if sep == "":          # 遍历到空串 = 没有语义边界可用了，字符级硬切兜底
                return [text[j:j + self.chunk_size] for j in range(0, len(text), self.chunk_size)]
            if sep in text:        # 命中第一个出现的分隔符
                rest = separators[i + 1:]
                break

        # parts 保留全部结果（含空串、"\n\n" 空白残片）——它们正好占末片位
        parts = text.split(sep)
        pieces = []
        for k, p in enumerate(parts):
            piece = p if k == len(parts) - 1 else p + sep    # 末片不焊（后面没东西/已带上层 sep），其余焊本层 sep
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


# ===== 前半的测试文本：4 段，1↔2 否定陷阱，3↔2 因果弱信号，文本4 天气正交 =====
EMBED_TEXTS = [
    "盐湖股份2024年净利润大幅增长。",
    "盐湖股份2024年净利润大幅下滑。",
    "碳酸锂价格在2024年持续下跌，从年初的10万元/吨跌至年末的7万元/吨。",
    "今天天气不错，适合出去散步。",
]

# ===== 后半的示例文本：6 段盐湖股份年报体 =====
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


def part_embedding():
    """前半：文本向量化 —— 看语义怎么变成可以算的距离。"""
    print("\n" + "=" * 64)
    print("前半：文本向量化 —— 理解 Embedding")
    print("=" * 64)

    embedder = Embedder()
    print(f"\n模型: {embedder.model}")
    print(f"Ollama 地址: {embedder.base_url}")

    print(f"\n正在向量化 {len(EMBED_TEXTS)} 段文本...")
    try:
        vectors = embedder.embed_batch(EMBED_TEXTS)
    except Exception as e:
        print(f"\n[WARN] 向量化失败：{e}")
        print("请确认 Ollama 已启动（ollama serve）且模型已下载（ollama pull qwen3-embedding:0.6b）")
        print("跳过前半，继续跑后半（分块不依赖任何服务）")
        return False

    # 打印每段文本的向量维度
    for i, (text, vec) in enumerate(zip(EMBED_TEXTS, vectors)):
        print(f"\n  文本{i+1}: {text[:40]}...")
        print(f"  向量维度: {len(vec)}, 前5个值: {[round(v, 4) for v in vec[:5]]}")

    # 计算两两之间的余弦相似度
    print("\n" + "=" * 64)
    print("余弦相似度矩阵（越接近 1 越相似）:")
    print("=" * 64)

    # 打印表头
    header = "      " + "".join(f" 文本{j+1} " for j in range(len(EMBED_TEXTS)))
    print(header)
    for i in range(len(EMBED_TEXTS)):
        row = f"文本{i+1} "
        for j in range(len(EMBED_TEXTS)):
            sim = Embedder.cosine_similarity(vectors[i], vectors[j])
            row += f" {sim:.3f} "
        print(row)

    # 解读
    print("\n" + "=" * 64)
    print("解读:")
    print("=" * 64)
    print("  文本1↔文本2: 净利润增长 vs 下滑，只差一词、意思相反 → 相似度仍很高（否定陷阱）")
    print("  文本3↔文本2: 碳酸锂跌 → 净利润下滑，因果相关 → 相似度却不高（弱信号）")
    print("  文本1/2/3 ↔ 文本4: 财务/商品 vs 天气，语义无关 → 相似度低")
    print()
    print("  这就是 Embedding 的核心价值：")
    print("  把文本映射到向量空间，语义相似的文本向量更近，")
    print("  从而可以用数学方法（余弦相似度）衡量语义相似性。")
    print("  RAG 检索的本质就是：找与用户查询向量最近的文档向量。")
    return True


def part_chunking():
    """后半：文档分块 —— 长文档怎么切成检索单位。"""
    print("\n" + "=" * 64)
    print("后半：文档分块 —— 理解 Chunking")
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


def main():
    print("=" * 60)
    print("Step 1: 基础件 —— Embedding 与 Chunking")
    print("=" * 60)
    part_embedding()
    part_chunking()


if __name__ == "__main__":
    main()
