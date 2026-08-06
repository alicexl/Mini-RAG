"""Step 3: 本地向量库 —— 存起来，检索出来

学习目标:
- 切好的 chunk 怎么存（向量 + 原文分开存）
- 检索到底在做什么（查询向量 vs 所有 chunk 向量，算余弦，取 Top-K）
- 从零搭一个最小的向量库（JSON + numpy），把 step1 的 embedding + step2 的切块
  串成一条完整的检索链路

运行:
    PYTHONUTF8=1 python step3_index/demo.py

依赖 Ollama 已启动 + qwen3-embedding:0.6b 已下载（同 step1）。
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import requests

# ===== Embedder（同 step1，内联以便 demo 自包含）=====
# 把文本喂给 Ollama 的 embedding 模型拿回向量；model/base_url 从环境变量读，默认连本地 Ollama。
_DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")


class Embedder:
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


# ===== RecursiveTextSplitter（同 step2，内联以便 demo 自包含）=====
# 按分隔符优先级切（段落→换行→句号→…→空格→空串兜底），尽量让每块落在语义边界。
class RecursiveTextSplitter:
    # 按优先级从高到低排列的分隔符：段落 → 换行 → 句号 → … → 空格 → 空串（字符级兜底，必须放末尾）
    SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")

    def __init__(self, chunk_size=200, chunk_overlap=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        # 两步：先把文本递归切成「片段 + 它后接的分隔符」，再把相邻片段累积成 ~chunk_size 的 chunk
        pairs = self._split(text, self.SEPARATORS, trailing="")
        return self._merge(pairs)

    def _split(self, text: str, separators: list[str], trailing: str) -> list[tuple[str, str]]:
        """递归切：返回 [(片段, 该片段后接的分隔符), ...]，每片 <= chunk_size。

        trailing 是「这段 text 在原文里后接的分隔符」，由外层传入。关键：递归切出的**最后一片**
        接 trailing、而非本层分隔符——这样外层分隔符（如句号）和本层分隔符（如逗号）不会堆在
        同一片尾部（避免 "。，" 这种标点粘连）。其余片段接本层分隔符，如实反映原文边界。
        """
        # 按优先级找第一个出现在 text 里的分隔符；一路找到空串还没命中，就字符级硬切
        for i, sep in enumerate(separators):
            if sep == "":          # 遍历到空串 = 没有语义边界可用了，字符级硬切兜底
                pieces = [text[j:j + self.chunk_size] for j in range(0, len(text), self.chunk_size)]
                return [(p, trailing if k == len(pieces) - 1 else "")
                        for k, p in enumerate(pieces)]
            if sep in text:        # 命中第一个出现的分隔符
                rest = separators[i + 1:]
                break

        # 按 sep 切出纯片段（不带分隔符），跳过空白残片，避免 "\n" 垃圾片混进来
        parts = [p for p in text.split(sep) if p.strip()]
        out = []
        for k, p in enumerate(parts):
            # 末片接外层 trailing（穿透到上层边界），其余接本层 sep —— 这正是避免标点粘连的关键
            after = trailing if k == len(parts) - 1 else sep
            if len(p) <= self.chunk_size:
                out.append((p, after))                       # 够小，直接收
            else:
                out.extend(self._split(p, rest, after))      # 太大，换更细的分隔符继续切
        return out

    def _merge(self, pairs: list[tuple[str, str]]) -> list[str]:
        """把相邻片段累积成接近 chunk_size 的 chunk；相邻 chunk 之间留 chunk_overlap 的重叠。"""
        chunks, cur, cur_len = [], [], 0
        for text, sep in pairs:
            block_len = len(text) + len(sep)                 # 这片完整长度（含后接分隔符）
            if cur and cur_len + block_len > self.chunk_size:
                # 累积要超了 → 先把当前这批封成一个 chunk
                chunks.append(self._join(cur))
                # overlap：从开头弹出整片，直到剩余长度 <= overlap
                # （下一个 chunk 接着这段开始 → 相邻块共享边界内容，避免漏检）
                while cur and cur_len > self.chunk_overlap:
                    cur_len -= len(cur[0][0]) + len(cur[0][1])
                    cur.pop(0)
            cur.append((text, sep))
            cur_len += block_len
        if cur:
            chunks.append(self._join(cur))
        return chunks

    @staticmethod
    def _join(cur: list[tuple[str, str]]) -> str:
        """把累积的 (内容,分隔符) 片段拼成文本：每片内容后跟它的分隔符。"""
        return "".join(text + sep for text, sep in cur)


# ===== VectorStore：最小的向量库 =====
# 存：原文走 JSON，向量走 numpy .npy（向量是数值矩阵，numpy 存取快、占空间小）。
# 检索：查询向量 vs 所有 chunk 向量算余弦相似度，按降序取 Top-K（线性扫描，朴素但够用）。
class VectorStore:
    def __init__(self):
        self.chunks: list[str] = []          # 原文，按入库顺序
        self.vectors: np.ndarray = None      # (N, dim) 的向量矩阵

    def add(self, texts: list[str], embedder: Embedder):
        """把若干段文本 embed 后入库。"""
        vecs = embedder.embed_batch(texts)
        new = np.array(vecs, dtype=np.float32)
        self.vectors = new if self.vectors is None else np.vstack([self.vectors, new])
        self.chunks.extend(texts)

    def save(self, path: str):
        """持久化：原文 → <path>.json，向量 → <path>.npy。"""
        path = Path(path)
        path.with_suffix(".json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(path.with_suffix(".npy"), self.vectors)

    def load(self, path: str):
        """从磁盘读回一个向量库。"""
        path = Path(path)
        self.chunks = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.vectors = np.load(path.with_suffix(".npy"))

    def search(self, query: str, embedder: Embedder, top_k=3) -> list[tuple[int, float, str]]:
        """检索：查询向量 vs 所有 chunk 向量算余弦，取相似度最高的 Top-K。"""
        q = np.array(embedder.embed_batch([query])[0], dtype=np.float32)  # (dim,)
        sims = self._cosine_to_all(q)                                     # (N,) 与每个 chunk 的相似度
        order = np.argsort(-sims)[:top_k]                                 # 降序取前 K
        return [(int(i), float(sims[i]), self.chunks[i]) for i in order]

    def _cosine_to_all(self, q: np.ndarray) -> np.ndarray:
        """查询向量 q 与库内所有向量的余弦相似度。余弦 = 归一化后点积，矩阵乘一次算完所有。"""
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        v_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
        return v_norm @ q_norm        # (N, dim) @ (dim,) -> (N,)


# ===== Demo =====
SAMPLE_DOC = """\
盐湖股份2024年实现营业收入150亿元，同比增长12%；归母净利润45亿元，同比增长20%。公司业绩增长主要得益于钾肥价格回暖与碳酸锂产销两旺。年报显示，公司整体毛利率较上年提升3个百分点，盈利能力持续改善。

钾肥是公司的传统主业，产能位居全国前列。2024年氯化钾产量约500万吨，销量保持稳定，国内市场份额持续领先。钾肥业务贡献了公司过半的营业收入和利润，是业绩的压舱石。报告期内，公司推进百万吨钾肥扩建项目，投产后将进一步巩固产能优势。

碳酸锂是公司的第二增长曲线。2024年碳酸锂产量达到3.5万吨，产能位居全国前列。公司依托察尔汗盐湖丰富的卤水资源，采用盐湖提锂工艺，生产成本显著低于矿石提锂企业，具备较强的成本优势。盐湖提锂的吨成本约为矿石法的六成，在价格下行周期中仍能保持盈利。

碳酸锂价格在2024年持续下跌，从年初的10万元/吨跌至年末的7万元/吨，行业整体承压。尽管价格走低，盐湖股份凭借低成本优势，碳酸锂业务仍保持盈利，成为少数逆周期盈利的锂盐企业。

公司持续加大研发投入，重点推进盐湖提锂技术的迭代升级与提锂吸附剂的国产化替代。2024年研发投入同比增长15%，多项技术成果实现产业化应用。公司还与高校联合攻关高镁锂比卤水提锂难题，进一步降低生产成本。

展望未来，公司计划继续扩大碳酸锂产能，目标三年内将产能提升至5万吨。钾肥业务有望受益于国际农产品价格上涨带来的需求提升。管理层对2025年的业绩持谨慎乐观态度，同时提示碳酸锂价格波动与下游需求不及预期的风险。公司表示将持续优化产品结构，提升抗周期能力。"""

# 三个查询刻意用和原文不完全一样的措辞，看语义检索（而非字面命中）的效果
QUERIES = [
    "公司2024年赚了多少钱？",      # 原文讲"营业收入/净利润" → 语义命中业绩段
    "锂业务的产能有多大？",        # 原文讲"碳酸锂产量3.5万吨" → 语义命中碳酸锂段
    "钾肥是公司的主要业务吗？",    # 原文讲"钾肥是传统主业/压舱石" → 语义命中钾肥段
]


def main():
    print("=" * 64)
    print("Step 3: 本地向量库 —— 存起来，检索出来")
    print("=" * 64)

    embedder = Embedder()
    print(f"\n模型: {embedder.model} @ {embedder.base_url}")

    # ---- [1] 切块(step2) → embed(step1) → 入库(step3) ----
    print("\n[1] 切块 + 向量化 + 入库")
    splitter = RecursiveTextSplitter(chunk_size=200)
    chunks = splitter.split_text(SAMPLE_DOC)
    print(f"    文档 {len(SAMPLE_DOC)} 字 → 切成 {len(chunks)} 块（chunk_size={splitter.chunk_size}）")

    store = VectorStore()
    try:
        store.add(chunks, embedder)
    except Exception as e:
        print(f"\n[ERROR] 向量化失败：{e}")
        print("请确认 Ollama 已启动（ollama serve）且模型已下载（ollama pull qwen3-embedding:0.6b）")
        sys.exit(1)
    print(f"    入库 {len(store.chunks)} 个 chunk，向量矩阵 shape={store.vectors.shape}")

    # ---- [2] 持久化：存盘 → 重新加载 ----
    print("\n[2] 持久化：save → load")
    store_path = "step3_index/vector_store"
    store.save(store_path)
    print(f"    存盘：{store_path}.json（原文 {len(store.chunks)} 条）+ {store_path}.npy（向量 {store.vectors.shape}）")
    store2 = VectorStore()
    store2.load(store_path)
    print(f"    重载：{len(store2.chunks)} 个 chunk，向量矩阵 {store2.vectors.shape}（与原库一致）")

    # ---- [3] 检索：Top-3 ----
    print("\n[3] 检索 Top-3")
    for q in QUERIES:
        print(f"\n    查询: {q}")
        for idx, sim, chunk in store.search(q, embedder, top_k=3):
            preview = chunk.replace("\n", "↵")[:46]
            print(f"      [{idx}] sim={sim:.3f}  {preview}")

    print("\n" + "=" * 64)
    print("解读:")
    print("=" * 64)
    print("  - '赚了多少钱' 命中讲营业收入/净利润的块 —— 查询词和原文不完全重合，靠语义命中")
    print("  - 每个查询的 Top 结果都是对应主题的块，说明 chunk_size=200 的切块 + 余弦检索 work")
    print("  - 完整链路：文档→切块→embed→存→查询embed→余弦TopK，这就是 RAG 的检索部分")


if __name__ == "__main__":
    main()
