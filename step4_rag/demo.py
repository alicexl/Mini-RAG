"""Step 4: 完整 RAG —— 检索增强生成

学习目标:
- 把 step1(embedding) + step2(切块) + step3(向量库检索) + LLM 生成 拼成一个完整的 RAG
- RAG 里「增强（Augmented）」那一步到底做了什么：把检索回来的 chunk 作为 context 塞进 prompt
- 「给 LLM 看检索结果再回答」的完整链路长什么样

运行:
    PYTHONUTF8=1 python step4_rag/demo.py

依赖:
- Ollama 已启动 + qwen3-embedding:0.6b 已下载（检索用，同 step1/3）
- 一个 chat 模型。默认走智谱 GLM 的 Anthropic 兼容网关（glm-5.2）。API Key 三种配置：
  ① 环境变量 ANTHROPIC_API_KEY（推荐）；② 改 demo.py 顶部 API_KEY 常量；
  ③ 都没设 → 运行时交互式输入（仅本次运行有效）。换模型/网关用
  ANTHROPIC_BASE_URL / ANTHROPIC_CHAT_MODEL 覆盖。
"""

import os
import sys

import numpy as np
import requests
from anthropic import Anthropic

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
        # 按优先级找第一个出现在 text 里的分隔符；一路找到空串还没命中，就字符级硬切
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


# ===== VectorStore（同 step3，内联以便 demo 自包含）=====
class VectorStore:
    def __init__(self):
        self.chunks: list[str] = []
        self.vectors: np.ndarray = None      # (N, dim)

    def add(self, texts: list[str], embedder: Embedder):
        vecs = embedder.embed_batch(texts)
        new = np.array(vecs, dtype=np.float32)
        self.vectors = new if self.vectors is None else np.vstack([self.vectors, new])
        self.chunks.extend(texts)

    def search(self, query: str, embedder: Embedder, top_k=3) -> list[tuple[int, float, str]]:
        """检索：查询向量 vs 所有 chunk 向量算余弦，取相似度最高的 Top-K。"""
        q = np.array(embedder.embed_batch([query])[0], dtype=np.float32)
        sims = self._cosine_to_all(q)
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i]), self.chunks[i]) for i in order]

    def _cosine_to_all(self, q: np.ndarray) -> np.ndarray:
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        v_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
        return v_norm @ q_norm        # (N, dim) @ (dim,) -> (N,)


# ===== Generator：调 LLM 生成（本章的新东西）=====
# 检索回来的 chunk 是文本，要交给 LLM 才能生成自然语言答案。这里走智谱 GLM 的 Anthropic 兼容网关。
# API Key 三种配置方式（与 mini-agent 一致）：
#   1. 环境变量 ANTHROPIC_API_KEY（推荐，设了 env 免改代码）
#   2. 直接改下面的 API_KEY 常量
#   3. 都没设 → 运行时交互式输入（仅本次运行有效，不持久化）
API_KEY = ""

_CHAT_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
_CHAT_MODEL = os.environ.get("ANTHROPIC_CHAT_MODEL", "glm-5.2")


class Generator:
    """把 prompt 喂给 chat 模型，拿回生成的文本。"""

    def __init__(self, base_url: str = _CHAT_BASE_URL, model: str = _CHAT_MODEL):
        api_key = os.environ.get("ANTHROPIC_API_KEY") or API_KEY
        if not api_key:                      # env 和常量都没设 → 交互式输入（仅本次运行有效）
            print("\n检测到尚未配置 API Key，请输入（仅本次运行有效）")
            print("如需持久化：请设置环境变量 ANTHROPIC_API_KEY，或改 demo.py 顶部的 API_KEY 常量")
            api_key = input("API Key: ").strip()
            if not api_key:
                raise SystemExit("未提供 API Key，退出")
        self.client = Anthropic(base_url=base_url, api_key=api_key)
        self.model = model

    def answer(self, system: str, user: str, max_tokens: int = 1024) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text


# ===== RAG：把「检索」和「生成」拼起来 =====
# system prompt 约束 LLM 只能用提供的资料回答 —— 这是「增强」的关键一半：
# 不光要把检索结果塞进 prompt，还要明确告诉模型「只用这些、没有就说没有」，压住它凭训练记忆瞎编的冲动。
SYSTEM_PROMPT = (
    "你是一个严谨的财报问答助手。只能根据下面【参考资料】里的内容回答用户问题，"
    "并在用到某条资料时标注它的编号（如「根据[1]」）。"
    "如果【参考资料】里没有相关信息，就如实回答「资料中未提及」，禁止编造。"
)


def build_context(retrieved: list[tuple[int, float, str]]) -> str:
    """把检索回来的 Top-K chunk 拼成一段带编号的 context，喂给 LLM 当作「参考资料」。"""
    parts = [f"[{i}] {chunk}" for i, (_, _, chunk) in enumerate(retrieved, 1)]
    return "\n\n".join(parts)


def rag_answer(question: str, store: VectorStore, embedder: Embedder,
               generator: Generator, top_k: int = 3):
    """一条完整的 RAG：检索 → 拼 prompt → 生成。"""
    retrieved = store.search(question, embedder, top_k=top_k)      # Retrieval
    context = build_context(retrieved)                             # Augmented（把检索结果拼进 prompt）
    user = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"
    answer = generator.answer(SYSTEM_PROMPT, user)                 # Generation
    return retrieved, answer


# ===== Demo =====
SAMPLE_DOC = """\
盐湖股份2024年实现营业收入150亿元，同比增长12%；归母净利润45亿元，同比增长20%。公司业绩增长主要得益于钾肥价格回暖与碳酸锂产销两旺。年报显示，公司整体毛利率较上年提升3个百分点，盈利能力持续改善。

钾肥是公司的传统主业，产能位居全国前列。2024年氯化钾产量约500万吨，销量保持稳定，国内市场份额持续领先。钾肥业务贡献了公司过半的营业收入和利润，是业绩的压舱石。报告期内，公司推进百万吨钾肥扩建项目，投产后将进一步巩固产能优势。

碳酸锂是公司的第二增长曲线。2024年碳酸锂产量达到3.5万吨，产能位居全国前列。公司依托察尔汗盐湖丰富的卤水资源，采用盐湖提锂工艺，生产成本显著低于矿石提锂企业，具备较强的成本优势。盐湖提锂的吨成本约为矿石法的六成，在价格下行周期中仍能保持盈利。

碳酸锂价格在2024年持续下跌，从年初的10万元/吨跌至年末的7万元/吨，行业整体承压。尽管价格走低，盐湖股份凭借低成本优势，碳酸锂业务仍保持盈利，成为少数逆周期盈利的锂盐企业。

公司持续加大研发投入，重点推进盐湖提锂技术的迭代升级与提锂吸附剂的国产化替代。2024年研发投入同比增长15%，多项技术成果实现产业化应用。公司还与高校联合攻关高镁锂比卤水提锂难题，进一步降低生产成本。

展望未来，公司计划继续扩大碳酸锂产能，目标三年内将产能提升至5万吨。钾肥业务有望受益于国际农产品价格上涨带来的需求提升。管理层对2025年的业绩持谨慎乐观态度，同时提示碳酸锂价格波动与下游需求不及预期的风险。公司表示将持续优化产品结构，提升抗周期能力。"""

# 三个查询：前两个的答案藏在原文里（看 LLM 能否从检索到的 chunk 抽出来），
# 第三个故意问原文没写的（看 LLM 是不是真能守住「资料中未提及」而不是瞎编）
QUERIES = [
    "公司2024年赚了多少钱？",
    "锂业务的产能有多大？",
    "公司2025年打算给股东分红多少？",   # 原文未提及 → 检验 LLM 守不守规矩
]


def main():
    print("=" * 64)
    print("Step 4: 完整 RAG —— 检索增强生成")
    print("=" * 64)

    embedder = Embedder()
    generator = Generator()
    print(f"\n检索模型: {embedder.model} @ {embedder.base_url}")
    print(f"生成模型: {generator.model}")

    # ---- [1] 建库：切块(step2) → embed(step1) → 入库(step3) ----
    print("\n[1] 建库：切块 + 向量化 + 入库")
    splitter = RecursiveTextSplitter(chunk_size=200)
    chunks = splitter.split_text(SAMPLE_DOC)
    store = VectorStore()
    try:
        store.add(chunks, embedder)
    except Exception as e:
        print(f"\n[ERROR] 向量化失败：{e}")
        print("请确认 Ollama 已启动且模型已下载（ollama pull qwen3-embedding:0.6b）")
        sys.exit(1)
    print(f"    文档 {len(SAMPLE_DOC)} 字 → {len(chunks)} 块，向量矩阵 {store.vectors.shape}")

    # ---- [2] RAG：检索 → 拼 prompt → 生成 ----
    print("\n[2] RAG 问答（检索 Top-3 → 拼 prompt → LLM 生成）")
    for q in QUERIES:
        print("\n" + "-" * 64)
        print(f"问：{q}")
        try:
            retrieved, answer = rag_answer(q, store, embedder, generator, top_k=3)
        except Exception as e:
            print(f"\n[ERROR] 生成失败：{e}")
            print("请确认已配置 chat 模型的鉴权（ANTHROPIC_API_KEY 或 ANTHROPIC_AUTH_TOKEN）")
            sys.exit(1)

        print("\n检索回来的 Top-3 chunk（作为【参考资料】喂给 LLM）：")
        for i, (idx, sim, chunk) in enumerate(retrieved, 1):
            preview = chunk.replace("\n", "↵")[:40]
            print(f"  [{i}] (chunk#{idx}, sim={sim:.3f}) {preview}")

        print("\nLLM 生成：")
        print(text_wrap(answer))

    print("\n" + "=" * 64)
    print("解读:")
    print("=" * 64)
    print("  - 前两问的答案藏在检索回来的 chunk 里 → LLM 能抽出来并标注编号")
    print("  - 第三问原文没写分红 → 守规矩的 LLM 应回答「资料中未提及」而非瞎编")
    print("  - 完整链路：文档→切块→embed→入库→查询检索→拼context→LLM生成")
    print("    这就是 RAG：检索(Retrieval) + 增强(Augmented) + 生成(Generation)")


def text_wrap(s: str, width: int = 60) -> str:
    """中文友好折行（按宽度切断，保留段落）。"""
    out = []
    for line in s.splitlines() or [s]:
        out.append("\n".join(line[i:i + width] for i in range(0, len(line), width) or [0]))
    return "\n".join(out)


if __name__ == "__main__":
    main()
