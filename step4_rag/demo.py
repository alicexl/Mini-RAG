"""Step 4: 完整 RAG —— 检索增强生成

学习目标:
- 把 step1(embedding) + step2(切块) + step3(向量库检索) + LLM 生成 拼成一个完整的 RAG
- RAG 里「增强（Augmented）」那一步到底做了什么：把检索回来的 chunk 作为 context 塞进 prompt
- 「给 LLM 看检索结果再回答」的完整链路长什么样

用法（两个命令，按用户场景划分）:
    # ① 导入单个文档：拆分 + 向量化 + 入库（幂等：同来源已导入则跳过，来源名取文件名）
    PYTHONUTF8=1 python step4_rag/demo.py ingest <文档路径>

    # ② 启动交互式问答 terminal（quit / exit 退出）
    PYTHONUTF8=1 python step4_rag/demo.py chat

依赖:
- Ollama 已启动 + qwen3-embedding:0.6b 已下载（检索用，同 step1/3）
- 一个 chat 模型。默认走智谱 GLM 的 Anthropic 兼容网关（glm-5.2）。API Key 三种配置：
  ① 环境变量 ANTHROPIC_API_KEY（推荐）；② 改 demo.py 顶部 API_KEY 常量；
  ③ 都没设 → 运行时交互式输入（仅本次运行有效）。换模型/网关用
  ANTHROPIC_BASE_URL / ANTHROPIC_CHAT_MODEL 覆盖。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from anthropic import Anthropic

# 向量库持久化目录（chunks.json + embeddings.npy + metadata.json 三件套，同 step3）
DATA_DIR = Path(__file__).parent / "data"

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
# data/ 目录下三个文件：原文 chunks.json、向量 embeddings.npy、元信息 metadata.json。
# 元信息记每块的来源（哪份文档），检索命中后知道答案出自哪。
class VectorStore:
    def __init__(self):
        self.chunks: list[str] = []
        self.vectors: np.ndarray = None      # (N, dim)
        self.metadata: list[dict] = []       # 与 chunks 等长：每块的来源等元信息

    def add(self, texts: list[str], embedder: Embedder, metadata: list[dict] | None = None):
        """把若干段文本 embed 后入库；metadata 与 texts 等长，记录每块的来源等信息。"""
        vecs = embedder.embed_batch(texts)
        new = np.array(vecs, dtype=np.float32)
        self.vectors = new if self.vectors is None else np.vstack([self.vectors, new])
        self.chunks.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])

    def save(self, data_dir: str):
        """持久化到 data/ 目录：原文、向量、元信息各一个文件。"""
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(data_dir / "embeddings.npy", self.vectors)
        (data_dir / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, data_dir: str):
        """从 data/ 目录读回一个向量库。"""
        data_dir = Path(data_dir)
        self.chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
        self.vectors = np.load(data_dir / "embeddings.npy")
        self.metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

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
# API Key 三种配置方式：
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
    return retrieved, answer, context


# ===== 命令一：ingest —— 导入单个文档 =====
def cmd_ingest(doc_path: str):
    """读文档 → 切块(step2) → 向量化(step1) → 入库(step3)。

    来源名取文件名（metadata 用它标注，检索结果里能看到出自哪份文档）。
    幂等：同来源已在库中则跳过（按 metadata 的 source 判定），重复导入不重复入库。
    """
    text = Path(doc_path).read_text(encoding="utf-8")
    source = Path(doc_path).stem

    store = VectorStore()
    if (DATA_DIR / "chunks.json").exists():
        store.load(str(DATA_DIR))
    if any(m.get("source") == source for m in store.metadata):
        print(f"[SKIP]「{source}」已在库中（共 {len(store.chunks)} chunks），跳过。")
        print(f"      要重新导入请先删掉 {DATA_DIR}/")
        return

    splitter = RecursiveTextSplitter(chunk_size=200)
    chunks = splitter.split_text(text)
    embedder = Embedder()
    try:
        store.add(chunks, embedder, metadata=[{"source": source} for _ in chunks])
    except Exception as e:
        print(f"[ERROR] 向量化失败：{e}")
        print("请确认 Ollama 已启动且模型已下载（ollama pull qwen3-embedding:0.6b）")
        sys.exit(1)
    store.save(str(DATA_DIR))
    print(f"[OK]「{source}」：{len(text)} 字 → {len(chunks)} 块，向量矩阵 {store.vectors.shape}")
    print(f"     库现有 {len(store.chunks)} 个 chunk（{DATA_DIR}/）")


# ===== 命令二：chat —— 交互式问答 terminal =====
# 示例问题：前两个的答案在语料里（看 LLM 能否从检索到的 chunk 抽出来），
# 最后一个语料没写（看 LLM 是不是真能守住「资料中未提及」而不是瞎编）。
QUERIES = [
    "公司2024年赚了多少钱？",
    "2023年和2024年，钾肥产销量分别是什么水平？",
    "公司2025年打算给股东分红多少？",   # 语料未提及 → 检验 LLM 守不守规矩
]


def cmd_chat():
    """交互式问答 terminal：输入问题 → 检索 Top-K（带来源）→ 拼 context → LLM 生成。"""
    embedder = Embedder()
    store = VectorStore()
    if not (DATA_DIR / "chunks.json").exists():
        print("[ERROR] 向量库为空。请先导入文档：")
        print("    python step4_rag/demo.py ingest <文档路径>")
        sys.exit(1)
    store.load(str(DATA_DIR))
    generator = Generator()

    print("=" * 64)
    print(f"向量库：{len(store.chunks)} 个 chunk（检索 {embedder.model} / 生成 {generator.model}）")
    print("输入问题开始问答，quit / exit 退出。试试：")
    for q in QUERIES:
        print(f"  · {q}")
    print("=" * 64)

    while True:
        q = input("\n查询> ").strip()
        if q.lower() in ("quit", "exit", "q", "退出"):
            break
        if not q:
            continue
        try:
            retrieved, answer, context = rag_answer(q, store, embedder, generator, top_k=3)
        except Exception as e:
            print(f"\n[ERROR] 生成失败：{e}")
            print("请确认已配置 chat 模型鉴权（ANTHROPIC_API_KEY / API_KEY 常量 / 交互输入）")
            continue

        print("\n检索 Top-3（[1][2][3] 即 LLM 回答里「根据[1]」的引用；方括号里是元信息来源）：")
        for i, (idx, sim, chunk) in enumerate(retrieved, 1):
            src = store.metadata[idx].get("source", "")
            raw = chunk.replace("\n", "↵")
            preview = raw[:46] + ("…" if len(raw) > 46 else "")
            print(f"  [{i}] sim={sim:.3f} [{src}] {preview}")
        # 增强：把检索结果拼成 prompt 喂给 LLM —— RAG 的关键一步（检索结果 → prompt）
        print("\n→ 拼 prompt 喂给 LLM（增强 = 检索结果 + system 约束）：")
        print("  [system] 只能用【参考资料】，引用编号（如「根据[1]」），没有就说没有")
        ctx_oneline = context.replace(chr(10), " ")
        print(f"  [参考资料] {ctx_oneline[:80]}{'…' if len(ctx_oneline) > 80 else ''}")
        print(f"  [用户问题] {q}")
        print("\n→ LLM 生成：")
        print(text_wrap(answer))


def main():
    parser = argparse.ArgumentParser(description="Step 4: 完整 RAG —— 检索增强生成")
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="{ingest,chat}")

    p_ingest = sub.add_parser("ingest", help="导入单个文档（拆分 + 向量化 + 入库，幂等）")
    p_ingest.add_argument("doc", help="文档路径（md/txt），来源名取文件名")

    sub.add_parser("chat", help="启动交互式问答 terminal")

    args = parser.parse_args()
    if args.cmd == "ingest":
        cmd_ingest(args.doc)
    else:
        cmd_chat()


def text_wrap(s: str, width: int = 60) -> str:
    """中文友好折行（按宽度切断，保留段落）。"""
    out = []
    for line in s.splitlines() or [s]:
        out.append("\n".join(line[i:i + width] for i in range(0, len(line), width) or [0]))
    return "\n".join(out)


if __name__ == "__main__":
    main()
