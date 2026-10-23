"""Step 4: 用 Flask 手写一个 Web 聊天界面 —— 把 RAG 做成能用的产品

学习目标:
- 把 step3 的命令行 RAG 做成一个 Web 聊天应用（Flask + 原生 HTML/JS，无前端框架）
- 三个产品化改造：① Web 聊天界面（Flask 路由 + 静态页）
  ② 流式生成（SSE，答案一个字一个字蹦出来）
  ③ 文档管理（真 HTTP 文件上传，替代 step3 的命令行 ingest）
- RAG 的核心链路（两级检索→拼 context→system 约束→生成）和 step3 完全一样，
  本章只在外面包一层 Web —— 看清「产品化」到底改了什么、没改什么。

启动:
    # 先备好两个外部依赖（同 step3）：
    #   · Ollama 已启动 + qwen3-embedding:0.6b 已下载（检索用）
    #   · 一个 chat 模型 API Key：set ANTHROPIC_API_KEY=xxx（智谱 glm-5.3 网关）
    python -X utf8 step4_app/demo.py
    # → 浏览器自动打开 http://127.0.0.1:7860

    # 换 chat 模型/网关用 ANTHROPIC_BASE_URL / ANTHROPIC_CHAT_MODEL 覆盖（同 step3）。
"""

import json
import os
import threading
import webbrowser
from pathlib import Path

import numpy as np
import requests
from anthropic import Anthropic
from flask import Flask, Response, jsonify, request, send_from_directory

# 向量库持久化目录（同 step2/3 的三件套：chunks.json + embeddings.npy + metadata.json）
DATA_DIR = Path(__file__).parent / "data"

# 两级检索的宽度（同 step3）：粗排宽进捞多少、精排收紧留几条
RECALL_K = 20
TOP_K = 3


# ================================================================
# 复用：Embedder / RecursiveTextSplitter / VectorStore / Reranker（同 step1/2/3，内联以便自包含）
# 几块逻辑和 step3 一致（省去前面 step 已讲过的 docstring）—— 本 step 不改 RAG 内核，只在外面包 Web。
# ================================================================
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


class RecursiveTextSplitter:
    # 按优先级从高到低排列的分隔符：段落 → 换行 → 句号 → … → 空格 → 空串（字符级兜底，必须放末尾）
    SEPARATORS = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")

    def __init__(self, chunk_size=200, chunk_overlap=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        pieces = self._split(text, self.SEPARATORS)
        return self._merge(pieces)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        for i, sep in enumerate(separators):
            if sep == "":
                return [text[j:j + self.chunk_size] for j in range(0, len(text), self.chunk_size)]
            if sep in text:
                rest = separators[i + 1:]
                break
        parts = text.split(sep)
        pieces = []
        for k, p in enumerate(parts):
            piece = p if k == len(parts) - 1 else p + sep
            if len(piece) <= self.chunk_size:
                pieces.append(piece)
            else:
                pieces.extend(self._split(piece, rest))
        return pieces

    def _merge(self, pieces: list[str]) -> list[str]:
        chunks, cur, cur_len = [], [], 0
        for p in pieces:
            if cur and cur_len + len(p) > self.chunk_size:
                chunks.append("".join(cur))
                while cur and cur_len > self.chunk_overlap:
                    cur_len -= len(cur[0])
                    cur.pop(0)
            cur.append(p)
            cur_len += len(p)
        if cur:
            chunks.append("".join(cur))
        return chunks


class VectorStore:
    def __init__(self):
        self.chunks: list[str] = []
        self.vectors: np.ndarray = None
        self.metadata: list[dict] = []

    def add(self, texts: list[str], embedder: Embedder, metadata: list[dict] | None = None):
        vecs = embedder.embed_batch(texts)
        new = np.array(vecs, dtype=np.float32)
        self.vectors = new if self.vectors is None else np.vstack([self.vectors, new])
        self.chunks.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])

    def save(self, data_dir):
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(data_dir / "embeddings.npy", self.vectors)
        (data_dir / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, data_dir):
        data_dir = Path(data_dir)
        self.chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
        self.vectors = np.load(data_dir / "embeddings.npy")
        self.metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))

    def search(self, query: str, embedder: Embedder, top_k=3) -> list[tuple[int, float, str]]:
        q = np.array(embedder.embed_batch([query])[0], dtype=np.float32)
        sims = self._cosine_to_all(q)
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i]), self.chunks[i]) for i in order]

    def _cosine_to_all(self, q: np.ndarray) -> np.ndarray:
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        v_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
        return v_norm @ q_norm


# ================================================================
# 本章新东西①：Generator 改成「流式」
# step3 的 Generator.answer() 是一次拿回整段答案；产品里用户想看着答案一个字一个字
# 蹦出来（"在思考"的反馈），所以改成 answer_stream —— 用 SDK 的 messages.stream，
# 拿到一个流，逐片 yield 文本增量。
# ================================================================
API_KEY = ""   # 没设 env 时兜底（推荐用环境变量 ANTHROPIC_API_KEY，免改代码）

_CHAT_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
_CHAT_MODEL = os.environ.get("ANTHROPIC_CHAT_MODEL", "glm-5.3")


class Generator:
    """流式生成器：answer_stream 把 prompt 喂给 chat 模型，逐片 yield 文本。

    和 step3 的差别只在「流式」：key 不再交互式 input()（Web 请求处理里没有 stdin），
    改为只认 env / 常量；没配 key 就在生成时抛错，由 Web 层接住、包成 SSE error 事件
    显示在聊天气泡里（见 /api/chat 的 stream()）。
    """

    def __init__(self, base_url: str = _CHAT_BASE_URL, model: str = _CHAT_MODEL):
        self.base_url = base_url
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY") or API_KEY

    def answer_stream(self, system: str, messages: list[dict], max_tokens: int = 2048):
        if not self.api_key:
            raise RuntimeError("未配置 chat 模型的 API Key（请设置环境变量 ANTHROPIC_API_KEY）")
        client = Anthropic(base_url=self.base_url, api_key=self.api_key)
        with client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:        # text_stream 逐片吐已累积的文本增量
                yield text


# ================================================================
# 复用：Reranker（同 step3）—— Cross-Encoder 精排，与生成共用同一个智谱 Key
# ================================================================
_RERANK_URL = "https://open.bigmodel.cn/api/paas/v4/rerank"


class Reranker:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("rerank 需要智谱 API Key（与生成共用 ANTHROPIC_API_KEY）")
        self.api_key = api_key

    def rerank(self, query: str, documents: list[str], top_n: int = TOP_K) -> list[dict]:
        """返回按相关性降序的 [{index, relevance_score}, ...]，index 指回 documents 的下标。"""
        resp = requests.post(
            _RERANK_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": "rerank", "query": query, "documents": documents, "top_n": top_n},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["results"]


# ================================================================
# 复用：SYSTEM_PROMPT + build_context + 两级检索（同 step3）—— RAG 的「增强」内核，Web 层不改它
# ================================================================
SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。只能根据下面【参考资料】里的内容回答用户问题，"
    "并在用到某条资料时标注它的编号（如「根据[1]」）。"
    "如果【参考资料】里没有相关信息，就如实回答「资料中未提及」，禁止编造。"
)


def build_context(retrieved: list[tuple[int, float, float, str]]) -> str:
    """把精排后的 Top-K chunk 拼成带编号的 context（同 step3）。"""
    parts = [f"[{i}] {chunk}" for i, (_, _, _, chunk) in enumerate(retrieved, 1)]
    return "\n\n".join(parts)


def retrieve(question: str, top_k: int = TOP_K):
    """两级检索（同 step3）：向量召回 RECALL_K 条 → rerank 精排取 top_k，失败降级回向量顺序。"""
    candidates = store.search(question, embedder, top_k=RECALL_K)
    try:
        results = reranker.rerank(question, [c[2] for c in candidates], top_n=top_k)
        retrieved = [(candidates[r["index"]][0], candidates[r["index"]][1],
                      r["relevance_score"], candidates[r["index"]][2]) for r in results]
        return retrieved, True
    except Exception:
        retrieved = [(idx, sim, sim, chunk) for idx, sim, chunk in candidates[:top_k]]
        return retrieved, False


# ================================================================
# 本章新东西②：Flask Web 层 —— 静态页 + JSON/SSE 端点
# static_folder=None：关掉 Flask 自带的 /static/<文件> 路由，页面统一从 / 进
#（send_from_directory 手动返回，把「静态文件是怎么被服务的」摊开来看）
# ================================================================
STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/status")
def status():
    """左栏库状态：现有多少 chunk、已导入哪些文档（按来源去重）。"""
    return jsonify(
        chunks=len(store.chunks),
        documents=sorted({m.get("source", "?") for m in store.metadata}),
    )


def _ingest_text(text: str, source: str) -> tuple[int, bool, str]:
    """切块 + 向量化 + 入库一份文档（同 step3 的 ingest）。幂等：source 已在库则跳过。返回 (块数, 是否新增, 提示)。"""
    if any(m.get("source") == source for m in store.metadata):
        return 0, False, f"[SKIP]「{source}」已在库中"
    chunks = splitter.split_text(text)
    store.add(chunks, embedder, metadata=[{"source": source} for _ in chunks])
    store.save(str(DATA_DIR))
    return len(chunks), True, f"[OK]「{source}」+{len(chunks)} 块"


@app.post("/api/upload")
def upload():
    """「导入」端点：multipart 真上传——文件由浏览器文件框选出、内容随请求发上来，
    服务端只收内容、不碰服务端文件系统（本地 / 远端部署行为一致）。"""
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="未收到文件（multipart 字段名须为 files）"), 400
    notes = []
    for f in files:
        source = Path(f.filename).stem        # 同 step3 ingest 口径：source 取文件名（去扩展名）
        text = f.read().decode("utf-8", errors="replace")
        notes.append(_ingest_text(text, source)[2])
    return jsonify(chunks=len(store.chunks), notes=notes)


@app.post("/api/clear")
def clear():
    """「清空」端点：内存清零 + 删掉 data/ 落盘三件套。"""
    store.chunks, store.metadata, store.vectors = [], [], None
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in DATA_DIR.glob("*"):
        try:
            f.unlink()
        except OSError:
            pass
    return jsonify(chunks=0, documents=[])


def _sse(event: str, data: dict) -> str:
    """SSE 帧：event 名 + JSON data 两行，空行结尾（前端按空行切帧）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
def chat():
    """聊天端点。错误分两层：请求级校验（空消息 / 空库）走 400 JSON——那时响应头还没定，
    可以随便返；流一旦开始（响应头已是 text/event-stream）中途出错就只能往下写
    SSE error 事件，前端把它显示在助手气泡里。"""
    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    history = body.get("history") or []
    if not message:
        return jsonify(error="message 不能为空"), 400
    if not store.chunks:
        return jsonify(error="向量库为空，请先在左侧导入文档"), 400

    retrieved, reranked = retrieve(message, top_k=TOP_K)              # 两级检索：召回 → 精排
    context = build_context(retrieved)                                # 增强①：拼 context
    user_msg = f"【参考资料】\n{context}\n\n【用户问题】\n{message}"
    # 把「前端带上来的对话历史 + 本轮 user（带参考资料）」一起喂给 LLM —— 多轮里能接住追问
    llm_messages = history + [{"role": "user", "content": user_msg}]
    sources_payload = [
        {"i": i, "sim": round(sim, 3), "score": round(score, 3),
         "source": store.metadata[idx].get("source", "?"),
         "preview": chunk.replace("\n", " ").strip()[:80]}
        for i, (idx, sim, score, chunk) in enumerate(retrieved, 1)
    ]

    def stream():
        yield _sse("sources", {"sources": sources_payload,
                               "reranked": reranked})                  # 先亮来源（rerank 失败时前端可提示）
        try:
            for delta in generator.answer_stream(SYSTEM_PROMPT, llm_messages):
                yield _sse("delta", {"text": delta})                 # 每个 LLM 增量一帧
            yield _sse("done", {})
        except Exception as e:
            yield _sse("error", {"message": f"生成失败：{e}\n请确认 chat 模型鉴权已配置（ANTHROPIC_API_KEY）"})

    # Flask 的流式响应：传入生成器，每 yield 一段就写一段到 socket——不需要把答案攒齐
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache"})


def main():
    # 进程级单例：embedder / splitter / store / generator（教学 demo 单用户，全局共享）。
    # 放在 main 里、key 确认之后初始化：import 本模块零副作用（不读盘、不打日志），
    # 启动顺序一条线——配置确认 → 初始化 → 起服务。路由处理函数引用的就是这几个模块级名字
    #（Python 调用时才解析全局名，所以在 main 里 global 赋值即可）。
    global embedder, splitter, generator, reranker, store
    generator = Generator()

    # 启动前确保 chat 模型 key 就位：env/常量都没设 → 交互式问一次；空着就直接退出。
    # 这里在主线程、起服务之前，终端 stdin 可用（不像请求处理线程里没 stdin）。
    if not generator.api_key:
        print("=" * 60)
        print("检测到尚未配置 chat 模型的 API Key")
        print("（智谱 glm-5.3，bigmodel.cn 生成，id.secret 格式）")
        print("持久化方式：设置环境变量 ANTHROPIC_API_KEY，或改 demo.py 顶部 API_KEY 常量")
        key = input("请输入 API Key（直接回车则退出）: ").strip()
        if not key:
            raise SystemExit("未提供 API Key，退出")
        generator.api_key = key

    embedder = Embedder()
    splitter = RecursiveTextSplitter(chunk_size=400, chunk_overlap=50)   # 同 step3 ingest 口径
    reranker = Reranker(generator.api_key)     # 精排与生成共用同一个智谱 Key
    store = VectorStore()
    if (DATA_DIR / "chunks.json").exists():
        try:
            store.load(str(DATA_DIR))
        except Exception as e:
            print(f"[WARN] 读回向量库失败（{e}），从空库开始")

    print("=" * 60)
    print(f"Mini-RAG 聊天应用  ·  检索 {embedder.model}  ·  生成 {generator.model}")
    print(f"向量库：{len(store.chunks)} 个 chunk  ·  API Key：已配置")
    print("浏览器打开 http://127.0.0.1:7860  ·  Ctrl+C 退出")
    print("=" * 60)
    # 等服务真的起来再开页（app.run 是阻塞的，直接 open 会抢在端口监听之前）
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:7860")).start()
    app.run(host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
