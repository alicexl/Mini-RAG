"""Step 4: 用 Flask 手写一个 Web 聊天界面 —— 把 RAG 做成能用的产品

学习目标:
- 把 step3 的命令行 RAG 做成一个 Web 聊天应用（Flask + 原生 HTML/JS，无前端框架）
- 四个产品化改造：① Web 聊天界面（Flask 路由 + 静态页）
  ② 流式生成（SSE，答案一个字一个字蹦出来）
  ③ 文档管理（真 HTTP 文件上传，替代 step3 的命令行 ingest）
  ④ 检索底座工程化：step3 手搓的「向量库 + BM25 + RRF」三件套，
    换成 Milvus Lite 单文件向量库——两路召回与融合都在库内完成。
    这套代码就是生产同款：把 db 路径换成服务 uri 即平移到
    Milvus Standalone / Zilliz Cloud，API 一行不改
- RAG 的链路概念（混合检索→拼 context→system 约束→生成）与 step3 一致：
  本章换的是零件，不是原理——对照 step3 看「同一件事的手搓版与工程版」

启动:
    # 先备好外部依赖（同 step3）：
    #   · Ollama 已启动 + qwen3-embedding:0.6b 已下载（检索用）
    #   · 一个 chat 模型 API Key：set ANTHROPIC_API_KEY=xxx（智谱 glm-5.3 网关）
    #   · pip install -r requirements.txt（含 pymilvus + milvus-lite）
    python -X utf8 step4_app/demo.py
    # → 浏览器自动打开 http://127.0.0.1:7860
    # 注意：Milvus Lite 一次只允许一个进程打开 data/mini_rag.db（文件锁）。
    # 本 demo 单进程启动，不要开 debug reloader（它会起父子两个进程抢锁）

    # 换 chat 模型/网关用 ANTHROPIC_BASE_URL / ANTHROPIC_CHAT_MODEL 覆盖（同 step3）。
"""

import json
import os
import threading
import webbrowser
from pathlib import Path

import requests
from anthropic import Anthropic
from flask import Flask, Response, jsonify, request, send_from_directory
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

# 向量库持久化：Milvus Lite 单文件库 data/mini_rag.db（step2/3 手搓三件套在此退休）
DATA_DIR = Path(__file__).parent / "data"

# 两级检索的宽度（同 step3）：粗排宽进捞多少、精排收紧留几条
RECALL_K = 20
TOP_K = 3
RRF_K = 60      # RRFRanker 的 k（同 step3 手写 rrf_merge 的默认值）


# ================================================================
# 复用：Embedder / RecursiveTextSplitter（同 step1，内联以便自包含）
# 几块逻辑和 step1 一致（省去前面 step 已讲过的 docstring）。
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


# ================================================================
# 本章新东西①：MilvusStore —— 检索底座从手搓三件套换成 Milvus Lite
# step3 里 VectorStore(JSON+numpy余弦) + BM25Index(jieba手搓Okapi) + rrf_merge(1/(k+rank))
# 三件各管一段；这里一张 collection 全包了：
#   · text 字段开 jieba 分析器 + BM25 函数 → 稀疏路：插的是原文，分词建索引库内自己干
#   · vector 字段 HNSW/COSINE 索引 → 稠密路：向量仍由 Ollama 算好喂进来
#   · hybrid_search(两路, RRFRanker) → 融合：step3 手写的名次归并，库里一个参数
# ================================================================
class MilvusStore:
    """Milvus Lite 向量库：稠密 + BM25 两路检索与 RRF 融合都在库内完成。"""

    COLLECTION = "docs"
    DIM = 1024                  # qwen3-embedding:0.6b 的维度（同 step1）

    def __init__(self, db_path: str):
        # 传本地 .db 路径 = 启动嵌入式 Milvus Lite（不用装服务；文件存在则接着用）
        self.client = MilvusClient(db_path)
        self._ensure_collection()

    def _ensure_collection(self):
        """首次运行建 collection（schema + 索引）；已存在则加载后直接用。"""
        if self.client.has_collection(self.COLLECTION):
            self.client.load_collection(self.COLLECTION)
            return
        schema = MilvusClient.create_schema(auto_id=True)            # 主键由库自动分配
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field(
            "text", DataType.VARCHAR, max_length=65535,
            enable_analyzer=True, analyzer_params={"type": "jieba"},  # 中文按词切（不按字切）
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)      # BM25 函数的输出位，不用手填
        schema.add_function(Function(
            name="bm25", function_type=FunctionType.BM25,
            input_field_names=["text"], output_field_names=["sparse"],
        ))
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.DIM)
        schema.add_field("source", DataType.VARCHAR, max_length=256)  # 来源文件名（幂等键 + 展示）

        index = self.client.prepare_index_params()
        index.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",
                        params={"M": 16, "efConstruction": 200})
        index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                        metric_type="BM25")
        self.client.create_collection(self.COLLECTION, schema=schema, index_params=index)
        self.client.load_collection(self.COLLECTION)

    def insert_chunks(self, chunks: list[str], vectors: list[list[float]], source: str):
        """入库即持久化（库内 WAL 落盘）——手搓版要手动 save 三件套，这里没有这一步。"""
        self.client.insert(self.COLLECTION,
                           [{"text": c, "vector": v, "source": source}
                            for c, v in zip(chunks, vectors)])

    def has_source(self, source: str) -> bool:
        """幂等判断：该来源是否已在库（标量过滤查询，step3 手搓版要扫 metadata 列表）。"""
        rows = self.client.query(self.COLLECTION, filter=f'source == "{source}"',
                                 output_fields=["id"])
        return bool(rows)

    def hybrid_search(self, query: str, query_vec: list[float], top_k: int):
        """两路召回 + RRF 融合，一步在库内完成。

        返回命中列表 [{id, distance(=RRF融合分), entity{text, source}}, ...] 按融合分降序。"""
        dense = AnnSearchRequest(          # 稠密路：语义相近
            data=[query_vec], anns_field="vector",
            param={"metric_type": "COSINE"}, limit=top_k)
        sparse = AnnSearchRequest(         # 稀疏路：字面命中（查询原文进来，库内 jieba 分词打 BM25）
            data=[query], anns_field="sparse",
            param={"metric_type": "BM25"}, limit=top_k)
        hits = self.client.hybrid_search(
            self.COLLECTION, [dense, sparse], ranker=RRFRanker(RRF_K),
            limit=top_k, output_fields=["text", "source"])
        return hits[0]

    def chunk_count(self) -> int:
        return int(self.client.get_collection_stats(self.COLLECTION)["row_count"])

    def sources(self) -> list[str]:
        """库内已有哪些来源（按 source 去重，左栏状态用）。"""
        rows = self.client.query(self.COLLECTION, filter='source != ""',
                                 output_fields=["source"], limit=16384)
        return sorted({r["source"] for r in rows})

    def clear(self):
        """清空：drop 掉 collection 重建——比删 .db 文件干净，也避开文件锁。"""
        self.client.drop_collection(self.COLLECTION)
        self._ensure_collection()


# ================================================================
# 本章新东西②：Generator 改成「流式」
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
# 检索链路里唯一留在库外的环节：rerank 要过一遍大模型，向量库不管这件事
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
# 复用：SYSTEM_PROMPT + build_context（同 step3）—— RAG 的「增强」内核
# retrieve 的三段式形状也同 step3（两路召回→融合→精排），底座换成 Milvus
# ================================================================
SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。只能根据下面【参考资料】里的内容回答用户问题，"
    "并在用到某条资料时标注它的编号（如「根据[1]」）。"
    "如果【参考资料】里没有相关信息，就如实回答「资料中未提及」，禁止编造。"
)


def build_context(retrieved: list[tuple[str, float, float | None, str]]) -> str:
    """把精排后的 Top-K chunk 拼成带编号的 context（同 step3）。"""
    parts = [f"[{i}] {chunk}" for i, (chunk, _, _, _) in enumerate(retrieved, 1)]
    return "\n\n".join(parts)


def retrieve(question: str, top_k: int = TOP_K):
    """混合检索（底座 = Milvus）：库内两路召回 + RRF 融合 Top-20 → rerank 精排 Top-3，
    失败降级回融合顺序（同 step3 的降级语义）。

    每条结果为 (chunk原文, RRF融合分, rerank分|降级时None, 来源)。
    """
    query_vec = embedder.embed_batch([question])[0]
    hits = store.hybrid_search(question, query_vec, top_k=RECALL_K)   # 稠密+BM25 两路 → RRF
    docs = [h["entity"]["text"] for h in hits]
    try:
        results = reranker.rerank(question, docs, top_n=top_k)
        retrieved = [(docs[r["index"]], hits[r["index"]]["distance"],
                      r["relevance_score"], hits[r["index"]]["entity"]["source"])
                     for r in results]
        return retrieved, True
    except Exception:
        retrieved = [(h["entity"]["text"], h["distance"], None, h["entity"]["source"])
                     for h in hits[:top_k]]
        return retrieved, False


# ================================================================
# 本章新东西③：Flask Web 层 —— 静态页 + JSON/SSE 端点
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
        chunks=store.chunk_count(),
        documents=store.sources(),
    )


def _ingest_text(text: str, source: str) -> tuple[int, bool, str]:
    """切块 + 向量化 + 入库一份文档。幂等：source 已在库则跳过。返回 (块数, 是否新增, 提示)。"""
    if store.has_source(source):
        return 0, False, f"[SKIP]「{source}」已在库中"
    chunks = splitter.split_text(text)
    store.insert_chunks(chunks, embedder.embed_batch(chunks), source)
    # 手搓版这里还要 save 三件套 + 重建 BM25 索引；Milvus 落库即持久化、稀疏路自动更新
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
    return jsonify(chunks=store.chunk_count(), notes=notes)


@app.post("/api/clear")
def clear():
    """「清空」端点：drop 掉 collection 重建（.db 文件保留，库回到空）。"""
    store.clear()
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
    if not store.chunk_count():
        return jsonify(error="向量库为空，请先在左侧导入文档"), 400

    retrieved, reranked = retrieve(message, top_k=TOP_K)              # 混合检索：两路召回→RRF→精排
    context = build_context(retrieved)                                # 增强①：拼 context
    user_msg = f"【参考资料】\n{context}\n\n【用户问题】\n{message}"
    # 把「前端带上来的对话历史 + 本轮 user（带参考资料）」一起喂给 LLM —— 多轮里能接住追问
    llm_messages = history + [{"role": "user", "content": user_msg}]
    sources_payload = [
        {"i": i, "rrf": round(rrf, 4),
         "score": (round(score, 3) if score is not None else None),
         "source": src, "preview": chunk.replace("\n", " ").strip()[:80]}
        for i, (chunk, rrf, score, src) in enumerate(retrieved, 1)
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = MilvusStore(str(DATA_DIR / "mini_rag.db"))   # 打开/创建 Milvus Lite 单文件库

    print("=" * 60)
    print(f"Mini-RAG 聊天应用  ·  检索 {embedder.model}  ·  生成 {generator.model}")
    print(f"向量库：{store.chunk_count()} 个 chunk  ·  Milvus Lite（data/mini_rag.db）·  API Key：已配置")
    print("浏览器打开 http://127.0.0.1:7860  ·  Ctrl+C 退出")
    print("=" * 60)
    # 等服务真的起来再开页（app.run 是阻塞的，直接 open 会抢在端口监听之前）
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:7860")).start()
    app.run(host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
