"""Step 5: 套一个 Gradio 聊天界面 —— 把 RAG 做成能用的产品

学习目标:
- 把 step4 的命令行 RAG 套成一个 Web 聊天界面（Gradio）
- 三个产品化改造：① Web 聊天界面 ② 流式生成（答案一个字一个字蹦出来）
  ③ 文档管理面板（点「导入」弹系统文件框选文档，本地 demo 特有）
- RAG 的核心链路（检索→拼 context→system 约束→生成）和 step4 完全一样，
  本章只在外面包一层 UI —— 看清「产品化」到底改了什么、没改什么。

启动:
    # 先备好两个外部依赖（同 step4）：
    #   · Ollama 已启动 + qwen3-embedding:0.6b 已下载（检索用）
    #   · 一个 chat 模型 API Key：set ANTHROPIC_API_KEY=xxx（智谱 glm-5.2 网关）
    PYTHONUTF8=1 python step5_app/demo.py
    # → 浏览器打开 http://127.0.0.1:7860

    # 换 chat 模型/网关用 ANTHROPIC_BASE_URL / ANTHROPIC_CHAT_MODEL 覆盖（同 step4）。
"""

import json
import os
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import gradio as gr
import numpy as np
import requests
from anthropic import Anthropic

# 向量库持久化目录（同 step3/4 的三件套：chunks.json + embeddings.npy + metadata.json）
DATA_DIR = Path(__file__).parent / "data"
# 示例语料目录：「导入」弹框的默认打开位置（从 step4 拷来的盐湖股份 3 年年报节选；不锁死，别处的 md/txt 也能选）
SAMPLE_DIR = Path(__file__).parent / "sample_reports"


# ================================================================
# 复用：Embedder / RecursiveTextSplitter / VectorStore（同 step1/2/3/4，内联以便自包含）
# 三块逻辑和 step4 一致（省去前面 step 已讲过的 docstring）—— step5 不改 RAG 内核，只在外面包 UI。
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
# step4 的 Generator.answer() 是一次拿回整段答案；产品里用户想看着答案一个字一个字
# 蹦出来（"在思考"的反馈），所以改成 answer_stream —— 用 SDK 的 messages.stream，
# 拿到一个流，逐片 yield 文本增量。
# ================================================================
API_KEY = ""   # 没设 env 时兜底（推荐用环境变量 ANTHROPIC_API_KEY，免改代码）

_CHAT_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
_CHAT_MODEL = os.environ.get("ANTHROPIC_CHAT_MODEL", "glm-5.2")


class Generator:
    """流式生成器：answer_stream 把 prompt 喂给 chat 模型，逐片 yield 文本。

    和 step4 的差别只在「流式」：key 不再交互式 input()（Web 服务没有 stdin），
    改为只认 env / 常量；没配 key 就在生成时抛错，由界面层接住、显示在聊天气泡里。
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
# 复用：SYSTEM_PROMPT + build_context（同 step4）—— RAG 的「增强」内核，UI 不改它
# ================================================================
SYSTEM_PROMPT = (
    "你是一个严谨的问答助手。只能根据下面【参考资料】里的内容回答用户问题，"
    "并在用到某条资料时标注它的编号（如「根据[1]」）。"
    "如果【参考资料】里没有相关信息，就如实回答「资料中未提及」，禁止编造。"
)


def build_context(retrieved: list[tuple[int, float, str]]) -> str:
    """把检索回来的 Top-K chunk 拼成带编号的 context（同 step4）。"""
    parts = [f"[{i}] {chunk}" for i, (_, _, chunk) in enumerate(retrieved, 1)]
    return "\n\n".join(parts)


# ================================================================
# 进程级单例：embedder / splitter / store / generator（教学 demo 单用户，全局共享）
# ================================================================
embedder = Embedder()
splitter = RecursiveTextSplitter(chunk_size=200)
generator = Generator()
store = VectorStore()
if (DATA_DIR / "chunks.json").exists():
    try:
        store.load(str(DATA_DIR))
    except Exception as e:
        print(f"[WARN] 读回向量库失败（{e}），从空库开始")


# ================================================================
# 本章新东西②：Gradio 回调（文档管理 + 聊天流式）
# ================================================================

def _status_text(note: str | None = None) -> str:
    """左栏库状态：现有多少 chunk、已导入哪些文档（按来源去重）。"""
    n = len(store.chunks)
    sources = sorted({m.get("source", "?") for m in store.metadata})
    src_lines = "\n".join(f"- {s}" for s in sources) or "_(空，请点「导入」选择文档)_"
    out = f"**向量库**：{n} 个 chunk\n\n**已导入文档**：\n{src_lines}"
    if note:
        out += f"\n\n_{note}_"
    return out


def _md_escape(s: str) -> str:
    """转义 Markdown 语法字符，让 chunk 原文按纯文本显示
    （否则原文里的 "## 标题" 会被渲染成大号标题、"*重点*" 变斜体，字号错乱）。"""
    return re.sub(r"([#*_`~|>\[\]()!\\])", r"\\\1", s)


def _format_sources(retrieved: list[tuple[int, float, str]]) -> str:
    """右栏检索来源：把本次 Top-K 列成带相似度和出处的清单（即 LLM 回答里「根据[1]」的出处）。
    标号单独一行，预览放进引用块（自带缩进 + 换行），条目之间空行隔开。"""
    if not retrieved:
        return "_(提问后，这里显示本次检索的来源 [1][2][3] 及出处文件)_"
    parts = ["**本次检索来源（Top-3，即 LLM 回答里「根据[1]」的出处）**："]
    for i, (idx, sim, chunk) in enumerate(retrieved, 1):
        src = store.metadata[idx].get("source", "?")
        preview = chunk.replace("\n", " ").strip()
        if len(preview) > 80:
            preview = preview[:80] + "…"
        parts.append(f"**[{i}]** `sim={sim:.3f}` · **{src}**\n\n> {_md_escape(preview)}")
    return "\n\n".join(parts)


def _ingest_text(text: str, source: str) -> tuple[int, bool, str]:
    """切块 + 向量化 + 入库一条文档。幂等：source 已在库则跳过。返回 (块数, 是否新增, 提示)。"""
    if any(m.get("source") == source for m in store.metadata):
        return 0, False, f"[SKIP]「{source}」已在库中"
    chunks = splitter.split_text(text)
    store.add(chunks, embedder, metadata=[{"source": source} for _ in chunks])
    store.save(str(DATA_DIR))
    return len(chunks), True, f"[OK]「{source}」+{len(chunks)} 块"


def ingest_dialog():
    """「导入」按钮回调：弹系统文件选择框（默认打开 sample_reports/），选中即导入。

    本地 demo 限定：服务端和浏览器在同一台机器，服务端弹的框用户才看得见——
    真部署到远端服务器，框会弹在服务器上没人点（产品化边界，见讲稿）。
    """
    root = tk.Tk()
    root.withdraw()                    # 不显示主窗口，只要对话框
    root.attributes("-topmost", True)  # 保证框弹在浏览器前面，不被挡住
    paths = filedialog.askopenfilenames(
        title="选择要导入的 md 文档",
        initialdir=str(SAMPLE_DIR),
        filetypes=[("Markdown/文本文档", "*.md *.txt")],
    )
    root.destroy()
    if not paths:
        return _status_text("未选择文件")
    notes, added = [], False
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8")
        except Exception as e:
            notes.append(f"[ERROR] 读取 {Path(p).name} 失败：{e}")
            continue
        _, ok, msg = _ingest_text(text, Path(p).stem)   # source 取文件名（同 step4 ingest 口径）
        added = added or ok
        notes.append(msg)
    if not added:
        return _status_text("已导入，无需重复导入")       # 选中的都已在库 → 提示已导入
    return _status_text("\n".join(notes))


def clear_store():
    """「清空向量库」按钮回调。"""
    store.chunks, store.metadata, store.vectors = [], [], None
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in DATA_DIR.glob("*"):
        try:
            f.unlink()
        except OSError:
            pass
    return _status_text("已清空向量库")


def respond(message, history):
    """聊天 submit 回调（生成器 → 流式）：检索 → 拼 context → 喂 LLM 流式生成。
    每一 yield 刷新 (chatbot, sources_md, msg)：chatbot 边收边显，msg 清空输入框。"""
    message = (message or "").strip()
    if not message:
        yield history, _format_sources([]), gr.update()
        return
    if not store.chunks:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ 向量库为空。请先在左侧点「导入」选择文档。"},
        ], "", gr.update()
        return

    retrieved = store.search(message, embedder, top_k=3)             # 检索
    context = build_context(retrieved)                               # 增强①：拼 context
    sources_md = _format_sources(retrieved)
    user_msg = f"【参考资料】\n{context}\n\n【用户问题】\n{message}"

    # 把「已有对话历史 + 本轮 user（带参考资料）」一起喂给 LLM —— 多轮里能接住追问
    llm_messages = history + [{"role": "user", "content": user_msg}]
    turned = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    yield turned, sources_md, ""                                      # 先亮出问题 + 来源，助手气泡先空着

    try:
        for chunk in generator.answer_stream(SYSTEM_PROMPT, llm_messages):   # 增强② + 生成（流式）
            turned[-1]["content"] += chunk
            yield turned, sources_md, ""
    except Exception as e:
        turned[-1]["content"] = f"⚠️ 生成失败：{e}\n请确认 chat 模型鉴权已配置（ANTHROPIC_API_KEY）"
        yield turned, sources_md, ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Mini-RAG · 问答助手") as app:
        gr.Markdown("# Mini-RAG · 问答助手\n把 Step 1-4 的 RAG 套成一个能用的聊天产品。")
        with gr.Row():
            # 左栏：文档管理（产品化改造③ —— 导入语料从敲命令变成点按钮弹框选文件）
            with gr.Column(scale=1):
                status = gr.Markdown(_status_text())
                gr.Markdown("---")
                btn_import = gr.Button("导入", variant="primary")
                btn_clear = gr.Button("清空", variant="stop")
            # 右栏：聊天（产品化改造① + ② —— Web 界面 + 流式）
            with gr.Column(scale=2):
                sources = gr.Markdown(_format_sources([]))
                chatbot = gr.Chatbot(height=460,
                                     placeholder="先在左侧导入文档，然后在这里提问…")
                msg = gr.Textbox(placeholder="问点关于已导入文档的…（Enter 发送）",
                                 label="问题", scale=4)
                send = gr.Button("发送", variant="primary", scale=1)

        # 事件绑定
        btn_import.click(ingest_dialog, [], [status])
        btn_clear.click(clear_store, [], [status])
        send_kwargs = dict(fn=respond, inputs=[msg, chatbot], outputs=[chatbot, sources, msg])
        msg.submit(**send_kwargs)
        send.click(**send_kwargs)
        # 每次页面加载/刷新都按「实际的」向量库重算状态（否则刷新会回退到服务启动那一刻的快照）
        app.load(_status_text, outputs=[status])
    return app


def main():
    # 启动前确保 chat 模型 key 就位：env/常量都没设 → 交互式问一次；空着就直接退出。
    # 这里在主线程、launch() 之前，终端 stdin 可用（不像 web 请求线程里没 stdin）。
    if not generator.api_key:
        print("=" * 60)
        print("检测到尚未配置 chat 模型的 API Key")
        print("（智谱 glm-5.2，bigmodel.cn 生成，id.secret 格式）")
        print("持久化方式：设置环境变量 ANTHROPIC_API_KEY，或改 demo.py 顶部 API_KEY 常量")
        key = input("请输入 API Key（直接回车则退出）: ").strip()
        if not key:
            raise SystemExit("未提供 API Key，退出")
        generator.api_key = key
    app = build_ui()
    print("=" * 60)
    print(f"Mini-RAG 聊天应用  ·  检索 {embedder.model}  ·  生成 {generator.model}")
    print(f"向量库：{len(store.chunks)} 个 chunk  ·  API Key：已配置")
    print("浏览器打开 http://127.0.0.1:7860  ·  Ctrl+C 退出")
    print("=" * 60)
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
              theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
