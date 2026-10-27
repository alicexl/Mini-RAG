"""Step 4: 用 Flask 手写一个 Web 聊天界面 —— 把 RAG 做成能用的产品

学习目标:
- 把 step3 的命令行 RAG 做成一个 Web 聊天应用（Flask + 原生 HTML/JS，无前端框架）
- 六个产品化改造：① Web 聊天界面（Flask 路由 + 静态页）
  ② 流式生成（SSE，答案一个字一个字蹦出来）
  ③ 文档管理（真 HTTP 文件上传，替代 step3 的命令行 ingest）
  ④ 检索底座工程化：step3 手搓的「向量库 + BM25 + RRF」三件套，
    换成 Milvus Lite 单文件向量库——两路召回与融合都在库内完成。
    这套代码就是生产同款：把 db 路径换成服务 uri 即平移到
    Milvus Standalone / Zilliz Cloud，API 一行不改
  ⑤ 多模态检索：PDF 直接上传——正文/表格/图片三种模态同一检索空间，
    表格转 HTML 保行列结构（命中回填原文）、图片走 VLM 生成描述、
    小表格跟正文一起切上下文不丢、大表格独立成块
  ⑥ 接入钉钉机器人（bot.py，可选）：Stream 模式出站 WebSocket 免公网；
    会话状态从「前端持有」换成服务端 SQLite（聊天平台替你藏掉的那部分
    又摊开一次）；history 表顺手落成使用日志（来源/延迟/token）
- RAG 的链路概念（混合检索→拼 context→system 约束→生成）与 step3 一致：
  本章换的是零件，不是原理——对照 step3 看「同一件事的手搓版与工程版」

启动:
    # 先备好外部依赖：
    #   · 一个智谱 API Key：set ANTHROPIC_API_KEY=xxx（embedding/生成/rerank/VLM 四者共用）
    #     （检索嵌入用智谱 embedding-3；step1-3 教学线的本地 Ollama 在本步退役，见改造④）
    #   · pip install -r requirements.txt（含 pymilvus + milvus-lite）
    python -X utf8 step4_app/demo.py
    # → 浏览器自动打开 http://127.0.0.1:7860
    # 注意：Milvus Lite 一次只允许一个进程打开 data/mini_rag.db（文件锁）。
    # 本 demo 单进程启动，不要开 debug reloader（它会起父子两个进程抢锁）

    # 换 chat 模型/网关用 ANTHROPIC_BASE_URL / ANTHROPIC_CHAT_MODEL 覆盖（同 step3）。
"""

import base64
import hashlib
import json
import os
import re
import sys
import threading
import webbrowser
from pathlib import Path

import fitz  # PyMuPDF：PDF 解析（文本块/表格/图片）
import pandas as pd
import requests
from anthropic import Anthropic
from flask import Flask, Response, jsonify, request, send_from_directory
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

# 向量库持久化：Milvus Lite 单文件库 data/mini_rag.db（step2/3 手搓三件套在此退休）
DATA_DIR = Path(__file__).parent / "data"
# 多模态产物：图片原件落这里（chunk 只存文件名，命中时前端经 /api/media/ 回显原图）
MEDIA_DIR = DATA_DIR / "media"

# 两级检索的宽度（同 step3）：粗排宽进捞多少、精排收紧留几条
RECALL_K = 20
TOP_K = 3
RRF_K = 60      # RRFRanker 的 k（同 step3 手写 rrf_merge 的默认值）


# ================================================================
# ================================================================
# 换件：embedding 从本地 Ollama 换成智谱 embedding-3 API（教学版见 step1）
# 产品线的取舍：实测单查嵌入 ~390ms vs 本地（GTX970 Vulkan）~2200ms——检索总延迟
# 2.7s 的八成花在本地嵌入上，切 API 后降到 ~0.9s；同一份评测集验证命中@3 持平
#（76.9%，见 README §3）。代价：检索也依赖 Key 与网络，且换模型=向量空间不兼容，
# 库必须全量重灌。
# ================================================================
_ZHIPU_EMBED_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
_ZHIPU_EMBED_DIM = 1024        # 必须与 schema 的 vector 维度一致（API 不指定默认 2048）


class ZhipuEmbedder:
    """智谱 embedding-3：1024 维与 schema 对齐；与生成 / rerank / VLM 共用同一个智谱 Key"""

    def __init__(self, api_key: str):
        self.model = "embedding-3"
        self.api_key = api_key

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(
            _ZHIPU_EMBED_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": list(texts), "dimensions": _ZHIPU_EMBED_DIM},
            timeout=60,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])   # 批量返回按 index 对齐
        return [d["embedding"] for d in data]


# 复用：RecursiveTextSplitter（同 step1，内联以便自包含；省去已讲过的 docstring）
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
#   · vector 字段 HNSW/COSINE 索引 → 稠密路：向量由智谱 embedding-3 算好喂进来
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
        """首次运行建 collection（schema + 索引）；已存在则加载后直接用。
        旧版（纯文本）schema 没有多模态字段 → drop 重建（demo 库，重传语料即可）。"""
        if self.client.has_collection(self.COLLECTION):
            fields = {f["name"] for f in self.client.describe_collection(self.COLLECTION)["fields"]}
            if "chunk_type" not in fields:
                self.client.drop_collection(self.COLLECTION)
            else:
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
        # —— 多模态三字段：chunk 不再只是纯文本 ——
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=8)   # text | table | image
        schema.add_field("content", DataType.VARCHAR, max_length=65535)  # 命中时回填给 LLM 的原文（表格=HTML）
        schema.add_field("page", DataType.INT64)                          # PDF 页码（非 PDF 语料 = 0）

        index = self.client.prepare_index_params()
        index.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",
                        params={"M": 16, "efConstruction": 200})
        index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX",
                        metric_type="BM25")
        self.client.create_collection(self.COLLECTION, schema=schema, index_params=index)
        self.client.load_collection(self.COLLECTION)

    def insert_rows(self, rows: list[dict], vectors: list[list[float]], source: str):
        """入库即持久化（库内 WAL 落盘）——手搓版要手动 save 三件套，这里没有这一步。
        rows 每项含 text/chunk_type/content/page（多模态统一行格式，text 分支同样走它），
        source 统一在这里补进每一行（幂等键 + 展示）。"""
        self.client.insert(self.COLLECTION,
                           [{**r, "source": source, "vector": v}
                            for r, v in zip(rows, vectors)])

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
            limit=top_k, output_fields=["text", "source", "chunk_type", "content", "page"])
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
# 本章新东西⑤：多模态 PDF 解析管线 —— 正文 / 表格 / 图片 三种模态进同一检索空间
# 思路（离线管道，上传时同步跑）：
#   · 每页抽三类元素（文本块 / 表格 / 图片占位），按 y 坐标排序还原成「元素流」，
#     表格区内的文本块剔除（表格整体转 HTML，不重复计字）
#   · 表格：HTML 保行列结构给 LLM 回填；文本化（表头+逐行拼接）给 embedding/BM25。
#     小表格（≤10 行）用边界标记跟正文一起切、上下文不丢；大表格独立成块，正文留引导句
#   · 图片：过滤装饰（短边过小 / 整页背景 / 同位重复 / 跨页同图），原件落 media/，
#     VLM 生成描述——描述是纯文本，独立成块参与检索，命中时前端回显原图
# ================================================================
_VLM_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
_VLM_MODEL = os.environ.get("ZHIPU_VLM_MODEL", "glm-4v-flash")

HEADER_Y = 80          # 页眉区（顶栏公司名/年份，y1 低于此的文本块丢弃）
FOOTER_Y = 60          # 页脚区（页码，y0 高于页高-此值的文本块丢弃）
SMALL_TABLE_ROWS = 10  # ≤此行数的表格跟正文一起切；更大的独立成块
MIN_IMG_PT = 50        # 图片最短边（pt）低于此 = 装饰线/小图标，不入库
IMG_PAGE_RATIO = 0.8   # 单图占页面面积超过此比例 = 整页背景，不入库


def describe_image(api_key: str, img_bytes: bytes, mime: str) -> str | None:
    """VLM（智谱 glm-4v，与 rerank/生成同一个 Key）给图片生成可检索的描述文本。
    失败返回 None——调用方降级为占位描述，不阻塞入库（离线管道的容错原则）。"""
    b64 = base64.b64encode(img_bytes).decode()
    try:
        resp = requests.post(
            _VLM_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=60,
            json={"model": _VLM_MODEL, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": "这是公司年报里的图片。用一两句话客观描述它：图表类型、"
                                         "主体是什么、关键趋势或结论。只描述看得到的内容，不推测具体数字。"},
            ]}]},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _overlap_ratio(rect: tuple, bbox: tuple) -> float:
    """rect 与 bbox 的交集面积占 rect 面积的比例（判定文本块是否落在表格里）。"""
    x0, y0 = max(rect[0], bbox[0]), max(rect[1], bbox[1])
    x1, y1 = min(rect[2], bbox[2]), min(rect[3], bbox[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = max((rect[2] - rect[0]) * (rect[3] - rect[1]), 1e-6)
    return (x1 - x0) * (y1 - y0) / area


def _page_elements(page) -> list[dict]:
    """一页 → 按 y 坐标排序的元素流 [{kind: text|table|image, y, ...}]。
    页眉页脚剔除；表格区内的文本块剔除；装饰图片剔除。"""
    tabs = page.find_tables().tables
    images, placed = [], []
    for x in page.get_images(full=True):
        for r in page.get_image_rects(x):
            if min(r.width, r.height) < MIN_IMG_PT:
                continue                            # 侧边装饰线（2pt 宽那种）
            if r.width * r.height > IMG_PAGE_RATIO * page.rect.width * page.rect.height:
                continue                            # 整页背景图
            if any(abs(r.x0 - p.x0) < 2 and abs(r.y0 - p.y0) < 2 for p in placed):
                continue                            # 同位叠放的双层图（阴影/遮罩），只取一个
            placed.append(r)
            images.append({"kind": "image", "y": r.y0, "xref": x[0]})
            break
    elements = []
    for x0, y0, x1, y1, txt, _no, btype in page.get_text("blocks"):
        if btype != 0 or not txt.strip():
            continue
        if y1 < HEADER_Y or y0 > page.rect.height - FOOTER_Y:
            continue                                # 页眉（公司名/年份）/ 页脚（页码）
        if any(_overlap_ratio((x0, y0, x1, y1), t.bbox) > 0.5 for t in tabs):
            continue                                # 表格区内的字——表格整体走 HTML，这里不重复计
        text = re.sub(r"[\r\n]+", "", txt)          # 块内硬换行是排版折行，拼回连续文本
        elements.append({"kind": "text", "y": y0, "text": text.strip()})
    elements.extend({"kind": "table", "y": t.bbox[1], "table": t} for t in tabs)
    elements.extend(images)
    elements.sort(key=lambda e: e["y"])
    return elements


def _table_parts(tab) -> tuple[str, str]:
    """表格 → (html, text)：HTML 保行列结构（LLM 读表格远好于线性化文本），
    text 是文本化版（表头一行 + 逐行竖线拼接），给 embedding 和 BM25 用。"""
    df = tab.to_pandas().fillna("")
    if isinstance(df.columns, pd.MultiIndex):       # 多级表头拉平成一行
        df.columns = [" ".join(map(str, c)) for c in df.columns]
    df.rename(columns=lambda c: "" if re.fullmatch(r"Col\d+", str(c)) else c,
              inplace=True)                         # PyMuPDF 没认出的表头占位（Col0/Col1…）清成空
    html = df.to_html(index=False)
    lines = ["表头：" + "、".join(str(c) for c in df.columns)]
    lines += ["｜".join(str(v) for v in row) for row in df.itertuples(index=False)]
    return html, "\n".join(lines)


def _refill_tables(chunk: str, registry: dict[str, str]) -> str:
    """把 chunk 里的〔表格Tn〕…〔/表格Tn〕文本化内容替换回 HTML 原文（content 版）。
    被切开的残缺标记替换不到——降级保留文本版：信息不丢，只是没还原成表格。"""
    return re.sub(r"〔表格T(\d+)〕.*?〔/表格T\1〕",
                  lambda m: registry.get(m.group(1), m.group(0)), chunk, flags=re.S)


_TABLE_MARK = re.compile(r"〔/?表格T?\d*〕")        # text 版里剥掉边界标记（embedding 不需要）


def parse_pdf(data: bytes, source: str, vlm_key: str) -> tuple[list[dict], int]:
    """PDF 字节 → 多模态 chunk 行 [{text, chunk_type, content, page}...] + VLM 成功张数。
    三类产物：正文+小表格（跟正文一起切，text 型）、大表格（独立 table 型）、图片（独立 image 型）。"""
    doc = fitz.open(stream=data, filetype="pdf")
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    stream, tables_html, standalone = [], {}, []
    seen_md5, vlm_ok = set(), 0
    tid = 0
    for pno, page in enumerate(doc, 1):
        page_parts = []
        for el in _page_elements(page):
            if el["kind"] == "text":
                page_parts.append(el["text"])
            elif el["kind"] == "table":
                html, text = _table_parts(el["table"])
                n_rows = len(el["table"].rows)
                if n_rows <= SMALL_TABLE_ROWS:      # 小表格：边界标记内联，跟正文一起切
                    page_parts.append(f"〔表格T{tid}〕\n{text}\n〔/表格T{tid}〕")
                    tables_html[str(tid)] = html
                else:                               # 大表格：正文留引导句，HTML 独立成块
                    page_parts.append(f"〔大表格：{text.splitlines()[0]}，共{n_rows}行，独立成块〕")
                    standalone.append({"text": text, "chunk_type": "table",
                                       "content": html, "page": pno})
                tid += 1
            else:                                   # 图片：VLM 描述独立成块，正文留引导句
                info = doc.extract_image(el["xref"])
                md5 = hashlib.md5(info["image"]).hexdigest()
                if md5 in seen_md5:                 # 跨页重复的同一张图，只入一次
                    continue
                seen_md5.add(md5)
                fname = f"{source}_p{pno}_{el['xref']}.{info['ext']}"
                (MEDIA_DIR / fname).write_bytes(info["image"])
                mime = f"image/{'jpeg' if info['ext'] in ('jpg', 'jpeg') else info['ext']}"
                desc = describe_image(vlm_key, info["image"], mime)
                if desc:
                    vlm_ok += 1
                else:
                    desc = "（图片，VLM 描述生成失败）"
                page_parts.append(f"〔图：{desc[:40]}…〕")
                standalone.append({"text": f"图片描述：{desc}", "chunk_type": "image",
                                   # 标记行不定界（无括号包裹）：文件名可含全角括号等字符，
                                   # 按行尾提取永不截断——用成对括号当定界符会被文件名里的括号截断
                                   "content": f"{desc}\n原图文件：{fname}", "page": pno})
        if page_parts:
            stream.append("\n\n".join(page_parts))
    # 正文流切块：text 版剥标记、content 版还原 HTML；正文 chunk 可能跨页，page 记 0（不标注）
    rows = standalone
    for chunk in splitter.split_text("\n\n".join(stream)):
        rows.append({"text": _TABLE_MARK.sub("", chunk),
                     "chunk_type": "text",
                     "content": _refill_tables(chunk, tables_html),
                     "page": 0})
    return rows, vlm_ok


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
    "参考资料带类型标注：[正文]是普通文本，[表格]是 HTML 表格（行列结构完整，按单元格读数），"
    "[图片]是对原图内容的文字描述。"
    "如果【参考资料】里没有相关信息，就如实回答「资料中未提及」，禁止编造。"
)

_TYPE_LABEL = {"text": "正文", "table": "表格", "image": "图片"}


def build_context(retrieved: list[dict]) -> str:
    """把精排后的 Top-K 拼成带编号 + 类型标注的 context。
    回填 content（多模态原文：表格 = HTML、图片 = 描述 + 原图文件名），而不是检索用的
    text（文本化版）——「embed/BM25 用文本化、生成用结构化原文」，两者在此分工。"""
    parts = [f"[{i}][{_TYPE_LABEL.get(e['type'], '正文')}] {e['content'] or e['text']}"
             for i, e in enumerate(retrieved, 1)]
    return "\n\n".join(parts)


def _to_entries(hits: list[dict], order: list[int], scores: dict[int, float]) -> list[dict]:
    """Milvus 命中 + 精排顺序 → 统一 entry 字典（chat 路由与 build_context 共用）。"""
    return [{"text": h["entity"]["text"], "content": h["entity"]["content"],
             "type": h["entity"]["chunk_type"], "page": h["entity"]["page"],
             "source": h["entity"]["source"], "rrf": h["distance"], "score": scores.get(i)}
            for i in order for h in [hits[i]]]


def retrieve(question: str, top_k: int = TOP_K):
    """混合检索（底座 = Milvus）：库内两路召回 + RRF 融合 Top-20 → rerank 精排 Top-3，
    失败降级回融合顺序（同 step3 的降级语义）。
    rerank 只看 text——表格/图片以文本化形态参与精排，多模态在此归一到同一竞争空间。"""
    query_vec = embedder.embed_batch([question])[0]
    hits = store.hybrid_search(question, query_vec, top_k=RECALL_K)   # 稠密+BM25 两路 → RRF
    docs = [h["entity"]["text"] for h in hits]
    try:
        results = reranker.rerank(question, docs, top_n=top_k)
        order = [r["index"] for r in results]
        return _to_entries(hits, order, {r["index"]: r["relevance_score"] for r in results}), True
    except Exception:
        return _to_entries(hits, list(range(top_k)), {}), False


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
    """切块 + 向量化 + 入库一份纯文本文档（md/txt 分支）。幂等：source 已在库则跳过。
    纯文本 chunk 三字段取默认值：type=text、content=text、page=0（无页码）。"""
    if store.has_source(source):
        return 0, False, f"[SKIP]「{source}」已在库中"
    chunks = splitter.split_text(text)
    rows = [{"text": c, "chunk_type": "text", "content": c, "page": 0} for c in chunks]
    store.insert_rows(rows, embedder.embed_batch(chunks), source)
    # 手搓版这里还要 save 三件套 + 重建 BM25 索引；Milvus 落库即持久化、稀疏路自动更新
    return len(chunks), True, f"[OK]「{source}」+{len(chunks)} 块"


def _ingest_pdf(data: bytes, source: str) -> tuple[int, bool, str]:
    """多模态分支（PDF）：解析 → 向量化 → 入库。幂等同上（media 文件也随 source 一起跳过）。"""
    if store.has_source(source):
        return 0, False, f"[SKIP]「{source}」已在库中"
    rows, vlm_ok = parse_pdf(data, source, generator.api_key)
    if not rows:
        return 0, True, f"[WARN]「{source}」未解析出内容（扫描件？本 demo 只吃有文字层的 PDF）"
    store.insert_rows(rows, embedder.embed_batch([r["text"] for r in rows]), source)
    by_type = {t: sum(1 for r in rows if r["chunk_type"] == t) for t in ("text", "table", "image")}
    imgs = by_type["image"]
    vlm_note = f"，VLM 描述 {vlm_ok}/{imgs}" if imgs else ""
    types = " + ".join(f"{v}{ {'text':'文本','table':'表格','image':'图片'}[k] }" for k, v in by_type.items() if v)
    return len(rows), True, f"[OK]「{source}」+{len(rows)} 块（{types}{vlm_note}）"


@app.post("/api/upload")
def upload():
    """「导入」端点：multipart 真上传——文件由浏览器文件框选出、内容随请求发上来，
    服务端只收内容、不碰服务端文件系统（本地 / 远端部署行为一致）。
    按扩展名分流：.pdf 走多模态解析管线，其余按纯文本。"""
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="未收到文件（multipart 字段名须为 files）"), 400
    notes = []
    for f in files:
        source = Path(f.filename).stem        # 同 step3 ingest 口径：source 取文件名（去扩展名）
        if f.filename.lower().endswith(".pdf"):
            notes.append(_ingest_pdf(f.read(), source)[2])
        else:
            text = f.read().decode("utf-8", errors="replace")
            notes.append(_ingest_text(text, source)[2])
    return jsonify(chunks=store.chunk_count(), notes=notes)


@app.get("/api/media/<name>")
def media(name):
    """图片原件回显：image chunk 的 content 里存文件名，前端经此路由取图（受控目录，无遍历风险）。"""
    return send_from_directory(MEDIA_DIR, name)


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
    context = build_context(retrieved)                                # 增强①：拼 context（content 原文回填）
    user_msg = f"【参考资料】\n{context}\n\n【用户问题】\n{message}"
    # 把「前端带上来的对话历史 + 本轮 user（带参考资料）」一起喂给 LLM —— 多轮里能接住追问
    llm_messages = history + [{"role": "user", "content": user_msg}]
    sources_payload = []
    for i, e in enumerate(retrieved, 1):
        m = re.search(r"原图文件：(.+)$", e["content"], re.M) if e["type"] == "image" else None
        sources_payload.append({
            "i": i, "rrf": round(e["rrf"], 4),
            "score": (round(e["score"], 3) if e["score"] is not None else None),
            "type": e["type"], "page": e["page"], "source": e["source"],
            "preview": e["text"].replace("\n", " ").strip()[:80],
            "image": m.group(1) if m else None,      # 图片 chunk → 文件名，前端回显原图
        })

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

    embedder = ZhipuEmbedder(generator.api_key)   # 检索嵌入与生成共用智谱 Key（Ollama 在本步退役）
    splitter = RecursiveTextSplitter(chunk_size=400, chunk_overlap=50)   # 同 step3 ingest 口径
    reranker = Reranker(generator.api_key)     # 精排与生成共用同一个智谱 Key
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)          # 图片原件落这里（VLM 描述入 chunk）
    store = MilvusStore(str(DATA_DIR / "mini_rag.db"))   # 打开/创建 Milvus Lite 单文件库

    print("=" * 60)
    print(f"Mini-RAG 聊天应用  ·  检索 {embedder.model}  ·  生成 {generator.model}")
    print(f"向量库：{store.chunk_count()} 个 chunk  ·  Milvus Lite（data/mini_rag.db）·  API Key：已配置")
    print("浏览器打开 http://127.0.0.1:7860  ·  Ctrl+C 退出")
    if os.environ.get("DINGTALK_CLIENT_ID") and os.environ.get("DINGTALK_CLIENT_SECRET"):
        print("检测到钉钉凭据：机器人线程将随服务启动（Stream 模式，见 bot.py）")
    print("=" * 60)
    # 钉钉机器人（可选，改造⑥）：两个凭据 env 都设了才起。同进程起线程而不是另起进程
    # ——Milvus Lite 单进程文件锁，bot 与 Flask 必须共库同进程（bot.py import demo 复用单例）
    if os.environ.get("DINGTALK_CLIENT_ID") and os.environ.get("DINGTALK_CLIENT_SECRET"):
        # 坑：本文件作为入口运行时模块名是 __main__，bot.py 里 import demo 会再加载
        # 一份**没跑过 main() 的裸副本**（单例全是 AttributeError）。把 __main__ 注册成
        # demo，bot 拿到的才是同一个正在跑的模块
        sys.modules["demo"] = sys.modules["__main__"]
        import bot                                  # 局部导入：bot 又 import demo，放模块级会成环
        threading.Thread(target=bot.bot_main, daemon=True, name="dingtalk-bot").start()
    # 等服务真的起来再开页（app.run 是阻塞的，直接 open 会抢在端口监听之前）
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:7860")).start()
    app.run(host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
