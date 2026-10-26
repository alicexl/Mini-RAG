# Step 4 — 手写一个 Web 聊天界面：把 RAG 做成产品

> Flask Web 界面 + SSE 流式生成 + 文档上传 + 检索底座换 Milvus Lite + **PDF 多模态检索（正文/表格/图片）**——浏览器打开就能问，答案一个字一个字往外吐。
> 四步路径与各步概览见仓库根目录 [README](../README.md)。

## 快速开始

前提同 [Step 3](../step3_rag/README.md)：Ollama 已启动（检索本地）+ 智谱 API Key（生成 / rerank / VLM 图片描述三者共用同一个 Key）+ `pip install -r requirements.txt`（本步新增 pymilvus / milvus-lite / pymupdf / pandas）。注意 Web 请求处理里没有 stdin，Key 只认环境变量 / 常量 / 启动时的交互确认（见 §2 改造②），没配会在生成时报错、以 SSE error 事件显示在聊天气泡里。另：Milvus Lite 一次只允许一个进程打开 `data/mini_rag.db`（文件锁），别开 Flask 的 debug reloader——它会起父子两个进程抢锁：

```bash
set ANTHROPIC_API_KEY=你的智谱key    # macOS/Linux 用 export
python -X utf8 step4_app/demo.py    # → 浏览器自动打开 http://127.0.0.1:7860
```

端口默认 7860，被占用时改 demo.py 底部 `app.run` 的 `port`。

---

## 1. 为什么要加界面，为什么手写 Flask

前三步我们手里已经有了一个完整的 RAG：两级检索（召回 + rerank 精排）→ 拼 context → system 约束 → LLM 生成。但它跑在命令行里（`step3_rag/demo.py chat`），作为 demo 够清楚，作为「产品」有两个硬伤：

- **上手门槛**：想用它的人得会开终端、装 Python、跑 `ingest`、再跑 `chat`——普通人到「打开终端」这一步就被劝退。
- **交互体验**：`Generator.answer()` 是阻塞的，问一个问题要干等几秒才有整段答案，没有任何「它在想」的反馈——不像 ChatGPT 那样一个字一个字往外吐。

这一步就是给 RAG 套一层 Web 聊天界面：**浏览器打开就能问、答案流式往外吐、左侧点「导入」从浏览器文件框里选文档入库**——降低上手门槛，提升交互体验。

界面怎么做？本 step 手写（Flask + 原生 HTML/JS）：多写一两百行，换来 Web 层完全摊开——URL 怎么映射到 Python 函数、静态页怎么被服务出去、流式答案怎么从后端一路推到浏览器的聊天气泡、文件怎么从浏览器传到服务端，每一环都看得见。

Flask 是 Python Web 生态的「最小公约数」，核心概念就四个：

| 概念 | 一句话 | 在 demo.py 里 |
|---|---|---|
| 路由（route） | 把 URL + 方法映射到一个 Python 函数 | `@app.post("/api/chat")` |
| 视图函数 | 被路由调用的函数，返回 HTML / JSON / 流 | `def chat(): ... return Response(...)` |
| 静态文件 | 浏览器加载的页面本身 | `send_from_directory(STATIC_DIR, "index.html")` |
| 请求对象 | 请求里带的东西（JSON 体 / 上传的文件） | `request.get_json()` / `request.files` |

整条链路就三段，各段都能单独抠出来看：

```
浏览器（static/index.html：渲染界面 + 发请求 + 解析 SSE）
   │  HTTP
Flask Web 层（demo.py：5 个路由）
   │  函数调用
RAG 内核（两级检索 → 拼 context → system 约束 → 生成；底座换 Milvus，见 §2 改造④）
```

---

## 2. 动手：四个产品化改造

前 3 步把原理摊开讲完了，这一步在产品化的同时做一次「零件升级」：`Embedder` / `RecursiveTextSplitter` / `Reranker` / `SYSTEM_PROMPT` / `build_context` 原样内联复用；**step3 手搓的检索三件套 `VectorStore`（JSON+numpy 余弦）/ `BM25Index`（手写 Okapi）/ `rrf_merge`（名次归并）整体退休**，换成一个 `MilvusStore`（见改造④）。新增的代码就两类：**流式的 Generator** 和 **Flask 的 Web 层**。

### 改造①：Web 聊天界面（路由 + 静态页）

Web 层一共 6 个路由，一张表看完：

| 方法 + 路径 | 干什么 |
|---|---|
| `GET /` | 返回 `static/index.html`（聊天页面） |
| `GET /api/status` | 库状态 JSON：chunk 数 + 已导入文档清单 |
| `POST /api/upload` | multipart 文件上传 → 切块 / 向量化 / 入库（幂等；PDF 走多模态解析，见改造⑤） |
| `POST /api/clear` | 清空向量库 |
| `POST /api/chat` | SSE 流式回答（见改造②） |
| `GET /api/media/<文件名>` | 回显图片 chunk 的原图（多模态，见改造⑤） |

```python
app = Flask(__name__, static_folder=None)   # 关掉 Flask 自带的 /static/<文件> 路由

@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")
```

`static_folder=None` 值得一提：Flask 默认会把 `static/` 目录自动挂到 `/static/<文件名>`。这里关掉默认行为、用 `send_from_directory` 手动返回页面——静态文件是怎么被服务的，亲眼看到这一次，后面再用默认行为就知其所以然。

前端就一个 `static/index.html`：无框架、无构建工具，CSS/JS 全内嵌。布局左右分栏——左栏文档管理（库状态 + 导入 / 清空），右栏聊天（检索来源面板 + 聊天气泡 + 输入框）。所有动态内容只用 `textContent` 写入（不走 `innerHTML`），chunk 原文按纯文本显示，顺带天然免疫 XSS。

### 改造②：流式生成（SSE）

`Generator.answer_stream` 和上一形态一样：用 SDK 的 `messages.stream` 拿到一个流，逐片 `yield` 文本增量。真正的新问题是：**HTTP 响应怎么「分多次送」？** Flask 的一次响应默认是送完即止的——那就让这一次响应本身变成流。这正是 **SSE（Server-Sent Events）**：`Content-Type: text/event-stream` + 一个生成器，Flask 每 `yield` 一段就往 socket 写一段。

协议自己定，四种事件，每帧两行头 + 空行结尾：

| 事件 | data | 时机 |
|---|---|---|
| `sources` | 本次检索来源（召回 20 → 精排 Top-3，带 `reranked` 标记） | 先亮来源 |
| `delta` | `{"text": "一小片答案"}` | 每个增量一帧 |
| `done` | `{}` | 回答结束 |
| `error` | `{"message": "..."}` | 生成中失败（无 Key / SDK 异常） |

```python
def stream():
    yield _sse("sources", {"sources": sources_payload, "reranked": reranked})
    try:
        for delta in generator.answer_stream(SYSTEM_PROMPT, llm_messages):
            yield _sse("delta", {"text": delta})
        yield _sse("done", {})
    except Exception as e:
        yield _sse("error", {"message": f"生成失败：{e}\n请确认 ANTHROPIC_API_KEY 已配置"})

return Response(stream(), mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache"})
```

错误分两层，这是个值得记住的 Web 实践：**请求级校验**（空消息 / 空库）走 `400 + JSON`——那时响应头还没定，可以随便返；**流一旦开始**（响应头已是 `text/event-stream`）中途出错就只能往下写 SSE `error` 事件，前端把它显示在助手气泡里。

前端消费 SSE 用的是 `fetch` 而不是浏览器原生的 `EventSource`——后者只支持 GET，我们要 POST 请求体。解析器核心就十五行：`resp.body.getReader()` 逐块读字节 → `TextDecoder` 转文本 → 按**空行**切帧 → 帧内取 `event:` / `data:` 两行：

```js
const reader = resp.body.getReader();
const decoder = new TextDecoder();
let buf = "";
while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  buf += decoder.decode(value, {stream: true});
  let cut;
  while ((cut = buf.indexOf("\n\n")) >= 0) {      // SSE 帧以空行分隔
    const frame = buf.slice(0, cut);
    buf = buf.slice(cut + 2);
    /* 解析 event:/data: → 更新来源面板 / 助手气泡 / history */
  }
}
```

**多轮对话的状态放哪？** 前端持有一个 `history` 数组，每次请求随 body 带上来；服务端不存任何会话状态（无状态服务，重启不丢对话）。把「已有历史 + 本轮 user（带参考资料）」一起喂 LLM，模型就能接住「那 2023 年呢？」这种追问；检索仍只针对最新问题（朴素 RAG 的简化）。这也是聊天框架的 Chatbot 组件替你藏掉的那部分——现在它就摊在你的 index.html 里，约 15 行。

> 为什么不用 WebSocket？这个场景是「请求小、响应流」的单向推送，SSE 就是纯 HTTP，够用且更简单；WebSocket 的双向能力用不上。顺带一提：LLM API 自家的流式返回（OpenAI / 智谱）走的就是 SSE——这里的做法和生产同构。

### 改造③：文档管理（真上传）

导入语料从「敲命令」变成「点按钮」。页面里一个隐藏的原生文件框（`<input type="file" multiple>`，label 当按钮用），选中文件后前端拼 `FormData` 发 `POST /api/upload`，服务端从 `request.files` 里拿：

```python
files = request.files.getlist("files")     # multipart 里的文件列表
for f in files:
    source = Path(f.filename).stem          # 同 step3 ingest 口径：source 取文件名
    text = f.read().decode("utf-8", errors="replace")
    notes.append(_ingest_text(text, source)[2])
```

`_ingest_text` 逻辑同 step3：切块 → 向量化 → 入库，幂等——按文件名判断，已在库就跳过，不重算 embedding：

```python
def _ingest_text(text, source):
    if store.has_source(source):        # 库内标量过滤查询，替代 step3 扫 metadata 列表
        return 0, False, f"[SKIP]「{source}」已在库中"
    ...
```

值得强调的是「真上传」三个字：文件框由**浏览器**弹出（在你自己的电脑上选文件），内容随请求传到服务端——服务端只收内容、不碰服务端文件系统，**本地跑和部署到远端行为完全一致**。demo 框架图省事时常见另一种做法（服务端代码直接弹个框、拿本地路径读文件），本地演示看不出差别，一旦部署到远端服务器，框会弹在服务器上没人点。

### 改造④：检索底座工程化（Milvus Lite）

前三步的检索底座是手搓的：`VectorStore` 把余弦摊成矩阵乘、`BM25Index` 把 Okapi 公式写成循环、`rrf_merge` 把名次融合写成十行——教学价值就在「没有黑盒」。但它们毕竟是教具：线性扫描扛不住百万 chunk、BM25 索引每次启动现建、三件套要自己 save/load。工程端的答案是**向量数据库**——这里换成 Milvus Lite（Milvus 的嵌入式单文件版，`pip install milvus-lite` 即用，不用装服务）。概念与 step3 一一对应，零件换生产件：

| step3 手搓件 | Milvus 里对应的东西 |
|---|---|
| `VectorStore`：npy 存向量 + 归一化点积 | `vector` 字段 + HNSW 索引（`metric_type="COSINE"`），大规模走近似检索 |
| `BM25Index`：jieba 分词 + 手写 Okapi | `text` 字段开 jieba 分析器 + BM25 函数，插原文、库内自动分词建稀疏索引 |
| `rrf_merge`：名次折算 1/(k+rank) | `hybrid_search(..., ranker=RRFRanker(60))` 一个参数 |
| 三件套 save/load + 上传后重建 BM25 | 落库即持久化（WAL），稀疏索引自动更新 |
| 扫 metadata 列表判幂等 | 标量过滤查询 `filter='source == "xxx"'` |

一张 collection 把「原文 + 向量 + 稀疏 + 元信息」全装下，schema 建一次、之后只管插和查：

```python
schema.add_field("text", DataType.VARCHAR, max_length=65535,
                 enable_analyzer=True, analyzer_params={"type": "jieba"})  # 中文按词切
schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)      # BM25 函数的输出位，不用手填
schema.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                             input_field_names=["text"], output_field_names=["sparse"]))
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)   # qwen3-embedding 维度
schema.add_field("source", DataType.VARCHAR, max_length=256)  # 来源文件名（幂等键）
```

检索侧两路各带一个 `AnnSearchRequest`，融合交给 `RRFRanker`——step3 里手写的那十行，在这里是一个参数：

```python
dense = AnnSearchRequest(data=[query_vec], anns_field="vector",
                         param={"metric_type": "COSINE"}, limit=20)
sparse = AnnSearchRequest(data=[query], anns_field="sparse",   # 查询原文进来，库内 jieba 分词
                          param={"metric_type": "BM25"}, limit=20)
hits = client.hybrid_search("docs", [dense, sparse],
                            ranker=RRFRanker(60), limit=20,
                            output_fields=["text", "source"])
```

**rerank 仍留在库外**（`retrieve()` 接着调智谱 rerank API）——精排要过一遍 Cross-Encoder 大模型，向量库不管这件事；失败降级回融合顺序，语义同 step3。

工程件不是白拿的，三个已知边界：

- **Windows 安装坑**：`pip install "pymilvus[milvus-lite]"` 在 Windows 上**装不上 milvus-lite**——pymilvus 给这个 extra 挂了 `sys_platform != 'win32'` 标记（C++ 老版本的遗留），直接单装 `milvus-lite` 即可（新一代是纯 Python 实现）。requirements.txt 里两行分开写。
- **BM25 的 IDF 是 segment 局部统计**（官方已知限制），不是 step3 手搓的全库 IDF——分数有微小差异，排序影响可忽略。
- **单进程文件锁**：一个 data 目录同一时刻只允许一个进程打开——本 demo 单进程没事，但别开 debug reloader（父子两进程抢锁）。

「生产同款」的真正含义：把 `MilvusStore` 里的 db 路径换成 `http://localhost:19530`（Milvus Standalone，Docker 起）或 Zilliz Cloud 的 uri，其余 API 一行不改——教学件到生产件的迁移路径就是换一个连接串。

来源面板也随之变干净：不再显示 `sim`（向量相似度）和路由标记（手搓三段式的讲解道具），只留 `rrf`（库内融合分）+ `score`（rerank 精排分）+ 出处。

### 改造⑤：多模态检索（PDF 的正文 / 表格 / 图片）

到上一节为止，语料还是「纯文本 md 节选」。这一步让 `/api/upload` 直接吃 **PDF**：服务端解析出三种模态，塞进**同一个检索空间**——上传真年报，表格问题命中表格块、图片问题命中图片块。样例语料是 `sample_reports/盐湖股份2024年报（精选页）.pdf`：从 177 页的真年报里抽 10 页（脚本 `tools/extract_pages.py`，页集覆盖三种模态的代表性形态），库里产出 35 块 = 20 文本 + 4 表格 + 11 图片。

**解析管线**（`parse_pdf`，上传时同步跑）：每页用 PyMuPDF 抽三类元素——文本块（`get_text("blocks")`）、表格（`find_tables()`）、图片（`get_images()` + 落盘）——按 y 坐标排序还原成「带位置信息的元素流」，页眉页脚剔除、表格区内的重复文本块剔除、装饰图片（2pt 宽的侧边线、整页背景、同位叠放）剔除。然后按模态分流：

| 模态 | 检索用什么（text 字段，进 embedding + BM25） | 生成回填什么（content 字段） | 成块方式 |
|---|---|---|---|
| 正文 | 原文 | 原文 | 跟随元素流一起切 |
| 小表格（≤10 行） | 表格**文本化**（表头 + 逐行拼接） | **HTML**（保行列结构） | 边界标记内联，跟正文一起切 |
| 大表格（>10 行） | 文本化全文 | HTML | **独立成块**，正文留引导句 |
| 图片 | **VLM 描述**（glm-4v-flash） | 描述 + 原图文件名 | **独立成块**，正文留引导句 |

三个关键设计，每个都对应一个「为什么」：

- **embed 用文本化、生成用 HTML**——LLM 读 HTML 表格的理解远好于线性化文本，但 embedding 和 BM25 吃 HTML 标签是纯噪声。所以同一个 chunk 存两份：`text` 给检索链路，`content` 给拼 context。小表格的内联机制：元素流里放边界标记 `〔表格T0〕文本化内容〔/表格T0〕`，切块后 text 版剥标记、content 版把标记对替换回 HTML——标记只活在切块瞬间，两头都不污染。
- **图片走 VLM 生成描述**——描述是纯文本，天然进同一个 embedding 空间；原件落 `data/media/`，chunk 的 content 里只存文件名。命中图片块时来源面板直接回显原图（`GET /api/media/<文件名>`）。VLM 失败不阻塞入库：描述降级为占位符，11 张图里挂一张不影响剩下十张（离线管道的容错原则）。
- **rerank 只看 text**——表格/图片以文本化形态参与精排，三种模态归一到同一竞争空间。这就是多模态检索的口径：模态差异在入库时消化（解析→文本化），检索链路一行不改。

schema 相应加三个字段（`chunk_type` / `content` / `page`），旧库自动检测重建。超长表格按行拆片这类扩展先不做——本语料最大表格 23 行、HTML 几 KB，VARCHAR 65535 远够；真遇到几百行的表再上。

### 跑起来

前提同 step3：Ollama 已启动（qwen3-embedding:0.6b），Key 设成环境变量，然后 `python -X utf8 step4_app/demo.py`。首次启动自动创建空库 `data/mini_rag.db`；库里留着上次的数据则直接接着用（落库即持久化，重启不丢）。点「导入」选 `sample_reports/盐湖股份2024年报（精选页）.pdf`（PDF 分支会先跑多模态解析 + 11 张图逐张过 VLM，约半分钟）：

```
[OK]「盐湖股份2024年报（精选页）」+35 块（20文本 + 4表格 + 11图片，VLM 描述 11/11）
```

然后在右栏问「公司2024年赚了多少钱？」——来源面板先亮出来（Milvus 两路召回 → RRF 融合 → rerank 精排 → Top-3，即 LLM 回答里「根据[N]」的出处；`rrf` 是库内融合分、`score` 是精排分；条目带类型标注），紧接着答案开始流式往外吐：

```
本次检索来源（Milvus 两路召回：稠密向量 + BM25 → RRF 融合 → rerank 精排 → Top-3）：
  [1]（表格）rrf=0.0328 score=1.0 · 盐湖股份2024年报（精选页） p4
      表头：、2024年、2023年、本年比上年增减、2022年 营业收入（元）｜15,134,119,500.89｜21,578,504…
  [2]（表格）rrf=0.0317 score=1.0 · 盐湖股份2024年报（精选页） p5
  [3]（正文）rrf=0.0308 score=1.0 · 盐湖股份2024年报（精选页）

助手（流式）：
根据[1]的财务数据表格：
- 2024年营业收入：15,134,119,500.89 元（约151.34亿元），比上年下降 29.86%
- 2024年归属于上市公司股东的净利润：4,663,116,528.38 元（约46.63亿元），比上年下降 41.07%
```

看多模态在工作：`[1]` 是**表格块**——「主要会计数据」大表（15 行）解析成独立 chunk，检索时以文本化形态参与两路召回与精排，生成时 LLM 拿到的却是 HTML 原文（行列结构完整，按单元格读数），所以数字读得又准又全。`rrf` 的量纲也读得出来：单路命中 ≈ 1/(60+名次) ≈ 0.016，两路命中叠加到 0.030+——表格块 rrf 最高正是因为文本化版在稠密、稀疏两路都命中了。再问一句「年报里那张饼状图展示了什么？」——这次 Top-3 里会出现 **（图片）** 条目（VLM 描述成块参与检索），来源面板下方直接回显饼图原图。

线路上传的是一帧一帧的 SSE（另一次提问的实测帧节选）：

```
event: delta
data: {"text": "资料"}

event: delta
data: {"text": "中"}

event: delta
data: {"text": "未"}

...（几十帧）

event: done
data: {}
```

顺手看两个边界行为（都是实测）：

- **防幻觉在工作**：问「公司2025年打算给股东分红多少？」——语料里没有分红信息，检索照样返回「最不差的」Top-3（精排分全是 1.0：Cross-Encoder 只会排「哪个更像答案」，证不了「没有」），但模型守住 system 约束如实拒答、还列举了资料里实际有什么：「资料中未提及……仅包含主要会计数据（营业收入、净利润、每股收益等）与成本构成数据」。检索送不存在的答案送不上来，LLM 也不编——这正是 step3 防幻觉两道关的产品化表现。
- **空库提问**：`POST /api/chat` 返回 `400 {"error": "向量库为空，请先在左侧导入文档"}`，前端显示成气泡里的 ⚠️ 提示。

这就是一个「能用」的 RAG 产品了：浏览器打开、选文档、提问、看答案一个字一个字出来、每条答案都能追溯到出自哪份年报的哪个 chunk。整个过程没碰过命令行（除了第一次启动）。

---

## 3. 四步走完

回头看这四步：

1. **[基础件](../step1_basics/README.md)**：分块（递归切分保语义边界）+ 向量化（Embedding，余弦相似度找语义接近）。
2. **[向量库](../step2_index/README.md)**：JSON + numpy 存 chunk 和向量，从零实现检索。
3. **[完整 RAG](../step3_rag/README.md)**：混合检索（向量 + BM25 两路召回、RRF 融合、rerank 精排）+ 拼 context + system 约束 + LLM 生成，闭环 + 防幻觉两道关。
4. **聊天界面 + 工程底座 + 多模态**（本 step）：Flask 手写 Web 层，SSE 流式 + 文档上传；检索底座从手搓三件套换成 Milvus Lite；PDF 直接入库，正文/表格/图片三种模态同一检索空间。
   
一套朴素 RAG 的骨架就这么搭起来了。前三步每个零件都最简、都看得见——embedding 是一次 HTTP 调用、分块是递归切分隔符、向量库是余弦点积、检索是召回 + 精排、生成是拼 prompt 喂 LLM。第四步在看清每个零件内部是什么之后，把存储与检索底座换成工业件（Milvus）、把语料从纯文本升级到多模态 PDF、外面加上 6 个路由 + 一段 SSE 的 Web 层——先没有黑盒，再心安理得地用黑盒。
