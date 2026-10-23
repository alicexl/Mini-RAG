# Step 4 — 手写一个 Web 聊天界面：把 RAG 做成产品

> Flask Web 界面 + SSE 流式生成 + 文档上传——浏览器打开就能问，答案一个字一个字往外吐。
> 四步路径与各步概览见仓库根目录 [README](../README.md)。

## 快速开始

前提同 [Step 3](../step3_rag/README.md)：Ollama 已启动（检索本地）+ 智谱 API Key（生成与 rerank 共用同一个 Key）。注意 Web 请求处理里没有 stdin，Key 只认环境变量 / 常量 / 启动时的交互确认（见 §2 改造②），没配会在生成时报错、以 SSE error 事件显示在聊天气泡里：

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

界面怎么做？两条路：

- **demo 框架**（Gradio / Streamlit）：纯 Python 几十行就能起一个带交互的网页。最快，但路由、前后端通信、流式推送、状态管理全被框架收进黑盒——对一个立身之本是「没有黑盒」的教学仓，等于在最后一步把前三步攒下的透明度全丢了。
- **手写**（Flask + 原生 HTML/JS）：多写一两百行，换来 Web 层完全摊开——URL 怎么映射到 Python 函数、静态页怎么被服务出去、流式答案怎么从后端一路推到浏览器的聊天气泡、文件怎么从浏览器传到服务端，每一环都看得见。

本 step 选手写。Flask 是 Python Web 生态的「最小公约数」，核心概念就四个：

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
RAG 内核（step3 原样：两级检索 → 拼 context → system 约束 → 生成）
```

---

## 2. 动手：三个产品化改造

RAG 的核心类（`Embedder` / `RecursiveTextSplitter` / `VectorStore` / `Reranker`）和 `SYSTEM_PROMPT` / `build_context` / `retrieve` 都和 step3 一样，本 step 不动它们（demo.py 里原样内联）。新增的代码就两类：**流式的 Generator** 和 **Flask 的 Web 层**。

### 改造①：Web 聊天界面（路由 + 静态页）

Web 层一共 5 个路由，一张表看完：

| 方法 + 路径 | 干什么 |
|---|---|
| `GET /` | 返回 `static/index.html`（聊天页面） |
| `GET /api/status` | 库状态 JSON：chunk 数 + 已导入文档清单 |
| `POST /api/upload` | multipart 文件上传 → 切块 / 向量化 / 入库（幂等） |
| `POST /api/clear` | 清空向量库 |
| `POST /api/chat` | SSE 流式回答（见改造②） |

```python
app = Flask(__name__, static_folder=None)   # 关掉 Flask 自带的 /static/<文件> 路由

@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")
```

`static_folder=None` 值得一提：Flask 默认会把 `static/` 目录自动挂到 `/static/<文件名>`。这里关掉默认行为、用 `send_from_directory` 手动返回页面——静态文件是怎么被服务的，亲眼看到这一次，后面再用默认行为就知其所以然。

前端就一个 `static/index.html`：无框架、无构建工具，CSS/JS 全内嵌。布局左右分栏——左栏文档管理（库状态 + 导入 / 清空），右栏聊天（检索来源面板 + 聊天气泡 + 输入框）。所有动态内容只用 `textContent` 写入（不走 `innerHTML`），chunk 原文按纯文本显示，顺带天然免疫 XSS。

### 改造②：流式生成（SSE）

`Generator.answer_stream` 和上一形态一样：用 SDK 的 `messages.stream` 拿到一个流，逐片 `yield` 文本增量。真正的新问题是：**HTTP 响应怎么「分多次送」？** Gradio 时代回调写成生成器界面就自动刷新；Flask 没有这层魔法，答案是一次 HTTP 响应送完的——那就让这一次响应本身变成流。这正是 **SSE（Server-Sent Events）**：`Content-Type: text/event-stream` + 一个生成器，Flask 每 `yield` 一段就往 socket 写一段。

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

**多轮对话的状态放哪？** 前端持有一个 `history` 数组，每次请求随 body 带上来；服务端不存任何会话状态（无状态服务，重启不丢对话）。把「已有历史 + 本轮 user（带参考资料）」一起喂 LLM，模型就能接住「那 2023 年呢？」这种追问；检索仍只针对最新问题（朴素 RAG 的简化）。这也是聊天框架的 Chatbot 组件替你藏掉的那部分——现在它就摊在你的 index.html 里，~15 行。

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
    if any(m.get("source") == source for m in store.metadata):
        return 0, False, f"[SKIP]「{source}」已在库中"
    ...
```

值得强调的是「真上传」三个字：文件框由**浏览器**弹出（在你自己的电脑上选文件），内容随请求传到服务端——服务端只收内容、不碰服务端文件系统，**本地跑和部署到远端行为完全一致**。demo 框架图省事时常见另一种做法（服务端代码直接弹个框、拿本地路径读文件），本地演示看不出差别，一旦部署到远端服务器，框会弹在服务器上没人点。

### 跑起来

前提同 step3：Ollama 已启动（qwen3-embedding:0.6b），Key 设成环境变量，然后 `python -X utf8 step4_app/demo.py`。首次启动状态栏是空的；若 `data/` 里留着上次入库的向量库，启动时会直接加载。点「导入」选中 `sample_reports/` 下的三份年报节选（可多选）：

```
[OK]「盐湖股份2023年年报（节选）」+39 块
[OK]「盐湖股份2024年年报（节选）」+34 块
[OK]「盐湖股份2025年年报（节选）」+42 块
```

然后在右栏问「公司2024年经营活动现金流量净额是多少？」——来源面板先亮出来（召回 20 → rerank 精排 → Top-3，即 LLM 回答里「根据[1]」的出处；`sim` 是向量相似度、`score` 是精排分），紧接着答案开始流式往外吐：

```
本次检索来源（召回 20 → rerank 精排 → Top-3）：
  [1] sim=0.580 score=1.0  盐湖股份2025年年报（节选）
  [2] sim=0.654 score=1.0  盐湖股份2024年年报（节选）
  [3] sim=0.658 score=1.0  盐湖股份2024年年报（节选）

助手（流式）：
公司2024年经营活动产生的现金流量净额为 **7,819,152,860.43元**（约78.19亿元）。

根据[1]和[3]，该数据在两份资料中一致。此外，根据[3]，相比2023年的
12,104,618,721.68元，2024年同比下降了35.40%。
```

看一眼 rerank 在这题干了什么：向量召回里 `sim` 最高的是 [3]（0.658，直接含答案的对比表），但精排把 **2025 年报的对比列**提为 [1]——那块同时装着 2024 和 2025 两年的现金流，对这个查询信息量最大。答案也因此带上了双来源校验（「根据[1]和[3]，两份资料一致」）。

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
3. **[完整 RAG](../step3_rag/README.md)**：两级检索（召回宽进 + rerank 精排收紧）+ 拼 context + system 约束 + LLM 生成，闭环 + 防幻觉两道关。
4. **聊天界面**（本 step）：Flask 手写 Web 层，SSE 流式 + 文档上传。UI 仅为产品化演示，RAG 内核仍是 step3 的实现。

一套朴素 RAG 的骨架就这么搭起来了。它每个零件都最简、都看得见——embedding 是一次 HTTP 调用、分块是递归切分隔符、向量库是余弦点积、检索是召回 + 精排、生成是拼 prompt 喂 LLM、Web 层是 5 个路由 + 一段 SSE。没有黑盒。
