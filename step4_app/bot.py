"""step4 附加层：钉钉机器人（Stream 模式）——把 RAG 接进聊天工具

通道为什么选 Stream 模式：钉钉机器人收消息有 Webhook / Stream 两种接法，
Webhook 要求公网可访问的 HTTPS 回调地址（家庭宽带 NAT 后面就得内网穿透）；
Stream 是**出站 WebSocket**——本进程主动连钉钉服务器，消息从平台推下来，
零公网 IP、零域名、零证书，家里常开的机器直接能跑。

与 Web 端的两处对照（都是聊天框架替你藏掉的东西，这里摊开）：
- 多轮对话的状态：Web 端 history 由前端持有、随请求上送（无状态服务）；
  机器人没有「前端」，会话状态必须在服务端——落在 SQLite，按 conversation_id
  隔离（单聊/群聊天然分开），每次取最近 3 轮
- 使用日志：history 表顺手就是使用日志（问题/答案/命中来源/延迟/token 全落库）——
  「检索质量」坏例的排查和真实 query 分布的积累从这里长出来

回复链路：收到消息 → 先 ACK 再异步回答（生成长尾 P95 可到 20s+，process
同步等完会触发钉钉超时重推）→ 先发一张 AI 卡片（「执行中」占位）→ 检索 →
**流式生成边收边写卡片**（打字机；append=False 全量替换 + ~1s 节流）→
ai_finish 定稿（答案 + 引用来源脚注）。生成关掉了思考块：语料数值问答不需要
长思考，实测单次生成快 ~30%、思考 tokens 全省；token 用量从流的 final
message 拿（get_final_message 在 with 块内，这正是流式也能记账的原因）。

启动（不单独跑，由 demo.py 自动接线；两个 env 都设了才起 bot 线程）：
    set DINGTALK_CLIENT_ID=xxx        # 钉钉开放平台应用的 Client ID（AppKey）
    set DINGTALK_CLIENT_SECRET=xxx    # 对应 Client Secret
    python -X utf8 step4_app/demo.py  # Flask 照常起，bot 线程同进程共享 Milvus 单进程锁

钉钉侧配置（一次性）：注册团队 → open-dev.dingtalk.com 创建应用 → 添加「机器人」
能力 → 消息接收模式选「Stream 模式」。不发布也能用（实测：未认证个人团队、
未发布应用，Stream 收发消息全部正常——认证挡的是正式发布，不是消息通道）。
"""

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
import time

import dingtalk_stream
from dingtalk_stream import AckMessage

import demo                                    # 单例（embedder/store/reranker）在 demo.main() 里接线，本模块同进程直接用
from anthropic import Anthropic

# bot 自己的日志走 stderr（实时）——print 在后台进程里是块缓冲，出问题时看不见现场
log = logging.getLogger("bot")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [BOT] %(levelname)s %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

# 会话历史：SQLite 落库（服务端持有，与 Web 端「前端持有」对照）；文件与 Milvus 库同目录
HISTORY_DB = str(demo.DATA_DIR / "bot_history.db")
HISTORY_WINDOW = 6          # 取最近 6 条消息（= 3 轮问答）拼 LLM 的 messages
MAX_REPLY_CHARS = 3000      # 钉钉 markdown 单条上限远大于此，先截断保阅读体验


def _conn() -> sqlite3.Connection:
    """每次操作开短连接：SQLite 跨线程共用连接易锁，这个量级短连接最省心"""
    c = sqlite3.connect(HISTORY_DB)
    c.execute("""CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,             -- user 存原始问题（不带参考资料，防多轮滚雪球）
        content TEXT NOT NULL,
        sources TEXT,                   -- assistant 行专属：命中来源 JSON（同时是使用日志）
        latency_ms INTEGER,             -- assistant 行专属：检索+生成总耗时
        tokens INTEGER,                 -- assistant 行专属：input+output token
        ts REAL NOT NULL)""")
    return c


def load_history(conversation_id: str) -> list[dict]:
    """取该会话最近 N 条（旧→新），口径与 Web 端前端上送的 history 一致"""
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? "
            "ORDER BY id DESC LIMIT ?", (conversation_id, HISTORY_WINDOW)).fetchall()
    return [{"role": r, "content": t} for r, t in reversed(rows)]


def append_history(conversation_id: str, role: str, content: str,
                   sources: str | None = None, latency_ms: int | None = None,
                   tokens: int | None = None) -> None:
    with _conn() as c:
        c.execute("INSERT INTO messages(conversation_id, role, content, sources, latency_ms, tokens, ts) "
                  "VALUES(?,?,?,?,?,?,?)",
                  (conversation_id, role, content, sources, latency_ms, tokens, time.time()))


def generate_stream(client: Anthropic, question: str, history: list[dict],
                    retrieved: list[dict], card) -> tuple[str, int]:
    """流式生成 + AI 卡片打字机。拼装口径与 chat 路由 / eval.py 同构。
    - thinking 关闭：数值问答不需要长思考，实测快 ~30% 且省全部思考 tokens
    - 卡片推送按 ~1s 节流、append=False 全量替换（SDK 语义：非追加）
    - usage 从 get_final_message() 拿——流式也能记账的关键（在 with 块内）"""
    context = demo.build_context(retrieved)
    user_msg = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"
    buf, last_push, n_push = "", 0.0, 0
    with client.messages.stream(model=demo._CHAT_MODEL, max_tokens=2048,
                                system=demo.SYSTEM_PROMPT,
                                messages=history + [{"role": "user", "content": user_msg}],
                                thinking={"type": "disabled"}) as s:
        for delta in s.text_stream:
            buf += delta
            now = time.time()
            if card and now - last_push >= 1.0:
                # 首帧走 ai_streaming（顺带把卡片置 INPUTING 打字机态），后续帧底层
                # 直发全量；SDK 的 ai_streaming 收不了尾（永远 finished=False，卡片
                # 会挂在流式会话里），isFinalize 终帧由调用方补——三要素缺一不可
                if n_push == 0:
                    card.ai_streaming(buf)
                else:
                    card.streaming(card.card_instance_id, "msgContent", buf,
                                   append=False, finished=False, failed=False)
                n_push += 1
                last_push = now
        usage = s.get_final_message().usage
    log.info("流式 %d 帧 / 共 %d 字", n_push, len(buf))
    return buf, usage.input_tokens + usage.output_tokens


def build_reply(answer: str, retrieved: list[dict]) -> str:
    """markdown 回复：答案 + 引用来源脚注（Web 端来源面板的机器人版）"""
    if len(answer) > MAX_REPLY_CHARS:
        answer = answer[:MAX_REPLY_CHARS] + "\n\n…（超长截断，完整答案见 Web 界面）"
    parts = [answer]
    if retrieved:
        parts.append("\n\n---\n**本次依据**")
        for i, e in enumerate(retrieved, 1):
            where = f" {e['source']} p{e['page']}" if e["page"] else f" {e['source']}"
            label = demo._TYPE_LABEL.get(e["type"], "正文")
            parts.append(f"\n[{i}]（{label}）{where}")
    return "".join(parts)


class RAGBotHandler(dingtalk_stream.ChatbotHandler):
    """钉钉消息 → RAG 链路 → markdown 回复。ACK 与回答分离：process 立刻返回，
    生成丢线程池异步做——同步等完长尾生成会触发钉钉超时重推（官方 issue 实锤）。"""

    def __init__(self, client: Anthropic):
        super().__init__()
        self.client = client
        self._seen: dict[str, float] = {}     # message_id -> ts，挡重推造成的重复回答
        self._lock = threading.Lock()

    async def process(self, callback: dingtalk_stream.CallbackMessage):
        msg = dingtalk_stream.ChatbotMessage.from_dict(callback.data)
        with self._lock:
            stale = [k for k, ts in self._seen.items() if time.time() - ts > 600]
            for k in stale:
                self._seen.pop(k, None)
            if msg.message_id and msg.message_id in self._seen:
                return AckMessage.STATUS_OK, "OK"      # 重推消息：已答过，直接 ACK
            if msg.message_id:
                self._seen[msg.message_id] = time.time()
        asyncio.get_running_loop().create_task(self._answer(msg))
        return AckMessage.STATUS_OK, "OK"

    async def _answer(self, msg: dingtalk_stream.ChatbotMessage):
        loop = asyncio.get_running_loop()
        holder = {"card": None}                          # 异常时拿得到卡片，好调 ai_fail
        try:
            await loop.run_in_executor(None, self._answer_sync, msg, holder)
        except Exception as e:                          # 异步任务里的异常没人接，自己兜
            self.logger.error(f"[BOT] 回答失败：{e}")
            if holder["card"]:
                try:
                    holder["card"].ai_fail()
                except Exception:
                    pass
            try:
                self.reply_text(f"回答失败：{e}", msg)
            except Exception:
                pass

    def _answer_sync(self, msg: dingtalk_stream.ChatbotMessage, holder: dict):
        question = (msg.text.content if msg.text else "").strip()
        question = re.sub(r"^@\S+\s*", "", question)     # 群聊 @机器人 前缀剥掉
        nick = msg.sender_nick or msg.sender_staff_id or "?"
        conv = msg.conversation_id or "default"
        if not question:
            self.reply_text("暂时只支持文字提问。", msg)
            return
        if not demo.store.chunk_count():
            self.reply_text("向量库是空的：先在 Web 界面导入文档，再回来提问。", msg)
            return

        history = load_history(conv)
        t0 = time.perf_counter()
        retrieved, reranked = demo.retrieve(question, top_k=demo.TOP_K)

        # AI 卡片先上（「执行中」占位）——失败降级为整条 markdown 回复，链路不断。
        # 注意 SDK 的坑：建卡片失败是**静默的**（只记日志不抛异常，card_instance_id
        # 留空），必须显式校验，否则废卡会吞掉降级路径——消息进来却一条都不回
        try:
            card = self.ai_markdown_card_start(msg, title="Mini-RAG")
            holder["card"] = card if getattr(card, "card_instance_id", None) else None
            if card and holder["card"] is None:
                self.logger.warning("AI 卡片创建静默失败（权限未开通？Card.Instance.Write），降级整条回复")
            log.info("卡片 %s", holder["card"].card_instance_id if holder["card"] else "降级整条回复")
        except Exception as e:
            self.logger.warning(f"AI 卡片创建失败，降级整条回复：{e}")

        answer, tokens = generate_stream(self.client, question, history, retrieved, holder["card"])
        latency_ms = int((time.perf_counter() - t0) * 1000)

        sources = json.dumps([{"i": i, "type": e["type"], "source": e["source"], "page": e["page"]}
                              for i, e in enumerate(retrieved, 1)], ensure_ascii=False)
        append_history(conv, "user", question)           # 入史的是原始问题（同 Web 端口径）
        append_history(conv, "assistant", answer, sources=sources,
                       latency_ms=latency_ms, tokens=tokens)
        reply_md = build_reply(answer, retrieved)
        if holder["card"]:
            c = holder["card"]
            # isFinalize 终帧：把流式会话按协议收掉（含来源脚注全文），再走 ai_finish
            # 刷完成态——两步都做，钉钉端哪个生效都能看到全文
            c.streaming(c.card_instance_id, "msgContent", reply_md,
                        append=False, finished=True, failed=False)
            log.info("终帧(isFinalize) %d 字", len(reply_md))
            c.ai_finish(reply_md)
            log.info("卡片定稿 %d 字", len(reply_md))
        else:
            self.reply_markdown("Mini-RAG", reply_md, msg)
            log.info("整条回复 %d 字", len(reply_md))
        log.info("%s: %s → %d 字 · %dms · %d tokens · rerank%s",
                 nick, question[:40], len(answer), latency_ms, tokens, "正常" if reranked else "降级")


def bot_main():
    """bot 线程入口：建事件循环（非主线程没有现成 loop）→ 连钉钉 → 常驻收消息"""
    cid = os.environ.get("DINGTALK_CLIENT_ID", "")
    secret = os.environ.get("DINGTALK_CLIENT_SECRET", "")
    key = os.environ.get("ANTHROPIC_API_KEY") or demo.API_KEY
    if not (cid and secret and key):
        print("[BOT] 未配置 DINGTALK_CLIENT_ID/SECRET（或 API Key），钉钉机器人不启动")
        return

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = dingtalk_stream.DingTalkStreamClient(dingtalk_stream.Credential(cid, secret))
    client.register_callback_handler(dingtalk_stream.ChatbotMessage.TOPIC,
                                     RAGBotHandler(Anthropic(base_url=demo._CHAT_BASE_URL,
                                                             api_key=demo._CHAT_API_KEY)))
    log.info("钉钉机器人已启动（Stream 模式，出站连接免公网）· 等待消息…")
    try:
        client.start_forever()          # 阻塞常驻，网络异常自动重连
    except Exception as e:
        log.error("Stream 连接退出：%s", e)
