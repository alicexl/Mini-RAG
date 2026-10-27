"""step4 评测：让「检索质量」从感觉变成数字

一次评测 = 给系统一个输入，用一段判分逻辑检查输出，看成没成。本脚本把这条
流程跑成批量、可重复的：

- 评测集 evalset.py：15 条分层 case（常规/边界/异常），每条带标准出处（检索层）
  和关键词断言（生成层）——「输入、初始状态、成功标准、参考解答」四要素齐备
- 多次试验：每条 case 独立跑 --runs 次（默认 3），每次走完整链路
  （检索 → 拼 context → 生成），生成是概率性的，跑一次说明不了什么
- 判分器全用代码：检索层判「标准出处进没进 Top-3」，生成层判关键词组 /
  拒答——财务数值题能写成断言的绝不靠肉眼
- 指标按环节拆开：检索层（命中@3、Top-1）与生成层（答对率、拒绝率、pass@N /
  pass^N）分开报——成功率上不去时，才分得清是检索没捞到还是生成没答好；
  外加过程指标（rerank 降级次数）、延迟（中位数 + P95）、成本（token）

启动（前提同 demo.py：智谱 Key 已设、库里已导入精选页 PDF）:
    python -X utf8 step4_app/eval.py             # 每条生成 3 次
    python -X utf8 step4_app/eval.py --runs 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))   # 让 import demo / evalset 找到本目录

import demo                                        # noqa: E402（单例在 main() 里接线，import 零副作用）
from evalset import CASES, REFUSE_PATTERNS        # noqa: E402

from anthropic import Anthropic                   # noqa: E402


def _percentile(xs: list[float], p: int) -> float:
    """nearest-rank 百分位：P95 看尾部——单次慢不说明什么，5% 的请求慢才是问题"""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.01 * p * len(xs)))]


def _match(entry: dict, cand: dict) -> bool:
    """检索层命中判定：表格/图片候选按 类型+页码，正文候选按 类型+关键词锚定
    （正文块没有页码，page 存的是 0）"""
    if entry["type"] != cand["type"]:
        return False
    if "kw" in cand:
        return cand["kw"] in entry["text"]
    return entry["page"] == cand["page"]


def _score_answer(answer: str, case: dict) -> bool:
    """生成层判分：普通题 = 每组关键词任一命中、全部组通过才算对；
    异常题（assert_groups=None）= 判拒答"""
    if case["assert_groups"] is None:
        return any(p in answer for p in REFUSE_PATTERNS)
    return all(any(k in answer for k in group) for group in case["assert_groups"])


def _generate(client: Anthropic, question: str, retrieved: list[dict]) -> tuple[str, int, int]:
    """非流式生成探针：prompt 拼装与 chat 路由完全同构（【参考资料】+【用户问题】），
    只是换成 messages.create 一把拿全文 + token 用量——流式 answer_stream 的 usage
    藏在生成器内部的 with 块里，外面拿不到。glm-5.3 会带 ThinkingBlock，只取文本块。"""
    context = demo.build_context(retrieved)
    user_msg = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"
    resp = client.messages.create(model=demo._CHAT_MODEL, max_tokens=2048,
                                  system=demo.SYSTEM_PROMPT,
                                  messages=[{"role": "user", "content": user_msg}])
    answer = "".join(b.text for b in resp.content if b.type == "text")
    return answer, resp.usage.input_tokens, resp.usage.output_tokens


def main():
    ap = argparse.ArgumentParser(description="Mini-RAG step4 评测")
    ap.add_argument("--runs", type=int, default=3,
                    help="每条 case 的生成次数（pass@N / pass^N 的 N，默认 3）")
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY") or demo.API_KEY
    if not key:
        raise SystemExit("未配置 API Key（set ANTHROPIC_API_KEY=你的智谱key），退出")

    # 接线：demo 的单例在它的 main() 里初始化（import 零副作用），这里按同一口径
    # 赋给模块级名字——retrieve() 调用时读的就是这几个全局
    demo.embedder = demo.ZhipuEmbedder(key)
    try:
        demo.embedder.embed_batch(["连通性检查"])
    except Exception as e:
        raise SystemExit(f"智谱 embedding 不可用（{e}）：检查 ANTHROPIC_API_KEY 与网络")
    demo.store = demo.MilvusStore(str(demo.DATA_DIR / "mini_rag.db"))
    demo.reranker = demo.Reranker(key)
    client = Anthropic(base_url=demo._CHAT_BASE_URL, api_key=key)

    n_chunks = demo.store.chunk_count()
    if not n_chunks:
        raise SystemExit("向量库为空：先启动 demo.py 导入 sample_reports/盐湖股份2024年报（精选页）.pdf，再跑评测")

    n_do = sum(1 for c in CASES if c["expect"] is not None)        # 该做的题（检索层也判）
    n_refuse = len(CASES) - n_do                                   # 不该做的题（只判拒答）
    print("=" * 68)
    print(f"Mini-RAG step4 评测 · 库内 {n_chunks} 块 · 评测集 {len(CASES)} 条"
          f"（该做 {n_do} / 不该做 {n_refuse}）· 每条生成 {args.runs} 次")
    print("=" * 68)

    rows = []                       # 每条 case 一行：检索命中 + 生成通过次数 + 答案样本
    ret_hits = ret_top1 = ret_runs = 0
    ret_by_layer: dict[str, list[int]] = {}      # layer -> [命中数, 总数]（按 run 计）
    gen_pass = gen_runs = 0                       # 该做的题（按 run 计）
    refuse_pass = refuse_runs = 0                 # 不该做的题（按 run 计）
    degraded = 0
    ret_ms: list[float] = []
    gen_s: list[float] = []
    in_tok = out_tok = 0

    for case in CASES:
        hit_any = top1_any = False
        k_passed = 0
        for _ in range(args.runs):
            t0 = time.perf_counter()
            retrieved, reranked = demo.retrieve(case["question"], top_k=demo.TOP_K)
            ret_ms.append((time.perf_counter() - t0) * 1000)
            degraded += (not reranked)

            if case["expect"] is not None:
                ret_runs += 1
                ret_by_layer.setdefault(case["layer"], [0, 0])[1] += 1
                hit = any(_match(e, c) for e in retrieved for c in case["expect"])
                top1 = any(_match(retrieved[0], c) for c in case["expect"])
                ret_hits += hit
                ret_top1 += top1
                if hit:
                    hit_any = True
                    ret_by_layer[case["layer"]][0] += 1
                top1_any = top1_any or top1

            t0 = time.perf_counter()
            answer, ti, to = _generate(client, case["question"], retrieved)
            gen_s.append(time.perf_counter() - t0)
            in_tok += ti
            out_tok += to
            ok = _score_answer(answer, case)
            k_passed += ok
            if case["assert_groups"] is None:
                refuse_runs += 1
                refuse_pass += ok
            else:
                gen_runs += 1
                gen_pass += ok

        rows.append((case, hit_any, top1_any, k_passed))
        ret_mark = "—" if case["expect"] is None else \
            ("✓Top1" if top1_any else ("✓@3" if hit_any else "✗"))
        print(f"  {case['id']} [{case['layer']}·{case['tag']}] 检索 {ret_mark}"
              f" · 生成 {k_passed}/{args.runs} · {case['note'].split('，')[0].split('：')[0]}")

    pass_at = sum(1 for c, _, _, k in rows if c["assert_groups"] is not None and k >= 1)
    pass_all = sum(1 for c, _, _, k in rows if c["assert_groups"] is not None and k == args.runs)
    n_gen_cases = sum(1 for c in CASES if c["assert_groups"] is not None)

    print("=" * 68)
    print("检索层（标准出处进 Top-3 判命中；异常题无标准出处，不判）")
    print(f"  命中@3  {ret_hits}/{ret_runs}（{ret_hits / ret_runs:.1%}）"
          f" · Top-1  {ret_top1}/{ret_runs}（{ret_top1 / ret_runs:.1%}）")
    print("  分层（按 run 计）：" + " · ".join(
        f"{k} {v[0]}/{v[1]}" for k, v in ret_by_layer.items()))
    print("生成层（代码断言）")
    print(f"  该做的题   答对 {gen_pass}/{gen_runs}（{gen_pass / gen_runs:.1%}）")
    print(f"  不该做的题 拒绝 {refuse_pass}/{refuse_runs}（{refuse_pass / refuse_runs:.1%}）")
    print(f"  pass@{args.runs}  {pass_at}/{n_gen_cases} · "
          f"pass^{args.runs}  {pass_all}/{n_gen_cases}"
          f"（做成一次就够 vs 每次都得对——生产上取哪个看场景）")
    print("过程与成本")
    print(f"  rerank 降级 {degraded}/{len(ret_ms)} · "
          f"检索 中位 {_percentile(ret_ms, 50):.0f}ms / P95 {_percentile(ret_ms, 95):.0f}ms · "
          f"生成 中位 {_percentile(gen_s, 50):.1f}s / P95 {_percentile(gen_s, 95):.1f}s")
    print(f"  token 输入 {in_tok:,} / 输出 {out_tok:,}（{len(gen_s)} 次生成，"
          f"均 {(in_tok + out_tok) // len(gen_s):,}/次）")


if __name__ == "__main__":
    main()
