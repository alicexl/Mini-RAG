# Mini-RAG

四步从零搭一个 RAG（检索增强生成）。每步一个自包含的 `demo.py` + 一份 `README.md`（快速开始 + 完整教学文档）：embedding 是一次 HTTP 调用、分块是递归切分隔符、向量库是余弦点积、检索是召回 + rerank 精排、生成是拼 prompt 喂 LLM——**没有黑盒**。

## 四步路径

| Step | 主题 | 教学文档 |
|---|---|---|
| 1 | **基础件**：分块（递归切分保语义边界 + overlap）与向量化（Embedding + 余弦相似度） | [step1_basics/README.md](step1_basics/README.md) |
| 2 | **向量库**：JSON + numpy 存 chunk 和向量，从零实现检索 | [step2_index/README.md](step2_index/README.md) |
| 3 | **完整 RAG**：混合检索（向量 + BM25 两路召回 → RRF 融合 → rerank 精排）+ 拼 context + system 约束 + 生成 | [step3_rag/README.md](step3_rag/README.md) |
| 4 | **聊天界面**：Flask 手写 Web 界面，SSE 流式 + 文档上传 | [step4_app/README.md](step4_app/README.md) |

每个 `demo.py` 自包含、可独立运行，不依赖共享库。

## 运行前提

- Python 3.10+（开发环境 3.12），`pip install -r requirements.txt`
- **检索（本地）**：[Ollama](https://ollama.com) 已启动，`ollama pull qwen3-embedding:0.6b`——离线、免 key
- **生成 + 精排（远程）**：智谱 API Key（[bigmodel.cn](https://bigmodel.cn) 生成，`id.secret` 格式）。生成走 GLM 的 Anthropic 兼容网关（默认 `glm-5.3`），rerank 走智谱 rerank API（`rerank` 模型）——两者共用同一个 Key

## 快速开始

```bash
set ANTHROPIC_API_KEY=你的智谱key      # macOS/Linux 用 export
python -X utf8 step2_index/demo.py
```

step1 的向量化部分与 step2 需要本地 Ollama（step1 的分块演示纯标准库、零依赖）；step3/4 另需智谱 key。

step3 是两条子命令：

```bash
python -X utf8 step3_rag/demo.py ingest <文档路径>   # 导入文档（幂等，重复导入自动跳过）
python -X utf8 step3_rag/demo.py chat                # 交互式问答
```

step4 起 Web 界面，浏览器自动打开：

```bash
python -X utf8 step4_app/demo.py    # → http://127.0.0.1:7860
```

## 示例语料

`step3_rag/sample_reports/` 与 `step4_app/sample_reports/` 内置盐湖股份 2023-2025 年报节选（`build_excerpts.py` 从年报 PDF 提取生成），拿来即可跑通完整链路。

## 朴素优先

本项目用裸代码覆盖了生产级检索的**完整三段式**：向量 + BM25 两路召回宽进（各 Top-20）→ RRF 融合（ID 归并去重、按排名融合）→ rerank 精排收紧（Top-3）再进生成（rerank 失败自动降级回 RRF 顺序）。每一环都是摊开的手写代码——余弦是矩阵乘、BM25 是手写 Okapi、RRF 是十行公式。剩下没做的（句子级引用、ANN 索引、query 改写）与实测发现的边界（rerank 分数饱和），在 [step3 README 的「剩下的边界」](step3_rag/README.md) 里展开。
