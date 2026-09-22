# Mini-RAG

五步从零搭一个 RAG（检索增强生成）。每步一个自包含的 `demo.py` + 一份讲稿：embedding 是一次 HTTP 调用、分块是递归切分隔符、向量库是余弦点积、检索是 Top-K、生成是拼 prompt 喂 LLM——**没有黑盒**。

## 五步路径

| Step | 主题 | 讲稿 |
|---|---|---|
| 1 | **Embedding**：文本变向量，余弦相似度找语义接近 | [step1_embedding/讲稿.md](step1_embedding/讲稿.md) |
| 2 | **分块**：长文档切 chunk，递归切分保语义边界 + overlap | [step2_chunking/讲稿.md](step2_chunking/讲稿.md) |
| 3 | **向量库**：JSON + numpy 存 chunk 和向量，从零实现检索 | [step3_index/讲稿.md](step3_index/讲稿.md) |
| 4 | **完整 RAG**：检索 + 拼 context + system 约束 + 生成，防幻觉两道关 | [step4_rag/讲稿.md](step4_rag/讲稿.md) |
| 5 | **聊天界面**：套 Gradio，流式生成 + 文档管理 | [step5_app/讲稿.md](step5_app/讲稿.md) |

每个 `demo.py` 自包含、可独立运行，不依赖共享库。

## 运行前提

- Python 3.10+（开发环境 3.12），`pip install -r requirements.txt`
- **检索（本地）**：[Ollama](https://ollama.com) 已启动，`ollama pull qwen3-embedding:0.6b`——离线、免 key
- **生成（远程）**：智谱 API Key（[bigmodel.cn](https://bigmodel.cn) 生成，`id.secret` 格式），走 GLM 的 Anthropic 兼容网关（默认 `glm-5.3`）

## 快速开始

```bash
set ANTHROPIC_API_KEY=你的智谱key      # macOS/Linux 用 export
python -X utf8 step3_index/demo.py
```

step2 纯标准库、零依赖；step1/3 需要本地 Ollama；step4/5 另需智谱 key。

step4 是两条子命令：

```bash
python -X utf8 step4_rag/demo.py ingest <文档路径>   # 导入文档（幂等，重复导入自动跳过）
python -X utf8 step4_rag/demo.py chat                # 交互式问答
```

step5 起 Web 界面，浏览器自动打开：

```bash
python -X utf8 step5_app/demo.py    # → http://127.0.0.1:7860
```

## 示例语料

`step4_rag/sample_reports/` 与 `step5_app/sample_reports/` 内置盐湖股份 2023-2025 年报节选（`build_excerpts.py` 从年报 PDF 提取生成），拿来即可跑通完整链路。

## 朴素优先

本项目止步于**朴素 RAG**：纯向量检索、Top-K、无混合检索、无 rerank、无 agent。这些生产级做法（两级流水线：混合召回 → 粗排 → 精排 → 生成）在 [step4 讲稿的「朴素 RAG 的边界」](step4_rag/讲稿.md) 里展开——先把看得见的骨架走通，再谈叠什么。
