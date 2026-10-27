# Mini-RAG

四步从零搭一个 RAG（检索增强生成）。每步一个自包含的 `demo.py` + 一份 `README.md`（快速开始 + 完整教学文档）：embedding 是一次 HTTP 调用、分块是递归切分隔符、向量库是余弦点积、检索是召回 + rerank 精排、生成是拼 prompt 喂 LLM——**没有黑盒**。

## 四步路径

| Step | 主题 | 教学文档 |
|---|---|---|
| 1 | **基础件**：分块（递归切分保语义边界 + overlap）与向量化（Embedding + 余弦相似度） | [step1_basics/README.md](step1_basics/README.md) |
| 2 | **向量库**：JSON + numpy 存 chunk 和向量，从零实现检索 | [step2_index/README.md](step2_index/README.md) |
| 3 | **完整 RAG**：混合检索（向量 + BM25 两路召回 → RRF 融合 → rerank 精排）+ 拼 context + system 约束 + 生成 | [step3_rag/README.md](step3_rag/README.md) |
| 4 | **聊天界面 + 工程底座 + 多模态 + 评测**：Flask 手写 Web（SSE 流式 + 上传），检索底座换 Milvus Lite 向量库，PDF 直接入库（正文/表格/图片三模态），分层评测集 + 两层判分让检索质量有数字 | [step4_app/README.md](step4_app/README.md) |

每个 `demo.py` 自包含、可独立运行，不依赖共享库。

## 运行前提

- Python 3.10+（开发环境 3.12），`pip install -r requirements.txt`
- **检索**：step1-3 用本地 [Ollama](https://ollama.com)（离线、免 key，`ollama pull qwen3-embedding:0.6b`）；step4 换智谱 embedding-3（与生成共用 Key，检索延迟 2.7s→0.9s，见 step4 README 改造④）
- **向量库（step4）**：Milvus Lite，`pip install -r requirements.txt` 自带（纯 Python 实现，单文件库，无需装服务）
- **生成 + 精排 + 图片描述（远程）**：智谱 API Key（[bigmodel.cn](https://bigmodel.cn) 生成，`id.secret` 格式）。生成走 GLM 的 Anthropic 兼容网关（默认 `glm-5.3`），rerank 走智谱 rerank API（`rerank` 模型），step4 的 PDF 图片描述走 VLM（默认 `glm-4v-flash`，免费档）——三者共用同一个 Key

## 快速开始

```bash
set ANTHROPIC_API_KEY=你的智谱key      # macOS/Linux 用 export
python -X utf8 step2_index/demo.py
```

step1 的向量化部分与 step2/3 需要本地 Ollama（step1 的分块演示纯标准库、零依赖）；step3 起另需智谱 key；step4 只需智谱 key（检索嵌入已换智谱 embedding-3）。

step3 是两条子命令：

```bash
python -X utf8 step3_rag/demo.py ingest <文档路径>   # 导入文档（幂等，重复导入自动跳过）
python -X utf8 step3_rag/demo.py chat                # 交互式问答
```

step4 起 Web 界面，浏览器自动打开：

```bash
python -X utf8 step4_app/demo.py    # → http://127.0.0.1:7860
python -X utf8 step4_app/eval.py    # 评测：15 条分层 case，检索命中 / 答案断言 / pass@N

# 可选：接入钉钉机器人（Stream 模式出站连接，免公网 IP/域名/证书，见 step4 README 改造⑥）
set DINGTALK_CLIENT_ID=xxx
set DINGTALK_CLIENT_SECRET=xxx
```

## 示例语料

`step3_rag/sample_reports/` 与 `step4_app/sample_reports/` 内置盐湖股份 2023-2025 年报节选（`build_excerpts.py` 从年报 PDF 提取生成），拿来即可跑通完整链路。step4 另有一份 `盐湖股份2024年报（精选页）.pdf`（`tools/extract_pages.py` 从真年报抽 10 页），用来演示多模态检索：表格问题命中表格块（HTML 原文回填）、图片问题命中图片块（VLM 描述 + 原图回显）。

## 朴素优先

本项目前三步用裸代码覆盖了生产级检索的**完整三段式**：向量 + BM25 两路召回宽进（各 Top-20）→ RRF 融合（ID 归并去重、按排名融合）→ rerank 精排收紧（Top-3）再进生成（rerank 失败自动降级回 RRF 顺序）。每一环都是摊开的手写代码——余弦是矩阵乘、BM25 是手写 Okapi、RRF 是十行公式。第四步产品化时把这套三件套换成 Milvus Lite 向量库（两路召回与融合收进库内、HNSW 索引，代码即生产同款）——先摊开看清，再换工业件。剩下没做的（句子级引用、query 改写）与实测发现的边界（rerank 分数饱和），在 [step3 README 的「剩下的边界」](step3_rag/README.md) 里展开。
