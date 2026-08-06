"""Step 1: 文本向量化 —— 理解 Embedding

学习目标:
- 什么是 Embedding？为什么文本可以变成向量？
- 语义相似的文本，向量在空间中更近
- 余弦相似度如何衡量向量之间的距离

运行:
    python step1_embedding/demo.py
"""

import os
import sys

import requests

# ===== Embedder：把文本变成向量 =====
# 设计要点：
# - model / base_url 从环境变量读，默认连本地 Ollama + qwen3-embedding:0.6b
# - embed_batch 一次请求嵌入多段文本（Ollama /api/embed 支持 input 为列表）
# - cosine_similarity 是纯数学，不依赖任何第三方库
_DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_DEFAULT_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")


class Embedder:
    """把文本喂给 Ollama 的 embedding 模型，拿回向量。"""

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

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """两个向量的余弦相似度，范围 [-1, 1]，越接近 1 越相似。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0  # 零向量防御，避免除零
        return dot / (norm_a * norm_b)


# ===== Demo：看 Embedding 的效果 =====
def main():
    print("=" * 60)
    print("Step 1: 文本向量化 —— 理解 Embedding")
    print("=" * 60)

    embedder = Embedder()
    print(f"\n模型: {embedder.model}")
    print(f"Ollama 地址: {embedder.base_url}")

    # 4 段测试文本 —— 1↔2 否定陷阱，3↔2 因果弱信号，文本4 天气正交
    texts = [
        "盐湖股份2024年净利润大幅增长。",
        "盐湖股份2024年净利润大幅下滑。",
        "碳酸锂价格在2024年持续下跌，从年初的10万元/吨跌至年末的7万元/吨。",
        "今天天气不错，适合出去散步。",
    ]

    print(f"\n正在向量化 {len(texts)} 段文本...")
    try:
        vectors = embedder.embed_batch(texts)
    except Exception as e:
        print(f"\n[ERROR] 向量化失败：{e}")
        print("请确认 Ollama 已启动（ollama serve）且模型已下载（ollama pull qwen3-embedding:0.6b）")
        sys.exit(1)

    # 打印每段文本的向量维度
    for i, (text, vec) in enumerate(zip(texts, vectors)):
        print(f"\n  文本{i+1}: {text[:40]}...")
        print(f"  向量维度: {len(vec)}, 前5个值: {[round(v, 4) for v in vec[:5]]}")

    # 计算两两之间的余弦相似度
    print("\n" + "=" * 60)
    print("余弦相似度矩阵（越接近 1 越相似）:")
    print("=" * 60)

    # 打印表头
    header = "      " + "".join(f" 文本{j+1} " for j in range(len(texts)))
    print(header)
    for i in range(len(texts)):
        row = f"文本{i+1} "
        for j in range(len(texts)):
            sim = Embedder.cosine_similarity(vectors[i], vectors[j])
            row += f" {sim:.3f} "
        print(row)

    # 解读
    print("\n" + "=" * 60)
    print("解读:")
    print("=" * 60)
    print("  文本1↔文本2: 净利润增长 vs 下滑，只差一词、意思相反 → 相似度仍很高（否定陷阱）")
    print("  文本3↔文本2: 碳酸锂跌 → 净利润下滑，因果相关 → 相似度却不高（弱信号）")
    print("  文本1/2/3 ↔ 文本4: 财务/商品 vs 天气，语义无关 → 相似度低")
    print()
    print("  这就是 Embedding 的核心价值：")
    print("  把文本映射到向量空间，语义相似的文本向量更近，")
    print("  从而可以用数学方法（余弦相似度）衡量语义相似性。")
    print("  RAG 检索的本质就是：找与用户查询向量最近的文档向量。")


if __name__ == "__main__":
    main()
