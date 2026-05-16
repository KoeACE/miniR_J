"""
Rerank 模块
功能：使用 BGE-Reranker-v2-M3 对检索结果进行重排序
"""

import os
import sys
from typing import List, Dict, Tuple

# 添加项目根目录到路径
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from scripts.config_manager import get_config


class Reranker:
    """Reranker 重排序器"""

    def __init__(self, model_path: str = None):
        """
        初始化 Reranker

        Args:
            model_path: Reranker 模型路径
        """
        config = get_config()
        if model_path is None:
            model_path = config.reranker_path
        self.model_path = model_path
        self._reranker = None

    def _init_reranker(self):
        """初始化 Reranker 模型"""
        if self._reranker is not None:
            return

        # 自动检测 GPU
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            device = "cpu"

        try:
            from FlagEmbedding import FlagReranker
            print(f"正在加载 BGE-Reranker-v2-M3 模型 (设备: {device})...")
            self._reranker = FlagReranker(
                model_name_or_path=self.model_path,
                use_fp16=False,
                device=device,
            )
            print("Reranker 模型加载完成")
        except ImportError:
            print("警告: FlagEmbedding 未安装，Rerank 功能不可用")
            self._reranker = None
        except Exception as e:
            print(f"加载 Reranker 模型失败: {e}")
            self._reranker = None

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        对检索结果进行重排序

        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回结果数

        Returns:
            重排序后的结果列表
        """
        if not results:
            return []

        self._init_reranker()

        if self._reranker is None:
            # Reranker 不可用，返回原始结果
            print("警告: Reranker 不可用，返回原始结果")
            return results[:top_k]

        # 准备文档对
        pairs = []
        for result in results:
            content = result.get("content", "")
            meta = result.get("meta", {})
            title = meta.get("title", "")
            # 使用标题+内容作为 passage
            passage = f"{title}\n{content}" if title else content
            passage = passage[:1000]  # 限制长度
            pairs.append((query, passage))

        # 计算 Rerank 分数
        try:
            scores = self._reranker.compute_score(pairs, normalize=True)
        except Exception as e:
            print(f"Rerank 计算失败: {e}")
            return results[:top_k]
        try:
            iter(scores)
        except TypeError:
            scores = [scores]

        # 组装结果
        reranked_results = []
        for i, (result, score) in enumerate(zip(results, scores)):
            score_val = float(score) if isinstance(score, (int, float)) else float(score[0])
            reranked_results.append({
                **result,
                "rerank_score": score_val,
                "original_rank": i + 1
            })

        # 按 Rerank 分数排序
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 更新排名
        for i, result in enumerate(reranked_results):
            result["rank"] = i + 1

        return reranked_results[:top_k]


def test_reranker():
    """测试 Reranker"""
    print("=" * 60)
    print("Reranker 测试")
    print("=" * 60)

    reranker = Reranker()

    # 模拟检索结果
    query = "EXAMPLE-A123 GPIO配置"
    results = [
        {
            "doc_id": 1,
            "content": "EXAMPLE-A123 的 GPIO 配置方法...",
            "meta": {"doc_name": "EXAMPLE-A123手册.md", "title": "GPIO配置"}
        },
        {
            "doc_id": 2,
            "content": "EXAMPLE-E020A 的 GPIO 配置方法...",
            "meta": {"doc_name": "EXAMPLE-E020A手册.md", "title": "GPIO配置"}
        },
        {
            "doc_id": 3,
            "content": "EXAMPLE-A123 的 PWM 配置方法...",
            "meta": {"doc_name": "EXAMPLE-A123手册.md", "title": "PWM配置"}
        },
    ]

    print(f"\n查询: {query}")
    print(f"原始结果数: {len(results)}")

    reranked = reranker.rerank(query, results, top_k=3)

    print("\n重排序结果:")
    for i, result in enumerate(reranked, 1):
        print(f"[{i}] 分数: {result.get('rerank_score', 0):.4f}, 原排名: {result.get('original_rank', '-')}")
        print(f"    文档: {result['meta'].get('doc_name', 'unknown')}")


if __name__ == "__main__":
    test_reranker()
