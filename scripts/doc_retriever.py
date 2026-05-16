"""
文档检索模块
功能：三路检索（BM25 + Dense + Sparse）+ Rerank 重排序
"""

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
import sys
import json
import re
from typing import List, Dict, Set, Optional, Tuple
import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from scripts.db_manager import DatabaseManager, init_db
from scripts.config_manager import get_config
from scripts.reranker import Reranker
from scripts.bm25_indexer import BM25Indexer


class DocRetriever:
    """三路文档检索器"""

    def __init__(
        self,
        db_manager: DatabaseManager = None,
        faiss_index_path: str = None,
        model_path: str = None,
        reranker_path: str = None
    ):
        self.config = get_config()

        if db_manager:
            self.db = db_manager
        else:
            self.db = DatabaseManager(self.config.db_path)
            self.db.init_database()

        if faiss_index_path is None:
            faiss_index_path = self.config.faiss_index_path
        self.faiss_index_path = faiss_index_path

        if model_path is None:
            model_path = self.config.model_path
        self.model_path = model_path

        if reranker_path is None:
            reranker_path = self.config.reranker_path
        self.reranker_path = reranker_path

        self._faiss_index = None
        self._embedding_model = None
        self._sparse_vectors = None
        self._sparse_index = None
        self._chunks_meta = None
        self._chunk_contents = None
        self._reranker = None
        self._bm25_indexer = None

    def _get_active_vector_ids(self) -> Optional[Set[int]]:
        try:
            conn = self.db._get_connection()
            conn.commit()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ch.vector_id
                FROM chunks ch
                JOIN corpus c ON ch.corpus_id = c.id
                WHERE ch.vector_id IS NOT NULL
                  AND c.is_active = 1
                  AND c.status = 'active'
            """)
            return {int(row["vector_id"]) for row in cursor.fetchall() if row["vector_id"] is not None}
        except Exception as e:
            print(f"  [警告] 读取有效文档向量ID失败: {e}")
            return None

    def _load_faiss_index(self):
        """加载 FAISS 索引和相关数据"""
        if self._faiss_index is not None:
            return

        import pickle

        index_path = os.path.join(self.faiss_index_path, "index.faiss")
        contents_path = os.path.join(self.faiss_index_path, "chunk_contents.pkl")
        meta_path = os.path.join(self.faiss_index_path, "chunks_meta.pkl")
        sparse_vectors_path = os.path.join(self.faiss_index_path, "sparse_vectors.pkl")
        sparse_index_path = os.path.join(self.faiss_index_path, "sparse_index.pkl")

        if not all(os.path.exists(p) for p in [index_path, contents_path, meta_path]):
            raise Exception("FAISS 索引文件不完整")

        self._faiss_index = faiss.read_index(index_path)

        with open(contents_path, "rb") as f:
            self._chunk_contents = pickle.load(f)

        with open(meta_path, "rb") as f:
            self._chunks_meta = pickle.load(f)

        if self._faiss_index.ntotal != len(self._chunk_contents) or self._faiss_index.ntotal != len(self._chunks_meta):
            raise Exception(
                f"FAISS 索引与元数据不一致: index={self._faiss_index.ntotal}, "
                f"contents={len(self._chunk_contents)}, meta={len(self._chunks_meta)}"
            )

        if os.path.exists(sparse_vectors_path) and os.path.exists(sparse_index_path):
            with open(sparse_vectors_path, "rb") as f:
                self._sparse_vectors = pickle.load(f)
            with open(sparse_index_path, "rb") as f:
                self._sparse_index = pickle.load(f)
            if len(self._sparse_vectors) != self._faiss_index.ntotal:
                print("  [警告] 稀疏向量数量与 FAISS 不一致，将禁用 Sparse 检索")
                self._sparse_vectors = None
                self._sparse_index = None

    def _init_embedding_model(self):
        """初始化嵌入模型"""
        if self._embedding_model is not None:
            return

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            device = "cpu"

        print(f"正在加载 BGE-M3 嵌入模型 (设备: {device})...")
        try:
            self._embedding_model = BGEM3FlagModel(
                model_name_or_path=self.model_path,
                use_fp16=False,
                local_files_only=True,
                trust_remote_code=True,
                devices=device,
            )
        except TypeError:
            self._embedding_model = BGEM3FlagModel(
                model_name_or_path=self.model_path,
                use_fp16=False,
            )
        print("嵌入模型加载完成")

    def _embed_query_dense(self, query: str) -> np.ndarray:
        """生成查询的稠密向量"""
        self._init_embedding_model()

        embedding = self._embedding_model.encode(
            [query],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            batch_size=1,
        )
        dense_vec = embedding["dense_vecs"][0]
        norm = np.linalg.norm(dense_vec)
        if norm > 0:
            dense_vec = dense_vec / norm
        return dense_vec.reshape(1, -1).astype(np.float32)

    def _embed_query_sparse(self, query: str) -> Dict[int, float]:
        """生成查询的稀疏向量"""
        self._init_embedding_model()

        embedding = self._embedding_model.encode(
            [query],
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=1,
        )
        sparse_vec = embedding["lexical_weights"][0]
        return {int(k): float(v) for k, v in sparse_vec.items()}

    def _load_bm25_index(self):
        if self._bm25_indexer is not None:
            return
        self._bm25_indexer = BM25Indexer(self.faiss_index_path)
        if not self._bm25_indexer.load():
            print("  [警告] BM25 索引不存在，将降级为两路融合（Dense + Sparse）")
            self._bm25_indexer = None

    def _retrieve_bm25(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        self._load_bm25_index()
        if self._bm25_indexer is None:
            return []
        return self._bm25_indexer.retrieve(query, top_k=top_k)

    def _retrieve_dense(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """稠密向量检索"""
        self._load_faiss_index()
        query_vec = self._embed_query_dense(query)
        distances, indices = self._faiss_index.search(query_vec, top_k)
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self._chunk_contents):
                results.append((int(idx), float(distances[0][rank])))
        return results

    def _retrieve_sparse(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """稀疏向量检索"""
        if self._sparse_index is None:
            self._load_faiss_index()
            if self._sparse_index is None:
                return []

        query_sparse = self._embed_query_sparse(query)
        doc_scores = {}

        for token_id, query_weight in query_sparse.items():
            token_id = int(token_id)
            if token_id in self._sparse_index:
                for doc_id, doc_weight in self._sparse_index[token_id].items():
                    doc_id = int(doc_id)
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0.0
                    doc_scores[doc_id] += query_weight * doc_weight

        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]] = None,
        top_k: int = 20,
        rrf_k: int = 30
    ) -> List[int]:
        scores = {}

        for rank, (doc_id, _) in enumerate(dense_results):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (rrf_k + rank + 1)

        for rank, (doc_id, _) in enumerate(sparse_results):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (rrf_k + rank + 1)

        if bm25_results:
            for rank, (doc_id, _) in enumerate(bm25_results):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (rrf_k + rank + 1)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]

    def _init_reranker(self):
        """初始化 Reranker"""
        if self._reranker is None:
            self._reranker = Reranker(self.reranker_path)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True
    ) -> List[Dict]:
        top_k = max(1, int(top_k or 5))
        bm25_results = self._retrieve_bm25(query, top_k=100)
        dense_results = self._retrieve_dense(query, top_k=100)
        sparse_results = self._retrieve_sparse(query, top_k=100)

        if bm25_results:
            print(f"  BM25 检索：返回 {len(bm25_results)} 个结果")
        else:
            print(f"  BM25 检索：无结果（索引不存在或未加载）")

        active_vector_ids = self._get_active_vector_ids()
        if active_vector_ids is not None:
            before_counts = (len(bm25_results), len(dense_results), len(sparse_results))
            bm25_results = [(doc_id, score) for doc_id, score in bm25_results if doc_id in active_vector_ids]
            dense_results = [(doc_id, score) for doc_id, score in dense_results if doc_id in active_vector_ids]
            sparse_results = [(doc_id, score) for doc_id, score in sparse_results if doc_id in active_vector_ids]
            after_counts = (len(bm25_results), len(dense_results), len(sparse_results))
            if before_counts != after_counts:
                print(f"  [过滤] 已排除删除/停用文档向量: {before_counts} → {after_counts}")

        rerank_faiss_topk = max(30, top_k * 5) if use_rerank else max(top_k * 3, top_k)
        fused_doc_ids = self._reciprocal_rank_fusion(dense_results, sparse_results, bm25_results, top_k=rerank_faiss_topk)

        self._load_faiss_index()
        results = []
        for rank, doc_id in enumerate(fused_doc_ids, 1):
            if 0 <= doc_id < len(self._chunk_contents):
                meta = self._chunks_meta[doc_id] if self._chunks_meta else {}
                images = meta.get("images", [])
                abs_images = [self.config.to_absolute_path(img) for img in images]

                results.append({
                    "doc_id": doc_id,
                    "content": self._chunk_contents[doc_id],
                    "meta": {
                        **meta,
                        "images": abs_images
                    },
                    "rank": rank
                })

        if active_vector_ids is not None:
            before_count = len(results)
            results = [r for r in results if int(r["doc_id"]) in active_vector_ids]
            if before_count != len(results):
                print(f"  [过滤] 结果阶段移除 {before_count - len(results)} 个无效分片")

        if use_rerank and results:
            self._init_reranker()
            results = self._reranker.rerank(query, results, top_k=top_k)

        return results[:top_k]


def test_doc_retriever():
    """测试文档检索器"""
    print("=" * 60)
    print("文档检索器测试")
    print("=" * 60)

    try:
        retriever = DocRetriever()

        test_queries = [
            "STM32F103 配置方法",
            "STM32F407 使用教程",
        ]

        print("\n检索测试:")
        print("-" * 60)
        for query in test_queries:
            print(f"查询: {query}")
            results = retriever.retrieve(query, top_k=3)
            print(f"  返回 {len(results)} 个结果")
            print()

        print("测试完成!")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_doc_retriever()
