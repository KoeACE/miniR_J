"""
BM25索引模块
功能：基于BM25Okapi的文本检索，支持中英文混合分词、索引构建与持久化
"""

import os
import sys
import re
import pickle
from typing import List, Set, Tuple, Optional, Dict

import jieba
from rank_bm25 import BM25Okapi

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)


def tokenize(text: str) -> List[str]:
    if not text:
        return []

    chinese_chars = re.findall(r'[\u4e00-\u9fa5]+', text)
    if chinese_chars:
        text_for_jieba = re.sub(r'([a-zA-Z0-9]+)', r' \1 ', text)
        tokens = jieba.lcut(text_for_jieba)
    else:
        tokens = text.split()

    result = []
    for token in tokens:
        token = token.strip().lower()
        if len(token) >= 2:
            result.append(token)

    return result


class BM25Indexer:

    def __init__(self, index_path: str):
        self.index_path = index_path
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_map: Dict[int, int] = {}
        self._tokenized_corpus: List[List[str]] = []

        os.makedirs(self.index_path, exist_ok=True)

    def build_index(self, chunk_contents: List[str], chunk_ids: List[int]):
        print(f"正在构建BM25索引，共 {len(chunk_contents)} 个文档...")

        self._tokenized_corpus = []
        self.corpus_map = {}

        if not chunk_contents or not chunk_ids:
            self.bm25 = None
            print("BM25索引为空")
            return

        for i, (content, chunk_id) in enumerate(zip(chunk_contents, chunk_ids)):
            tokens = tokenize(content)
            self._tokenized_corpus.append(tokens)
            self.corpus_map[i] = chunk_id

        if not self._tokenized_corpus:
            self.bm25 = None
            print("BM25索引为空")
            return

        self.bm25 = BM25Okapi(self._tokenized_corpus)

        print(f"BM25索引构建完成，共索引 {len(self._tokenized_corpus)} 个文档")

    def append_documents(self, new_contents: List[str], new_ids: List[int]):
        if self.bm25 is None:
            self.build_index(new_contents, new_ids)
            return

        print(f"正在追加 {len(new_contents)} 个文档到BM25索引...")

        existing_count = len(self._tokenized_corpus)

        for i, (content, chunk_id) in enumerate(zip(new_contents, new_ids)):
            tokens = tokenize(content)
            self._tokenized_corpus.append(tokens)
            self.corpus_map[existing_count + i] = chunk_id

        self.bm25 = BM25Okapi(self._tokenized_corpus)

        print(f"BM25索引追加完成，当前共索引 {len(self._tokenized_corpus)} 个文档")

    def save(self):
        index_file = os.path.join(self.index_path, "bm25_index.pkl")
        corpus_map_file = os.path.join(self.index_path, "bm25_corpus_map.pkl")

        if self.bm25 is None:
            for file_path in (index_file, corpus_map_file):
                if os.path.exists(file_path):
                    os.remove(file_path)
            print("BM25索引为空，已清理旧索引文件")
            return

        with open(index_file, "wb") as f:
            pickle.dump(self.bm25, f)

        with open(corpus_map_file, "wb") as f:
            pickle.dump({
                "corpus_map": self.corpus_map,
                "tokenized_corpus": self._tokenized_corpus,
            }, f)

        print(f"BM25索引已保存到 {self.index_path}")

    def load(self) -> bool:
        index_file = os.path.join(self.index_path, "bm25_index.pkl")
        corpus_map_file = os.path.join(self.index_path, "bm25_corpus_map.pkl")

        if not os.path.exists(index_file) or not os.path.exists(corpus_map_file):
            return False

        with open(index_file, "rb") as f:
            self.bm25 = pickle.load(f)

        with open(corpus_map_file, "rb") as f:
            data = pickle.load(f)
            self.corpus_map = data["corpus_map"]
            self._tokenized_corpus = data["tokenized_corpus"]

        print(f"BM25索引已加载，共 {len(self._tokenized_corpus)} 个文档")
        return True

    def retrieve(self, query: str, top_k: int = 100, candidate_ids: Set[int] = None) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            print("警告：BM25索引未加载，无法检索")
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        results: List[Tuple[int, float]] = []
        for i, score in enumerate(scores):
            chunk_id = self.corpus_map.get(i)
            if chunk_id is None:
                continue
            if candidate_ids is not None and chunk_id not in candidate_ids:
                continue
            results.append((chunk_id, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]
