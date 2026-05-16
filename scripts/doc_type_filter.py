"""
文档类型过滤模块
功能：根据文档类型过滤检索结果
"""

from typing import List, Dict, Optional, Set
from scripts.db_manager import DatabaseManager


class DocTypeFilter:
    """文档类型过滤器"""

    def __init__(self, db_manager: DatabaseManager = None):
        if db_manager:
            self.db = db_manager
        else:
            self.db = DatabaseManager()

    def get_chunks_by_doc_type(self, doc_type: str, use_vector_id: bool = True) -> Set[int]:
        corpus_ids = self.db.get_corpus_ids_by_category(doc_type)

        chunk_ids = set()
        for corpus_id in corpus_ids:
            chunks = self.db.get_chunks_by_corpus(corpus_id)
            if use_vector_id:
                chunk_ids.update([c.vector_id for c in chunks if c.vector_id is not None])
            else:
                chunk_ids.update([c.id for c in chunks])

        return chunk_ids

    def filter_results_by_doc_type(
        self,
        results: List[Dict],
        doc_type: str,
        top_k: int = 5,
        id_field: str = "doc_id"
    ) -> List[Dict]:
        use_vector_id = (id_field == "doc_id")
        allowed_chunk_ids = self.get_chunks_by_doc_type(doc_type, use_vector_id=use_vector_id)

        filtered = []
        for result in results:
            chunk_id = result.get(id_field)
            if chunk_id in allowed_chunk_ids:
                filtered.append(result)
            if len(filtered) >= top_k:
                break

        return filtered

    def detect_query_doc_type(self, query: str) -> Optional[str]:
        query_lower = query.lower()

        requirement_keywords = ['需求', 'cl', 'prd', '产品需求', '功能需求', '用户故事', '吸顶灯']
        entity_keywords = ['example-a', 'example-b', 'example-d', 'example-e', 'example-f', 'example-g', 'example-h', '实体', 'mcu', '手册', 'datasheet']
        tool_keywords = ['tool', '仿真器', '烧录器', '下载器', '工具', '软件', '调试']

        for keyword in requirement_keywords:
            if keyword in query_lower:
                return '需求文档'

        for keyword in entity_keywords:
            if keyword in query_lower:
                return '实体文档'

        for keyword in tool_keywords:
            if keyword in query_lower:
                return '工具文档'

        return None


def test_doc_type_filter():
    filter = DocTypeFilter()

    test_queries = [
        "CL04 吸顶灯需求分析",
        "EXAMPLE-A123 实体烧录方法",
        "TOOL-1 工具使用说明",
        "仿真器连接不上怎么办",
    ]

    print("=" * 60)
    print("查询意图检测测试")
    print("=" * 60)

    for query in test_queries:
        doc_type = filter.detect_query_doc_type(query)
        print(f"查询: {query}")
        print(f"  检测到类型: {doc_type}")
        print()


if __name__ == "__main__":
    test_doc_type_filter()
