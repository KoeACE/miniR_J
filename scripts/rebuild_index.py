"""
重建 FAISS 索引脚本
功能：使用标题增强嵌入和子标题分片重新生成所有向量索引
用法：python scripts/rebuild_index.py [--clean] [--skip-embedding]
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import faiss
from tqdm import tqdm

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from scripts.db_manager import DatabaseManager
from scripts.config_manager import get_config


def rebuild_index(clean: bool = False, skip_embedding: bool = False):
    """
    重建 FAISS 索引

    Args:
        clean: 是否清除旧索引（重新生成向量）
        skip_embedding: 是否跳过向量生成（仅重建 FAISS 索引和关键词索引）
    """
    config = get_config()
    db = DatabaseManager(config.db_path)
    db.init_database()
    faiss_index_path = config.faiss_index_path

    print("=" * 70)
    print("重建 FAISS 索引")
    print("=" * 70)
    print(f"FAISS 索引路径: {faiss_index_path}")

    version_path = os.path.join(faiss_index_path, "index_version.json")
    if os.path.exists(version_path):
        with open(version_path, "r", encoding="utf-8") as f:
            version_info = json.load(f)
        print(f"当前索引版本: {version_info}")
    else:
        print("当前索引版本: v1（无版本标记）")

    print("\n[1/4] 读取所有分片...")
    conn = db._get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ch.id, ch.corpus_id, ch.chunk_index, ch.content, ch.title,
                   ch.title_level, ch.images, ch.vector_id,
                   c.name as doc_name, c.file_path
            FROM chunks ch
            JOIN corpus c ON ch.corpus_id = c.id
            WHERE c.status = 'active'
            ORDER BY CASE WHEN ch.vector_id IS NULL THEN 1 ELSE 0 END, ch.vector_id, ch.id
        """)
        chunk_rows = cursor.fetchall()
    finally:
        cursor.close()

    print(f"  找到 {len(chunk_rows)} 个分片")

    if not chunk_rows:
        print("没有分片数据，退出")
        return

    chunk_texts = []
    chunks_meta = []
    chunk_images = []

    for row in chunk_rows:
        doc_name = row['doc_name'] or 'unknown'
        title = row['title'] or ''
        content = row['content'] or ''
        title_level = row['title_level']
        images = json.loads(row['images']) if isinstance(row['images'], str) else (row['images'] or [])

        if title:
            chunk_text = f"{doc_name} > {title}\n{content}"
        else:
            chunk_text = f"{doc_name}\n{content}"

        chunk_texts.append(chunk_text)
        chunks_meta.append({
            "corpus_id": row["corpus_id"],
            "doc_name": doc_name,
            "title": title,
            "title_level": title_level,
            "images": images
        })
        chunk_images.append(images)

    if skip_embedding:
        print("\n[跳过] 向量生成（使用现有向量）")
        dense_path = os.path.join(faiss_index_path, "index.faiss")
        if not os.path.exists(dense_path):
            print("错误：现有 FAISS 索引不存在，无法跳过向量生成")
            return

        existing_index = faiss.read_index(dense_path)
        if existing_index.ntotal != len(chunk_rows):
            print(f"错误：现有 FAISS 向量数({existing_index.ntotal})与数据库分片数({len(chunk_rows)})不一致，无法跳过向量生成")
            return
        dense_vectors = existing_index.reconstruct_n(0, existing_index.ntotal)

        sparse_path = os.path.join(faiss_index_path, "sparse_vectors.pkl")
        if os.path.exists(sparse_path):
            with open(sparse_path, "rb") as f:
                sparse_vectors = pickle.load(f)
            if len(sparse_vectors) != len(chunk_rows):
                print("警告：稀疏向量数量与分片数不一致，将重建为空稀疏索引")
                sparse_vectors = [{} for _ in chunk_rows]
        else:
            sparse_vectors = [{} for _ in chunk_rows]
    else:
        print("\n[2/4] 生成增强嵌入向量...")
        from scripts.add_documents import DocumentProcessor

        processor = DocumentProcessor(append_mode=False)

        print(f"  生成 {len(chunk_texts)} 个向量...")
        dense_vectors, sparse_vectors = processor.generate_embeddings(chunk_texts)

    print("\n[3/4] 构建 FAISS 索引...")
    os.makedirs(faiss_index_path, exist_ok=True)

    norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    dense_vectors = dense_vectors / norms

    dimension = dense_vectors.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(dense_vectors.astype(np.float32))

    sparse_index = {}
    if sparse_vectors:
        for doc_id, sparse_vec in enumerate(sparse_vectors):
            if isinstance(sparse_vec, dict):
                for token_id, weight in sparse_vec.items():
                    token_id = int(token_id)
                    if token_id not in sparse_index:
                        sparse_index[token_id] = {}
                    sparse_index[token_id][doc_id] = float(weight)

    print("\n[4/4] 保存索引...")

    faiss.write_index(faiss_index, os.path.join(faiss_index_path, "index.faiss"))

    chunk_contents = chunk_texts

    with open(os.path.join(faiss_index_path, "chunk_contents.pkl"), "wb") as f:
        pickle.dump(chunk_contents, f)

    with open(os.path.join(faiss_index_path, "chunks_meta.pkl"), "wb") as f:
        pickle.dump(chunks_meta, f)

    with open(os.path.join(faiss_index_path, "sparse_vectors.pkl"), "wb") as f:
        pickle.dump(sparse_vectors, f)

    with open(os.path.join(faiss_index_path, "sparse_index.pkl"), "wb") as f:
        pickle.dump(sparse_index, f)

    with open(os.path.join(faiss_index_path, "chunk_images.pkl"), "wb") as f:
        pickle.dump(chunk_images, f)

    with open(os.path.join(faiss_index_path, "index_version.json"), "w", encoding="utf-8") as f:
        json.dump({
            "version": 2,
            "features": ["title_enhanced_embedding", "sub_heading_chunking"],
            "total_chunks": len(chunk_rows)
        }, f, ensure_ascii=False, indent=2)

    cursor = conn.cursor()
    try:
        for new_vector_id, row in enumerate(chunk_rows):
            cursor.execute("UPDATE chunks SET vector_id = ? WHERE id = ?", (new_vector_id, row["id"]))
        conn.commit()
        print("  数据库 vector_id 已同步")
    finally:
        cursor.close()

    from scripts.bm25_indexer import BM25Indexer
    bm25_indexer = BM25Indexer(faiss_index_path)
    all_ids = list(range(len(chunk_contents)))
    bm25_indexer.build_index(chunk_contents, all_ids)
    bm25_indexer.save()
    print(f"  BM25索引构建完成: {len(chunk_contents)} 个文档")

    print("\n" + "=" * 70)
    print("索引重建完成!")
    print(f"  FAISS 索引: {faiss_index.ntotal} 个向量")
    print(f"  索引版本: v2 (title_enhanced_embedding + sub_heading_chunking)")
    print("=" * 70)

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重建 FAISS 索引")
    parser.add_argument("--clean", action="store_true", help="清除旧索引（重新生成向量）")
    parser.add_argument("--skip-embedding", action="store_true", help="跳过向量生成（仅重建 FAISS 索引和关键词索引）")

    args = parser.parse_args()

    rebuild_index(clean=args.clean, skip_embedding=args.skip_embedding)
