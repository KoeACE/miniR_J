"""
文档和切片管理工具
功能：
1. 列出所有已入库文档
2. 查看文档详情
3. 启用/禁用文档
4. 删除文档
5. 统计信息
"""

import os
import sys
import argparse
from typing import List, Optional
from tabulate import tabulate

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from scripts.db_manager import DatabaseManager
from scripts.config_manager import get_config


class DocumentManager:
    """文档管理器"""

    def __init__(self, db_path: str = None, db=None):
        if db is not None:
            self.db = db
        else:
            self.db = DatabaseManager(db_path)
            self.db.init_database()

    def list_documents(self, category: str = None, limit: int = None) -> List[dict]:
        conn = self.db._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT id, name, chunk_count, chunk_strategy, is_active, created_at
            FROM corpus
            ORDER BY created_at DESC
        """
        cursor.execute(query)

        rows = cursor.fetchall()

        documents = []
        for row in rows:
            doc = {
                'id': row['id'],
                'name': row['name'],
                'chunk_count': row['chunk_count'],
                'chunk_strategy': row['chunk_strategy'],
                'is_active': bool(row['is_active']) if row['is_active'] is not None else True,
                'created_at': row['created_at'],
            }
            documents.append(doc)

        if limit:
            documents = documents[:limit]

        return documents

    def get_document_detail(self, corpus_id: str) -> Optional[dict]:
        corpus = self.db.get_corpus_by_id(corpus_id)
        if not corpus:
            return None

        chunks = self.db.get_chunks_by_corpus_id(corpus_id)

        return {
            'id': corpus.id,
            'name': corpus.name,
            'description': corpus.description,
            'type': corpus.type,
            'source': corpus.source,
            'file_path': corpus.file_path,
            'chunk_count': corpus.chunk_count,
            'chunk_strategy': corpus.chunk_strategy,
            'status': corpus.status,
            'is_active': bool(corpus.is_active) if corpus.is_active is not None else True,
            'created_at': corpus.created_at,
            'updated_at': corpus.updated_at,
            'chunks': [
                {
                    'id': chunk.id,
                    'chunk_index': chunk.chunk_index,
                    'title': chunk.title,
                    'vector_id': chunk.vector_id,
                    'content_preview': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content
                }
                for chunk in chunks
            ]
        }

    def get_chunk_detail(self, chunk_id: int) -> Optional[dict]:
        chunk = self.db.get_chunk_by_id(chunk_id)
        if not chunk:
            return None

        corpus = self.db.get_corpus_by_id(chunk.corpus_id)

        return {
            'id': chunk.id,
            'corpus_id': chunk.corpus_id,
            'corpus_name': corpus.name if corpus else '未知',
            'chunk_index': chunk.chunk_index,
            'title': chunk.title,
            'title_level': chunk.title_level,
            'content': chunk.content,
            'content_length': len(chunk.content),
            'vector_id': chunk.vector_id,
            'images': chunk.images,
            'created_at': chunk.created_at
        }

    def get_document_by_index(self, index: int) -> Optional[dict]:
        conn = self.db._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, chunk_count, chunk_strategy, is_active, created_at
            FROM corpus
            ORDER BY created_at DESC
            LIMIT 1 OFFSET ?
        """, (index - 1,))

        row = cursor.fetchone()
        if not row:
            return None

        return {
            'id': row['id'],
            'name': row['name'],
            'chunk_count': row['chunk_count'],
            'chunk_strategy': row['chunk_strategy'],
            'is_active': bool(row['is_active']) if row['is_active'] is not None else True,
            'created_at': row['created_at'],
        }

    def toggle_document_status(self, corpus_id: str) -> dict:
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name, is_active FROM corpus WHERE id = ?", (corpus_id,))
        row = cursor.fetchone()
        if not row:
            return {"success": False, "message": f"Document not found: {corpus_id}"}

        new_status = 0 if row['is_active'] else 1
        status_text = "disabled" if new_status == 0 else "enabled"

        cursor.execute("UPDATE corpus SET is_active = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (new_status, corpus_id))
        conn.commit()

        return {
            "success": True,
            "corpus_id": corpus_id,
            "name": row['name'],
            "is_active": bool(new_status),
            "message": f"Document {status_text}: {row['name']}"
        }

    def delete_document(self, corpus_id: str, confirm: bool = True,
                         faiss_index_path: str = None, update_index: bool = True) -> bool:
        import faiss
        import numpy as np

        corpus = self.db.get_corpus_by_id(corpus_id)
        if not corpus:
            print(f"错误：找不到文档 ID {corpus_id}")
            return False

        chunks = self.db.get_chunks_by_corpus_id(corpus_id)
        vector_ids_to_delete = [chunk.vector_id for chunk in chunks if chunk.vector_id is not None]

        created_at_str = ""
        if hasattr(corpus, 'created_at') and corpus.created_at:
            if isinstance(corpus.created_at, str):
                created_at_str = corpus.created_at[:10]
            else:
                created_at_str = corpus.created_at.strftime('%Y-%m-%d')

        if confirm:
            print(f"\n即将删除以下文档：")
            print(f"  名称: {corpus.name}")
            print(f"  ID: {corpus_id}")
            print(f"  入库日期: {created_at_str}")
            print(f"  分片数: {len(chunks)}")
            print(f"  向量数: {len(vector_ids_to_delete)}")
            print(f"\n警告：此操作将同时删除：")
            print(f"  - 数据库: 文档记录、{len(chunks)} 个分片")
            if update_index and faiss_index_path:
                print(f"  - FAISS: {len(vector_ids_to_delete)} 个向量")
                print(f"  - Pickle: 更新 chunk_contents.pkl、chunks_meta.pkl 等")

            response = input(f"\n确认删除? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("取消删除")
                return False

        try:
            # 先更新 FAISS 索引（这样可以确保索引和其他文件，但暂不删除数据库
            if update_index and faiss_index_path is None:
                faiss_index_path = get_config().faiss_index_path

            if update_index and faiss_index_path and vector_ids_to_delete:
                self._update_faiss_index(faiss_index_path, vector_ids_to_delete)

            # 然后删除数据库记录
            conn = self.db._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM corpus WHERE id = ?", (corpus_id,))
            conn.commit()
            print("✓ SQLite 数据已删除")

            print(f"\n成功删除文档: {corpus.name}")
            print("-" * 50)
            return True

        except Exception as e:
            print(f"删除失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _update_faiss_index(self, faiss_index_path: str, vector_ids_to_delete: List[int]):
        import faiss
        import numpy as np
        import pickle

        try:
            print(f"\n正在更新FAISS索引...")

            index_file = os.path.join(faiss_index_path, "index.faiss")
            contents_file = os.path.join(faiss_index_path, "chunk_contents.pkl")
            meta_file = os.path.join(faiss_index_path, "chunks_meta.pkl")
            sparse_file = os.path.join(faiss_index_path, "sparse_vectors.pkl")
            sparse_index_file = os.path.join(faiss_index_path, "sparse_index.pkl")
            bm25_index_file = os.path.join(faiss_index_path, "bm25_index.pkl")
            bm25_corpus_map_file = os.path.join(faiss_index_path, "bm25_corpus_map.pkl")

            if not os.path.exists(index_file):
                print(f"  警告：FAISS索引文件不存在，跳过索引更新")
                return
            for required_file in (contents_file, meta_file):
                if not os.path.exists(required_file):
                    raise FileNotFoundError(f"索引元数据文件不存在: {required_file}")

            index = faiss.read_index(index_file)
            total_vectors = index.ntotal

            with open(contents_file, 'rb') as f:
                chunk_contents = pickle.load(f)
            with open(meta_file, 'rb') as f:
                chunks_meta = pickle.load(f)

            if len(chunk_contents) != total_vectors or len(chunks_meta) != total_vectors:
                raise ValueError(
                    f"索引不一致: FAISS={total_vectors}, contents={len(chunk_contents)}, meta={len(chunks_meta)}"
                )

            sparse_vectors = None
            if os.path.exists(sparse_file):
                with open(sparse_file, 'rb') as f:
                    sparse_vectors = pickle.load(f)

            sparse_index = None
            if os.path.exists(sparse_index_file):
                with open(sparse_index_file, 'rb') as f:
                    sparse_index = pickle.load(f)

            # 首先，获取需要保留的向量
            keep_mask = np.ones(total_vectors, dtype=bool)
            for vid in vector_ids_to_delete:
                if 0 <= vid < total_vectors:
                    keep_mask[vid] = False

            # 计算新的 vector_id 映射关系
            old_to_new_id = {}
            new_id = 0
            for old_id in range(total_vectors):
                if keep_mask[old_id]:
                    old_to_new_id[old_id] = new_id
                    new_id += 1

            # 创建新的索引
            all_vectors = index.reconstruct_n(0, total_vectors)
            new_vectors = all_vectors[keep_mask]
            dimension = all_vectors.shape[1]
            new_index = faiss.IndexFlatIP(dimension)
            if len(new_vectors) > 0:
                new_index.add(new_vectors)

            # 更新数据库中剩余文档的 vector_id（如果它们存在
            if old_to_new_id:
                conn = self.db._get_connection()
                cursor = conn.cursor()
                try:
                    cursor.execute("BEGIN TRANSACTION")
                    # 更新所有保留的 chunks 的 vector_id
                    for old_id, new_id in old_to_new_id.items():
                        cursor.execute(
                            "UPDATE chunks SET vector_id = ? WHERE vector_id = ?",
                            (new_id, old_id)
                        )
                    conn.commit()
                    print(f"  ✓ 数据库 vector_id 已同步更新")
                except Exception as e:
                    conn.rollback()
                    raise e

            # 保存 FAISS 索引
            faiss.write_index(new_index, index_file)
            print(f"  ✓ FAISS索引已更新: {total_vectors} → {new_index.ntotal} 个向量")

            # 更新 chunk 内容和元数据
            new_contents = [chunk_contents[i] for i in range(len(chunk_contents)) if keep_mask[i]]
            new_meta = [chunks_meta[i] for i in range(len(chunks_meta)) if keep_mask[i]]

            with open(contents_file, 'wb') as f:
                pickle.dump(new_contents, f)
            with open(meta_file, 'wb') as f:
                pickle.dump(new_meta, f)
            print(f"  ✓ Pickle文件已更新")

            # 更新稀疏向量
            if sparse_vectors is not None:
                if len(sparse_vectors) != total_vectors:
                    raise ValueError(f"稀疏向量数量不一致: sparse={len(sparse_vectors)}, FAISS={total_vectors}")
                new_sparse = [sparse_vectors[i] for i in range(len(sparse_vectors)) if keep_mask[i]]
                with open(sparse_file, 'wb') as f:
                    pickle.dump(new_sparse, f)
                print(f"  ✓ 稀疏向量已更新")

            # 更新稀疏索引
            if sparse_index is not None:
                new_sparse_index = {}
                for token_id, doc_dict in sparse_index.items():
                    new_doc_dict = {}
                    for doc_id, weight in doc_dict.items():
                        if doc_id < len(keep_mask) and keep_mask[doc_id]:
                            new_doc_id = old_to_new_id.get(doc_id, doc_id)
                            new_doc_dict[new_doc_id] = weight
                    if new_doc_dict:
                        new_sparse_index[token_id] = new_doc_dict
                with open(sparse_index_file, 'wb') as f:
                    pickle.dump(new_sparse_index, f)
                print(f"  ✓ 稀疏索引已更新")

            # 重建 BM25 索引
            if os.path.exists(bm25_index_file) and os.path.exists(bm25_corpus_map_file):
                from scripts.bm25_indexer import BM25Indexer
                bm25_indexer = BM25Indexer(faiss_index_path)
                all_ids = list(range(len(new_contents)))
                bm25_indexer.build_index(new_contents, all_ids)
                bm25_indexer.save()
                print(f"  ✓ BM25索引已重建")

            print(f"✓ FAISS索引和Pickle文件更新完成")

        except Exception as e:
            print(f"  警告：更新FAISS索引时出错: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_statistics(self) -> dict:
        conn = self.db._get_connection()
        cursor = conn.cursor()

        stats = {}

        tables = ['corpus', 'chunks']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            row = cursor.fetchone()
            stats[table] = row['count'] if row else 0

        cursor.execute("""
            SELECT name, created_at FROM corpus
            ORDER BY created_at DESC LIMIT 5
        """)
        stats['recent'] = [(row['name'], row['created_at']) for row in cursor.fetchall()]

        return stats

    def search_documents(self, keyword: str) -> List[dict]:
        conn = self.db._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT id, name, chunk_count, chunk_strategy, is_active, created_at
            FROM corpus
            WHERE name LIKE ? OR description LIKE ?
            ORDER BY created_at DESC
        """
        pattern = f"%{keyword}%"
        cursor.execute(query, (pattern, pattern))

        rows = cursor.fetchall()

        documents = []
        for row in rows:
            doc = {
                'id': row['id'],
                'name': row['name'],
                'chunk_count': row['chunk_count'],
                'chunk_strategy': row['chunk_strategy'],
                'is_active': bool(row['is_active']) if row['is_active'] is not None else True,
                'created_at': row['created_at'],
            }
            documents.append(doc)

        return documents


def print_document_list(documents: List[dict]):
    if not documents:
        print("没有找到文档")
        return

    headers = ['序号', '文档名称', '分片数', '分片策略', '状态', '创建时间']
    rows = []

    for i, doc in enumerate(documents, 1):
        is_active = "启用" if doc['is_active'] else "禁用"
        rows.append([
            i,
            doc['name'][:40] + '...' if len(doc['name']) > 40 else doc['name'],
            doc['chunk_count'],
            doc['chunk_strategy'] or '-',
            is_active,
            str(doc['created_at'])[:10] if doc['created_at'] else '-'
        ])

    print(tabulate(rows, headers=headers, tablefmt='grid'))
    print(f"\n共 {len(documents)} 个文档")


def print_document_detail(detail: dict):
    print("\n" + "=" * 80)
    print(f"文档详情: {detail['name']}")
    print("=" * 80)

    print(f"\n【基本信息】")
    print(f"  ID: {detail['id']}")
    print(f"  名称: {detail['name']}")
    print(f"  类型: {detail['type']}")
    print(f"  状态: {detail['status']}")
    print(f"  来源: {detail['source']}")
    print(f"  文件路径: {detail['file_path']}")
    print(f"  分片策略: {detail['chunk_strategy']}")
    print(f"  启用状态: {'启用' if detail['is_active'] else '禁用'}")

    print(f"\n【分片信息】")
    print(f"  总分片数: {detail['chunk_count']}")

    if detail['chunks']:
        headers = ['切片ID', '序号', '标题', '向量ID', '内容预览']
        rows = []
        for chunk in detail['chunks'][:10]:
            rows.append([
                chunk['id'],
                chunk['chunk_index'],
                chunk['title'][:30] + '...' if len(chunk['title']) > 30 else chunk['title'],
                chunk['vector_id'],
                chunk['content_preview'][:50] + '...' if len(chunk['content_preview']) > 50 else chunk['content_preview']
            ])
        print(tabulate(rows, headers=headers, tablefmt='grid'))

        if len(detail['chunks']) > 10:
            print(f"\n... 还有 {len(detail['chunks']) - 10} 个分片")

    print(f"\n【时间信息】")
    print(f"  创建时间: {detail['created_at']}")
    print(f"  更新时间: {detail['updated_at']}")

    print("=" * 80)


def print_chunk_detail(chunk: dict):
    print("\n" + "=" * 80)
    print(f"切片详情: ID {chunk['id']}")
    print("=" * 80)

    print(f"\n【基本信息】")
    print(f"  切片ID: {chunk['id']}")
    print(f"  所属文档: {chunk['corpus_name']} ({chunk['corpus_id']})")
    print(f"  切片序号: {chunk['chunk_index']}")
    print(f"  标题: {chunk['title']}")
    print(f"  标题级别: {chunk['title_level']}")
    print(f"  向量ID: {chunk['vector_id']}")

    print(f"\n【图片】")
    if chunk['images']:
        for img in chunk['images']:
            print(f"  - {img}")
    else:
        print("  无")

    print(f"\n【内容】({chunk['content_length']} 字符)")
    print("-" * 80)
    print(chunk['content'])
    print("-" * 80)

    print(f"\n创建时间: {chunk['created_at']}")
    print("=" * 80)


def print_statistics(stats: dict):
    print("\n" + "=" * 80)
    print("数据库统计信息")
    print("=" * 80)

    print(f"\n【基础统计】")
    print(f"  文档总数: {stats['corpus']}")
    print(f"  分片总数: {stats['chunks']}")

    print(f"\n【最近添加】")
    for name, created_at in stats['recent']:
        if hasattr(created_at, 'strftime'):
            date_str = created_at.strftime('%Y-%m-%d')
        else:
            date_str = str(created_at)[:10]
        print(f"  {date_str} - {name}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='文档和切片管理工具')
    parser.add_argument('--db', help='数据库路径', default=None)

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    list_parser = subparsers.add_parser('list', help='列出所有文档')
    list_parser.add_argument('-l', '--limit', type=int, help='限制数量')

    detail_parser = subparsers.add_parser('detail', help='查看文档详情')
    detail_parser.add_argument('id', help='文档ID')

    chunk_parser = subparsers.add_parser('chunk', help='查看切片详情')
    chunk_parser.add_argument('chunk_id', type=int, help='切片ID')

    toggle_parser = subparsers.add_parser('toggle', help='启用/禁用文档')
    toggle_parser.add_argument('id', help='文档ID或序号')
    toggle_parser.add_argument('--by-index', action='store_true', help='将ID解释为序号')

    delete_parser = subparsers.add_parser('delete', help='删除文档（包括SQLite、FAISS索引和Pickle文件）')
    delete_parser.add_argument('id', help='文档ID或序号（如：abc-123 或 1,2,3）')
    delete_parser.add_argument('-y', '--yes', action='store_true', help='跳过确认')
    delete_parser.add_argument('--faiss', help='FAISS索引目录路径', default='faiss_index')
    delete_parser.add_argument('--no-index', action='store_true', help='不更新FAISS索引和Pickle文件')
    delete_parser.add_argument('--by-index', action='store_true', help='将ID解释为序号（从1开始）')

    stats_parser = subparsers.add_parser('stats', help='统计信息')

    search_parser = subparsers.add_parser('search', help='搜索文档')
    search_parser.add_argument('keyword', help='搜索关键词')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = DocumentManager(args.db)

    if args.command == 'list':
        documents = manager.list_documents(limit=args.limit)
        print_document_list(documents)

    elif args.command == 'detail':
        detail = manager.get_document_detail(args.id)
        if detail:
            print_document_detail(detail)
        else:
            print(f"错误：找不到文档 ID {args.id}")

    elif args.command == 'chunk':
        chunk = manager.get_chunk_detail(args.chunk_id)
        if chunk:
            print_chunk_detail(chunk)
        else:
            print(f"错误：找不到切片 ID {args.chunk_id}")

    elif args.command == 'toggle':
        doc_id = args.id
        if args.by_index or doc_id.isdigit():
            index = int(doc_id)
            doc = manager.get_document_by_index(index)
            if not doc:
                print(f"错误：找不到序号 {index} 的文档")
                return
            doc_id = doc['id']
            print(f"序号 {index} 对应文档: {doc['name']}")

        result = manager.toggle_document_status(doc_id)
        if result['success']:
            print(f"\n✓ {result['message']}")
        else:
            print(f"\n✗ {result['message']}")

    elif args.command == 'delete':
        id_list = [x.strip() for x in args.id.split(',')]

        faiss_path = None
        if not args.no_index:
            if os.path.isabs(args.faiss):
                faiss_path = args.faiss
            else:
                faiss_path = os.path.join(WORKSPACE_ROOT, args.faiss)

        for doc_id in id_list:
            if args.by_index or doc_id.isdigit():
                index = int(doc_id)
                doc = manager.get_document_by_index(index)
                if not doc:
                    print(f"错误：找不到序号 {index} 的文档")
                    continue
                real_id = doc['id']
                print(f"\n序号 {index} 对应文档: {doc['name']}")
            else:
                real_id = doc_id

            manager.delete_document(
                real_id,
                confirm=not args.yes,
                faiss_index_path=faiss_path,
                update_index=not args.no_index
            )

    elif args.command == 'stats':
        stats = manager.get_statistics()
        print_statistics(stats)

    elif args.command == 'search':
        documents = manager.search_documents(args.keyword)
        print(f"\n搜索关键词: {args.keyword}")
        print(f"找到 {len(documents)} 个结果\n")
        print_document_list(documents)


if __name__ == '__main__':
    main()
