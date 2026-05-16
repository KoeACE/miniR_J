"""
SQLite 数据库管理模块
功能：管理 SQLite 数据库连接，支持语料和分片的基本 CRUD 操作
"""

import sqlite3
import uuid
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import os


@dataclass
class Corpus:
    name: str
    description: str = ""
    type: str = "文档"
    data_summary: str = ""
    source: str = ""
    file_path: str = ""
    relative_path: str = ""
    chunk_count: int = 0
    status: str = "active"
    chunk_strategy: str = "title"
    is_active: bool = True
    id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class Chunk:
    corpus_id: str
    chunk_index: int
    content: str
    title: str = ""
    title_level: int = 0
    images: List[str] = None
    vector_id: Optional[int] = None
    original_content: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.images is None:
            self.images = []


class DatabaseManager:

    def __init__(self, db_path: str = None):
        if db_path is None:
            try:
                from scripts.config_manager import get_config
                db_path = get_config().db_path
            except Exception:
                workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                db_path = os.path.join(workspace_root, "rag_data.db")
        self.db_path = db_path
        self._connection = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            # 启用外键约束
            cursor = self._connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            self._connection.commit()
        return self._connection

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None

    def init_database(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corpus (
                id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                type VARCHAR(50) NOT NULL,
                data_summary TEXT,
                source VARCHAR(255),
                file_path VARCHAR(500),
                relative_path VARCHAR(500) DEFAULT '',
                chunk_count INTEGER DEFAULT 0,
                status VARCHAR(20) DEFAULT 'active',
                chunk_strategy TEXT DEFAULT 'title',
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                corpus_id VARCHAR(36) NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                title VARCHAR(500),
                title_level INTEGER,
                images TEXT,
                vector_id INTEGER,
                original_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (corpus_id) REFERENCES corpus(id) ON DELETE CASCADE,
                UNIQUE(corpus_id, chunk_index)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_groups (
                id VARCHAR(36) PRIMARY KEY,
                group_name VARCHAR(255) NOT NULL,
                description TEXT DEFAULT '',
                active_version_id VARCHAR(36),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_versions (
                id VARCHAR(36) PRIMARY KEY,
                group_id VARCHAR(36) NOT NULL,
                corpus_id VARCHAR(36) NOT NULL,
                version_label VARCHAR(100) DEFAULT '',
                version_order INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (group_id) REFERENCES document_groups(id),
                FOREIGN KEY (corpus_id) REFERENCES corpus(id) ON DELETE CASCADE,
                UNIQUE(group_id, corpus_id)
            )
        """)

        cursor.execute("PRAGMA table_info(corpus)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'is_active' not in columns:
            cursor.execute("ALTER TABLE corpus ADD COLUMN is_active INTEGER DEFAULT 1")
        if 'chunk_strategy' not in columns:
            cursor.execute("ALTER TABLE corpus ADD COLUMN chunk_strategy TEXT DEFAULT 'title'")

        cursor.execute("PRAGMA table_info(chunks)")
        chunk_columns = [row[1] for row in cursor.fetchall()]
        if 'original_content' not in chunk_columns:
            cursor.execute("ALTER TABLE chunks ADD COLUMN original_content TEXT")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_corpus_id ON chunks(corpus_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_vector_id ON chunks(vector_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_corpus_relative_path ON corpus(relative_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_corpus_is_active ON corpus(is_active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_versions_group_id ON document_versions(group_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_versions_corpus_id ON document_versions(corpus_id)")

        conn.commit()
        print(f"数据库初始化完成: {self.db_path}")

    def add_corpus(self, corpus: Corpus) -> str:
        if corpus.id is None:
            corpus.id = str(uuid.uuid4())

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO corpus (id, name, description, type, data_summary, source, file_path,
                              relative_path, chunk_count, status, chunk_strategy, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (corpus.id, corpus.name, corpus.description, corpus.type, corpus.data_summary,
             corpus.source, corpus.file_path, corpus.relative_path, corpus.chunk_count,
             corpus.status, corpus.chunk_strategy, 1 if corpus.is_active else 0)
        )
        conn.commit()
        return corpus.id

    def get_corpus_by_id(self, corpus_id: str) -> Optional[Corpus]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM corpus WHERE id = ?", (corpus_id,))
        row = cursor.fetchone()
        if row:
            return Corpus(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                type=row["type"],
                data_summary=row["data_summary"],
                source=row["source"],
                file_path=row["file_path"],
                relative_path=row["relative_path"],
                chunk_count=row["chunk_count"],
                status=row["status"],
                chunk_strategy=row["chunk_strategy"] if "chunk_strategy" in row.keys() else "title",
                is_active=bool(row["is_active"]) if row["is_active"] is not None else True,
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )
        return None

    def get_corpus_by_name(self, name: str) -> Optional[Corpus]:
        corpora = self.get_corpora_by_name(name)
        return corpora[0] if corpora else None

    def get_corpora_by_name(self, name: str) -> List[Corpus]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM corpus WHERE name = ? AND status = 'active' ORDER BY created_at DESC", (name,))
        rows = cursor.fetchall()
        return [
            Corpus(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                type=row["type"],
                data_summary=row["data_summary"],
                source=row["source"],
                file_path=row["file_path"],
                relative_path=row["relative_path"],
                chunk_count=row["chunk_count"],
                status=row["status"],
                chunk_strategy=row["chunk_strategy"] if "chunk_strategy" in row.keys() else "title",
                is_active=bool(row["is_active"]) if row["is_active"] is not None else True,
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )
            for row in rows
        ]

    def get_corpus_by_path(self, file_path: str, relative_path: str = None) -> Optional[Corpus]:
        conn = self._get_connection()
        cursor = conn.cursor()
        paths = [file_path]
        if relative_path:
            paths.append(relative_path)
        normalized_abs = os.path.normpath(file_path) if file_path else ""
        if normalized_abs and normalized_abs not in paths:
            paths.append(normalized_abs)

        placeholders = ",".join("?" for _ in paths)
        cursor.execute(
            f"""
            SELECT * FROM corpus
            WHERE status = 'active'
              AND (file_path IN ({placeholders}) OR relative_path IN ({placeholders}))
            ORDER BY created_at DESC
            LIMIT 1
            """,
            tuple(paths + paths)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return Corpus(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            type=row["type"],
            data_summary=row["data_summary"],
            source=row["source"],
            file_path=row["file_path"],
            relative_path=row["relative_path"],
            chunk_count=row["chunk_count"],
            status=row["status"],
            chunk_strategy=row["chunk_strategy"] if "chunk_strategy" in row.keys() else "title",
            is_active=bool(row["is_active"]) if row["is_active"] is not None else True,
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

    def delete_corpus(self, corpus_id: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM corpus WHERE id = ?", (corpus_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            raise e

    def get_all_corpus_ids(self) -> List[str]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM corpus WHERE status = 'active'")
        rows = cursor.fetchall()
        return [row["id"] for row in rows]

    def get_corpus_ids_by_category(self, doc_type: str) -> List[str]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM corpus WHERE type = ? AND status = 'active' AND is_active = 1",
            (doc_type,)
        )
        rows = cursor.fetchall()
        return [row["id"] for row in rows]

    def update_corpus_chunk_count(self, corpus_id: str, chunk_count: int):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE corpus SET chunk_count = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (chunk_count, corpus_id)
        )
        conn.commit()

    def toggle_corpus_active(self, corpus_id: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT is_active FROM corpus WHERE id = ?", (corpus_id,))
        row = cursor.fetchone()
        if not row:
            return False
        new_status = 0 if row['is_active'] else 1
        cursor.execute("UPDATE corpus SET is_active = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (new_status, corpus_id))
        conn.commit()
        return True

    def add_chunk(self, chunk: Chunk) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()

        images_json = json.dumps(chunk.images) if chunk.images else "[]"
        original_content = chunk.original_content if chunk.original_content is not None else chunk.content

        cursor.execute(
            """
            INSERT INTO chunks (corpus_id, chunk_index, content, title, title_level, images, vector_id, original_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk.corpus_id, chunk.chunk_index, chunk.content, chunk.title,
             chunk.title_level, images_json, chunk.vector_id, original_content)
        )
        conn.commit()
        return cursor.lastrowid

    def get_chunk_by_id(self, chunk_id: int) -> Optional[Chunk]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        if row:
            return Chunk(
                id=row["id"],
                corpus_id=row["corpus_id"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                title=row["title"],
                title_level=row["title_level"],
                images=json.loads(row["images"]) if row["images"] else [],
                vector_id=row["vector_id"],
                original_content=row["original_content"] if "original_content" in row.keys() else None,
                created_at=row["created_at"]
            )
        return None

    def get_chunks_by_corpus_id(self, corpus_id: str) -> List[Chunk]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE corpus_id = ? ORDER BY chunk_index", (corpus_id,))
        rows = cursor.fetchall()
        return [
            Chunk(
                id=row["id"],
                corpus_id=row["corpus_id"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                title=row["title"],
                title_level=row["title_level"],
                images=json.loads(row["images"]) if row["images"] else [],
                vector_id=row["vector_id"],
                original_content=row["original_content"] if "original_content" in row.keys() else None,
                created_at=row["created_at"]
            )
            for row in rows
        ]

    def update_chunk_vector_id(self, chunk_id: int, vector_id: int):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE chunks SET vector_id = ? WHERE id = ?",
            (vector_id, chunk_id)
        )
        conn.commit()

    def revert_chunk(self, chunk_id: int) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT original_content FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        if not row or row["original_content"] is None:
            return False
        cursor.execute(
            "UPDATE chunks SET content = ? WHERE id = ?",
            (row["original_content"], chunk_id)
        )
        conn.commit()
        return cursor.rowcount > 0

    def get_chunk_by_vector_id(self, vector_id: int) -> Optional[Chunk]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE vector_id = ?", (vector_id,))
        row = cursor.fetchone()
        if row:
            return Chunk(
                id=row["id"],
                corpus_id=row["corpus_id"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                title=row["title"],
                title_level=row["title_level"],
                images=json.loads(row["images"]) if row["images"] else [],
                vector_id=row["vector_id"],
                original_content=row["original_content"] if "original_content" in row.keys() else None,
                created_at=row["created_at"]
            )
        return None

    def get_chunks_by_corpus(self, corpus_id: str) -> List[Chunk]:
        return self.get_chunks_by_corpus_id(corpus_id)

    def get_stats(self) -> Dict:
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        tables = ["corpus", "chunks"]

        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
            except Exception:
                stats[table] = 0

        return stats


def init_db(db_path: str = None) -> DatabaseManager:
    db = DatabaseManager(db_path)
    db.init_database()
    return db


init_db_v3 = init_db
DatabaseManagerV3 = DatabaseManager
migrate_db_v2_to_v3 = lambda db_path=None: None


if __name__ == "__main__":
    print("=" * 60)
    print("SQLite 数据库管理器测试")
    print("=" * 60)

    db = init_db()

    stats = db.get_stats()
    print(f"\n数据库统计:")
    for table, count in stats.items():
        print(f"  {table}: {count}")

    db.close()
    print("\n测试完成!")
