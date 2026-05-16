"""
清空数据库脚本
功能：清空 FAISS 索引和 SQLite 数据库
用法：python scripts/reset_database.py
"""

import os
import sys
import json
import shutil
import sqlite3

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)


def _close_all_db_connections(db_path):
    if not os.path.exists(db_path):
        return
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
    except Exception:
        pass


def reset_all():
    results = []
    from scripts.config_manager import get_config
    config = get_config()

    faiss_dir = config.faiss_index_path
    if os.path.isdir(faiss_dir):
        shutil.rmtree(faiss_dir)
        os.makedirs(faiss_dir, exist_ok=True)
        results.append(f"[OK] FAISS 索引已清空: {faiss_dir}")
    else:
        os.makedirs(faiss_dir, exist_ok=True)
        results.append(f"[WARN] FAISS 目录不存在，已创建: {faiss_dir}")

    db_paths = [
        config.db_path,
        os.path.join(WORKSPACE_ROOT, "rag_data.db"),
        os.path.join(WORKSPACE_ROOT, "corpus.db"),
        os.path.join(WORKSPACE_ROOT, "faiss_index", "corpus.db"),
    ]
    seen = set()
    for db_path in db_paths:
        db_path = os.path.normpath(db_path)
        if db_path in seen:
            continue
        seen.add(db_path)
        db_name = os.path.relpath(db_path, WORKSPACE_ROOT) if db_path.startswith(WORKSPACE_ROOT) else db_path
        if os.path.exists(db_path):
            _close_all_db_connections(db_path)
            try:
                os.remove(db_path)
                results.append(f"[OK] SQLite 文件已删除: {db_name}")
            except PermissionError:
                results.append(f"[WARN] SQLite 文件被占用，尝试清空内容: {db_name}")
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    for (table_name,) in tables:
                        if not table_name.startswith("sqlite_"):
                            cursor.execute(f"DELETE FROM {table_name}")
                    conn.commit()
                    conn.close()
                    results.append(f"[OK] SQLite 文件内容已清空: {db_name}")
                except Exception as e2:
                    results.append(f"[ERROR] 清空失败: {e2}")

    try:
        from scripts.db_manager import DatabaseManager
        db = DatabaseManager(config.db_path)
        db.init_database()
        db.close()
        results.append("[OK] SQLite 数据库已重新初始化")
    except Exception as e:
        results.append(f"[WARN] SQLite 重新初始化失败: {e}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("清空数据库")
    print("=" * 60)
    results = reset_all()
    for r in results:
        print(r)
    print("\n[DONE] 删库完成！")
