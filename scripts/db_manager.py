"""
数据库管理模块
功能：管理 SQLite 数据库连接
"""

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.db_manager_sqlite import DatabaseManager
from scripts.db_manager_sqlite import init_db
from scripts.db_manager_sqlite import Corpus, Chunk

DatabaseManagerV3 = DatabaseManager
init_db_v3 = init_db
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
