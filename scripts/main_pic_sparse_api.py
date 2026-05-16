"""
API 模式（纯检索）
功能：三路混合检索（BM25 + Dense + Sparse），Rerank 精排
"""

import os
import sys
from typing import List, Dict

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from scripts.doc_retriever import DocRetriever
from scripts.config_manager import get_config

RERANK_FINAL_TOPK = 5


def print_separator(title: str) -> None:
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_step(step_num: int, title: str) -> None:
    print(f"\n【步骤 {step_num}】{title}")
    print("-" * 80)


def main():
    print("=" * 80)
    print("API 模式（纯检索）")
    print("=" * 80)
    print("\n命令说明：")
    print("  直接输入查询：三路混合检索（BM25 + Dense + Sparse）")
    print("  r:查询：使用 Rerank 模式检索文档（更精确）")
    print("  q/quit：退出")
    print("=" * 80)

    print("\n正在初始化检索器...")
    doc_retriever = DocRetriever()
    config = get_config()
    print("检索器初始化完成！\n")

    while True:
        try:
            user_input = input("\n请输入检索问题（输入 'q' 退出）：").strip()

            if not user_input:
                continue

            if user_input.lower() in ["q", "quit", "exit"]:
                print("再见!")
                break

            use_rerank = False

            if user_input.startswith("r:"):
                use_rerank = True
                query = user_input[2:].strip()
            else:
                query = user_input

            if not query:
                print("请输入有效问题")
                continue

            print_separator("开始处理")
            print(f"用户输入：{query}")

            print_step(1, "文档检索")
            if use_rerank:
                print("执行三路混合检索 + Rerank 重排序...")
            else:
                print("执行三路混合检索（BM25 + Dense + Sparse）...")

            docs = doc_retriever.retrieve(
                query=query,
                top_k=RERANK_FINAL_TOPK,
                use_rerank=use_rerank
            )

            print(f"召回文档：{len(docs)}个")
            for i, doc in enumerate(docs, 1):
                meta = doc.get("meta", {})
                doc_name = meta.get("doc_name", "unknown")
                title = meta.get("title", "")
                rerank_info = ""
                if use_rerank and "rerank_score" in doc:
                    rerank_info = f" (Rerank分数: {doc['rerank_score']:.4f})"
                print(f"  [{i}] {doc_name} - {title}{rerank_info}")

            print_step(2, "文档内容")
            for i, doc in enumerate(docs, 1):
                meta = doc.get("meta", {})
                doc_name = meta.get("doc_name", "unknown")
                title = meta.get("title", "")
                content = doc.get("content", "")
                print(f"\n{'='*60}")
                print(f"[文档 {i}] {doc_name} - {title}")
                print(f"{'-'*60}")
                print(content[:1000])
                if len(content) > 1000:
                    print(f"... (截断，共 {len(content)} 字符)")

            print_separator("处理完成")

        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as error:
            print(f"检索出错：{error}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
