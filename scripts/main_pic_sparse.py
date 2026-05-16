"""
本地检索测试 V2（使用两阶段检索）
功能：支持关键词过滤的混合检索测试工具，支持图片自动打开

使用说明:
1. 直接输入查询: 使用两阶段检索（关键词过滤 + 向量检索）
2. 添加新实体型号: 修改 entity_patterns 配置或调用 KeywordIndexer
"""

import os
import sys
from typing import List, Dict, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from scripts.retrieval_pipeline import RetrievalPipeline

MAX_OPEN_IMAGES_PER_RESULT = 3
MAX_OPEN_IMAGES_TOTAL = 8


def get_image_abs_path(image) -> str:
    if isinstance(image, str):
        return image
    if isinstance(image, dict):
        return image.get("abs_path", "")
    return ""


def collect_openable_images(results: List[Dict]) -> Tuple[List[str], List[str], List[str]]:
    openable_paths = []
    skipped_paths = []
    limited_paths = []
    seen_paths = set()

    for result in results:
        current_result_count = 0
        meta = result.get("meta", {})
        images = meta.get("images", [])

        for image in images:
            abs_path = get_image_abs_path(image)
            if not abs_path or abs_path in seen_paths:
                continue

            seen_paths.add(abs_path)
            if not os.path.exists(abs_path):
                skipped_paths.append(abs_path)
                continue

            if current_result_count >= MAX_OPEN_IMAGES_PER_RESULT or len(openable_paths) >= MAX_OPEN_IMAGES_TOTAL:
                limited_paths.append(abs_path)
                continue

            openable_paths.append(abs_path)
            current_result_count += 1

    return openable_paths, skipped_paths, limited_paths


def open_image_paths(image_paths: List[str]):
    if not image_paths:
        print("\n未命中可打开的图片")
        return

    if not hasattr(os, "startfile"):
        print("\n当前环境不支持自动打开图片")
        return

    print("\n准备打开以下图片:")
    for image_path in image_paths:
        print(f"  {image_path}")

    for image_path in image_paths:
        try:
            os.startfile(image_path)
        except OSError as error:
            print(f"打开失败: {image_path} ({error})")


def format_result(result: dict, index: int, use_rerank: bool = False) -> str:
    meta = result.get("meta", {})
    doc_name = meta.get("doc_name", "unknown")
    title = meta.get("title", "")
    content = result.get("content", "")
    images = meta.get("images", [])
    image_count = meta.get("image_count", len(images))

    output = []
    output.append(f"\n{'='*60}")

    if use_rerank and "rerank_score" in result:
        output.append(f"【第{index}名】Rerank分数：{result['rerank_score']:.4f}")
    else:
        output.append(f"【第{index}名】doc_id: {result.get('doc_id', 'N/A')}")

    output.append(f"文档名：{doc_name}")
    output.append(f"标题：{title}")

    if "dense_score" in result:
        output.append(f"稠密相似度：{result['dense_score']}")
    if "sparse_score" in result:
        output.append(f"稀疏相似度：{result['sparse_score']}")

    if "debug" in result:
        debug = result["debug"]
        if "stage1_keyword_filter" in debug:
            stage1 = debug["stage1_keyword_filter"]
            output.append(f"关键词：{stage1.get('keywords', [])}")
            output.append(f"候选文档数：{stage1.get('candidate_count', 0)}")

    output.append(f"图片数量：{image_count}")

    if images:
        output.append("图片路径：")
        for image in images[:5]:
            abs_path = get_image_abs_path(image)
            if abs_path:
                output.append(f"  {abs_path}")
        if len(images) > 5:
            output.append(f"  ... 还有 {len(images) - 5} 张图片")
    else:
        output.append("图片路径：无")

    output.append(f"\n内容：{content[:1000]}")
    output.append("-" * 60)

    return "\n".join(output)


def display_results(results: List[Dict], use_rerank: bool = False):
    print(f"\n检索结果（共 {len(results)} 条）：")
    for i, result in enumerate(results, 1):
        print(format_result(result, i, use_rerank))


def main():
    print("=" * 60)
    print("本地检索测试（两阶段检索 + 图片自动打开）")
    print("=" * 60)
    print("\n命令说明:")
    print("  直接输入查询: 使用两阶段检索（关键词过滤 + 向量检索）")
    print("  r:查询: 使用两阶段检索 + Rerank")
    print("  n:查询: 不使用关键词过滤的检索（对比测试）")
    print("  local:查询: 本地模式（跳过行业召回，不调用LLM）")
    print("  debug:查询: 显示调试信息")
    print("  type:类型:查询: 指定文档类型检索（如 type:实体文档:EXAMPLE-A123 烧录）")
    print("  quit/exit: 退出")
    print("\n支持的文档类型: 实体文档、需求文档、工具文档")
    print("=" * 60)

    print("\n正在初始化检索器...")
    retriever = RetrievalPipeline()
    print("检索器初始化完成！\n")

    while True:
        try:
            user_input = input("\n▶️  请输入查询 (或 'quit' 退出): ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("再见!")
                break

            skip_industry = False
            debug_mode = False
            use_rerank = True

            if user_input.startswith("n:"):
                skip_industry = True
                query = user_input[2:].strip()
            elif user_input.startswith("r:"):
                use_rerank = True
                query = user_input[2:].strip()
            elif user_input.startswith("local:"):
                skip_industry = True
                query = user_input[6:].strip()
            elif user_input.startswith("debug:"):
                debug_mode = True
                query = user_input[6:].strip()
            else:
                query = user_input

            if not query:
                print("查询不能为空")
                continue

            if not skip_industry:
                print(f"\n[两阶段检索] 查询：{query}")
            else:
                print(f"\n[本地检索] 查询：{query}")

            result = retriever.retrieve(
                query=query,
                top_k=5,
                use_rerank=use_rerank
            )

            print(result.formatted_output)

            openable_paths, skipped_paths, limited_paths = collect_openable_images(result.doc_docs)

            if skipped_paths:
                print("\n以下图片不存在或不可用，已跳过打开：")
                for skipped_path in skipped_paths:
                    print(f"  {skipped_path}")

            if limited_paths:
                print(f"\n以下图片因自动开图限流被跳过：")
                print(f"每条结果最多 {MAX_OPEN_IMAGES_PER_RESULT} 张，总计最多 {MAX_OPEN_IMAGES_TOTAL} 张")
                for limited_path in limited_paths:
                    print(f"  {limited_path}")

            open_image_paths(openable_paths)

        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
