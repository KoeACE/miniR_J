"""
检索流程整合模块
功能：整合文档召回，实现检索流程
"""

import os
import sys
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from scripts.doc_retriever import DocRetriever
from scripts.config_manager import get_config


@dataclass
class RetrievalResult:
    query: str
    docs: List[Dict]
    formatted_output: str


def convert_image_path(image_path: str) -> str:
    if not image_path:
        return ""

    if WORKSPACE_ROOT.lower() in image_path.lower():
        return image_path

    old_project_patterns = [
        r"^[a-zA-Z]:\\Users\\[^\\]+\\rag-try-[^\\]+\\",
        r"^[a-zA-Z]:\\[^\\]+\\rag-try-[^\\]+\\",
    ]

    for pattern in old_project_patterns:
        if re.match(pattern, image_path, re.IGNORECASE):
            relative_path = re.sub(pattern, "", image_path, flags=re.IGNORECASE)
            new_path = os.path.join(WORKSPACE_ROOT, relative_path)
            if os.path.exists(new_path):
                return new_path
            basename = os.path.basename(image_path)
            new_path = os.path.join(WORKSPACE_ROOT, basename)
            if os.path.exists(new_path):
                return new_path
            return image_path

    return image_path


def format_doc_detail(doc: Dict, index: int) -> str:
    meta = doc.get("meta", {})
    content = doc.get("content", "")
    images = meta.get("images", [])

    output = []
    output.append(f"\n{'=' * 70}")
    output.append(f"【结果 {index}】")

    output.append(f"文档名：{meta.get('doc_name', 'Unknown')}")
    output.append(f"标题：{meta.get('title', 'N/A')}")

    category = meta.get('category', '')
    knowledge_point = meta.get('knowledge_point', '')
    custom = meta.get('custom', '')
    if category:
        tag_str = f"标签：{category}"
        if knowledge_point:
            tag_str += f" > {knowledge_point}"
        if custom:
            tag_str += f" > {custom}"
        output.append(tag_str)

    if "dense_score" in doc:
        output.append(f"向量相似度：{doc['dense_score']:.4f}")
    if "rerank_score" in doc:
        output.append(f"Rerank分数：{doc['rerank_score']:.4f}")

    output.append(f"图片数量：{len(images)}")
    if images:
        output.append("图片路径：")
        for img in images[:3]:
            if isinstance(img, str):
                img_path = img
            elif isinstance(img, dict):
                img_path = img.get("abs_path", img.get("path", ""))
            else:
                img_path = str(img)

            img_path = convert_image_path(img_path)
            if os.path.exists(img_path):
                output.append(f"  ✓ {img_path}")
            else:
                output.append(f"  ✗ {img_path} (不存在)")

    if content:
        output.append(f"\n内容：")
        output.append(content[:800])
        if len(content) > 800:
            output.append("...")

    output.append("-" * 70)

    return "\n".join(output)


class RetrievalPipeline:

    def __init__(self, doc_retriever: DocRetriever = None):
        self.config = get_config()
        self.doc_retriever = doc_retriever or DocRetriever()

    def _format_output(self, query: str, docs: List[Dict]) -> str:
        output_parts = []

        output_parts.append("=" * 70)
        output_parts.append("检索结果")
        output_parts.append("=" * 70)

        if docs:
            output_parts.append("\n" + "=" * 70)
            output_parts.append(f"【文档】({len(docs)}个)")
            output_parts.append("=" * 70)
            for i, doc in enumerate(docs, 1):
                meta = doc.get("meta", {})
                doc_name = meta.get("doc_name", "Unknown")
                title = meta.get("title", "")
                content = doc.get("content", "")[:200]
                output_parts.append(f"\n{i}. {doc_name}")
                if title:
                    output_parts.append(f"   标题: {title}")
                output_parts.append(f"   内容: {content}...")

        output_parts.append("\n" + "-" * 70)

        return "\n".join(output_parts)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        verbose: bool = True
    ) -> RetrievalResult:
        """
        检索主接口

        流程：
        1. 文档召回：基于查询召回文档

        Args:
            query: 用户查询
            top_k: 返回结果数
            use_rerank: 是否使用 Rerank 重排序
            verbose: 是否打印详细结果

        Returns:
            检索结果
        """
        if verbose:
            print("[阶段1] 执行文档召回...")
            print(f"  使用query: {query[:100]}...")

        docs = self.doc_retriever.retrieve(
            query=query,
            top_k=top_k,
            use_rerank=use_rerank
        )
        if verbose:
            print(f"  召回文档: {len(docs)}个")

        formatted_output = self._format_output(
            query=query,
            docs=docs
        )

        if verbose:
            print(formatted_output)

        if verbose and docs:
            print("\n" + "=" * 70)
            print("详细结果")
            print("=" * 70)
            for i, doc in enumerate(docs, 1):
                print(format_doc_detail(doc, i))

        return RetrievalResult(
            query=query,
            docs=docs,
            formatted_output=formatted_output
        )
