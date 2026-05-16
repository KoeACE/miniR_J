"""
文档入库脚本
功能：支持数据库存储和关键词索引的入库脚本，支持交互式标签选择
"""

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
import sys
import re
import json
import pickle
import time
import signal
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
import numpy as np
import faiss
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

from scripts.db_manager import DatabaseManager, Corpus, Chunk
from scripts.bm25_indexer import BM25Indexer
from scripts.config_manager import get_config


class DocumentProcessor:
    """文档处理器"""

    TITLE_PATTERN = re.compile(r'^(#{1,})\s+(?:\*{0,2}\s*(\d+(?:\.\d+)*)\s*\*{0,2}\s+\.?\s*)?(.+)$')

    SUB_TITLE_PATTERN = re.compile(r'^(#{2,6})\s+(?!\*{0,2}\s*\d+(?:\.\d+)*)\s*(.+)$')

    WORD_NUMBERED_HEADING_PATTERN = re.compile(r'^((?:[一二三四五六七八九十]+、)|(?:第[一二三四五六七八九十\d]+[章节])|(?:\d{1,2}(?:\.\d{1,2})*[、.．]))\s*(.{1,40})$')

    def __init__(
        self,
        md_directory: str = None,
        faiss_index_path: str = None,
        model_path: str = None,
        db_path: str = None,
        append_mode: bool = True,
        resume: bool = False,
        reset: bool = False,
        chunk_strategy: str = "length",
        chunk_size: int = 500,
        overlap_ratio: float = 0.1
    ):
        config = get_config()
        if md_directory is None:
            md_directory = config.md_directory
        self.md_directory = md_directory

        if faiss_index_path is None:
            faiss_index_path = config.faiss_index_path
        self.faiss_index_path = faiss_index_path

        if model_path is None:
            model_path = config.model_path
        self.model_path = model_path

        self.progress_file = os.path.join(self.faiss_index_path, "progress.json")
        self.processed_files: List[str] = []
        self.resume_mode = resume
        self.reset_mode = reset

        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio

        if self.reset_mode:
            self._reset_progress()

        self._setup_signal_handlers()

        print("[初始化] 正在初始化 SQLite 数据库...")
        self.db = DatabaseManager(db_path)
        self.db.init_database()
        self.bm25_indexer = BM25Indexer(self.faiss_index_path)

        self.model = None

        self.chunks_meta: List[Dict] = []
        self.chunk_contents: List[str] = []
        self.chunk_images: List[List[str]] = []
        self.dense_vectors: np.ndarray = None
        self.sparse_vectors: List[Dict] = []

        self.append_mode = append_mode
        self.existing_vector_count = 0

        self.processed_documents: List[Dict] = []

        self.log_file = os.path.join(self.faiss_index_path, f"import_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.import_logs: List[Dict] = []

    def _setup_signal_handlers(self):
        import threading
        if threading.current_thread() is not threading.main_thread():
            return

        def signal_handler(signum, frame):
            print("\n\n[中断信号] 检测到 Ctrl+C，正在保存进度...")
            self.save_progress()
            print(f"[中断信号] 进度已保存到: {self.progress_file}")
            print("[中断信号] 程序已安全退出，下次可使用 --resume 参数恢复")
            sys.exit(0)

        try:
            signal.signal(signal.SIGINT, signal_handler)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, signal_handler)
        except (ValueError, OSError):
            pass

    def save_progress(self):
        try:
            progress_data = {
                "processed_files": self.processed_files,
                "last_update": datetime.now().isoformat()
            }
            os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[警告] 保存进度失败: {e}")

    def save_import_logs(self):
        if not self.import_logs:
            return
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump({
                    "import_session": {
                        "start_time": self.import_logs[0].get("timestamp") if self.import_logs else None,
                        "end_time": datetime.now().isoformat(),
                        "total_documents": len(self.import_logs),
                        "total_chunks": sum(log.get("chunk_count", 0) for log in self.import_logs),
                        "log_file": self.log_file
                    },
                    "documents": self.import_logs
                }, f, ensure_ascii=False, indent=2)
            print(f"\n[日志] 详细入库日志已保存到: {self.log_file}")
        except Exception as e:
            print(f"[警告] 保存入库日志失败: {e}")

    def load_progress(self) -> List[str]:
        if not os.path.exists(self.progress_file):
            return []

        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
                processed_files = progress_data.get("processed_files", [])
                last_update = progress_data.get("last_update", "未知")
                print(f"  找到进度文件，上次更新: {last_update}")
                print(f"  已处理文档数: {len(processed_files)}")
                return processed_files
        except json.JSONDecodeError as e:
            print(f"[警告] 进度文件格式错误: {e}")
            return []
        except Exception as e:
            print(f"[警告] 加载进度失败: {e}")
            return []

    def _reset_progress(self):
        if os.path.exists(self.progress_file):
            try:
                os.remove(self.progress_file)
                print(f"[重置] 已删除进度文件: {self.progress_file}")
            except Exception as e:
                print(f"[警告] 删除进度文件失败: {e}")
        else:
            print(f"[重置] 进度文件不存在: {self.progress_file}")

    def get_last_processed_document(self) -> Optional[Dict]:
        if not self.processed_documents:
            return None
        return self.processed_documents[-1]

    def rollback_last_document(self) -> bool:
        last_doc = self.get_last_processed_document()
        if not last_doc:
            print("  [回退] 没有可回退的文档")
            return False

        corpus_id = last_doc.get("corpus_id")
        doc_name = last_doc.get("name", "未知")
        chunk_count = last_doc.get("chunk_count", 0)

        print(f"\n  [回退] 正在删除上一条文档: {doc_name}")
        print(f"  [回退] 语料ID: {corpus_id}, 分片数: {chunk_count}")

        try:
            print(f"  [回退] 删除数据库记录...")
            self.db.delete_corpus(corpus_id)
            print(f"    [回退] 语料记录已删除")

            self.processed_documents.pop()

            print(f"  [回退] 文档 '{doc_name}' 已成功删除")
            print(f"  [提示] FAISS 向量未删除（不影响检索，如需清理请重建索引）")

            return True

        except Exception as e:
            print(f"  [错误] 回退文档失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_existing_index(self) -> bool:
        index_path = os.path.join(self.faiss_index_path, "index.faiss")

        if not os.path.exists(index_path):
            print("未找到已有索引，将创建新索引")
            return False

        try:
            print("正在加载已有索引...")

            self.faiss_index = faiss.read_index(index_path)
            self.existing_vector_count = self.faiss_index.ntotal
            print(f"  已有向量数: {self.existing_vector_count}")

            contents_path = os.path.join(self.faiss_index_path, "chunk_contents.pkl")
            if os.path.exists(contents_path):
                with open(contents_path, "rb") as f:
                    self.chunk_contents = pickle.load(f)
                print(f"  已有分片内容: {len(self.chunk_contents)} 条")

            meta_path = os.path.join(self.faiss_index_path, "chunks_meta.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self.chunks_meta = pickle.load(f)
                print(f"  已有分片元数据: {len(self.chunks_meta)} 条")

            sparse_path = os.path.join(self.faiss_index_path, "sparse_vectors.pkl")
            if os.path.exists(sparse_path):
                with open(sparse_path, "rb") as f:
                    self.sparse_vectors = pickle.load(f)
                print(f"  已有稀疏向量: {len(self.sparse_vectors)} 条")

            images_path = os.path.join(self.faiss_index_path, "chunk_images.pkl")
            if os.path.exists(images_path):
                with open(images_path, "rb") as f:
                    self.chunk_images = pickle.load(f)
                print(f"  已有图片数据: {len(self.chunk_images)} 条")

            print("已有索引加载完成")
            return True

        except Exception as e:
            print(f"加载已有索引失败: {e}")
            print("将创建新索引")
            self.chunks_meta = []
            self.chunk_contents = []
            self.chunk_images = []
            self.dense_vectors = None
            self.sparse_vectors = []
            self.existing_vector_count = 0
            return False

    def _init_model(self):
        if self.model is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda":
                    print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
            except ImportError:
                device = "cpu"

            print(f"正在加载 BGE-M3 模型 (设备: {device})...")
            try:
                self.model = BGEM3FlagModel(
                    model_name_or_path=self.model_path,
                    use_fp16=False,
                    trust_remote_code=True,
                    devices=device,
                )
            except TypeError:
                self.model = BGEM3FlagModel(
                    model_name_or_path=self.model_path,
                    use_fp16=True,
                )
            print("模型加载完成")

    def find_markdown_files(self) -> List[str]:
        md_files = []
        for root, dirs, files in os.walk(self.md_directory):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))
        return sorted(md_files)

    def find_files(self) -> List[str]:
        files = []
        for root, dirs, filenames in os.walk(self.md_directory):
            for file in filenames:
                if file.endswith('.md') or file.endswith('.docx'):
                    files.append(os.path.join(root, file))
        return sorted(files)

    def estimate_chunks(self, content: str) -> int:
        title_pattern = re.compile(r'^#{1,}\s+', re.MULTILINE)
        titles = title_pattern.findall(content)
        return max(1, len(titles))

    def _prepare_content_preview(self, content: str, max_chars: int = 500, max_lines: int = 10) -> str:
        preview = content[:max_chars]

        if len(content) > max_chars:
            last_newline = preview.rfind('\n')
            if last_newline > 0:
                preview = preview[:last_newline]
            preview += "\n..."

        lines = preview.split('\n')
        if len(lines) > max_lines:
            lines = lines[:max_lines-1]
            lines.append("...")

        cleaned_lines = []
        for line in lines:
            line = ''.join(char for char in line if char.isprintable() or char == '\n')
            if len(line) > 100:
                line = line[:97] + "..."
            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    _PH_RE = re.compile(r'<<IMAGE:([0-9a-f]+)>>')
    _MD_IMG_RE = re.compile(r'!\[.*?\]\((.+?)\)')
    _HTML_IMG_RE = re.compile(r'<img\b[^>]*\bsrc=["\']?([^"\'\s>]+)', re.IGNORECASE)

    @staticmethod
    def _is_remote_path(path: str) -> bool:
        parsed = urlparse((path or "").strip())
        return parsed.scheme in {"http", "https", "data"}

    @staticmethod
    def _clean_image_ref(img_ref: str) -> str:
        img_ref = (img_ref or "").strip()
        if img_ref.startswith("<") and img_ref.endswith(">"):
            img_ref = img_ref[1:-1].strip()
        return img_ref

    def _extract_images_from_content(self, content: str, file_path: str = None, images_list: List[str] = None, doc_format: str = None, image_map: Dict[str, str] = None) -> List[str]:
        result = []
        config = get_config()
        if doc_format == 'markdown' or (doc_format is None and not self._PH_RE.search(content)):
            file_dir = os.path.dirname(file_path) if file_path else ''
            for img_path in self._iter_markdown_image_refs(content):
                if self._is_remote_path(img_path):
                    result.append(img_path)
                    continue
                if os.path.isabs(img_path):
                    abs_img_path = img_path
                else:
                    abs_img_path = os.path.normpath(os.path.join(file_dir, img_path))
                result.append(config.to_relative_path(abs_img_path))
        else:
            if image_map is not None:
                for match in self._PH_RE.finditer(content):
                    ph_full = match.group(0)
                    ph_id = match.group(1)
                    if ph_id in image_map:
                        result.append(config.to_relative_path(image_map[ph_id]))
            elif images_list is not None:
                order = 0
                for match in self._PH_RE.finditer(content):
                    if order < len(images_list):
                        result.append(config.to_relative_path(images_list[order]))
                        order += 1
        return result

    def _iter_markdown_image_refs(self, text: str):
        for match in self._MD_IMG_RE.finditer(text):
            yield self._clean_image_ref(match.group(1))
        for match in self._HTML_IMG_RE.finditer(text):
            yield self._clean_image_ref(match.group(1))

    @staticmethod
    def _reindex_placeholders(content: str, images: List[str]) -> Tuple[str, List[str]]:
        return content, images

    def read_file(self, file_path: str) -> Tuple[str, str, List[str], Dict[str, str]]:
        if file_path.endswith('.md'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            images = []
            file_dir = os.path.dirname(file_path)
            for img_path in self._iter_markdown_image_refs(content):
                if self._is_remote_path(img_path):
                    images.append(img_path)
                    continue
                if os.path.isabs(img_path):
                    abs_img_path = img_path
                else:
                    abs_img_path = os.path.normpath(os.path.join(file_dir, img_path))
                images.append(abs_img_path)
            return content, 'markdown', images, {}
        elif file_path.endswith('.docx'):
            import hashlib
            from docx import Document as DocxDocument

            file_dir = os.path.dirname(file_path)
            doc_stem = os.path.splitext(os.path.basename(file_path))[0]
            safe_stem = "doc_" + hashlib.md5(os.path.abspath(file_path).encode("utf-8", errors="ignore")).hexdigest()[:12]
            img_dir = os.path.join(file_dir, "images")
            os.makedirs(img_dir, exist_ok=True)

            W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
            W_P = f"{{{W_NS}}}p"
            W_T = f"{{{W_NS}}}t"
            W_PPR = f"{{{W_NS}}}pPr"
            W_PSTYLE = f"{{{W_NS}}}pStyle"
            A_BLIP = f"{{{A_NS}}}blip"
            R_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"

            VML_NS = "urn:schemas-microsoft-com:vml"
            V_IMAGEDATA = f"{{{VML_NS}}}imagedata"
            R_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

            try:
                doc = DocxDocument(file_path)
            except Exception as e:
                print(f"  [警告] python-docx 打开失败: {e}，尝试 zipfile 回退")
                try:
                    import zipfile
                    from xml.etree import ElementTree as ET
                    content_lines_fallback = []
                    with zipfile.ZipFile(file_path, 'r') as zf:
                        if 'word/document.xml' in zf.namelist():
                            doc_xml = zf.read('word/document.xml')
                            root = ET.fromstring(doc_xml)
                            body = root.find(f'{{{W_NS}}}body')
                            if body is not None:
                                for para in body:
                                    if para.tag == W_P:
                                        texts = []
                                        for t in para.iter(W_T):
                                            if t.text:
                                                texts.append(t.text)
                                        line = ''.join(texts).strip()
                                        if line:
                                            content_lines_fallback.append(line)
                    if content_lines_fallback:
                        print(f"  [Word] zipfile 回退成功，提取 {len(content_lines_fallback)} 行文本（无图片）")
                        return '\n'.join(content_lines_fallback), 'word', [], {}
                except Exception as e2:
                    print(f"  [警告] zipfile 回退也失败: {e2}")
                return '', 'word', [], {}

            images = []
            image_map = {}
            bound_rids = {}
            img_count = 0

            def _extract_image(r_id: str) -> Optional[str]:
                nonlocal img_count
                if r_id in bound_rids:
                    return bound_rids[r_id]
                try:
                    img_part = doc.part.related_parts[r_id]
                    img_bytes = img_part.blob
                    if len(img_bytes) < 500:
                        return None
                    ct = img_part.content_type
                    ext = ct.split("/")[-1].replace("jpeg", "jpg")
                    img_path = os.path.join(img_dir, f"{safe_stem}_img_{img_count}.{ext}")
                    while os.path.exists(img_path) and os.path.getsize(img_path) != len(img_bytes):
                        img_count += 1
                        img_path = os.path.join(img_dir, f"{safe_stem}_img_{img_count}.{ext}")
                    if os.path.exists(img_path):
                        abs_img_path = os.path.normpath(img_path)
                        bound_rids[r_id] = abs_img_path
                        img_count += 1
                        return abs_img_path
                    with open(img_path, 'wb') as f:
                        f.write(img_bytes)
                    abs_img_path = os.path.normpath(img_path)
                    bound_rids[r_id] = abs_img_path
                    img_count += 1
                    return abs_img_path
                except Exception as e:
                    print(f"  [Word] 图片提取失败 rId={r_id}: {e}")
                    return None

            def _heading_level(p_node) -> int:
                pPr = p_node.find(W_PPR)
                if pPr is None:
                    return 0
                pStyle = pPr.find(W_PSTYLE)
                if pStyle is None:
                    return 0
                style_val = pStyle.get(f'{{{W_NS}}}val', pStyle.get('val', ''))
                if style_val.startswith('Heading'):
                    try:
                        return max(1, min(6, int(style_val.replace('Heading', '').strip())))
                    except ValueError:
                        return 1
                return 0

            def _placeholder_for(r_id: str, abs_path: str) -> str:
                seed = f"{os.path.basename(file_path)}:{r_id}:{len(image_map)}:{abs_path}"
                ph_id = hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]
                while ph_id in image_map and image_map[ph_id] != abs_path:
                    seed += ":x"
                    ph_id = hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]
                image_map[ph_id] = abs_path
                images.append(abs_path)
                return f"<<IMAGE:{ph_id}>>"

            def _paragraph_text_with_images(p_node) -> str:
                parts = []
                for node in p_node.iter():
                    if node.tag == W_T and node.text:
                        parts.append(node.text)
                    elif node.tag == A_BLIP:
                        r_id = node.get(R_EMBED)
                        if r_id:
                            abs_path = _extract_image(r_id)
                            if abs_path:
                                parts.append(_placeholder_for(r_id, abs_path))
                        else:
                            r_id_link = node.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}link")
                            if r_id_link:
                                print(f"  [Word] 跳过链接图片 rId={r_id_link}")
                    elif node.tag == V_IMAGEDATA:
                        r_id = node.get(R_ID) or node.get(f"{{{VML_NS}}}id")
                        if r_id:
                            abs_path = _extract_image(r_id)
                            if abs_path:
                                parts.append(_placeholder_for(r_id, abs_path))
                return ''.join(parts).strip()

            content_lines = []
            for p_node in doc.element.body.iter(W_P):
                text = _paragraph_text_with_images(p_node)
                level = _heading_level(p_node)
                if not text and level <= 0:
                    continue
                if level > 0:
                    content_lines.append('#' * level + ' ' + text)
                else:
                    content_lines.append(text)

            print(f"  [Word] 文本行数: {len(content_lines)}, 插入图片: {img_count}, 占位符映射: {len(image_map)}")
            return '\n'.join(content_lines), 'word', images, image_map
        return '', 'unknown', [], {}

    def parse_word(self, content: str, file_path: str = None, images: List[str] = None, image_map: Dict[str, str] = None) -> List[Dict]:
        sections = []
        lines = content.split('\n')
        current_section = {"title": "", "title_level": 0, "content": [], "images": [], "title_path": ""}
        title_stack = []

        config = get_config()
        if images is None:
            images = []

        global_ph_idx = 0

        def _images_for_text(text: str) -> List[str]:
            nonlocal global_ph_idx
            found = []
            for ph_id in self._PH_RE.findall(text):
                img_path = None
                if image_map and ph_id in image_map:
                    img_path = image_map[ph_id]
                elif global_ph_idx < len(images):
                    img_path = images[global_ph_idx]
                    global_ph_idx += 1
                if img_path:
                    rel = config.to_relative_path(img_path)
                    if rel not in found:
                        found.append(rel)
            return found

        for line in lines:
            match = self.TITLE_PATTERN.match(line)
            word_match = None if match else self.WORD_NUMBERED_HEADING_PATTERN.match(line.strip())
            if match or word_match:
                if current_section["content"]:
                    raw_content = '\n'.join(current_section["content"]).strip()
                    sections.append({
                        "title": current_section["title"],
                        "title_level": current_section["title_level"],
                        "content": raw_content,
                        "images": current_section["images"],
                        "title_path": current_section.get("title_path", "")
                    })

                if match:
                    level = len(match.group(1))
                    number = match.group(2)
                    title = match.group(3)
                else:
                    prefix = word_match.group(1)
                    title = word_match.group(2)
                    number = prefix.rstrip("、.．")
                    level = 1 if prefix.startswith(("一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "第")) else 2
                full_title = f"{number} {title}" if number else title

                section_level = number.count('.') + 1 if number else 1

                title_stack = [entry for entry in title_stack
                              if not number or number.startswith(entry["number"] + ".")]

                title_path = " > ".join(entry["full_title"] for entry in title_stack)

                title_stack.append({"number": number or "", "full_title": full_title})

                current_section = {
                    "title": full_title,
                    "title_level": level,
                    "section_level": section_level,
                    "content": [],
                    "images": [],
                    "title_path": title_path
                }
            else:
                for img_path in _images_for_text(line):
                    if img_path not in current_section["images"]:
                        current_section["images"].append(img_path)

                current_section["content"].append(line)

        if current_section["content"]:
            raw_content = '\n'.join(current_section["content"]).strip()
            sections.append({
                "title": current_section["title"],
                "title_level": current_section["title_level"],
                "section_level": current_section.get("section_level", 1),
                "content": raw_content,
                "images": current_section["images"],
                "title_path": current_section.get("title_path", "")
            })

        if not sections:
            sections.append({
                "title": "文档内容",
                "title_level": 1,
                "section_level": 1,
                "content": content,
                "images": [],
                "title_path": ""
            })

        return sections

    def parse_markdown(self, content: str, file_path: str = None, images: List[str] = None) -> List[Dict]:
        lines = content.split('\n')
        sections = []
        current_section = {"title": "", "title_level": 0, "content": [], "images": [], "title_path": ""}

        title_stack = []

        config = get_config()
        file_dir = os.path.dirname(file_path) if file_path else ''

        def _resolve_md_images(text):
            resolved = []
            for img_path in self._iter_markdown_image_refs(text):
                if self._is_remote_path(img_path):
                    resolved.append(img_path)
                    continue
                if os.path.isabs(img_path):
                    abs_img_path = img_path
                else:
                    abs_img_path = os.path.normpath(os.path.join(file_dir, img_path))
                resolved.append(config.to_relative_path(abs_img_path))
            return resolved

        for line in lines:
            match = self.TITLE_PATTERN.match(line)
            if match:
                if current_section["content"]:
                    sections.append({
                        "title": current_section["title"],
                        "title_level": current_section["title_level"],
                        "content": '\n'.join(current_section["content"]).strip(),
                        "images": current_section["images"],
                        "title_path": current_section.get("title_path", "")
                    })

                level = len(match.group(1))
                number = match.group(2)
                title = match.group(3)
                full_title = f"{number} {title}" if number else title

                section_level = number.count('.') + 1 if number else 1

                title_stack = [entry for entry in title_stack
                              if not number or number.startswith(entry["number"] + ".")]

                title_path = " > ".join(entry["full_title"] for entry in title_stack)

                title_stack.append({"number": number, "full_title": full_title})

                current_section = {
                    "title": full_title,
                    "title_level": level,
                    "section_level": section_level,
                    "content": [],
                    "images": [],
                    "title_path": title_path
                }
            else:
                for img_path in self._iter_markdown_image_refs(line):
                    if self._is_remote_path(img_path):
                        current_section["images"].append(img_path)
                        continue
                    if os.path.isabs(img_path):
                        abs_img_path = img_path
                    else:
                        abs_img_path = os.path.normpath(os.path.join(file_dir, img_path))
                    current_section["images"].append(config.to_relative_path(abs_img_path))

                current_section["content"].append(line)

        if current_section["content"]:
            sections.append({
                "title": current_section["title"],
                "title_level": current_section["title_level"],
                "section_level": current_section.get("section_level", 1),
                "content": '\n'.join(current_section["content"]).strip(),
                "images": current_section["images"],
                "title_path": current_section.get("title_path", "")
            })

        final_sections = []
        for section in sections:
            content_text = section['content']

            if len(content_text) <= 500:
                final_sections.append(section)
                continue

            content_lines = content_text.split('\n')
            sub_matches = []
            for i, line in enumerate(content_lines):
                sub_match = self.SUB_TITLE_PATTERN.match(line.strip())
                if sub_match:
                    sub_matches.append((i, sub_match))

            if not sub_matches:
                final_sections.append(section)
                continue

            sub_sections = []
            parent_title = section['title']
            parent_title_path = section.get('title_path', '')

            first_sub_idx = sub_matches[0][0]
            if first_sub_idx > 0:
                pre_content = '\n'.join(content_lines[:first_sub_idx]).strip()
                if pre_content:
                    pre_section = {
                        "title": parent_title,
                        "title_level": section['title_level'],
                        "section_level": section.get('section_level', 1),
                        "content": pre_content,
                        "images": _resolve_md_images(pre_content),
                        "title_path": parent_title_path
                    }
                    sub_sections.append(pre_section)

            for j, (match_idx, sub_match) in enumerate(sub_matches):
                sub_title = sub_match.group(2).strip()
                sub_title = re.sub(r'\*{1,2}', '', sub_title).strip()

                start_idx = match_idx + 1
                if j + 1 < len(sub_matches):
                    end_idx = sub_matches[j + 1][0]
                else:
                    end_idx = len(content_lines)

                sub_content = '\n'.join(content_lines[start_idx:end_idx]).strip()

                full_sub_title = f"{parent_title} > {sub_title}"

                sub_title_path = parent_title_path

                sub_content_full = '\n'.join(content_lines[match_idx:end_idx])
                sub_images = _resolve_md_images(sub_content_full)

                sub_sections.append({
                    "title": full_sub_title,
                    "title_level": section['title_level'],
                    "section_level": section.get('section_level', 1),
                    "content": sub_content,
                    "images": sub_images,
                    "title_path": sub_title_path
                })

            merged = []
            for sub_sec in sub_sections:
                if len(sub_sec['content']) < 100 and merged:
                    merged[-1]['content'] += '\n' + sub_sec['content']
                    if sub_sec.get('images'):
                        existing_imgs = set(merged[-1].get('images', []))
                        for img in sub_sec['images']:
                            if img not in existing_imgs:
                                merged[-1]['images'].append(img)
                                existing_imgs.add(img)
                else:
                    merged.append(sub_sec)

            final_sections.extend(merged)

        return final_sections

    def split_by_length(self, content: str, chunk_size: int = 500, overlap_ratio: float = 0.1) -> List[str]:
        if not content or len(content) <= chunk_size:
            return [content] if content else []

        separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " "]

        def _token_spans(text: str):
            spans = []
            for m in self._PH_RE.finditer(text):
                spans.append((m.start(), m.end()))
            for m in self._MD_IMG_RE.finditer(text):
                spans.append((m.start(), m.end()))
            for m in self._HTML_IMG_RE.finditer(text):
                spans.append((m.start(), m.end()))
            return sorted(spans)

        def _protect_end(text: str, end: int) -> int:
            for start_pos, end_pos in _token_spans(text):
                if start_pos < end < end_pos:
                    return end_pos
            return end

        def _hard_split(text: str) -> List[str]:
            result = []
            start = 0
            while start < len(text):
                end = min(len(text), start + chunk_size)
                end = min(len(text), _protect_end(text, end))
                if end <= start:
                    end = min(len(text), start + chunk_size)
                result.append(text[start:end])
                start = end
            return result

        def _recursive_split(text: str) -> List[str]:
            text = text.strip()
            if not text:
                return []
            if len(text) <= chunk_size:
                return [text]
            for sep in separators:
                if sep not in text:
                    continue
                parts = text.split(sep)
                chunks = []
                current = ""
                for part in parts:
                    candidate = current + (sep if current else "") + part
                    if len(candidate) <= chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        if len(part) > chunk_size:
                            chunks.extend(_recursive_split(part))
                            current = ""
                        else:
                            current = part
                if current:
                    chunks.append(current)
                if chunks:
                    return chunks
            return _hard_split(text)

        def _merge_short(chunks: List[str], min_size: int = 80) -> List[str]:
            if not chunks:
                return []
            merged = [chunks[0]]
            for chunk in chunks[1:]:
                prev = merged[-1]
                if len(chunk) < min_size and len(prev) + len(chunk) <= int(chunk_size * 1.25):
                    merged[-1] = prev + chunk
                else:
                    merged.append(chunk)
            return merged

        def _overlap_tail(text: str, overlap: int) -> str:
            if overlap <= 0 or len(text) <= overlap:
                return ""
            tail = text[-(overlap * 2):]
            matches = list(re.finditer(r'[。！？.!?\n]', tail))
            if matches:
                candidate = tail[matches[-1].end():].lstrip()
                if candidate:
                    return candidate
            return text[-overlap:]

        raw = _merge_short(_recursive_split(content))
        overlap = max(0, min(int(chunk_size * overlap_ratio), chunk_size // 2))
        if overlap <= 0 or len(raw) <= 1:
            return raw

        result = [raw[0]]
        for idx, chunk in enumerate(raw[1:], 1):
            tail = _overlap_tail(raw[idx - 1], overlap)
            result.append((tail + chunk) if tail else chunk)
        return result

    def _is_document_exists(self, file_path: str) -> bool:
        config = get_config()
        relative_path = config.to_relative_path(file_path)
        doc_name_no_ext = os.path.splitext(os.path.basename(file_path))[0]
        doc_name_with_ext = os.path.basename(file_path)

        existing = self.db.get_corpus_by_path(file_path, relative_path)
        if existing:
            return True

        for name in (doc_name_no_ext, doc_name_with_ext):
            for existing in self.db.get_corpora_by_name(name):
                if self._check_path_match(existing, file_path, relative_path):
                    return True

        return False

    def _check_path_match(self, existing: Corpus, file_path: str, relative_path: str) -> bool:
        if existing.relative_path and existing.relative_path == relative_path:
            return True
        if existing.file_path == file_path:
            return True
        if existing.file_path == relative_path:
            return True
        if file_path.endswith(existing.file_path) or existing.file_path.endswith(os.path.basename(file_path)):
            return True
        return False

    def process_document(self, file_path: str) -> Tuple[str, List[Dict], Dict]:
        doc_name = os.path.splitext(os.path.basename(file_path))[0]

        if self._is_document_exists(file_path):
            print(f"  [跳过] 文档已存在: {doc_name}")
            return None, [], None

        content, doc_type, images, image_map = self.read_file(file_path)
        if not content.strip():
            print(f"  [跳过] 文档内容为空: {doc_name}")
            return None, [], None

        if self.chunk_strategy == "title":
            if doc_type == "word":
                sections = self.parse_word(content, file_path, images, image_map)
            else:
                sections = self.parse_markdown(content, file_path, images)
            print(f"  [策略] 使用标题分片策略")
        else:
            chunks_content = self.split_by_length(content, self.chunk_size, self.overlap_ratio)
            sections = []
            for idx, chunk_content in enumerate(chunks_content):
                chunk_images = self._extract_images_from_content(chunk_content, file_path, images, doc_type, image_map)
                sections.append({
                    "title": f"分片 {idx + 1}",
                    "title_level": 1,
                    "section_level": 1,
                    "content": chunk_content,
                    "images": chunk_images,
                    "title_path": ""
                })
            print(f"  [策略] 使用长度分片策略 (chunk_size={self.chunk_size}, overlap_ratio={self.overlap_ratio})")

        if not sections:
            print(f"  [跳过] 未生成有效分片: {doc_name}")
            return None, [], None

        config = get_config()
        relative_path = config.to_relative_path(file_path)

        corpus = Corpus(
            name=doc_name,
            description="",
            type="文档",
            data_summary=content[:500],
            source="本地导入",
            file_path=file_path,
            relative_path=relative_path,
            chunk_count=len(sections),
            chunk_strategy=self.chunk_strategy,
        )

        corpus_id = self.db.add_corpus(corpus)

        chunks = []
        chunk_logs = []

        for i, section in enumerate(sections):
            if section.get('title_path'):
                chunk_text = f"{doc_name} > {section['title_path']} > {section['title']}\n{section['content']}"
            elif section['title']:
                chunk_text = f"{doc_name} > {section['title']}\n{section['content']}"
            else:
                chunk_text = f"{doc_name}\n{section['content']}"

            try:
                chunk_log = {
                    "chunk_index": i,
                    "title": section["title"] if section["title"] else "无标题",
                    "title_path": section.get('title_path', ''),
                    "section_level": section.get("section_level", 1),
                    "content_preview": section["content"][:200] if section["content"] else "",
                    "content_length": len(section["content"]) if section["content"] else 0
                }
                chunk_logs.append(chunk_log)
            except Exception as e:
                chunk_logs.append({
                    "chunk_index": i,
                    "title": section["title"] if section["title"] else "无标题",
                    "error": str(e)
                })

            chunk = {
                "corpus_id": corpus_id,
                "chunk_index": i,
                "title": section["title"],
                "title_level": section["title_level"],
                "content": section["content"],
                "images": section["images"],
                "chunk_text": chunk_text
            }
            chunks.append(chunk)

        doc_log = {
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "doc_name": doc_name,
            "corpus_id": corpus_id,
            "chunk_strategy": self.chunk_strategy,
            "chunk_count": len(sections),
            "chunks": chunk_logs
        }
        self.import_logs.append(doc_log)

        print(f"  [文档] {doc_name}")
        print(f"  [分片] {len(sections)} 个 chunk")

        return corpus_id, chunks, {}

    def process_document_web(self, file_path: str, tags: Dict = None) -> Dict:
        doc_name = os.path.splitext(os.path.basename(file_path))[0]

        if self._is_document_exists(file_path):
            return {"success": False, "corpus_id": None, "doc_name": doc_name,
                    "message": f"文档已存在: {doc_name}", "tags": None}

        content, doc_type, images, image_map = self.read_file(file_path)
        if not content.strip():
            return {"success": False, "corpus_id": None, "doc_name": doc_name,
                    "message": f"文档内容为空: {doc_name}", "tags": None}

        if self.chunk_strategy == "title":
            if doc_type == "word":
                sections = self.parse_word(content, file_path, images, image_map)
            else:
                sections = self.parse_markdown(content, file_path, images)
        else:
            chunks_content = self.split_by_length(content, self.chunk_size, self.overlap_ratio)
            sections = []
            for idx, chunk_content in enumerate(chunks_content):
                chunk_images = self._extract_images_from_content(chunk_content, file_path, images, doc_type, image_map)
                sections.append({
                    "title": f"分片 {idx + 1}",
                    "title_level": 1,
                    "section_level": 1,
                    "content": chunk_content,
                    "images": chunk_images,
                    "title_path": ""
                })

        if not sections:
            return {"success": False, "corpus_id": None, "doc_name": doc_name,
                    "message": f"未生成有效分片: {doc_name}", "tags": None}

        config = get_config()
        relative_path = config.to_relative_path(file_path)

        corpus = Corpus(
            name=doc_name,
            description="",
            type="文档",
            data_summary=content[:500],
            source="Web上传",
            file_path=file_path,
            relative_path=relative_path,
            chunk_count=len(sections),
            chunk_strategy=self.chunk_strategy,
        )

        corpus_id = self.db.add_corpus(corpus)

        chunks = []
        for i, section in enumerate(sections):
            if section.get('title_path'):
                chunk_text = f"{doc_name} > {section['title_path']} > {section['title']}\n{section['content']}"
            elif section['title']:
                chunk_text = f"{doc_name} > {section['title']}\n{section['content']}"
            else:
                chunk_text = f"{doc_name}\n{section['content']}"

            chunk = {
                "corpus_id": corpus_id,
                "chunk_index": i,
                "title": section["title"],
                "title_level": section["title_level"],
                "content": section["content"],
                "images": section["images"],
                "chunk_text": chunk_text,
            }
            chunks.append(chunk)

        doc_info = {
            "corpus_id": corpus_id,
            "file_path": file_path,
            "name": doc_name,
            "chunk_count": len(chunks),
            "vector_ids": [],
        }
        self.processed_documents.append(doc_info)

        return {
            "success": True,
            "corpus_id": corpus_id,
            "doc_name": doc_name,
            "message": f"入库成功: {doc_name} ({len(chunks)} 个分片)",
            "chunks": chunks,
        }

    def generate_embeddings(self, texts: List[str]) -> Tuple[np.ndarray, List[Dict]]:
        self._init_model()

        print("正在生成向量嵌入...")
        embeddings = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=4,
        )

        dense_vecs = embeddings["dense_vecs"]
        sparse_vecs = embeddings["lexical_weights"]

        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        dense_vecs = dense_vecs / norms

        sparse_list = [{int(k): float(v) for k, v in sv.items()} for sv in sparse_vecs]

        return dense_vecs.astype(np.float32), sparse_list

    def build_sparse_index(self, sparse_vectors: List[Dict]) -> Dict:
        sparse_index = {}
        for doc_id, sparse_vec in enumerate(sparse_vectors):
            for token_id, weight in sparse_vec.items():
                token_id = int(token_id)
                if token_id not in sparse_index:
                    sparse_index[token_id] = {}
                sparse_index[token_id][doc_id] = weight
        return sparse_index

    def save_faiss_index(self):
        os.makedirs(self.faiss_index_path, exist_ok=True)

        index_path = os.path.join(self.faiss_index_path, "index.faiss")
        faiss.write_index(self.faiss_index, index_path)

        with open(os.path.join(self.faiss_index_path, "chunk_contents.pkl"), "wb") as f:
            pickle.dump(self.chunk_contents, f)

        with open(os.path.join(self.faiss_index_path, "chunks_meta.pkl"), "wb") as f:
            pickle.dump(self.chunks_meta, f)

        with open(os.path.join(self.faiss_index_path, "sparse_vectors.pkl"), "wb") as f:
            pickle.dump(self.sparse_vectors, f)

        with open(os.path.join(self.faiss_index_path, "sparse_index.pkl"), "wb") as f:
            pickle.dump(self.sparse_index, f)

        with open(os.path.join(self.faiss_index_path, "chunk_images.pkl"), "wb") as f:
            pickle.dump(self.chunk_images, f)

        with open(os.path.join(self.faiss_index_path, "index_version.json"), "w", encoding="utf-8") as f:
            json.dump({
                "version": 2,
                "features": ["title_enhanced_embedding", "sub_heading_chunking", "bm25_index"],
                "chunk_strategy": self.chunk_strategy,
                "chunk_size": self.chunk_size,
                "overlap_ratio": self.overlap_ratio
            }, f, ensure_ascii=False, indent=2)

        self.bm25_indexer.save()

        print(f"索引保存完成: {self.faiss_index_path}")

    def process_all(self):
        md_files = self.find_files()
        print(f"找到 {len(md_files)} 个 Markdown/Word 文件")
        print(f"[分片策略] chunk_strategy={self.chunk_strategy}, chunk_size={self.chunk_size}, overlap_ratio={self.overlap_ratio}")

        stats = self.db.get_stats()
        if stats.get('corpus', 0) > 0:
            print(f"数据库中已有 {stats.get('corpus', 0)} 个语料")

        if not md_files:
            print("没有找到 Markdown 或 Word 文件")
            return

        if self.append_mode:
            self.load_existing_index()

        if self.resume_mode:
            print("\n[恢复模式] 正在加载上次进度...")
            self.processed_files = self.load_progress()
            if self.processed_files:
                original_count = len(md_files)
                md_files = [f for f in md_files if f not in self.processed_files]
                filtered_count = original_count - len(md_files)
                print(f"[恢复模式] 已过滤 {filtered_count} 个已处理文档")
                print(f"[恢复模式] 剩余 {len(md_files)} 个文档待处理")
            else:
                print("[恢复模式] 没有找到进度记录，将处理所有文档")
        else:
            if os.path.exists(self.progress_file):
                print(f"\n[提示] 发现进度文件: {self.progress_file}")
                print("  如需恢复上次进度，请使用 --resume 参数")
                print("  如需重新开始，请使用 --reset 参数")

        if not md_files:
            print("没有需要处理的文档（所有文档已处理完成）")
            return

        all_chunks = []
        corpus_count = 0
        pending_chunks_map: Dict[str, List[Dict]] = {}

        print("\n正在处理文档...")
        i = 0
        while i < len(md_files):
            file_path = md_files[i]
            try:
                corpus_id, chunks, metadata = self.process_document(file_path)

                if corpus_id == "ROLLBACK":
                    if self.processed_documents:
                        rollback_success = self.rollback_last_document()
                        if rollback_success:
                            last_doc = self.get_last_processed_document()
                            if last_doc:
                                last_file_path = last_doc.get("file_path")
                                if last_file_path in pending_chunks_map:
                                    del pending_chunks_map[last_file_path]
                                try:
                                    last_idx = md_files.index(last_file_path)
                                    i = last_idx
                                    print(f"\n  [回退] 重新处理文档: {os.path.basename(last_file_path)}")
                                except ValueError:
                                    print(f"\n  [警告] 找不到上一条文档在文件列表中的位置")
                        else:
                            print("\n  [警告] 回退失败，继续处理当前文档")
                    else:
                        print("\n  [警告] 没有可回退的文档，继续处理当前文档")
                    continue

                if corpus_id is None:
                    i += 1
                    continue

                doc_info = {
                    "corpus_id": corpus_id,
                    "file_path": file_path,
                    "name": os.path.splitext(os.path.basename(file_path))[0],
                    "chunk_count": len(chunks),
                    "vector_ids": []
                }
                self.processed_documents.append(doc_info)
                pending_chunks_map[file_path] = chunks
                all_chunks.extend(chunks)
                corpus_count += 1

                self.processed_files.append(file_path)
                self.save_progress()

                i += 1

            except Exception as e:
                print(f"\n处理文件失败 {file_path}: {e}")
                self.save_progress()
                print(f"  进度已保存，可使用 --resume 参数从该文档继续")
                i += 1
                continue

        print(f"处理完成: {corpus_count} 个语料, {len(all_chunks)} 个分片")

        if not all_chunks:
            print("没有生成分片")
            return

        chunk_texts = [chunk["chunk_text"] for chunk in all_chunks]
        dense_vecs, sparse_vecs = self.generate_embeddings(chunk_texts)

        print("正在保存到数据库...")

        corpus_id_to_doc_idx = {}
        for idx, doc_info in enumerate(self.processed_documents):
            corpus_id_to_doc_idx[doc_info["corpus_id"]] = idx

        vector_id = self.existing_vector_count
        for i, chunk in enumerate(tqdm(all_chunks, desc="保存分片")):
            chunk_record = Chunk(
                corpus_id=chunk["corpus_id"],
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                title=chunk["title"],
                title_level=chunk["title_level"],
                images=chunk["images"],
                vector_id=vector_id
            )

            if chunk["corpus_id"] in corpus_id_to_doc_idx:
                doc_idx = corpus_id_to_doc_idx[chunk["corpus_id"]]
                self.processed_documents[doc_idx]["vector_ids"].append(vector_id)

            self.db.add_chunk(chunk_record)

            corpus_info = self.db.get_corpus_by_id(chunk["corpus_id"])

            self.chunks_meta.append({
                "corpus_id": chunk["corpus_id"],
                "doc_name": os.path.basename(corpus_info.file_path) if corpus_info else "unknown",
                "title": chunk["title"],
                "title_level": chunk["title_level"],
                "images": chunk["images"]
            })
            self.chunk_contents.append(chunk["chunk_text"])
            self.chunk_images.append(chunk["images"])

            vector_id += 1

        if self.append_mode and self.existing_vector_count > 0:
            print("正在合并向量数据...")
            self.dense_vectors = np.vstack([self.faiss_index.reconstruct_n(0, self.existing_vector_count), dense_vecs])
            self.sparse_vectors = self.sparse_vectors + sparse_vecs
        else:
            self.dense_vectors = dense_vecs
            self.sparse_vectors = sparse_vecs

        self.sparse_index = self.build_sparse_index(self.sparse_vectors)

        print("正在构建 BM25 索引...")
        if self.append_mode and self.existing_vector_count > 0:
            existing_bm25 = BM25Indexer(self.faiss_index_path)
            if existing_bm25.load():
                new_contents = [chunk["chunk_text"] for chunk in all_chunks]
                new_ids = list(range(self.existing_vector_count, self.existing_vector_count + len(all_chunks)))
                existing_bm25.append_documents(new_contents, new_ids)
                self.bm25_indexer = existing_bm25
            else:
                all_ids = list(range(len(self.chunk_contents)))
                self.bm25_indexer.build_index(self.chunk_contents, all_ids)
        else:
            all_ids = list(range(len(self.chunk_contents)))
            self.bm25_indexer.build_index(self.chunk_contents, all_ids)

        print("正在创建 FAISS 索引...")
        dimension = self.dense_vectors.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.dense_vectors)

        self.save_faiss_index()

        print("\n数据库统计:")
        stats = self.db.get_stats()
        for table, count in stats.items():
            print(f"  {table}: {count}")

        print(f"\n处理完成!")
        print(f"  本次新增语料: {corpus_count}")
        print(f"  本次新增分片: {len(all_chunks)}")
        print(f"  向量总数: {self.faiss_index.ntotal}")
        print(f"  向量维度: {dimension}")

        self.save_import_logs()

def main():
    parser = argparse.ArgumentParser(
        description="文档入库脚本 - 支持数据库存储和 BM25 索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python add_documents.py                              # 使用默认标题分片策略
  python add_documents.py --chunk-strategy length      # 使用长度分片策略
  python add_documents.py --chunk-size 1000            # 使用更大的分片大小
  python add_documents.py --resume                     # 恢复上次进度
  python add_documents.py --reset                      # 重置进度并重新开始
        """
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="恢复上次进度（断点续传）"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置进度（删除进度文件，重新开始）"
    )
    parser.add_argument(
        "--chunk-strategy",
        type=str,
        default="title",
        choices=["title", "length"],
        help="分片策略: title (按标题分片) 或 length (按长度分片), 默认: title"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="分片大小(字符数), 仅用于 length 策略, 默认: 500"
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.1,
        help="重叠比例, 仅用于 length 策略, 默认: 0.1"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("文档入库脚本")
    print("=" * 60)

    resume = args.resume
    reset = args.reset
    chunk_strategy = args.chunk_strategy
    chunk_size = args.chunk_size
    overlap_ratio = args.overlap_ratio

    if resume and reset:
        print("\n[错误] --resume 和 --reset 参数不能同时使用")
        sys.exit(1)

    if resume:
        print("\n[恢复模式] 将恢复上次处理进度")
    elif reset:
        print("\n[重置模式] 将删除进度文件并重新开始")

    print(f"\n[分片策略] chunk_strategy={chunk_strategy}, chunk_size={chunk_size}, overlap_ratio={overlap_ratio}")

    try:
        processor = DocumentProcessor(
            resume=resume,
            reset=reset,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio
        )
        processor.process_all()
        print("\n完成!")
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n[错误] 程序运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
