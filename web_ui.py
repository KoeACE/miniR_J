"""
Gradio Web 管理界面
功能：文档入库、文档列表、检索、统计
启动方式：python web_ui.py
"""

import os
import sys
import re
import json
import copy
import threading
import base64
import mimetypes
import html as html_lib
import hashlib
from urllib.parse import urlparse


WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_ROOT)

import gradio as gr
import numpy as np
import faiss
from tqdm import tqdm
from scripts.db_manager import DatabaseManager, Corpus, Chunk
from scripts.manage_documents import DocumentManager
from scripts.add_documents import DocumentProcessor
from scripts.bm25_indexer import BM25Indexer
from scripts.config_manager import get_config

LANG = {
    "zh": {
        "app_title": "# 📚 RAG 知识库管理系统",
        "tab_import": "文档入库",
        "tab_docs": "文档列表",
        "tab_search": "检索测试",
        "tab_system": "系统管理",
        "import_desc": "输入服务器上的文件夹路径，系统自动递归扫描所有 `.md` 和 `.docx` 文件并入库。\n\n✅ 支持 Markdown (.md) 和 Word (.docx) 文档\n✅ 图片引用完整保留\n✅ 扫描时已入库的文档会自动跳过。",
        "folder_path": "文件夹路径",
        "folder_placeholder": "如 D:\\docs 或 /home/user/docs",
        "scan_btn": "📂 扫描文件",
        "scan_result": "扫描结果",
        "file_list": "扫描到的文件列表",
        "import_settings": "### 入库设置",
        "chunk_strategy": "分块策略",
        "chunk_strategy_info": "title: 按标题分块, length: 按长度分块",
        "chunk_size": "分块大小（字符数）",
        "chunk_size_info": "仅 length 策略生效，默认 512",
        "overlap_ratio": "重叠比例",
        "overlap_ratio_info": "仅 length 策略生效，默认 0.1（10%）",
        "import_btn": "🚀 开始入库",
        "import_log": "入库日志",
        "keyword_search": "关键词搜索",
        "keyword_placeholder": "输入文档名关键词",
        "status_filter": "状态筛选",
        "all": "全部",
        "active": "启用",
        "inactive": "停用",
        "refresh": "🔄 刷新",
        "doc_id": "Corpus ID",
        "doc_id_placeholder": "点击表格行自动填充",
        "toggle_btn": "⏸ 停用/启用",
        "delete_btn": "🗑 删除",
        "action_result": "操作结果",
        "search_title": "### 🔍 检索测试\n\n测试 RAG 检索效果，使用混合检索模式（BM25 + Dense + Sparse）。",
        "query_label": "查询内容",
        "query_placeholder": "如 机器学习基础概念、论文摘要检索",
        "top_k": "返回结果数",
        "use_rerank": "使用 Rerank",
        "yes": "是",
        "no": "否",
        "search_btn": "🔍 开始检索",
        "clear_btn": "🗑 清空",
        "search_result": "检索结果",
        "stats_tab": "统计信息",
        "stats_refresh": "🔄 刷新统计",
        "stats_info": "统计信息",
        "reset_tab": "一键删库",
        "reset_desc": "### ⚠️ 危险操作：清空所有数据\n\n此操作将：\n1. 清空 FAISS 索引文件\n2. 删除 SQLite 数据库文件并重新初始化\n\n**此操作不可恢复！**",
        "reset_step1": "🗑 开始删库",
        "reset_warning": "警告",
        "confirm_code": "输入确认码",
        "confirm_placeholder": "输入上方显示的确认码",
        "reset_step2": "🔴 确认删库",
        "reset_result": "删库结果",
        "col_full_path": "完整路径",
        "col_rel_path": "相对路径",
        "col_status": "状态",
        "col_id": "ID",
        "col_name": "文档名",
        "col_state": "状态",
        "active_mark": "● 启用",
        "inactive_mark": "○ 停用",
        "total_docs": "文档总数",
        "total_chunks": "分片总数",
        "enter_query": "请输入查询内容",
        "enter_folder": "请输入文件夹路径",
        "enter_corpus_id": "请输入或选择 corpus_id",
        "doc_not_found": "文档不存在",
        "db_conn_fail": "数据库连接失败",
        "deleted": "已删除",
        "delete_fail": "删除失败",
        "enabled": "启用",
        "disabled": "停用",
        "scan_files_first": "请先扫描文件夹",
        "path_not_exist": "路径不存在或不是文件夹",
        "no_md_files": "文件夹中未找到 .md 或 .docx 文件",
        "found_files": "找到 {count} 个文档（⏩新文档 {new}，✅已入库 {exist}）",
        "reset_warning_text": "⚠️ 即将清空所有数据（SQLite + FAISS），此操作不可恢复！\n\n请输入确认码 ** {code} ** 后点击「确认删库」",
        "confirm_wrong": "❌ 确认码不正确，操作已取消",
        "reset_done": "[DONE] 数据库已重置为空库。",
        "reset_timeout": "[ERROR] 删库脚本执行超时（60秒）",
    }
}

_current_lang = "zh"


ACADEMIC_PRIMARY = gr.themes.Color(
    name="minir_blue",
    c50="#EFF6FF", c100="#DBEAFE", c200="#BFDBFE",
    c300="#93C5FD", c400="#60A5FA", c500="#3B82F6",
    c600="#2563EB", c700="#1D4ED8", c800="#1E40AF",
    c900="#1E3A8A", c950="#172554",
)

ACADEMIC_THEME = gr.themes.Default(
    primary_hue=ACADEMIC_PRIMARY,
    secondary_hue="slate",
).set(
    body_background_fill="#F6F8FB",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F3F6FA",
    border_color_accent="#D6DEE8",
    shadow_spread="0",
)

ACADEMIC_CSS = """
.gradio-container { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif !important; }
h1, h2, h3, h4 { color: #111827; letter-spacing: 0; }
h1 { font-size: 1.8rem !important; font-weight: 700 !important; }
.markdown h1 { border-bottom: 1px solid #E5E7EB; padding-bottom: 8px; }
.tab-nav button { font-weight: 600 !important; }
table thead th { background-color: #F3F6FA !important; color: #111827 !important; font-weight: 600; }
.mr-overview { display: grid; gap: 12px; }
.mr-doc-group { border: 1px solid #D6DEE8; border-radius: 8px; overflow: hidden; background: #fff; }
.mr-doc-head { display: flex; justify-content: space-between; gap: 12px; padding: 10px 12px; background: #F3F6FA; border-bottom: 1px solid #D6DEE8; font-weight: 700; color: #111827; }
.mr-chunk-row { padding: 9px 12px; border-bottom: 1px solid #EEF2F7; }
.mr-chunk-row:last-child { border-bottom: 0; }
.mr-chunk-title { display: flex; align-items: center; justify-content: space-between; gap: 10px; color: #111827; font-weight: 650; }
.mr-chunk-title small { color: #64748B; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mr-status { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; border-radius: 4px; padding: 1px 6px; font-size: 12px; }
.mr-chunk-meta { margin-top: 4px; color: #2563EB; font-size: 12px; }
.mr-snippet { margin-top: 4px; color: #475569; font-size: 13px; line-height: 1.5; }
.mr-current-preview, .mr-result-card { border: 1px solid var(--border-color-primary); border-radius: 8px; background: var(--background-fill-primary); overflow: hidden; margin-bottom: 12px; }
.mr-current-head, .mr-result-head { display: flex; justify-content: space-between; gap: 10px; padding: 10px 12px; background: var(--background-fill-secondary); border-bottom: 1px solid var(--border-color-primary); color: var(--body-text-color); font-weight: 700; }
.mr-current-head span, .mr-result-title { color: var(--body-text-color-subdued); font-size: 13px; font-weight: 500; }
.mr-current-body, .mr-result-body { padding: 12px; max-height: 420px; overflow: auto; background: var(--background-fill-primary); color: var(--body-text-color); font-size: 14px; line-height: 1.7; }
.mr-current-images { display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 12px; border-top: 1px solid #E5E7EB; background: #F8FAFC; }
.mr-empty-images { color: #64748B; font-size: 13px; border: 1px solid #E5E7EB; border-radius: 8px; }
.mr-thumb { width: 104px; text-decoration: none; color: #475569; font-size: 12px; }
.mr-thumb img { width: 104px; height: 76px; object-fit: contain; display: block; border: 1px solid #D6DEE8; border-radius: 6px; background: #fff; }
.mr-thumb span { display: block; margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mr-ph-token { display: inline-block; padding: 1px 7px; margin: 0 2px; border-radius: 4px; background: #DBEAFE; border: 1px solid #93C5FD; color: #1D4ED8; font-size: 12px; font-weight: 650; white-space: nowrap; vertical-align: baseline; }
.mr-missing-image { display: inline-block; padding: 1px 7px; border-radius: 4px; background: #FEE2E2; color: #B91C1C; font-size: 12px; }
.mr-inline-image { display: block; width: fit-content; max-width: 100%; margin: 8px 0; cursor: zoom-in; }
.mr-inline-image img { display: block; max-width: min(100%, 720px); max-height: 420px; object-fit: contain; border: 1px solid #D6DEE8; border-radius: 6px; background: #fff; }
.mr-lightbox { display: none; position: fixed; inset: 0; z-index: 9999; background: rgba(15, 23, 42, 0.86); align-items: center; justify-content: center; padding: 24px; cursor: zoom-out; }
.mr-lightbox:target { display: flex; }
.mr-lightbox img { max-width: 92vw; max-height: 88vh; width: auto; height: auto; object-fit: contain; border-radius: 8px; background: #fff; box-shadow: 0 20px 60px rgba(0,0,0,0.35); }
.mr-lightbox-close { position: fixed; top: 18px; right: 24px; color: #fff; font-size: 32px; line-height: 1; font-weight: 700; }
.mr-result-head span { font-size: 15px; }
.mr-result-title { padding: 7px 12px; border-bottom: 1px solid #E5E7EB; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #EEF2F7; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
"""


def t(key, **kwargs):
    text = LANG[_current_lang].get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text


_processor = None
_cached_db = None
_cached_dm = None
_retriever = None
_managers_lock = threading.Lock()


def _get_managers():
    global _cached_db, _cached_dm
    with _managers_lock:
        if _cached_db is None:
            try:
                print("[WebUI] 正在初始化数据库连接...")
                _cached_db = DatabaseManager()
                _cached_db.init_database()
                _cached_dm = DocumentManager(db=_cached_db)
                print("[WebUI] 数据库连接初始化完成")
            except Exception as e:
                print(f"[WebUI] 数据库连接失败: {e}")
                raise
    return _cached_db, _cached_dm


def _reset_managers():
    global _cached_db, _cached_dm, _processor, _retriever
    with _managers_lock:
        if _cached_db is not None:
            try:
                _cached_db.close()
            except Exception:
                pass
        _cached_db = None
        _cached_dm = None
        _processor = None
        _retriever = None


def _reset_retriever():
    global _retriever
    with _retriever_lock:
        _retriever = None


def _get_processor():
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor


def _get_paths_from_table(table_data):
    if table_data is None:
        return []
    if hasattr(table_data, 'empty') and table_data.empty:
        return []
    if len(table_data) == 0:
        return []
    if isinstance(table_data, list):
        return [row[0] for row in table_data]
    return table_data.iloc[:, 0].tolist()


def _get_new_paths_from_table(table_data):
    if table_data is None:
        return []
    if hasattr(table_data, 'empty') and table_data.empty:
        return []
    if len(table_data) == 0:
        return []
    if isinstance(table_data, list):
        return [row[0] for row in table_data if len(row) > 2 and "新文档" in str(row[2])]
    if table_data.shape[1] < 3:
        return table_data.iloc[:, 0].tolist()
    return table_data[table_data.iloc[:, 2].astype(str).str.contains("新文档")].iloc[:, 0].tolist()


def scan_server_folder(folder_path):
    if not folder_path:
        return t("enter_folder"), []
    folder_path = os.path.abspath(folder_path.strip())
    if not os.path.isdir(folder_path):
        return t("path_not_exist"), []
    print(f"[WebUI] 扫描文件夹: {folder_path}")

    doc_files = []
    for root, dirs, files in os.walk(folder_path):
        for f in sorted(files):
            if f.endswith('.md') or f.endswith('.docx'):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, folder_path)
                doc_files.append((full_path, rel_path))

    if not doc_files:
        return t("no_md_files") + f": {folder_path}", []

    rows = []
    try:
        db, _ = _get_managers()
    except Exception:
        db = None

    for full_path, rel_path in doc_files:
        doc_name = os.path.splitext(os.path.basename(full_path))[0]
        exists = False
        if db is not None:
            try:
                cfg = get_config()
                exists = db.get_corpus_by_path(full_path, cfg.to_relative_path(full_path)) is not None
            except Exception:
                exists = False
        status_mark = "✅已入库" if exists else "⏩新文档"
        rows.append([full_path, rel_path, status_mark])

    new_count = sum(1 for r in rows if "新文档" in r[2])
    exist_count = sum(1 for r in rows if "已入库" in r[2])
    print(f"[WebUI] 扫描完成，找到 {len(doc_files)} 个文件（新 {new_count}，已入库 {exist_count}）")
    info = t("found_files", count=len(doc_files), new=new_count, exist=exist_count)
    return info, rows


_pending_chunks = []


def _render_preview_html(chunks_data):
    if not chunks_data:
        return "<p style='color:#888;'>无分片数据</p>"

    doc_groups = {}
    for i, chunk in enumerate(chunks_data):
        doc_name = chunk.get("doc_name", "未知文档")
        if doc_name not in doc_groups:
            doc_groups[doc_name] = []
        doc_groups[doc_name].append((i, chunk))

    html = ["<div class='mr-overview'>"]
    for doc_idx, (doc_name, doc_chunks) in enumerate(doc_groups.items()):
        html.append(f"<div class='mr-doc-group'>")
        html.append(f"<div class='mr-doc-head'>")
        html.append(f"<span>{html_lib.escape(doc_name)}</span>")
        html.append(f"<span>{len(doc_chunks)} 个分片</span>")
        html.append(f"</div>")

        for chunk_idx, chunk in doc_chunks:
            i = chunk_idx
            title = chunk.get("title", "")
            content = chunk.get("content", "")
            images_rel = chunk.get("images", [])
            original_content = chunk.get("original_content", "")
            original_images = chunk.get("original_images", [])
            modified = (original_content and original_content != content) or (original_images and original_images != images_rel)

            html.append(f"<div class='mr-chunk-row'>")
            html.append(f"<div class='mr-chunk-title'><span>分片 {i+1}</span>")
            if modified:
                html.append(f" <span class='mr-status'>已修改</span>")
            if title:
                html.append(f"<small>{html_lib.escape(title)}</small>")
            html.append(f"</div>")
            if images_rel:
                html.append(f"<div class='mr-chunk-meta'>{len(images_rel)} 张图片</div>")
            snippet = re.sub(r'\s+', ' ', content).strip()
            html.append(f"<div class='mr-snippet'>{html_lib.escape(snippet[:180])}{'...' if len(snippet) > 180 else ''}</div>")
            html.append(f"</div>")

        html.append(f"</div>")
    html.append("</div>")
    return "".join(html)


def _render_single_chunk_preview(chunk):
    if not chunk:
        return "<p style='color:#888;'>请选择分片</p>"
    cfg = get_config()
    images = [cfg.to_absolute_path(p) for p in chunk.get("images", [])]
    file_dir = os.path.dirname(chunk.get("file_path", "")) if chunk.get("file_path") else None
    title = chunk.get("title", "")
    html = ["<div class='mr-current-preview'>"]
    html.append("<div class='mr-current-head'>")
    html.append(f"<strong>{html_lib.escape(chunk.get('doc_name', '未知文档'))}</strong>")
    if title:
        html.append(f"<span>{html_lib.escape(title)}</span>")
    html.append("</div>")
    html.append("<div class='mr-current-body'>")
    html.append(_render_content_html(chunk.get("content", ""), images, file_dir, show_placeholder_tags=True))
    html.append("</div>")
    if images:
        html.append("<div class='mr-current-images'>")
        for idx, img_path in enumerate(images):
            b64 = _image_to_base64(img_path)
            if not b64:
                continue
            html.append(_render_zoomable_image(b64, f"图片{idx}", img_path, idx, "mr-thumb"))
        html.append("</div>")
    html.append("</div>")
    return "".join(html)


def _render_chunk_images_html(chunk):
    if not chunk:
        return "<div class='mr-current-images mr-empty-images'>当前分片无图片</div>"
    cfg = get_config()
    images = [cfg.to_absolute_path(p) for p in chunk.get("images", [])]
    valid_images = [(idx, img_path, _image_to_base64(img_path)) for idx, img_path in enumerate(images)]
    valid_images = [(idx, img_path, b64) for idx, img_path, b64 in valid_images if b64]
    if not valid_images:
        return "<div class='mr-current-images mr-empty-images'>当前分片无图片</div>"
    html = ["<div class='mr-current-images'>"]
    for idx, img_path, b64 in valid_images:
        name = html_lib.escape(os.path.basename(img_path))
        html.append(_render_zoomable_image(b64, f"图片{idx} {name}", img_path, idx, "mr-thumb"))
    html.append("</div>")
    return "".join(html)


def preview_chunks(table_data, chunk_strategy, chunk_size, overlap_ratio):
    global _pending_chunks
    file_paths = _get_new_paths_from_table(table_data)
    if not file_paths:
        return "<p style='color:#888;'>请先扫描文件</p>", gr.update(visible=False), gr.update(visible=False), gr.update(choices=[], value=None), "", "<div class='mr-current-images mr-empty-images'>当前分片无图片</div>", json.dumps([])

    processor = _get_processor()
    processor.chunk_strategy = chunk_strategy
    processor.chunk_size = int(chunk_size) if chunk_size else 512
    processor.overlap_ratio = float(overlap_ratio) if overlap_ratio else 0.1

    if processor.append_mode:
        processor.load_existing_index()

    _pending_chunks = []

    for file_path in file_paths:
        doc_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            content, doc_format, images, image_map = processor.read_file(file_path)
            if not content.strip():
                continue

            if chunk_strategy == "length":
                chunks_content = processor.split_by_length(content, processor.chunk_size, processor.overlap_ratio)
                sections = []
                for idx, chunk_content in enumerate(chunks_content):
                    chunk_images = processor._extract_images_from_content(chunk_content, file_path, images, doc_format, image_map)
                    sections.append({
                        "title": f"分片 {idx + 1}",
                        "title_level": 1,
                        "section_level": 1,
                        "content": chunk_content,
                        "images": chunk_images,
                        "title_path": ""
                    })
            else:
                if doc_format == 'markdown':
                    sections = processor.parse_markdown(content, file_path, images)
                elif doc_format == 'word':
                    sections = processor.parse_word(content, file_path, images, image_map)
                else:
                    continue

            if not sections:
                continue

            for section in sections:
                chunk_text = section.get("content", "")
                if not chunk_text.strip():
                    continue
                _pending_chunks.append({
                    "doc_name": doc_name,
                    "file_path": file_path,
                    "title": section.get("title", ""),
                    "title_level": section.get("title_level", 0),
                    "content": chunk_text,
                    "original_content": chunk_text,
                    "images": section.get("images", []),
                    "original_images": list(section.get("images", [])),
                    "title_path": section.get("title_path", ""),
                })
        except Exception as e:
            print(f"  [预览] 读取 {doc_name} 失败: {e}")

    if not _pending_chunks:
        return "<p style='color:#888;'>未能生成任何分片</p>", gr.update(visible=False), gr.update(visible=False), gr.update(choices=[], value=None), "", "<div class='mr-current-images mr-empty-images'>当前分片无图片</div>", json.dumps([])

    preview_html = _render_preview_html(_pending_chunks)
    chunk_choices = []
    for i, chunk in enumerate(_pending_chunks):
        doc_name = chunk.get("doc_name", "未知文档")
        title = chunk.get("title", "")
        label = f"{doc_name} / 分片{i+1}"
        if title:
            label += f": {title}"
        chunk_choices.append(label)

    return (
        preview_html,
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(choices=chunk_choices, value=chunk_choices[0] if chunk_choices else None),
        _pending_chunks[0]["content"] if _pending_chunks else "",
        _render_chunk_images_html(_pending_chunks[0] if _pending_chunks else None),
        json.dumps(_pending_chunks, ensure_ascii=False),
    )


def _parse_chunk_idx(chunk_selection):
    if not chunk_selection:
        return -1
    match = re.search(r'分片(\d+)', chunk_selection)
    if match:
        return int(match.group(1)) - 1
    prefix = str(chunk_selection).split(":", 1)[0]
    nums = re.findall(r'(\d+)', prefix)
    if nums:
        return int(nums[-1]) - 1
    return -1


def _abs_images(image_list):
    cfg = get_config()
    return [cfg.to_absolute_path(p) for p in (image_list or [])]


def _sync_images_after_text_edit(chunk, new_content):
    old_content = chunk.get("content", "")
    old_images = list(chunk.get("images", []) or [])
    old_ph = [m.group(0) for m in _PLACEHOLDER_PATTERN.finditer(old_content)]
    new_ph = [m.group(0) for m in _PLACEHOLDER_PATTERN.finditer(new_content)]

    if new_ph:
        by_placeholder = {}
        for idx, ph in enumerate(old_ph):
            if idx < len(old_images):
                by_placeholder[ph] = old_images[idx]
        next_images = []
        for idx, ph in enumerate(new_ph):
            img = by_placeholder.get(ph)
            if img is None and idx < len(old_images):
                img = old_images[idx]
            if img and img not in next_images:
                next_images.append(img)
        chunk["images"] = next_images
        return

    inline_images = sorted(_iter_inline_image_matches(new_content), key=lambda item: item[1])
    if inline_images:
        cfg = get_config()
        file_dir = os.path.dirname(chunk.get("file_path", "")) if chunk.get("file_path") else ""
        next_images = []
        for _kind, _start, _end, _match, img_path, _label in inline_images:
            abs_path = _resolve_image_ref(img_path, file_dir)
            rel = cfg.to_relative_path(abs_path)
            if rel not in next_images:
                next_images.append(rel)
        chunk["images"] = next_images
        return

    chunk["images"] = []


def _on_chunk_selected(chunk_selection, chunks_json):
    try:
        chunks_data = json.loads(chunks_json)
    except Exception:
        return "", "<div class='mr-current-images mr-empty-images'>当前分片无图片</div>", gr.update(choices=[], value=None)
    if not chunk_selection or not chunks_data:
        return "", "<div class='mr-current-images mr-empty-images'>当前分片无图片</div>", gr.update(choices=[], value=None)
    try:
        idx = _parse_chunk_idx(chunk_selection)
    except (ValueError, IndexError):
        return "", "<div class='mr-current-images mr-empty-images'>当前分片无图片</div>", gr.update(choices=[], value=None)
    if 0 <= idx < len(chunks_data):
        chunk = chunks_data[idx]
        img_choices = [f"图片{i}" for i in range(len(chunk.get("images", [])))]
        return chunk.get("content", ""), _render_chunk_images_html(chunk), gr.update(choices=img_choices, value=img_choices[0] if img_choices else None)
    return "", "<div class='mr-current-images mr-empty-images'>当前分片无图片</div>", gr.update(choices=[], value=None)


def _on_next_chunk(chunk_selection, chunks_json):
    try:
        chunks_data = json.loads(chunks_json)
    except Exception:
        return gr.update(), "", "<div class='mr-current-images mr-empty-images'>当前分片无图片</div>", gr.update(choices=[], value=None)
    if not chunks_data:
        return gr.update(), "", "<div class='mr-current-images mr-empty-images'>当前分片无图片</div>", gr.update(choices=[], value=None)
    idx = _parse_chunk_idx(chunk_selection)
    next_idx = min(max(idx, -1) + 1, len(chunks_data) - 1)
    chunk = chunks_data[next_idx]
    label = f"{chunk.get('doc_name', '未知文档')} / 分片{next_idx + 1}"
    if chunk.get("title"):
        label += f": {chunk.get('title')}"
    img_choices = [f"图片{i}" for i in range(len(chunk.get("images", [])))]
    return gr.update(value=label), chunk.get("content", ""), _render_chunk_images_html(chunk), gr.update(choices=img_choices, value=img_choices[0] if img_choices else None)


def _on_update_chunk(chunk_selection, new_content, chunks_json):
    try:
        chunks_data = json.loads(chunks_json)
    except Exception:
        return chunks_json, "<p style='color:#888;'>更新失败</p>", "<div class='mr-current-images mr-empty-images'>更新失败</div>", gr.update(choices=[], value=None)
    if not chunk_selection or not chunks_data:
        return chunks_json, "<p style='color:#888;'>更新失败</p>", "<div class='mr-current-images mr-empty-images'>更新失败</div>", gr.update(choices=[], value=None)
    try:
        idx = _parse_chunk_idx(chunk_selection)
    except (ValueError, IndexError):
        return chunks_json, "<p style='color:#888;'>更新失败</p>", "<div class='mr-current-images mr-empty-images'>更新失败</div>", gr.update(choices=[], value=None)
    if 0 <= idx < len(chunks_data):
        _sync_images_after_text_edit(chunks_data[idx], new_content)
        chunks_data[idx]["content"] = new_content
        preview_html = _render_preview_html(chunks_data)
        img_choices = [f"图片{i}" for i in range(len(chunks_data[idx].get("images", [])))]
        return json.dumps(chunks_data, ensure_ascii=False), preview_html, _render_chunk_images_html(chunks_data[idx]), gr.update(choices=img_choices, value=img_choices[0] if img_choices else None)
    return chunks_json, "<p style='color:#888;'>更新失败</p>", "<div class='mr-current-images mr-empty-images'>更新失败</div>", gr.update(choices=[], value=None)


def _on_revert_chunk(chunk_selection, chunks_json):
    try:
        chunks_data = json.loads(chunks_json)
    except Exception:
        return chunks_json, "<p style='color:#888;'>还原失败</p>", "", "<div class='mr-current-images mr-empty-images'>还原失败</div>", gr.update(choices=[], value=None)
    if not chunk_selection or not chunks_data:
        return chunks_json, "<p style='color:#888;'>还原失败</p>", "", "<div class='mr-current-images mr-empty-images'>还原失败</div>", gr.update(choices=[], value=None)
    try:
        idx = _parse_chunk_idx(chunk_selection)
    except (ValueError, IndexError):
        return chunks_json, "<p style='color:#888;'>还原失败</p>", "", "<div class='mr-current-images mr-empty-images'>还原失败</div>", gr.update(choices=[], value=None)
    if 0 <= idx < len(chunks_data):
        original = chunks_data[idx].get("original_content")
        if original is not None:
            chunks_data[idx]["content"] = original
        original_images = chunks_data[idx].get("original_images")
        if original_images is not None:
            chunks_data[idx]["images"] = list(original_images)
        preview_html = _render_preview_html(chunks_data)
        img_choices = [f"图片{i}" for i in range(len(chunks_data[idx].get("images", [])))]
        return json.dumps(chunks_data, ensure_ascii=False), preview_html, chunks_data[idx]["content"], _render_chunk_images_html(chunks_data[idx]), gr.update(choices=img_choices, value=img_choices[0] if img_choices else None)
    return chunks_json, "<p style='color:#888;'>还原失败</p>", "", "<div class='mr-current-images mr-empty-images'>还原失败</div>", gr.update(choices=[], value=None)


def _on_delete_chunk_image(chunk_selection, image_index, chunks_json):
    try:
        chunks_data = json.loads(chunks_json)
    except Exception:
        return chunks_json, "<p style='color:#888;'>删除失败</p>", "", "<div class='mr-current-images mr-empty-images'>删除失败</div>", gr.update(choices=[], value=None)
    if not chunk_selection or not chunks_data:
        return chunks_json, "<p style='color:#888;'>删除失败</p>", "", "<div class='mr-current-images mr-empty-images'>删除失败</div>", gr.update(choices=[], value=None)
    try:
        idx = _parse_chunk_idx(chunk_selection)
    except (ValueError, IndexError):
        return chunks_json, "<p style='color:#888;'>删除失败</p>", "", "<div class='mr-current-images mr-empty-images'>删除失败</div>", gr.update(choices=[], value=None)
    if 0 <= idx < len(chunks_data):
        images = chunks_data[idx].get("images", [])
        content = chunks_data[idx].get("content", "")
        img_idx = -1
        if isinstance(image_index, str) and image_index.startswith("图片"):
            try:
                img_idx = int(image_index.replace("图片", ""))
            except ValueError:
                img_idx = -1
        elif isinstance(image_index, str):
            nums = re.findall(r'\d+', image_index)
            img_idx = int(nums[-1]) if nums else -1
        elif isinstance(image_index, (int, float)):
            img_idx = int(image_index)
        if 0 <= img_idx < len(images):
            images.pop(img_idx)
            chunks_data[idx]["images"] = images

            if _PLACEHOLDER_PATTERN.search(content):
                placeholders = list(_PLACEHOLDER_PATTERN.finditer(content))
                if img_idx < len(placeholders):
                    target_ph = placeholders[img_idx].group(0)
                    chunks_data[idx]["content"] = content.replace(target_ph, "")
            else:
                inline_imgs = sorted(_iter_inline_image_matches(content), key=lambda item: item[1])
                if img_idx < len(inline_imgs):
                    _kind, start, end, _match, _img_path, _label = inline_imgs[img_idx]
                    content = content[:start] + content[end:]
                    chunks_data[idx]["content"] = content

        preview_html = _render_preview_html(chunks_data)
        img_choices = [f"图片{i}" for i in range(len(chunks_data[idx].get("images", [])))]
        return json.dumps(chunks_data, ensure_ascii=False), preview_html, chunks_data[idx]["content"], _render_chunk_images_html(chunks_data[idx]), gr.update(choices=img_choices, value=img_choices[0] if img_choices else None)
    return chunks_json, "<p style='color:#888;'>删除失败</p>", "", "<div class='mr-current-images mr-empty-images'>删除失败</div>", gr.update(choices=[], value=None)


def confirm_import(chunks_json, chunk_strategy, chunk_size, overlap_ratio):
    global _pending_chunks
    try:
        chunks_data = json.loads(chunks_json)
    except Exception:
        return "❌ 无可入库的分片数据"
    if not chunks_data:
        return "❌ 无可入库的分片数据"

    processor = _get_processor()
    processor.chunk_strategy = chunk_strategy
    processor.chunk_size = int(chunk_size) if chunk_size else 512
    processor.overlap_ratio = float(overlap_ratio) if overlap_ratio else 0.1

    if processor.append_mode:
        processor.load_existing_index()

    results = []
    all_chunks = []
    doc_names_added = {}
    config = get_config()

    for chunk in chunks_data:
        file_path = chunk["file_path"]
        doc_name = chunk["doc_name"]

        if doc_name not in doc_names_added:
            if processor._is_document_exists(file_path):
                results.append(f"⏭ 文档已存在，跳过: {doc_name}")
                continue
            corpus = Corpus(
                file_path=file_path,
                name=doc_name,
                type="文档",
                data_summary=chunk.get("content", "")[:500],
                source="Web审核入库",
                relative_path=config.to_relative_path(file_path),
                chunk_strategy=chunk_strategy,
            )
            corpus_id = processor.db.add_corpus(corpus)
            doc_names_added[doc_name] = {"corpus_id": corpus_id, "chunk_count": 0}
        else:
            corpus_id = doc_names_added[doc_name]["corpus_id"]

        doc_names_added[doc_name]["chunk_count"] += 1
        chunk_index = doc_names_added[doc_name]["chunk_count"] - 1

        title_path = chunk.get("title_path", "")
        title = chunk.get("title", "")
        if title_path:
            chunk_text = f"{doc_name} > {title_path} > {title}\n{chunk['content']}"
        elif title:
            chunk_text = f"{doc_name} > {title}\n{chunk['content']}"
        else:
            chunk_text = f"{doc_name}\n{chunk['content']}"

        all_chunks.append({
            "corpus_id": corpus_id,
            "chunk_index": chunk_index,
            "content": chunk["content"],
            "title": chunk.get("title", ""),
            "title_level": chunk.get("title_level", 0),
            "images": chunk.get("images", []),
            "title_path": title_path,
            "chunk_text": chunk_text,
            "original_content": chunk.get("original_content", chunk["content"]),
        })

    for doc_name, info in doc_names_added.items():
        processor.db.update_corpus_chunk_count(info["corpus_id"], info["chunk_count"])

    if not all_chunks:
        return "❌ 无有效分片可入库"

    try:
        chunk_texts = [c["chunk_text"] for c in all_chunks]
        dense_vecs, sparse_vecs = processor.generate_embeddings(chunk_texts)

        vector_id = processor.existing_vector_count

        for chunk in tqdm(all_chunks, desc="保存分片"):
            chunk_record = Chunk(
                corpus_id=chunk["corpus_id"],
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                title=chunk["title"],
                title_level=chunk["title_level"],
                images=chunk["images"],
                vector_id=vector_id,
                original_content=chunk.get("original_content"),
            )
            processor.db.add_chunk(chunk_record)
            corpus_info = processor.db.get_corpus_by_id(chunk["corpus_id"])
            processor.chunks_meta.append({
                "corpus_id": chunk["corpus_id"],
                "doc_name": os.path.basename(corpus_info.file_path) if corpus_info else "unknown",
                "title": chunk["title"],
                "title_level": chunk["title_level"],
                "images": chunk["images"],
            })
            processor.chunk_contents.append(chunk["chunk_text"])
            processor.chunk_images.append(chunk["images"])
            vector_id += 1

        if processor.append_mode and processor.existing_vector_count > 0:
            processor.dense_vectors = np.vstack([
                processor.faiss_index.reconstruct_n(0, processor.existing_vector_count),
                dense_vecs
            ])
            processor.sparse_vectors = processor.sparse_vectors + sparse_vecs
        else:
            processor.dense_vectors = dense_vecs
            processor.sparse_vectors = sparse_vecs

        processor.sparse_index = processor.build_sparse_index(processor.sparse_vectors)

        print("正在构建 BM25 索引...")
        if processor.append_mode and processor.existing_vector_count > 0:
            existing_bm25 = BM25Indexer(processor.faiss_index_path)
            if existing_bm25.load():
                new_ids = list(range(processor.existing_vector_count, processor.existing_vector_count + len(all_chunks)))
                existing_bm25.append_documents(chunk_texts, new_ids)
                processor.bm25_indexer = existing_bm25
            else:
                all_ids = list(range(len(processor.chunk_contents)))
                processor.bm25_indexer.build_index(processor.chunk_contents, all_ids)
        else:
            all_ids = list(range(len(processor.chunk_contents)))
            processor.bm25_indexer.build_index(processor.chunk_contents, all_ids)

        dimension = processor.dense_vectors.shape[1]
        processor.faiss_index = faiss.IndexFlatIP(dimension)
        processor.faiss_index.add(processor.dense_vectors)
        processor.save_faiss_index()
        _reset_retriever()

        results.append(f"✅ 入库完成: {len(all_chunks)} 个分片, {len(doc_names_added)} 个文档, {processor.faiss_index.ntotal} 条向量")
    except Exception as e:
        for info in doc_names_added.values():
            try:
                processor.db.delete_corpus(info["corpus_id"])
            except Exception:
                pass
        results.append(f"❌ 向量索引更新失败: {str(e)}")

    _pending_chunks = []
    return "\n".join(results)


def cancel_import():
    global _pending_chunks
    _pending_chunks = []
    return (
        "<p style='color:#888;'>已取消，无待入库数据</p>",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(choices=[], value=None),
        "",
        "<div class='mr-current-images mr-empty-images'>当前分片无图片</div>",
        json.dumps([]),
    )


def do_import(table_data, chunk_strategy, chunk_size, overlap_ratio):
    file_paths = _get_new_paths_from_table(table_data)
    if not file_paths:
        return t("scan_files_first")
    print(f"[WebUI] 入库: {len(file_paths)} 个文件")

    processor = _get_processor()
    processor.chunk_strategy = chunk_strategy
    processor.chunk_size = int(chunk_size) if chunk_size else 512
    processor.overlap_ratio = float(overlap_ratio) if overlap_ratio else 0.1

    if processor.append_mode:
        processor.load_existing_index()

    results = []
    all_chunks = []

    for file_path in file_paths:
        doc_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            result = processor.process_document_web(
                file_path=file_path,
                tags=None,
            )
            if result["success"]:
                results.append(f"✅ {result['message']}")
                if result.get("chunks"):
                    all_chunks.extend(result["chunks"])
            else:
                results.append(f"⏭ {result['message']}")
        except Exception as e:
            results.append(f"❌ {doc_name}: {str(e)}")

    if all_chunks:
        try:
            chunk_texts = [c["chunk_text"] for c in all_chunks]
            dense_vecs, sparse_vecs = processor.generate_embeddings(chunk_texts)

            vector_id = processor.existing_vector_count

            for chunk in tqdm(all_chunks, desc="保存分片"):
                chunk_record = Chunk(
                    corpus_id=chunk["corpus_id"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"],
                    title=chunk["title"],
                    title_level=chunk["title_level"],
                    images=chunk["images"],
                    vector_id=vector_id,
                )

                processor.db.add_chunk(chunk_record)
                corpus_info = processor.db.get_corpus_by_id(chunk["corpus_id"])

                processor.chunks_meta.append({
                    "corpus_id": chunk["corpus_id"],
                    "doc_name": os.path.basename(corpus_info.file_path) if corpus_info else "unknown",
                    "title": chunk["title"],
                    "title_level": chunk["title_level"],
                    "images": chunk["images"],
                })
                processor.chunk_contents.append(chunk["chunk_text"])
                processor.chunk_images.append(chunk["images"])
                vector_id += 1

            if processor.append_mode and processor.existing_vector_count > 0:
                processor.dense_vectors = np.vstack([
                    processor.faiss_index.reconstruct_n(0, processor.existing_vector_count),
                    dense_vecs
                ])
                processor.sparse_vectors = processor.sparse_vectors + sparse_vecs
            else:
                processor.dense_vectors = dense_vecs
                processor.sparse_vectors = sparse_vecs

            processor.sparse_index = processor.build_sparse_index(processor.sparse_vectors)

            print("正在构建 BM25 索引...")
            if processor.append_mode and processor.existing_vector_count > 0:
                existing_bm25 = BM25Indexer(processor.faiss_index_path)
                if existing_bm25.load():
                    new_ids = list(range(processor.existing_vector_count, processor.existing_vector_count + len(all_chunks)))
                    existing_bm25.append_documents(chunk_texts, new_ids)
                    processor.bm25_indexer = existing_bm25
                else:
                    all_ids = list(range(len(processor.chunk_contents)))
                    processor.bm25_indexer.build_index(processor.chunk_contents, all_ids)
            else:
                all_ids = list(range(len(processor.chunk_contents)))
                processor.bm25_indexer.build_index(processor.chunk_contents, all_ids)

            dimension = processor.dense_vectors.shape[1]
            processor.faiss_index = faiss.IndexFlatIP(dimension)
            processor.faiss_index.add(processor.dense_vectors)
            processor.save_faiss_index()
            _reset_retriever()

            results.append(f"\n📊 向量索引已更新: 共 {len(all_chunks)} 个分片, {processor.faiss_index.ntotal} 条向量")
        except Exception as e:
            for chunk in all_chunks:
                try:
                    processor.db.delete_corpus(chunk["corpus_id"])
                except Exception:
                    pass
            results.append(f"\n❌ 向量索引更新失败: {str(e)}")

    return "\n".join(results)


def load_documents(keyword="", status_filter="全部"):
    global _cached_docs_data
    try:
        _, dm = _get_managers()
    except Exception as e:
        return [[f"{t('db_conn_fail')}: {e}", "", ""]]
    docs = dm.list_documents(limit=500)
    if keyword:
        docs = [d for d in docs if keyword.lower() in d.get("name", "").lower()]

    filter_active = status_filter in ("启用", "Active")
    filter_inactive = status_filter in ("停用", "Inactive")

    rows = []
    for doc in docs:
        is_active = doc.get("is_active", True)
        if filter_active and not is_active:
            continue
        if filter_inactive and is_active:
            continue
        rows.append([
            doc["id"],
            doc["name"][:80],
            t("active_mark") if is_active else t("inactive_mark"),
        ])
    _cached_docs_data = rows
    return rows


def delete_doc(corpus_id):
    if not corpus_id:
        return t("enter_corpus_id")
    try:
        _, dm = _get_managers()
    except Exception as e:
        return f"{t('db_conn_fail')}: {e}"
    faiss_path = get_config().faiss_index_path
    success = dm.delete_document(corpus_id, confirm=False, faiss_index_path=faiss_path)
    if success:
        _reset_retriever()
    return f"{t('deleted')}: {corpus_id}" if success else f"{t('delete_fail')}: {corpus_id}"


def toggle_doc_status(corpus_id):
    if not corpus_id:
        return t("enter_corpus_id")
    try:
        db, dm = _get_managers()
    except Exception as e:
        return f"{t('db_conn_fail')}: {e}"
    doc = dm.get_document_detail(corpus_id)
    if not doc:
        return f"{t('doc_not_found')}: {corpus_id}"
    new_status = not doc.get("is_active", True)
    db.toggle_corpus_active(corpus_id)
    status_text = t("enabled") if new_status else t("disabled")
    return f"{status_text}: {corpus_id}"


def load_stats():
    try:
        _, dm = _get_managers()
    except Exception as e:
        return f"{t('db_conn_fail')}: {e}"
    stats = dm.get_statistics()
    lines = [
        f"{t('total_docs')}: {stats.get('corpus', 0)}",
        f"{t('total_chunks')}: {stats.get('chunks', 0)}",
    ]
    return "\n".join(lines)


_retriever_lock = threading.Lock()
_cached_docs_data = []


def _get_retriever():
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            from scripts.retrieval_pipeline import RetrievalPipeline
            _retriever = RetrievalPipeline()
    return _retriever


def _image_to_base64(img_path):
    if _is_remote_image_path(img_path):
        return img_path.strip()
    if not img_path or not os.path.isfile(img_path):
        return None
    mime_type, _ = mimetypes.guess_type(img_path)
    if not mime_type:
        mime_type = "image/png"
    try:
        with open(img_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{data}"
    except Exception:
        return None


def _image_lightbox_id(source_key, idx):
    raw = f"{source_key}:{idx}".encode("utf-8", errors="ignore")
    return "mr-img-" + hashlib.md5(raw).hexdigest()[:12]


def _render_zoomable_image(b64, label, source_key, idx, link_class):
    safe_label = html_lib.escape(label)
    safe_src = html_lib.escape(b64, quote=True)
    box_id = _image_lightbox_id(source_key, idx)
    return (
        f"<a href='#{box_id}' class='{link_class}' title='点击放大'>"
        f"<img src='{safe_src}' alt='{safe_label}' />"
        f"{f'<span>{safe_label}</span>' if link_class == 'mr-thumb' else ''}"
        f"</a>"
        f"<a href='#' id='{box_id}' class='mr-lightbox' title='点击关闭'>"
        f"<span class='mr-lightbox-close'>×</span>"
        f"<img src='{safe_src}' alt='{safe_label}' />"
        f"</a>"
    )


_IMG_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_HTML_IMG_PATTERN = re.compile(r'<img\b[^>]*\bsrc=["\']?([^"\'\s>]+)[^>]*>', re.IGNORECASE)
_PLACEHOLDER_PATTERN = re.compile(r'<<IMAGE:([0-9a-f]+)>>')


def _is_remote_image_path(path):
    parsed = urlparse((path or "").strip())
    return parsed.scheme in {"http", "https", "data"}


def _clean_image_ref(img_ref):
    img_ref = (img_ref or "").strip()
    if img_ref.startswith("<") and img_ref.endswith(">"):
        img_ref = img_ref[1:-1].strip()
    return img_ref


def _resolve_image_ref(img_path, file_dir=None):
    img_path = _clean_image_ref(img_path)
    if _is_remote_image_path(img_path):
        return img_path
    if not os.path.isabs(img_path) and file_dir:
        return os.path.normpath(os.path.join(file_dir, img_path))
    return img_path


def _iter_inline_image_matches(content):
    for m in _IMG_PATTERN.finditer(content):
        yield ("md_img", m.start(), m.end(), m, m.group(2), m.group(1) or m.group(2))
    for m in _HTML_IMG_PATTERN.finditer(content):
        yield ("html_img", m.start(), m.end(), m, m.group(1), m.group(1))


def _render_content_html(content, images=None, file_dir=None, show_placeholder_tags=False):
    parts = []
    last_end = 0

    combined = []
    ph_counter = 0
    for m in _PLACEHOLDER_PATTERN.finditer(content):
        combined.append(('placeholder', m.start(), m.end(), m, ph_counter))
        ph_counter += 1
    md_img_counter = 0
    for kind, start, end, m, _img_path, _label in _iter_inline_image_matches(content):
        combined.append((kind, start, end, m, md_img_counter))
        md_img_counter += 1
    combined.sort(key=lambda x: x[1])

    for item in combined:
        kind, start, end, match, idx = item
        if start < last_end:
            continue
        text_before = content[last_end:start]
        if text_before:
            escaped = html_lib.escape(text_before).replace("\n", "<br>")
            parts.append(escaped)

        if kind == 'placeholder':
            ph_text = match.group(0)
            if show_placeholder_tags:
                parts.append(
                    f'<span class="mr-ph-token" title="{html_lib.escape(ph_text)}">图片{idx}</span>'
                )
            elif images is not None and 0 <= idx < len(images):
                b64 = _image_to_base64(images[idx])
                if b64:
                    parts.append(_render_zoomable_image(b64, f"图片{idx}", images[idx], idx, "mr-inline-image"))
                else:
                    parts.append(
                        f'<span class="mr-missing-image">{html_lib.escape(ph_text)}(缺失)</span>'
                    )
            elif images is not None:
                parts.append(
                    f'<span class="mr-missing-image">{html_lib.escape(ph_text)}(越界)</span>'
                )
            else:
                parts.append(
                    f'<span class="mr-ph-token">{html_lib.escape(ph_text)}</span>'
                )
        elif kind in ('md_img', 'html_img'):
            if kind == 'md_img':
                alt_text = match.group(1)
                img_path = match.group(2)
            else:
                alt_text = ""
                img_path = match.group(1)
            abs_path = _resolve_image_ref(img_path, file_dir)
            if not _is_remote_image_path(abs_path) and not os.path.isabs(abs_path) and images is not None and idx < len(images):
                abs_path = images[idx]
            b64 = _image_to_base64(abs_path)
            if b64:
                label = html_lib.escape(alt_text or img_path)
                if show_placeholder_tags:
                    parts.append(f"<span class='mr-ph-token' title='{html_lib.escape(img_path)}'>图片{idx}: {label}</span>")
                else:
                    parts.append(_render_zoomable_image(b64, label, abs_path, idx, "mr-inline-image"))
            else:
                parts.append(
                    f'<span class="mr-missing-image">{html_lib.escape(alt_text or img_path)}</span>'
                )

        last_end = end

    remaining = content[last_end:]
    if remaining:
        escaped = html_lib.escape(remaining).replace("\n", "<br>")
        parts.append(escaped)
    return "".join(parts)


def do_retrieve(query, top_k, use_rerank):
    if not query or not query.strip():
        return t("enter_query")
    query = query.strip()
    try:
        top_k = int(top_k) if top_k else 5
    except (ValueError, TypeError):
        top_k = 5

    use_rerank = (use_rerank in ("是", "Yes"))

    try:
        retriever = _get_retriever()
        result = retriever.retrieve(
            query=query,
            top_k=top_k,
            use_rerank=use_rerank,
            verbose=False,
        )

        docs = result.docs
        if not docs:
            return "<p style='color:#888;'>未找到相关文档</p>"

        html_parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("meta", {})
            doc_name = meta.get("doc_name", "Unknown")
            title = meta.get("title", "")
            content = doc.get("content", "")
            images = meta.get("images", [])
            score_info = ""
            if "rerank_score" in doc:
                score_info = f" <span style='color:#888;font-size:0.85em;'>(Rerank: {doc['rerank_score']:.4f})</span>"

            html_parts.append(f"<div class='mr-result-card'>")
            html_parts.append(f"<div class='mr-result-head'><span>{i}. {html_lib.escape(doc_name)}</span>{score_info}</div>")
            if title:
                html_parts.append(f"<div class='mr-result-title'>标题: {html_lib.escape(title)}</div>")
            html_parts.append(f"<div class='mr-result-body'>")
            html_parts.append(_render_content_html(content, images))
            html_parts.append("</div>")
            html_parts.append("</div>")

        return "".join(html_parts)
    except Exception as e:
        import traceback
        return f"<p style='color:red;'>[ERROR] {html_lib.escape(str(e))}</p><pre>{html_lib.escape(traceback.format_exc())}</pre>"


def _on_doc_table_select(evt: gr.SelectData):
    try:
        if evt.index:
            row_idx = evt.index[0]
            docs_data = _cached_docs_data
            if docs_data and row_idx < len(docs_data):
                return str(docs_data[row_idx][0])
    except Exception:
        pass
    return ""


def reset_database_step1():
    global _reset_code
    import secrets
    _reset_code = secrets.token_hex(4).upper()
    warn_text = t("reset_warning_text", code=_reset_code)
    return warn_text, gr.update(visible=True)


def reset_database_step2(confirm_text):
    global _reset_code
    if confirm_text.strip().upper() != _reset_code:
        _reset_code = ""
        return t("confirm_wrong"), gr.update(visible=False)

    _reset_managers()

    try:
        import subprocess
        script_path = os.path.join(WORKSPACE_ROOT, "scripts", "reset_database.py")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=60,
            cwd=WORKSPACE_ROOT,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            error = result.stderr.strip()
            return f"[ERROR] Reset failed (exit code {result.returncode})\n\n{output}\n\n{error}", gr.update(visible=False)
        return output + f"\n\n{t('reset_done')}", gr.update(visible=False)
    except subprocess.TimeoutExpired:
        return t("reset_timeout"), gr.update(visible=False)
    except Exception as e:
        return f"[ERROR] {e}", gr.update(visible=False)


with gr.Blocks(title="RAG 知识库管理", analytics_enabled=False) as app:
    gr.Markdown(t("app_title"))

    with gr.Tabs():
        with gr.TabItem("文档入库"):
            import_desc_md = gr.Markdown(t("import_desc"))

            with gr.Row():
                server_folder = gr.Textbox(
                    label=t("folder_path"),
                    placeholder=t("folder_placeholder"),
                    scale=4,
                )
                scan_btn = gr.Button(t("scan_btn"), variant="secondary", scale=1)

            scan_info = gr.Textbox(label=t("scan_result"), interactive=False)
            server_file_table = gr.Dataframe(
                headers=[t("col_full_path"), t("col_rel_path"), t("col_status")],
                datatype=["str", "str", "str"],
                interactive=False,
                wrap=True,
                label=t("file_list"),
            )

            import_settings_md = gr.Markdown(t("import_settings"))
            with gr.Row():
                chunk_strategy = gr.Dropdown(
                    choices=["title", "length"],
                    value="length",
                    label=t("chunk_strategy"),
                    info=t("chunk_strategy_info")
                )
                with gr.Column(visible=True, scale=1) as length_params_col:
                    chunk_size = gr.Number(
                        value=512,
                        label=t("chunk_size"),
                        info=t("chunk_size_info"),
                    )
                    overlap_ratio = gr.Number(
                        value=0.1,
                        label=t("overlap_ratio"),
                        info=t("overlap_ratio_info"),
                    )

            def on_chunk_strategy_change(strategy):
                return gr.update(visible=(strategy == "length"))

            chunk_strategy.change(
                fn=on_chunk_strategy_change,
                inputs=chunk_strategy,
                outputs=[length_params_col],
            )

            import_btn = gr.Button("👀 预览分片", variant="primary")
            preview_html = gr.HTML(visible=False)
            chunks_state = gr.State("[]")

            with gr.Column(visible=False) as edit_col:
                gr.Markdown("### ✏️ 分片编辑面板")
                with gr.Row():
                    chunk_selector = gr.Dropdown(label="选择分片", choices=[], scale=3)
                    btn_next_chunk = gr.Button("下一片", variant="secondary", scale=1)
                with gr.Row():
                    edit_content = gr.Textbox(label="分片内容（删除 <<IMAGE:xxxx>> 或 Markdown 图片语法后保存，即删除对应图片）", lines=14, scale=3)
                with gr.Row():
                    edit_images_html = gr.HTML(label="当前分片图片")
                with gr.Row():
                    img_delete_idx = gr.Dropdown(label="删除第几张图片", choices=[], scale=1)
                    btn_delete_img = gr.Button("🗑 删除图片", variant="secondary", scale=1)
                    btn_update_chunk = gr.Button("💾 保存修改", variant="primary", scale=1)
                    btn_revert_chunk = gr.Button("↩️ 还原", variant="secondary", scale=1)

            with gr.Row(visible=False) as confirm_row:
                btn_confirm_import = gr.Button("✅ 确认入库", variant="primary")
                btn_cancel_import = gr.Button("❌ 取消", variant="secondary")
            import_log = gr.Textbox(label=t("import_log"), interactive=False, lines=6)

            scan_btn.click(
                fn=scan_server_folder,
                inputs=server_folder,
                outputs=[scan_info, server_file_table],
            )

            import_btn.click(
                fn=preview_chunks,
                inputs=[server_file_table, chunk_strategy, chunk_size, overlap_ratio],
                outputs=[preview_html, edit_col, confirm_row, chunk_selector, edit_content, edit_images_html, chunks_state],
            )

            chunk_selector.change(
                fn=_on_chunk_selected,
                inputs=[chunk_selector, chunks_state],
                outputs=[edit_content, edit_images_html, img_delete_idx],
            )

            btn_next_chunk.click(
                fn=_on_next_chunk,
                inputs=[chunk_selector, chunks_state],
                outputs=[chunk_selector, edit_content, edit_images_html, img_delete_idx],
            )

            btn_update_chunk.click(
                fn=_on_update_chunk,
                inputs=[chunk_selector, edit_content, chunks_state],
                outputs=[chunks_state, preview_html, edit_images_html, img_delete_idx],
            )

            btn_delete_img.click(
                fn=_on_delete_chunk_image,
                inputs=[chunk_selector, img_delete_idx, chunks_state],
                outputs=[chunks_state, preview_html, edit_content, edit_images_html, img_delete_idx],
            )

            btn_revert_chunk.click(
                fn=_on_revert_chunk,
                inputs=[chunk_selector, chunks_state],
                outputs=[chunks_state, preview_html, edit_content, edit_images_html, img_delete_idx],
            )

            btn_confirm_import.click(
                fn=confirm_import,
                inputs=[chunks_state, chunk_strategy, chunk_size, overlap_ratio],
                outputs=import_log,
            )

            btn_cancel_import.click(
                fn=cancel_import,
                outputs=[preview_html, edit_col, confirm_row, chunk_selector, edit_content, edit_images_html, chunks_state],
            )

        with gr.TabItem("文档列表"):
            with gr.Row():
                doc_keyword = gr.Textbox(label=t("keyword_search"), placeholder=t("keyword_placeholder"))
                doc_status = gr.Dropdown(
                    choices=[t("all"), t("active"), t("inactive")],
                    value=t("all"), label=t("status_filter")
                )
                doc_refresh = gr.Button(t("refresh"), variant="secondary")

            doc_table = gr.Dataframe(
                headers=[t("col_id"), t("col_name"), t("col_state")],
                datatype=["str", "str", "str"],
                interactive=False,
                wrap=True,
            )

            with gr.Row():
                doc_id_input = gr.Textbox(label=t("doc_id"), placeholder=t("doc_id_placeholder"))
                btn_toggle = gr.Button(t("toggle_btn"), variant="secondary")
                btn_delete = gr.Button(t("delete_btn"), variant="stop")

            doc_action_msg = gr.Textbox(label=t("action_result"), interactive=False)

            doc_table.select(
                fn=_on_doc_table_select,
                inputs=None,
                outputs=doc_id_input,
            )
            doc_refresh.click(
                fn=load_documents,
                inputs=[doc_keyword, doc_status],
                outputs=doc_table,
            )
            doc_keyword.submit(
                fn=load_documents,
                inputs=[doc_keyword, doc_status],
                outputs=doc_table,
            )
            btn_delete.click(
                fn=delete_doc,
                inputs=doc_id_input,
                outputs=doc_action_msg,
            )
            btn_toggle.click(
                fn=toggle_doc_status,
                inputs=doc_id_input,
                outputs=doc_action_msg,
            )

        with gr.TabItem("检索测试"):
            search_desc_md = gr.Markdown(t("search_title"))
            with gr.Row():
                retrieve_query = gr.Textbox(
                    label=t("query_label"),
                    placeholder=t("query_placeholder"),
                    scale=4,
                )
            with gr.Row():
                retrieve_topk = gr.Slider(minimum=1, maximum=20, value=5, step=1, label=t("top_k"))
                retrieve_rerank = gr.Radio(choices=[t("yes"), t("no")], value=t("yes"), label=t("use_rerank"))
            with gr.Row():
                retrieve_btn = gr.Button(t("search_btn"), variant="primary")
                btn_clear_retrieve = gr.Button(t("clear_btn"), variant="secondary")
            retrieve_result = gr.HTML(label=t("search_result"))

            retrieve_btn.click(
                fn=do_retrieve,
                inputs=[retrieve_query, retrieve_topk, retrieve_rerank],
                outputs=retrieve_result,
            )
            btn_clear_retrieve.click(
                fn=lambda: ("", ""),
                outputs=[retrieve_query, retrieve_result],
            )

        with gr.TabItem("系统管理"):
            with gr.Tabs():
                with gr.TabItem("统计信息"):
                    stats_refresh = gr.Button(t("stats_refresh"), variant="secondary")
                    stats_text = gr.Textbox(label=t("stats_info"), interactive=False, lines=10)
                    stats_refresh.click(fn=load_stats, outputs=stats_text)

                with gr.TabItem("一键删库"):
                    reset_desc_md = gr.Markdown(t("reset_desc"))
                    btn_reset_step1 = gr.Button(t("reset_step1"), variant="stop")
                    reset_warning = gr.Textbox(label=t("reset_warning"), interactive=False)
                    with gr.Row(visible=False) as reset_confirm_row:
                        reset_confirm_text = gr.Textbox(label=t("confirm_code"), placeholder=t("confirm_placeholder"))
                        btn_reset_step2 = gr.Button(t("reset_step2"), variant="stop")
                    reset_result = gr.Textbox(label=t("reset_result"), interactive=False)

                    btn_reset_step1.click(
                        fn=reset_database_step1,
                        outputs=[reset_warning, reset_confirm_row],
                    )
                    btn_reset_step2.click(
                        fn=reset_database_step2,
                        inputs=reset_confirm_text,
                        outputs=[reset_result, reset_confirm_row],
                    )



ACADEMIC_PRIMARY = gr.themes.Color(
    name="minir_blue",
    c50="#EFF6FF", c100="#DBEAFE", c200="#BFDBFE",
    c300="#93C5FD", c400="#60A5FA", c500="#3B82F6",
    c600="#2563EB", c700="#1D4ED8", c800="#1E40AF",
    c900="#1E3A8A", c950="#172554",
)

ACADEMIC_THEME = gr.themes.Default(
    primary_hue=ACADEMIC_PRIMARY,
    secondary_hue="stone",
).set(
    body_background_fill="#F6F8FB",
    body_background_fill_dark="#111827",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F3F6FA",
    border_color_accent="#D6DEE8",
    shadow_spread="0",
)

ACADEMIC_CSS = """
.gradio-container { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', sans-serif !important; }
h1, h2, h3, h4 { color: #111827; letter-spacing: 0; }
h1 { font-size: 1.8rem !important; font-weight: 700 !important; }
h2 { font-size: 1.3rem !important; font-weight: 600 !important; }
h3 { font-size: 1.1rem !important; }
.markdown h1 { border-bottom: 1px solid #E5E7EB; padding-bottom: 8px; }
.tab-nav button { font-weight: 600 !important; }
table thead th { background-color: #F3F6FA !important; color: #111827 !important; font-weight: 600; }
.mr-overview { display: grid; gap: 12px; }
.mr-doc-group { border: 1px solid #D6DEE8; border-radius: 8px; overflow: hidden; background: #fff; }
.mr-doc-head { display: flex; justify-content: space-between; gap: 12px; padding: 10px 12px; background: #F3F6FA; border-bottom: 1px solid #D6DEE8; font-weight: 700; color: #111827; }
.mr-chunk-row { padding: 9px 12px; border-bottom: 1px solid #EEF2F7; }
.mr-chunk-row:last-child { border-bottom: 0; }
.mr-chunk-title { display: flex; align-items: center; justify-content: space-between; gap: 10px; color: #111827; font-weight: 650; }
.mr-chunk-title small { color: #64748B; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mr-status { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; border-radius: 4px; padding: 1px 6px; font-size: 12px; }
.mr-chunk-meta { margin-top: 4px; color: #2563EB; font-size: 12px; }
.mr-snippet { margin-top: 4px; color: #475569; font-size: 13px; line-height: 1.5; }
.mr-current-preview, .mr-result-card { border: 1px solid var(--border-color-primary); border-radius: 8px; background: var(--background-fill-primary); overflow: hidden; margin-bottom: 12px; }
.mr-current-head, .mr-result-head { display: flex; justify-content: space-between; gap: 10px; padding: 10px 12px; background: var(--background-fill-secondary); border-bottom: 1px solid var(--border-color-primary); color: var(--body-text-color); font-weight: 700; }
.mr-current-head span, .mr-result-title { color: var(--body-text-color-subdued); font-size: 13px; font-weight: 500; }
.mr-current-body, .mr-result-body { padding: 12px; max-height: 420px; overflow: auto; background: var(--background-fill-primary); color: var(--body-text-color); font-size: 14px; line-height: 1.7; }
.mr-current-images { display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 12px; border-top: 1px solid #E5E7EB; background: #F8FAFC; }
.mr-empty-images { color: #64748B; font-size: 13px; border: 1px solid #E5E7EB; border-radius: 8px; }
.mr-thumb { width: 104px; text-decoration: none; color: #475569; font-size: 12px; }
.mr-thumb img { width: 104px; height: 76px; object-fit: contain; display: block; border: 1px solid #D6DEE8; border-radius: 6px; background: #fff; }
.mr-thumb span { display: block; margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mr-ph-token { display: inline-block; padding: 1px 7px; margin: 0 2px; border-radius: 4px; background: #DBEAFE; border: 1px solid #93C5FD; color: #1D4ED8; font-size: 12px; font-weight: 650; white-space: nowrap; vertical-align: baseline; }
.mr-missing-image { display: inline-block; padding: 1px 7px; border-radius: 4px; background: #FEE2E2; color: #B91C1C; font-size: 12px; }
.mr-inline-image { display: block; width: fit-content; max-width: 100%; margin: 8px 0; cursor: zoom-in; }
.mr-inline-image img { display: block; max-width: min(100%, 720px); max-height: 420px; object-fit: contain; border: 1px solid #D6DEE8; border-radius: 6px; background: #fff; }
.mr-lightbox { display: none; position: fixed; inset: 0; z-index: 9999; background: rgba(15, 23, 42, 0.86); align-items: center; justify-content: center; padding: 24px; cursor: zoom-out; }
.mr-lightbox:target { display: flex; }
.mr-lightbox img { max-width: 92vw; max-height: 88vh; width: auto; height: auto; object-fit: contain; border-radius: 8px; background: #fff; box-shadow: 0 20px 60px rgba(0,0,0,0.35); }
.mr-lightbox-close { position: fixed; top: 18px; right: 24px; color: #fff; font-size: 32px; line-height: 1; font-weight: 700; }
.mr-result-head span { font-size: 15px; }
.mr-result-title { padding: 7px 12px; border-bottom: 1px solid #E5E7EB; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #EEF2F7; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
"""


def main():
    print("=" * 60)
    print("RAG 知识库管理界面")
    print("=" * 60)
    print("访问地址: http://localhost:8001")
    print("=" * 60)
    app.launch(
        server_name="127.0.0.1",
        server_port=8001,
        share=False,
        inbrowser=False,
        theme=ACADEMIC_THEME,
        css=ACADEMIC_CSS,
    )


if __name__ == "__main__":
    main()
