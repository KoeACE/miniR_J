"""
FastAPI 服务
功能：提供文档检索 API 接口
"""

import os
import sys
import time
from typing import List, Optional, Dict
from pydantic import BaseModel

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_ROOT)

from scripts.retrieval_pipeline import RetrievalPipeline
from scripts.doc_retriever import DocRetriever
from scripts.config_manager import get_config
from scripts.db_manager import DatabaseManager, init_db
from scripts.manage_documents import DocumentManager

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("警告: FastAPI 未安装，API 服务不可用")
    print("请运行: pip install fastapi uvicorn")


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    use_rerank: bool = True


class RetrieveResponse(BaseModel):
    query: str
    docs: List[dict]
    total_docs: int
    debug_info: Optional[dict] = None


class DocRetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    use_rerank: bool = True


class DocRetrieveResponse(BaseModel):
    query: str
    docs: List[dict]
    total_docs: int
    timings: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="文档检索服务",
        description="文档检索系统",
        version="3.0.0"
    )

    pipeline: Optional[RetrievalPipeline] = None
    doc_retriever: Optional[DocRetriever] = None

    def _serialize_doc(doc: dict, include_full_meta: bool = True) -> dict:
        meta = doc.get("meta", {}) or {}
        if include_full_meta:
            serialized_meta = dict(meta)
        else:
            serialized_meta = {
                "doc_name": meta.get("doc_name", ""),
                "title": meta.get("title", ""),
                "entity": meta.get("entity", ""),
            }
        return {
            "doc_id": doc.get("doc_id", ""),
            "content": doc.get("content", ""),
            "meta": serialized_meta,
            "rank": doc.get("rank"),
            **({"rerank_score": doc["rerank_score"]} if "rerank_score" in doc else {}),
        }

    def _reset_retrieval_services():
        global pipeline, doc_retriever
        pipeline = RetrievalPipeline()
        doc_retriever = DocRetriever()

    @app.on_event("startup")
    async def startup_event():
        global pipeline, doc_retriever
        print("=" * 60)
        print("正在初始化文档检索服务...")
        print("=" * 60)
        try:
            _reset_retrieval_services()
            print("检索管道初始化完成!")
        except Exception as e:
            print(f"检索管道初始化失败: {e}")
            import traceback
            traceback.print_exc()

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        components = {
            "retrieval_pipeline": "ok" if pipeline is not None else "error",
            "doc_retriever": "ok" if doc_retriever is not None else "error",
        }

        status = "healthy" if all(v == "ok" for v in components.values()) else "degraded"

        return HealthResponse(
            status=status,
            version="3.0.0",
            components=components
        )

    @app.post("/kb/retrieve", response_model=RetrieveResponse)
    async def retrieve(request: RetrieveRequest):
        """
        文档检索接口

        执行文档召回，返回原始文档，
        适用于上层 Agent 自有 LLM 处理逻辑的场景。
        """
        if pipeline is None or doc_retriever is None:
            raise HTTPException(status_code=503, detail="检索服务未初始化")

        try:
            total_start = time.time()
            timings = {}

            print("\n" + "=" * 60)
            print(f"[API] 收到检索请求: {request.query}")
            print("=" * 60)

            query = request.query

            step_start = time.time()
            print("[步骤1] 文档召回...")
            if request.use_rerank:
                print("  使用Rerank重排序模式...")

            doc_retrieve_start = time.time()
            docs = doc_retriever.retrieve(
                query=query,
                top_k=request.top_k,
                use_rerank=request.use_rerank
            )
            timings['doc_retrieve'] = round(time.time() - doc_retrieve_start, 3)

            print(f"  ✓ 召回文档：{len(docs)}个 ({timings['doc_retrieve']}s)")

            timings['total'] = round(time.time() - total_start, 3)

            debug_info = {
                "timings": timings,
                "recall_summary": {
                    "use_rerank": request.use_rerank,
                }
            }

            print(f"\n[计时] 文档召回: {timings.get('doc_retrieve', 0)}s")
            print(f"  总耗时: {timings['total']}s")

            response = RetrieveResponse(
                query=query,
                docs=[_serialize_doc(doc) for doc in docs],
                total_docs=len(docs),
                debug_info=debug_info
            )

            print(f"\n[API] 检索完成，返回 {response.total_docs} 个文档")
            return response

        except Exception as e:
            print(f"[API] 检索失败: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/kb/docs/retrieve", response_model=DocRetrieveResponse)
    async def docs_retrieve(request: DocRetrieveRequest):
        """
        文档召回接口（独立）

        仅执行文档召回，不调用LLM
        """
        if doc_retriever is None:
            raise HTTPException(status_code=503, detail="文档召回服务未初始化")

        try:
            start_time = time.time()

            print("\n" + "=" * 60)
            print(f"[API] 文档召回请求: {request.query}")
            print("=" * 60)

            step_start = time.time()
            if request.use_rerank:
                print("[步骤1] 执行文档召回（使用Rerank）...")
            else:
                print("[步骤1] 执行文档召回...")

            docs = doc_retriever.retrieve(
                query=request.query,
                top_k=request.top_k,
                use_rerank=request.use_rerank
            )
            retrieve_time = round(time.time() - step_start, 3)

            print(f"  召回文档：{len(docs)}个 ({retrieve_time}s)")

            total_time = round(time.time() - start_time, 3)

            response = DocRetrieveResponse(
                query=request.query,
                docs=[_serialize_doc(doc) for doc in docs],
                total_docs=len(docs),
                timings={
                    "doc_retrieve": retrieve_time,
                    "total": total_time
                }
            )

            print(f"[API] 文档召回完成，总耗时: {total_time}s")
            return response

        except Exception as e:
            print(f"[API] 文档召回失败: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    _doc_manager: Optional[DocumentManager] = None

    def _get_doc_manager() -> DocumentManager:
        global _doc_manager
        if _doc_manager is None:
            _doc_manager = DocumentManager(get_config().db_path)
        return _doc_manager

    @app.get("/api/documents")
    async def api_list_documents(
        is_active: Optional[bool] = None,
        keyword: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ):
        dm = _get_doc_manager()
        docs = dm.list_documents(limit=limit + offset)
        if keyword:
            docs = [d for d in docs if keyword.lower() in d.get("name", "").lower()]
        if is_active is not None:
            docs = [d for d in docs if d.get("is_active", True) == is_active]
        total = len(docs)
        results = []
        for doc in docs[offset:offset + limit]:
            results.append({
                "corpus_id": doc.get("id", ""),
                "name": doc.get("name", ""),
                "is_active": doc.get("is_active", True),
                "created_at": doc.get("created_at", ""),
            })
        return {"documents": results, "total": total}

    @app.get("/api/documents/{corpus_id}")
    async def api_get_document(corpus_id: str):
        dm = _get_doc_manager()
        detail = dm.get_document_detail(corpus_id)
        if not detail:
            raise HTTPException(status_code=404, detail=f"文档不存在: {corpus_id}")
        return {
            "corpus_id": detail.get("id", ""),
            "name": detail.get("name", ""),
            "is_active": detail.get("is_active", True),
            "created_at": detail.get("created_at", ""),
            "updated_at": detail.get("updated_at", ""),
            "chunk_count": detail.get("chunk_count", 0),
            "chunk_strategy": detail.get("chunk_strategy", ""),
            "chunks": detail.get("chunks", []),
        }

    @app.delete("/api/documents/{corpus_id}")
    async def api_delete_document(corpus_id: str, update_index: bool = True):
        dm = _get_doc_manager()
        faiss_path = get_config().faiss_index_path
        success = dm.delete_document(
            corpus_id, confirm=False,
            faiss_index_path=faiss_path,
            update_index=update_index,
        )
        if not success:
            raise HTTPException(status_code=400, detail="删除失败")
        if update_index:
            _reset_retrieval_services()
        return {"deleted": corpus_id}

    @app.put("/api/documents/{corpus_id}/toggle")
    async def api_toggle_document(corpus_id: str):
        dm = _get_doc_manager()
        result = dm.toggle_document_status(corpus_id)
        if not result.get("success", False):
            raise HTTPException(status_code=404, detail=result.get("message", "Toggle failed"))
        return result

    @app.get("/api/stats")
    async def api_stats():
        dm = _get_doc_manager()
        stats = dm.get_statistics()
        return stats

    @app.get("/")
    async def root():
        return {
            "message": "文档检索服务",
            "version": "3.0.0",
            "description": "文档检索系统",
            "docs": "/docs"
        }

else:
    class DummyApp:
        pass
    app = DummyApp()


def main():
    if not FASTAPI_AVAILABLE:
        print("错误: FastAPI 未安装，无法启动服务")
        print("请运行: pip install fastapi uvicorn")
        return

    print("=" * 60)
    print("启动文档检索 FastAPI 服务")
    print("=" * 60)
    print("API 文档: http://localhost:8765/docs")
    print("健康检查: http://localhost:8765/health")
    print("=" * 60)
    print("检索接口:")
    print("  POST /kb/retrieve       - 文档检索")
    print("  POST /kb/docs/retrieve   - 文档召回")
    print("=" * 60)
    print("管理接口:")
    print("  GET  /api/documents      - 文档列表")
    print("  PUT  /api/documents/{id}/toggle - 切换文档状态")
    print("  GET  /api/stats          - 统计信息")
    print("=" * 60)

    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
