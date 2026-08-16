"""混合检索：向量（Chroma）+ BM25（jieba 中文分词）→ RRF 融合 → 可选重排序。

BM25 索引基于「语料版本号」懒重建：入库 / 删除 / 清空后调用
``invalidate_retrieval_cache()``，下一次查询时自动重建，
避免每次查询都全表扫描。
"""

from dataclasses import dataclass, field
from typing import Any

import jieba
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import (
    BM25_TOP_K,
    DEFAULT_TOP_K,
    RERANK_MODEL,
    RERANK_TOP_N,
    RRF_K,
)

# 语料版本号：知识库内容变化后递增，触发所有检索器重建 BM25 索引
_corpus_version = 0


@dataclass
class _BM25IndexState:
    """模块级共享的 BM25 索引状态（多个检索器共用一份，避免重复构建）。"""

    version: int | None = None
    documents: list[Document] = field(default_factory=list)
    index: BM25Okapi | None = None


_bm25_state = _BM25IndexState()
_reranker = None  # Cross-Encoder 重排序模型（全局单例，约 2.2GB）


def invalidate_retrieval_cache() -> None:
    """知识库内容变化后调用，强制检索器在下一次查询时重建 BM25 索引。"""
    global _corpus_version
    _corpus_version += 1


def _get_reranker():
    """懒加载重排序模型；首次使用时才加载，避免拖慢应用启动。"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _tokenize(text: str) -> list[str]:
    """jieba 中文分词（BM25 需要词级 token，纯空格切分对中文无效）。"""
    return [token for token in jieba.lcut(text) if token.strip()]


def _document_key(document: Document) -> tuple[Any, ...]:
    """文档稳定标识：来源 + 页码 + 正文，用于跨检索结果去重。"""
    return (
        document.metadata.get("source", ""),
        document.metadata.get("page"),
        document.page_content,
    )


def rrf_merge(
    result_lists: list[list[Document]],
    top_k: int = DEFAULT_TOP_K,
    k: int = RRF_K,
) -> list[Document]:
    """Reciprocal Rank Fusion：按文档在多个结果列表中的排名加权求和。"""
    scores: dict[tuple[Any, ...], float] = {}
    documents: dict[tuple[Any, ...], Document] = {}
    for result_list in result_lists:
        for rank, document in enumerate(result_list, start=1):
            key = _document_key(document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            documents.setdefault(key, document)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [documents[key] for key, _ in ranked[:top_k]]


class HybridRetriever:
    """向量 + BM25 混合检索器，输出 RRF 融合后的 top-k 文档。"""

    def __init__(
        self,
        vectorstore,
        top_k: int = DEFAULT_TOP_K,
        bm25_k: int = BM25_TOP_K,
        rerank: bool = True,
        rerank_top_n: int = RERANK_TOP_N,
    ) -> None:
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.bm25_k = bm25_k
        self.rerank = rerank
        self.rerank_top_n = rerank_top_n

    def invoke(self, query: str) -> list[Document]:
        """混合检索主入口：向量 + BM25 → RRF 融合 →（可选）重排序 → top-k。"""
        query = (query or "").strip()
        if not query:
            return []

        vector_documents = self.vectorstore.similarity_search(
            query, k=self.bm25_k
        )
        bm25_documents = self._bm25_search(query)
        fused = rrf_merge(
            [vector_documents, bm25_documents],
            top_k=self.rerank_top_n if self.rerank else self.top_k,
        )
        if not fused:
            return []
        if not self.rerank:
            return fused
        return self._rerank(query, fused)

    # ---------- 内部实现 ----------

    def _ensure_index(self) -> None:
        """语料版本变化时重建 BM25 索引（首次查询或 invalidate 后触发）。"""
        if _bm25_state.version == _corpus_version:
            return
        data = self.vectorstore.get(include=["documents", "metadatas"])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        docs = [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in zip(documents, metadatas)
        ]
        _bm25_state.documents = docs
        _bm25_state.index = BM25Okapi(
            [_tokenize(doc.page_content) for doc in docs]
        )
        _bm25_state.version = _corpus_version

    def _bm25_search(self, query: str) -> list[Document]:
        """BM25 检索：分词打分后取前 bm25_k 条。"""
        self._ensure_index()
        documents = _bm25_state.documents
        index = _bm25_state.index
        if not documents or index is None:
            return []
        scores = index.get_scores(_tokenize(query))
        top_indexes = scores.argsort()[::-1][: self.bm25_k]
        return [documents[i] for i in top_indexes if scores[i] > 0]

    def _rerank(self, query: str, candidates: list[Document]) -> list[Document]:
        """用 Cross-Encoder 对候选文档重新打分排序。"""
        reranker = _get_reranker()
        pairs = [(query, doc.page_content) for doc in candidates]
        raw_scores = reranker.predict(pairs)
        if len(raw_scores) > 0 and isinstance(raw_scores[0], dict):
            scores = [
                item.get("score", item.get("logits", 0.0))
                for item in raw_scores
            ]
        else:
            scores = np.asarray(raw_scores).reshape(-1).tolist()
        ranked = [
            doc
            for _, doc in sorted(
                zip(scores, candidates), key=lambda item: item[0], reverse=True
            )
        ]
        return ranked[: self.top_k]
