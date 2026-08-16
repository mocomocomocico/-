"""向量库：本地嵌入模型 + Chroma 持久化存储。

本模块只负责「嵌入模型 / 向量库」的构建与基础操作，
文档解析与入库流程见 app/ingestion.py。
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import CHROMA_DIR, COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL
from app.models import CollectionStats


def build_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """构建本地嵌入模型（首次使用会自动下载，约 100 MB）。

    更换模型名会改变向量维度，需要清空 ``chroma_db`` 目录后重新入库。
    """
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise RuntimeError(
            f"无法加载嵌入模型 {model_name}。\n"
            "若模型未下载，请先执行：huggingface-cli download BAAI/bge-small-zh-v1.5\n"
            "或在联网环境设置 HF_HUB_OFFLINE=0 后重启应用。"
        ) from exc


def create_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """打开（不存在则创建）持久化向量库，数据保存在 chroma_db/ 目录。"""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        create_collection_if_not_exists=True,
    )


def get_collection_stats(vector_store: Chroma) -> CollectionStats:
    """统计知识库的分片总数与来源文件列表。"""
    try:
        data = vector_store.get(limit=5000, include=["metadatas"])
    except Exception:
        return CollectionStats(chunk_count=0, source_names=[])
    metadatas = data.get("metadatas") or []
    source_names = sorted(
        {m.get("source") for m in metadatas if m and m.get("source")}
    )
    return CollectionStats(
        chunk_count=len(data.get("ids") or []),
        source_names=source_names,
    )


def delete_source_documents(vector_store: Chroma, source: str) -> None:
    """删除指定来源文件的所有分片。"""
    vector_store.delete(where={"source": source})


def clear_vector_store(vector_store: Chroma) -> None:
    """清空知识库的全部分片。"""
    vector_store.reset_collection()
