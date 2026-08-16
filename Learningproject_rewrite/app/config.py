"""全局配置：集中管理路径、模型名与默认参数。

所有可调常量都应放在这里，避免散落在各模块中难以查找。
"""

from pathlib import Path

# ---------- 路径 ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "knowledge_base"

# ---------- 大模型 ----------
DEFAULT_LLM_MODEL = "deepseek-v4-flash"
LLM_MODEL_OPTIONS = ["deepseek-v4-flash", "deepseek-v4-pro"]
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TEMPERATURE = 0.2
MAX_HISTORY_TURNS = 10  # 多轮对话最多保留的最近轮数

# ---------- 嵌入与重排序模型 ----------
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# ---------- 检索参数 ----------
DEFAULT_TOP_K = 4  # 最终返回给模型的文档条数
BM25_TOP_K = 10  # BM25 单独召回的条数
RERANK_TOP_N = 10  # 进入重排序的候选条数
RRF_K = 60  # RRF 融合常数

# ---------- 文档入库参数 ----------
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 120
