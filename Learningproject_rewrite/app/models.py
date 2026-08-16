"""领域模型：统一各模块之间传递的数据结构。

使用 dataclass 描述界面展示所需的数据，使用 TypedDict 描述 LangGraph
图的状态，避免到处传递无结构的 dict。
"""

from dataclasses import dataclass, field
from typing import TypedDict

from langchain_core.documents import Document


@dataclass
class SourceReference:
    """一条引用来源：回答所依据的知识库片段。"""

    content: str
    source: str
    page: int | None = None


@dataclass
class ToolCallRecord:
    """Agent 的一次工具调用记录（用于界面展示）。"""

    name: str
    args: dict
    result: str


@dataclass
class AssistantReply:
    """一次完整的助手回复：回答正文 + 附加信息。"""

    answer: str = ""
    reasoning: str = ""  # 深度思考模式下的推理过程
    sources: list[SourceReference] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    @classmethod
    def from_message_dict(cls, message: dict) -> "AssistantReply":
        """从会话历史中存储的助手消息字典还原出对象。"""
        return cls(
            answer=message.get("content", ""),
            reasoning=message.get("reasoning", ""),
            sources=[SourceReference(**s) for s in message.get("sources", [])],
            tool_calls=[ToolCallRecord(**c) for c in message.get("tool_calls", [])],
        )


@dataclass
class CollectionStats:
    """知识库统计信息（分片总数 + 来源文件列表）。"""

    chunk_count: int
    source_names: list[str]


@dataclass
class IngestionResult:
    """一次批量入库的结果。"""

    added_chunks: int
    failed: list[tuple[str, str]]  # (文件名, 失败原因)


class RAGState(TypedDict):
    """标准 RAG 图的节点状态。"""

    question: str
    history: list[dict]
    documents: list[Document]
    reply: AssistantReply
