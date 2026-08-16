"""LangGraph 编排：标准 RAG 流程与 Agent 检索流程。

- 标准 RAG：固定执行 检索 → 生成；
- Agent 检索：由模型自主决定是否调用检索工具（ReAct 循环）。
"""

import json
from dataclasses import asdict
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from app.config import DEFAULT_TOP_K, MAX_HISTORY_TURNS
from app.models import AssistantReply, RAGState, SourceReference, ToolCallRecord
from app.retrieval import HybridRetriever
from app.tools import AGENT_TOOLS

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手，请优先基于知识库内容回答用户问题。\n"
    "规则：\n"
    "1. 只依据知识库内容作答，不要编造知识库中不存在的信息；\n"
    "2. 回答引用来源时标注 [来源1]、[来源2] 等；\n"
    "3. 如果知识库内容不足以回答问题，请明确说明“知识库中没有找到相关内容”，"
    "再简要给出基于常识的提示；\n"
    "4. 使用中文回答，结构清晰、简洁；\n"
    "5. 当问题涉及当前时间、地点或天气等实时信息时，先调用对应工具获取数据，"
    "再基于工具返回结果回答；用户提到具体城市时，必须把城市名传给天气工具，"
    "不能留空。"
)


def history_to_messages(history: list[dict]) -> list[Any]:
    """把会话历史转成 LangChain 消息（截断到最近 N 轮，保留思考内容）。"""
    messages = []
    for item in history[-MAX_HISTORY_TURNS * 2 :]:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
            continue
        reply = item.get("reply")
        reasoning = reply.reasoning if reply else item.get("reasoning", "")
        kwargs = (
            {"additional_kwargs": {"reasoning_content": reasoning}}
            if reasoning
            else {}
        )
        messages.append(AIMessage(content=item["content"], **kwargs))
    return messages


def document_to_source(document: Document) -> SourceReference:
    """把检索到的文档转成界面可展示的来源引用。"""
    return SourceReference(
        content=document.page_content,
        source=document.metadata.get("source", "未知来源"),
        page=document.metadata.get("page"),
    )


def format_context(documents: list[Document]) -> str:
    """把多个文档拼成带 [来源N] 前缀的上下文，供模型参考。"""
    blocks = [
        f"[来源{index}]（{doc.metadata.get('source', '未知')}）\n{doc.page_content}"
        for index, doc in enumerate(documents, start=1)
    ]
    return "\n\n".join(blocks)


def make_retriever(
    vectorstore,
    top_k: int = DEFAULT_TOP_K,
    hybrid: bool = True,
    rerank: bool = True,
):
    """按配置构建检索器：混合检索（向量+BM25+RRF+重排序）或纯向量检索。"""
    if hybrid:
        return HybridRetriever(vectorstore, top_k=top_k, rerank=rerank)
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def build_rag_graph(
    llm,
    vectorstore,
    top_k: int = DEFAULT_TOP_K,
    hybrid: bool = True,
    rerank: bool = True,
):
    """标准 RAG 图：检索 → 生成。

    说明：向量检索在知识库非空时总会返回结果，因此不需要「无结果」分支，
    「无相关内容」的场景由系统提示词规则兜底。
    """
    retriever = make_retriever(vectorstore, top_k, hybrid, rerank)

    def retrieve(state: RAGState) -> dict:
        return {"documents": retriever.invoke(state["question"])}

    def generate(state: RAGState) -> dict:
        context = format_context(state["documents"])
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *history_to_messages(state.get("history", [])),
            HumanMessage(
                content=(
                    "以下是知识库检索到的相关内容：\n\n"
                    f"{context}\n\n请回答用户问题：{state['question']}"
                )
            ),
        ]
        response = llm.invoke(messages)
        return {
            "reply": AssistantReply(
                answer=response.content,
                reasoning=response.additional_kwargs.get("reasoning_content")
                or "",
                sources=[
                    document_to_source(doc) for doc in state["documents"]
                ],
            )
        }

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


def build_agent_graph(
    llm,
    vectorstore,
    top_k: int = DEFAULT_TOP_K,
    hybrid: bool = True,
    rerank: bool = True,
):
    """Agent 检索图：由模型自主决定是否调用检索工具（ReAct 循环）。"""
    retriever = make_retriever(vectorstore, top_k, hybrid, rerank)

    @tool
    def retrieve_knowledge(query: str) -> str:
        """在知识库中检索与 query 最相关的内容，返回 JSON 数组。

        当用户问题需要依据知识库信息回答时调用本工具。
        """
        documents = retriever.invoke(query)
        snippets = [asdict(document_to_source(doc)) for doc in documents]
        return json.dumps(snippets, ensure_ascii=False)

    return create_react_agent(
        model=llm,
        tools=[retrieve_knowledge, *AGENT_TOOLS],
        prompt=SYSTEM_PROMPT,
    )


# ---------- 流式输出 ----------


def stream_reply(
    graph,
    inputs: dict,
    callbacks=None,
    metadata: dict | None = None,
) -> tuple[Any, AssistantReply]:
    """同步消费 LangGraph 流式输出。

    返回 (文本生成器, reply)：
    - 生成器逐 token 产出回答正文，供 st.write_stream 实时渲染；
    - reply 在流结束后包含完整回答、思考过程、引用来源与工具调用记录。
    """
    reply = AssistantReply()
    pending_tool_calls: dict[str, ToolCallRecord] = {}

    def text_stream():
        stream_kwargs: dict[str, Any] = {"stream_mode": ["messages", "updates"]}
        config: dict[str, Any] = {}
        if callbacks:
            config["callbacks"] = callbacks
        if metadata:
            config["metadata"] = metadata
        if config:
            stream_kwargs["config"] = config
        for mode, payload in graph.stream(inputs, **stream_kwargs):
            if mode == "messages":
                token = _consume_token_chunk(payload, reply)
                if token:
                    yield token
            elif mode == "updates":
                for node_update in payload.values():
                    if isinstance(node_update, dict):
                        _consume_node_update(
                            node_update, reply, pending_tool_calls
                        )

    return text_stream(), reply


def _consume_token_chunk(payload: tuple, reply: AssistantReply) -> str | None:
    """累积模型流式产出的 token 与思考内容，返回本次产出的 token 文本。"""
    chunk, _ = payload
    if not isinstance(chunk, AIMessage):
        return None
    if isinstance(chunk.content, str) and chunk.content:
        reply.answer += chunk.content
    reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
    if reasoning:
        reply.reasoning += reasoning
    return chunk.content if isinstance(chunk.content, str) and chunk.content else None


def _consume_node_update(
    update: dict,
    reply: AssistantReply,
    pending_tool_calls: dict[str, ToolCallRecord],
) -> None:
    """处理一次节点更新：合并完整回答，并追踪工具调用。"""
    node_reply = update.get("reply")
    if isinstance(node_reply, AssistantReply):
        _merge_reply(node_reply, reply)
    for message in update.get("messages") or []:
        _track_tool_call(message, pending_tool_calls)
        _track_tool_result(message, reply, pending_tool_calls)


def _merge_reply(node_reply: AssistantReply, reply: AssistantReply) -> None:
    """合并节点一次性产出的完整回答（如 RAG 图的 generate 节点）。"""
    if node_reply.answer:
        reply.answer = node_reply.answer
    if node_reply.reasoning:
        reply.reasoning = node_reply.reasoning
    if node_reply.sources:
        reply.sources = node_reply.sources


def _track_tool_call(
    message,
    pending_tool_calls: dict[str, ToolCallRecord],
) -> None:
    """记录模型发起的工具调用，等待对应的工具结果。"""
    if message.type != "ai" or not getattr(message, "tool_calls", None):
        return
    for call in message.tool_calls:
        pending_tool_calls[call["id"]] = ToolCallRecord(
            name=call.get("name", ""),
            args=call.get("args") or {},
            result="",
        )


def _track_tool_result(
    message,
    reply: AssistantReply,
    pending_tool_calls: dict[str, ToolCallRecord],
) -> None:
    """把工具结果与调用记录配对；知识库检索结果同时解析为引用来源。"""
    if message.type != "tool" or not message.content:
        return
    call_id = getattr(message, "tool_call_id", None)
    if call_id in pending_tool_calls:
        record = pending_tool_calls.pop(call_id)
        record.result = message.content[:300]
        reply.tool_calls.append(record)
    # 检索工具返回的是来源 JSON 数组，可直接作为引用来源展示
    try:
        parsed = json.loads(message.content)
        if isinstance(parsed, list):
            reply.sources = [SourceReference(**item) for item in parsed]
    except (TypeError, json.JSONDecodeError):
        pass
