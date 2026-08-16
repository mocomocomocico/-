"""Streamlit 界面组件。

把「侧边栏配置」「知识库管理」「聊天历史渲染」从入口脚本中拆出，
让 streamlit_app.py 只负责页面组装与交互流程。
"""

import json
import os
from dataclasses import dataclass
from typing import Callable

import streamlit as st

from app.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    LLM_MODEL_OPTIONS,
)
from app.ingestion import ingest_files
from app.models import AssistantReply, CollectionStats, SourceReference, ToolCallRecord
from app.retrieval import invalidate_retrieval_cache
from app.store import clear_vector_store, delete_source_documents

# 界面标签 → 内部标识
MODE_OPTIONS = {"标准 RAG": "rag", "Agent 检索": "agent"}
RETRIEVAL_OPTIONS = {"混合检索（推荐）": True, "仅向量": False}

CONTENT_PREVIEW_LIMIT = 400  # 引用来源预览长度
TOOL_RESULT_PREVIEW_LIMIT = 200  # 工具结果预览长度


@dataclass
class ChatSettings:
    """用户在一次会话中选择的全部配置。"""

    api_key: str
    mode: str
    model: str
    temperature: float
    thinking: bool
    reasoning_effort: str
    top_k: int
    hybrid: bool
    rerank: bool
    tracing_enabled: bool


# ---------- 侧边栏 ----------


def render_sidebar(
    embedding_model: str,
    vector_store,
    stats: CollectionStats,
    on_knowledge_base_changed: Callable[[], None],
) -> ChatSettings:
    """渲染整个侧边栏，返回用户本次会话的全部配置。"""
    with st.sidebar:
        st.title("知识库问答")
        st.caption(f"嵌入模型：{embedding_model}")

        api_key = _render_api_key_input()
        _render_new_chat_button()
        conversation = _render_conversation_settings()
        retrieval = _render_retrieval_settings()
        _render_knowledge_base_manager(vector_store, stats, on_knowledge_base_changed)
        tracing_enabled = _render_langfuse_panel()
        _render_architecture_notes()

        return ChatSettings(
            api_key=api_key,
            tracing_enabled=tracing_enabled,
            **conversation,
            **retrieval,
        )


def _render_api_key_input() -> str:
    """API Key 输入框（优先读取环境变量，仅保存在本次会话）。"""
    return st.text_input(
        "DeepSeek API Key",
        type="password",
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        key="api_key",
        help="sk- 开头的 DeepSeek API Key，仅保存在本次会话中，不会写入磁盘。",
    )


def _render_new_chat_button() -> None:
    """新建对话：清空当前会话历史。"""
    st.button(
        "新建对话",
        icon=":material/chat:",
        width="stretch",
        on_click=lambda: st.session_state.pop("messages", None),
    )


def _render_conversation_settings() -> dict:
    """对话设置：流程模式 / 模型 / 温度 / 深度思考。"""
    with st.expander("对话设置", expanded=True):
        mode_label = st.segmented_control(
            "流程模式",
            options=list(MODE_OPTIONS),
            default="标准 RAG",
            key="mode",
        )
        col_model, col_temp = st.columns(2)
        model = col_model.selectbox("模型", LLM_MODEL_OPTIONS, key="model")
        temperature = col_temp.slider(
            "温度", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05, key="temperature"
        )
        thinking = st.toggle(
            "深度思考",
            value=False,
            key="thinking",
            help="开启后质量更高但响应更慢；V4 思考模式会输出推理过程。",
        )
        reasoning_effort = "medium"
        if thinking:
            reasoning_effort = st.selectbox(
                "思考强度",
                ["low", "medium", "high"],
                index=1,
                key="reasoning_effort",
            )
        return {
            "mode": MODE_OPTIONS[mode_label],
            "model": model,
            "temperature": temperature,
            "thinking": thinking,
            "reasoning_effort": reasoning_effort,
        }


def _render_retrieval_settings() -> dict:
    """检索参数：检索方式 / top-k / 重排序。"""
    with st.expander("检索参数", expanded=False):
        retrieval_label = st.segmented_control(
            "检索方式",
            options=list(RETRIEVAL_OPTIONS),
            default="混合检索（推荐）",
            key="retrieval_mode",
            help="混合检索 = 向量 + BM25 中文检索 + RRF 融合，召回更全面。",
        )
        top_k = st.slider("检索条数 top-k", 1, 8, DEFAULT_TOP_K, key="top_k")
        rerank = st.toggle(
            "重排序（Cross-Encoder）",
            value=False,
            key="rerank",
            help=(
                "用 bge-reranker-v2-m3 对候选重新打分，更精准；"
                "代价：首次加载约 2.2GB 模型，每次查询增加数秒延迟。"
            ),
        )
        return {
            "hybrid": RETRIEVAL_OPTIONS[retrieval_label],
            "top_k": top_k,
            "rerank": rerank,
        }


def _render_knowledge_base_manager(
    vector_store,
    stats: CollectionStats,
    on_changed: Callable[[], None],
) -> None:
    """知识库管理：上传入库 / 删除文档 / 清空知识库。"""
    with st.expander("知识库管理", expanded=False):
        uploaded_files = st.file_uploader(
            "上传文档（txt / md / pdf / docx）",
            type=["txt", "md", "markdown", "pdf", "docx"],
            accept_multiple_files=True,
            key="kb_uploader",
        )
        col_size, col_overlap = st.columns(2)
        chunk_size = col_size.slider(
            "分块大小", 300, 1500, DEFAULT_CHUNK_SIZE, 50, key="chunk_size"
        )
        chunk_overlap = col_overlap.slider(
            "分块重叠", 0, 300, DEFAULT_CHUNK_OVERLAP, 20, key="chunk_overlap"
        )

        st.caption(
            f"知识库现有 **{stats.chunk_count}** 个分片，"
            f"来自 **{len(stats.source_names)}** 个文件。"
        )

        if st.button(
            "开始入库",
            type="primary",
            icon=":material/database:",
            width="stretch",
            disabled=not uploaded_files,
        ):
            files = [(f.name, f.getvalue()) for f in uploaded_files]
            with st.spinner("正在解析并写入向量库…"):
                result = ingest_files(
                    vector_store,
                    files,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            st.toast(f"入库完成：新增 {result.added_chunks} 个分片")
            if result.failed:
                st.warning(
                    "以下文件处理失败："
                    + "；".join(
                        f"{name}（{error}）" for name, error in result.failed
                    )
                )
            _refresh_knowledge_base(on_changed)

        if stats.source_names:
            del_source = st.selectbox(
                "删除文档",
                ["— 选择文档 —"] + stats.source_names,
                key="del_source",
            )
            if del_source != "— 选择文档 —" and st.button(
                "删除该文档", icon=":material/delete:", width="stretch"
            ):
                delete_source_documents(vector_store, del_source)
                st.toast(f"已删除 {del_source}")
                _refresh_knowledge_base(on_changed)

        confirm_clear = st.checkbox("确认清空整个知识库", key="confirm_clear")
        if st.button(
            "清空知识库",
            icon=":material/delete_sweep:",
            width="stretch",
            disabled=not confirm_clear,
        ):
            clear_vector_store(vector_store)
            st.toast("知识库已清空")
            _refresh_knowledge_base(on_changed)


def _refresh_knowledge_base(on_changed: Callable[[], None]) -> None:
    """入库 / 删除 / 清空后统一处理：失效检索与统计缓存，刷新页面。"""
    invalidate_retrieval_cache()
    on_changed()
    st.rerun()


def _render_langfuse_panel() -> bool:
    """Langfuse 配置面板，返回是否启用追踪。"""
    with st.expander("可观测性（Langfuse）", expanded=False):
        public_key = st.text_input(
            "Public Key",
            value=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            key="lf_public",
        )
        secret_key = st.text_input(
            "Secret Key",
            type="password",
            value=os.getenv("LANGFUSE_SECRET_KEY", ""),
            key="lf_secret",
        )
        host = st.text_input(
            "Host",
            value=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            key="lf_host",
        )
        configured = _write_langfuse_env(public_key, secret_key, host)
        tracing_enabled = st.toggle(
            "启用追踪",
            value=configured,
            key="tracing_enabled",
            help="记录每次提问的完整调用链（检索、工具、模型）到 Langfuse。",
        )
        if configured:
            st.caption(":material/cloud_done: 已配置，追踪可用")
        else:
            st.caption(
                "填写 Public Key / Secret Key / Host 三项后即可启用，"
                "配置仅保存在本次会话。"
            )
        return tracing_enabled


def _write_langfuse_env(public_key: str, secret_key: str, host: str) -> bool:
    """把界面填写的 Langfuse 配置写入进程环境变量（SDK 从环境变量读取）。"""
    if not (public_key and secret_key and host):
        return False
    os.environ["LANGFUSE_PUBLIC_KEY"] = public_key.strip()
    os.environ["LANGFUSE_SECRET_KEY"] = secret_key.strip()
    os.environ["LANGFUSE_HOST"] = host.strip()
    return True


def _render_architecture_notes() -> None:
    """侧边栏底部的架构说明。"""
    with st.expander("架构说明"):
        st.markdown(
            """
- **前端**：Streamlit 聊天界面
- **编排**：LangGraph（标准 RAG 图 / Agent 检索图）
- **检索**：向量（Chroma）+ BM25 + RRF + Cross-Encoder 重排序
- **嵌入**：sentence-transformers（本地）
- **大模型**：DeepSeek V4 Flash（OpenAI 兼容接口）
- **Agent 工具**：知识库检索 · 当前时间 · 当前地点 · 当前天气
"""
        )


# ---------- 聊天渲染 ----------


def render_chat_history(messages: list[dict]) -> None:
    """渲染已保存在会话状态中的全部消息。"""
    for message in messages:
        avatar = (
            ":material/person:"
            if message["role"] == "user"
            else ":material/robot:"
        )
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])
            if message["role"] == "assistant":
                reply = message.get("reply") or AssistantReply.from_message_dict(
                    message
                )
                render_reply_metadata(reply)


def render_reply_metadata(reply: AssistantReply) -> None:
    """渲染助手回复附带的引用来源、思考过程与工具调用记录。"""
    _render_sources(reply.sources)
    _render_reasoning(reply.reasoning)
    _render_tool_calls(reply.tool_calls)


def _render_sources(sources: list[SourceReference]) -> None:
    """引用来源：编号 + 来源文件 + 页码 + 内容预览。"""
    if not sources:
        return
    with st.expander(f"引用来源（{len(sources)} 条）"):
        for index, source in enumerate(sources, start=1):
            label = f"**{index}. {source.source}**"
            if source.page:
                label += f" · 第 {source.page} 页"
            st.markdown(label)
            st.caption(_truncate(source.content, CONTENT_PREVIEW_LIMIT))


def _render_reasoning(reasoning: str) -> None:
    """深度思考模式下的推理过程。"""
    if reasoning:
        with st.expander("思考过程"):
            st.write(reasoning)


def _render_tool_calls(calls: list[ToolCallRecord]) -> None:
    """工具调用记录：工具名 + 参数 + 结果预览。"""
    if not calls:
        return
    with st.expander(f"工具调用（{len(calls)} 次）"):
        for index, call in enumerate(calls, start=1):
            args_text = (
                json.dumps(call.args, ensure_ascii=False)
                if call.args
                else "（无参数）"
            )
            st.markdown(f"**{index}. {call.name}**")
            st.caption(f"参数：{args_text}")
            st.caption(
                f"结果：{_truncate(call.result, TOOL_RESULT_PREVIEW_LIMIT)}"
            )


def _truncate(text: str, limit: int) -> str:
    """截断长文本并在末尾加省略号。"""
    return text[:limit] + ("…" if len(text) > limit else "")
