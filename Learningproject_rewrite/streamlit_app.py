"""知识库问答系统 · 应用入口。

运行：streamlit run streamlit_app.py

页面流程：渲染侧边栏配置 → 展示聊天历史 → 处理新提问。
界面细节（侧边栏、知识库管理、聊天渲染）见 app/ui.py。
"""

import os

import streamlit as st
from langchain_core.messages import HumanMessage

from app.config import DEFAULT_EMBEDDING_MODEL
from app.graphs import (
    build_agent_graph,
    build_rag_graph,
    history_to_messages,
    stream_reply,
)
from app.llm import build_llm
from app.models import AssistantReply
from app.store import build_embeddings, create_vector_store, get_collection_stats
from app.tracing import get_langfuse_handler
from app.ui import ChatSettings, render_chat_history, render_reply_metadata, render_sidebar

st.set_page_config(
    page_title="知识库问答系统",
    page_icon=":material/menu_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

# ---------------- 会话状态 ----------------
st.session_state.setdefault("messages", [])


# ---------------- 资源缓存 ----------------


@st.cache_resource(show_spinner="正在加载本地嵌入模型…")
def get_embeddings():
    return build_embeddings(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_store():
    return create_vector_store(get_embeddings())


@st.cache_data(ttl="1m")
def get_stats():
    """知识库统计（缓存 1 分钟；入库/删除/清空后显式失效）。"""
    return get_collection_stats(get_store())


@st.cache_resource(max_entries=8, show_spinner=False)
def get_llm(api_key, model, temperature, thinking, reasoning_effort):
    return build_llm(
        api_key=api_key,
        model=model,
        temperature=temperature,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )


@st.cache_resource(max_entries=4, show_spinner=False)
def get_graph(
    mode,
    api_key,
    model,
    temperature,
    thinking,
    reasoning_effort,
    top_k,
    hybrid,
    rerank,
):
    """按当前配置构建 LangGraph 图（资源级缓存，配置变化时才重建）。"""
    llm = get_llm(
        api_key, model, temperature, thinking, reasoning_effort
    )
    store = get_store()
    if mode == "agent":
        return build_agent_graph(
            llm, store, top_k=top_k, hybrid=hybrid, rerank=rerank
        )
    return build_rag_graph(
        llm, store, top_k=top_k, hybrid=hybrid, rerank=rerank
    )


def get_tracing_handler(settings: ChatSettings):
    """按需创建 Langfuse 回调处理器（在会话内只创建一次）。"""
    if not settings.tracing_enabled:
        return None
    if "langfuse_handler" not in st.session_state:
        st.session_state.langfuse_handler = get_langfuse_handler()
    return st.session_state.langfuse_handler


def build_graph_inputs(mode: str, question: str, history: list[dict]) -> dict:
    """按流程模式构造 LangGraph 输入。"""
    if mode == "agent":
        return {
            "messages": history_to_messages(history)
            + [HumanMessage(content=question)]
        }
    return {"question": question, "history": history}


def run_assistant_reply(question: str, settings: ChatSettings) -> None:
    """执行一次问答：调用 LangGraph → 流式渲染回答 → 存入会话历史。"""
    with st.chat_message("assistant", avatar=":material/robot:"):
        status = st.empty()
        status.caption("正在检索知识库…")
        try:
            graph = get_graph(
                settings.mode,
                settings.api_key,
                settings.model,
                settings.temperature,
                settings.thinking,
                settings.reasoning_effort,
                settings.top_k,
                settings.hybrid,
                settings.rerank,
            )
            inputs = build_graph_inputs(
                settings.mode, question, st.session_state.messages[:-1]
            )
            handler = get_tracing_handler(settings)
            callbacks = [handler] if handler else None
            generator, reply = stream_reply(
                graph, inputs, callbacks=callbacks
            )
            streamed = st.write_stream(generator)
            status.empty()
            if not streamed and reply.answer:
                # 兜底：模型未走流式通道时，直接展示完整回答
                st.write(reply.answer)
        except Exception as exc:  # noqa: BLE001
            status.empty()
            st.error(f"生成失败：{exc}")
            streamed = ""
            reply = AssistantReply(answer=f"（回答生成失败：{exc}）")

        answer = reply.answer or streamed or "（未生成回答）"
        render_reply_metadata(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "reply": reply}
    )


# ---------------- 页面 ----------------
stats = get_stats()
settings = render_sidebar(
    embedding_model=EMBEDDING_MODEL,
    vector_store=get_store(),
    stats=stats,
    on_knowledge_base_changed=lambda: get_stats.clear(),
)

st.title("知识库问答系统")
st.caption("上传文档 → 开始入库 → 提问。回答会标注引用来源。")

if not settings.api_key:
    st.info("请在左侧填写 DeepSeek API Key 后开始使用。")
    st.stop()

render_chat_history(st.session_state.messages)

if stats.chunk_count == 0:
    st.info(
        "知识库还是空的：在左侧上传 txt / md / pdf / docx 文档，"
        "点击“开始入库”后即可提问。"
    )

# 聊天输入框始终渲染，避免生成回答后输入框消失
if prompt := st.chat_input("请输入你的问题…", submit_mode="disable"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.write(prompt)
    run_assistant_reply(prompt, settings)
