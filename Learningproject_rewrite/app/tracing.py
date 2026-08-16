"""Langfuse 可观测性集成（可选）。

通过环境变量启用（LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
LANGFUSE_HOST）；未配置时所有函数返回 None / False，
应用行为与未接入时完全一致。
"""

import os

from langfuse.langchain import CallbackHandler


def is_langfuse_configured() -> bool:
    """是否已配置 Langfuse（三个环境变量齐全才认为可用）。"""
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_HOST")
    )


def get_langfuse_handler() -> CallbackHandler | None:
    """创建 LangChain/LangGraph 回调处理器；未配置则返回 None。

    处理器会从环境变量读取凭证，可传入 LangGraph 的
    config["callbacks"] 以记录完整调用链（检索、工具、模型）。
    """
    if not is_langfuse_configured():
        return None
    return CallbackHandler()
