"""DeepSeek 大模型封装。

langchain-deepseek 1.0.x 存在一个已知问题（GitHub issue #37713）：
DeepSeek V4 系列开启思考模式后，多轮对话必须把上一轮的
``reasoning_content`` 原样回传给 API，否则返回 400。官方集成在序列化请求时
会丢弃该字段，这里通过子类在请求 payload 中补回。
"""

from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

from app.config import DEFAULT_BASE_URL, DEFAULT_LLM_MODEL


class ReasoningCompatibleChatDeepSeek(ChatDeepSeek):
    """多轮对话中自动回传 ``reasoning_content`` 的 ChatDeepSeek 兼容子类。"""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        reasoning_contents = [
            message.additional_kwargs.get("reasoning_content")
            if isinstance(message, AIMessage)
            else None
            for message in messages
        ]
        for api_message, reasoning in zip(
            payload.get("messages", []), reasoning_contents
        ):
            if reasoning and api_message.get("role") == "assistant":
                api_message["reasoning_content"] = reasoning
        return payload


def build_llm(
    api_key: str,
    model: str = DEFAULT_LLM_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.2,
    thinking: bool = False,
    reasoning_effort: str = "medium",
    max_tokens: int = 8192,
) -> ReasoningCompatibleChatDeepSeek:
    """构建 DeepSeek 聊天模型。

    默认关闭思考模式以换取响应速度；开启后通过 ``reasoning_effort``
    控制思考强度。
    """
    return ReasoningCompatibleChatDeepSeek(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort if thinking else None,
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        timeout=120,
        max_retries=2,
    )
