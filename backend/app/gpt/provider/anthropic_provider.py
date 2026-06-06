"""Anthropic Messages API 适配层：对外暴露与 OpenAI SDK 兼容的接口。

使 UniversalGPT 和 chat_service 无需修改即可调用 Anthropic 协议端点。
转换流程：
  OpenAI 格式 messages/tools → Anthropic 格式请求
  Anthropic 响应 → OpenAI 格式 response 对象
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from anthropic import Anthropic

from app.services.proxy_config_manager import ProxyConfigManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ─── OpenAI 兼容的响应数据结构 ────────────────────────────────────────────────

@dataclass
class FunctionCall:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str
    type: str = "function"
    function: FunctionCall = field(default_factory=lambda: FunctionCall("", ""))


@dataclass
class Message:
    content: Optional[str] = None
    role: str = "assistant"
    tool_calls: Optional[list] = None


@dataclass
class Choice:
    index: int = 0
    message: Message = field(default_factory=Message)
    finish_reason: str = "stop"


@dataclass
class ChatCompletion:
    choices: list = field(default_factory=list)
    model: str = ""
    id: str = ""


@dataclass
class ModelObject:
    id: str
    object: str = "model"

    def dict(self):
        return {"id": self.id, "object": self.object}


@dataclass
class ModelList:
    data: list = field(default_factory=list)
    object: str = "list"


# ─── 转换工具 ─────────────────────────────────────────────────────────────────

def _convert_messages_for_anthropic(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
    """将 OpenAI 格式的 messages 转为 Anthropic 格式。

    Returns:
        (system_prompt, anthropic_messages)
    """
    system_prompt = None
    anthropic_messages = []

    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            # Anthropic 将 system 作为独立参数
            system_prompt = msg.get("content", "")
            continue

        if role == "tool":
            # OpenAI tool result → Anthropic tool_result content block
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }]
            })
            continue

        if role == "assistant":
            # 可能包含 tool_calls（从上一轮 LLM 响应追加的 msg 对象）
            tool_calls = getattr(msg, "tool_calls", None) or msg.get("tool_calls")
            if tool_calls:
                content_blocks = []
                # 先添加文本部分（如有）
                text = getattr(msg, "content", None) or msg.get("content")
                if text:
                    content_blocks.append({"type": "text", "text": text})
                for tc in tool_calls:
                    fn = getattr(tc, "function", None) or tc.get("function", {})
                    fn_name = getattr(fn, "name", None) or fn.get("name", "")
                    fn_args_raw = getattr(fn, "arguments", None) or fn.get("arguments", "{}")
                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except json.JSONDecodeError:
                        fn_args = {}
                    tc_id = getattr(tc, "id", None) or tc.get("id", "")
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc_id,
                        "name": fn_name,
                        "input": fn_args,
                    })
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
            else:
                content = getattr(msg, "content", None) or msg.get("content", "")
                anthropic_messages.append({"role": "assistant", "content": content})
            continue

        # user message
        content = msg.get("content", "")
        if isinstance(content, list):
            # 多模态 content 数组 → Anthropic 格式
            anthropic_content = []
            for block in content:
                if block.get("type") == "text":
                    anthropic_content.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image_url":
                    url = block["image_url"]["url"]
                    if url.startswith("data:"):
                        # base64 内联图片
                        # 格式: data:image/png;base64,xxxxx
                        media_type = url.split(";")[0].split(":")[1]
                        data = url.split(",", 1)[1]
                        anthropic_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            }
                        })
                    else:
                        # URL 引用图片
                        anthropic_content.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url,
                            }
                        })
            anthropic_messages.append({"role": "user", "content": anthropic_content})
        else:
            anthropic_messages.append({"role": "user", "content": content})

    return system_prompt, anthropic_messages


def _convert_tools_for_anthropic(tools: Optional[list[dict]]) -> Optional[list[dict]]:
    """将 OpenAI tools 格式转为 Anthropic tools 格式。"""
    if not tools:
        return None

    anthropic_tools = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool["function"]
        anthropic_tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return anthropic_tools or None


def _convert_response_to_openai(response) -> ChatCompletion:
    """将 Anthropic Messages 响应转为 OpenAI ChatCompletion 格式。"""
    text_parts = []
    tool_calls = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                type="function",
                function=FunctionCall(
                    name=block.name,
                    arguments=json.dumps(block.input, ensure_ascii=False),
                ),
            ))

    content = "\n".join(text_parts) if text_parts else None
    finish_reason = "tool_calls" if tool_calls else "stop"

    message = Message(
        content=content,
        role="assistant",
        tool_calls=tool_calls if tool_calls else None,
    )

    return ChatCompletion(
        choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
        model=response.model,
        id=response.id,
    )


# ─── Shim Client ─────────────────────────────────────────────────────────────

class _Completions:
    """模拟 client.chat.completions 接口。"""

    def __init__(self, anthropic_client: Anthropic):
        self._client = anthropic_client

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: Optional[float] = None,
        tools: Optional[list[dict]] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatCompletion:
        system_prompt, anthropic_messages = _convert_messages_for_anthropic(messages)
        anthropic_tools = _convert_tools_for_anthropic(tools)

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 8192,
        }

        if system_prompt:
            create_kwargs["system"] = system_prompt
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools

        response = self._client.messages.create(**create_kwargs)
        return _convert_response_to_openai(response)


class _Chat:
    """模拟 client.chat 命名空间。"""

    def __init__(self, anthropic_client: Anthropic):
        self.completions = _Completions(anthropic_client)


class _Models:
    """模拟 client.models 接口（Anthropic 无此端点，返回空列表）。"""

    def list(self) -> ModelList:
        return ModelList(data=[])


class AnthropicShimClient:
    """对外暴露 OpenAI SDK 兼容接口，内部使用 Anthropic SDK。

    用法与 openai.OpenAI() 客户端一致：
        client.chat.completions.create(...)
        client.models.list()
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            # 用户填写的可能是 http://host:port 或 http://host:port/v1/messages
            # Anthropic SDK 只要 base_url 到根路径即可，它会自动拼 /v1/messages
            clean_url = base_url.rstrip("/")

            # 容错：处理 "https://http://..." 这种畸形双协议前缀
            if clean_url.startswith("https://http://") or clean_url.startswith("https://https://"):
                clean_url = clean_url[len("https://"):]
            elif clean_url.startswith("http://http://") or clean_url.startswith("http://https://"):
                clean_url = clean_url[len("http://"):]

            if clean_url.endswith("/v1/messages"):
                clean_url = clean_url[: -len("/v1/messages")]
            elif clean_url.endswith("/v1"):
                clean_url = clean_url[: -len("/v1")]
            client_kwargs["base_url"] = clean_url

        # 处理代理：对 HTTP 局域网/本地地址不走代理，避免代理对 HTTP 做 TLS 导致失败
        effective_url = client_kwargs.get("base_url", "https://api.anthropic.com")
        parsed = urlparse(effective_url)
        is_plain_http = parsed.scheme == "http"

        if is_plain_http:
            # 本地 HTTP 端点：显式禁用代理（trust_env=False 阻止从环境变量读取代理）
            client_kwargs["http_client"] = httpx.Client(trust_env=False, timeout=600.0)
            logger.info(f"Anthropic 客户端连接 HTTP 端点（无代理）: {effective_url}")
        else:
            # HTTPS 端点：尊重全局代理设置
            proxy_url = ProxyConfigManager().get_proxy_url()
            if proxy_url:
                client_kwargs["http_client"] = httpx.Client(proxy=proxy_url, timeout=600.0)
                logger.info(f"Anthropic 客户端走代理: {proxy_url}")

        self._anthropic = Anthropic(**client_kwargs)
        self.chat = _Chat(self._anthropic)
        self.models = _Models()

    @staticmethod
    def test_connection(api_key: str, base_url: Optional[str], model: str) -> bool:
        """连通性测试：发一条最小化请求验证可达性。"""
        try:
            client = AnthropicShimClient(api_key=api_key, base_url=base_url)
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
            )
            logger.info(f"Anthropic 连通性测试成功（model={model}）")
            return True
        except Exception as e:
            logger.warning(f"Anthropic 连通性测试失败（model={model}）：{e}")
            return False
