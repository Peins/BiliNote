from app.gpt.base import GPT
from app.gpt.provider.OpenAI_compatible_provider import OpenAICompatibleProvider
from app.gpt.provider.anthropic_provider import AnthropicShimClient
from app.gpt.universal_gpt import UniversalGPT
from app.models.model_config import ModelConfig

# 用于判断是否走 Anthropic 协议的供应商标识（小写比较）
_ANTHROPIC_PROVIDER_IDS = {"claude"}


def _is_anthropic_provider(config: ModelConfig) -> bool:
    """判断该配置是否应使用 Anthropic 协议。

    note.py 传 provider=type（"built-in"）、name="Claude"；
    model.py 传 provider=name（"Claude"）。
    两个字段都检查以兼容所有调用场景。
    """
    provider_lower = (config.provider or "").strip().lower()
    name_lower = (config.name or "").strip().lower()
    return provider_lower in _ANTHROPIC_PROVIDER_IDS or name_lower in _ANTHROPIC_PROVIDER_IDS


class GPTFactory:
    @staticmethod
    def from_config(config: ModelConfig) -> GPT:
        if _is_anthropic_provider(config):
            client = AnthropicShimClient(api_key=config.api_key, base_url=config.base_url)
        else:
            client = OpenAICompatibleProvider(api_key=config.api_key, base_url=config.base_url).get_client
        return UniversalGPT(client=client, model=config.model_name)
