from __future__ import annotations

from .region import CUSTOM, Region
from .base import ProviderBase
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider
from .opencode import OpenCodeProvider


REGISTRY = {
    cls.name: cls
    for cls in (
        OpenAIProvider,
        AnthropicProvider,
        GoogleProvider,
        OpenCodeProvider,
    )
}
