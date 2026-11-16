from __future__ import annotations

from odoo.addons.muk_ai.providers.openai import OpenAIProvider


class OpenCodeProvider(OpenAIProvider):
    """OpenCode gateway adapter speaking the OpenAI Responses wire.

    The OpenCode Console gateway (``https://opencode.ai/inference``) serves
    its whole multi-vendor catalog — GPT, Claude, Gemini, GLM, Kimi,
    DeepSeek, … — behind a single API key, exposing every text model
    through the OpenAI-compatible endpoints (``/responses`` and
    ``/chat/completions``) and translating between vendors server-side.
    Requests are authenticated with a plain bearer token, exactly like
    OpenAI's own.
    """

    name = 'opencode'
    label = 'OpenCode'
    default_model = 'glm-5.3-flash'
    default_url = 'https://opencode.ai/inference/openai/v1'

    # The gateway is a pure inference proxy: it serves no built-in
    # OpenAI connectors (web search, code interpreter) and exposes no
    # Files API, so those capabilities stay off and uploads never run.
    supports_web_search = False
    supports_code_interpreter = False
    sandbox_files = False

    def request(
        self,
        inputs,
        tools_schema=None,
        text_schema=None,
        on_delta=None,
        model=None,
        enable_web_search=False,
        enable_code_interpreter=False,
        extra=None,
    ) -> dict:
        """Run the request on the gateway, dropping OpenAI-only request parts.

        Built-in tool routing is disabled even when asked for — the
        capability flags of the provider record already keep the session
        from requesting it, and this keeps a direct call honest. The
        ``prompt_cache_key`` is dropped as well: the gateway does not
        document OpenAI's cache-routing hint and a strict translation
        would reject the unknown field.
        """
        extra = dict(extra or {})
        extra.pop('cache_key', None)
        return super().request(
            inputs,
            tools_schema=tools_schema,
            text_schema=text_schema,
            on_delta=on_delta,
            model=model,
            enable_web_search=False,
            enable_code_interpreter=False,
            extra=extra,
        )
