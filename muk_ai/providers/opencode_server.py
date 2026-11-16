from __future__ import annotations

import base64
import time
from collections.abc import Callable

import requests

from odoo import models
from odoo.exceptions import UserError

from odoo.addons.muk_ai.providers.base import ProviderBase
from odoo.addons.muk_ai.providers.region import CUSTOM

CATALOG_TTL = 300.0
FALLBACK_CONTEXT_WINDOW = 8192
REASONING_EFFORTS = ('low', 'medium', 'high')


class OpenCodeServerProvider(ProviderBase):
    """Self-hosted OpenCode server adapter acting as catalog and key broker.

    The OpenCode server (``opencode serve``) owns the vendor credentials —
    NVIDIA, OpenRouter, … — and resolves them in ``GET /config/providers``
    alongside the model catalog (context windows, pricing, capability
    flags). This adapter keeps it that way: the sync action on the provider
    copies the catalog into ``muk_ai.model`` records, and inference calls
    each vendor's own OpenAI-compatible ``/chat/completions`` endpoint
    directly with the key the server reports, so no vendor secret is ever
    stored in Odoo. The server itself is only asked again (cached for a
    few minutes) when a chat turn starts.
    """

    name = 'opencode_server'
    label = 'OpenCode Server'
    default_model = ''
    default_url = ''
    regions = (CUSTOM,)

    supports_web_search = False
    supports_code_interpreter = False
    sandbox_files = False

    reasoning_error_tokens = ('reasoning', 'effort')

    # Process-wide resolved catalog, keyed by the server credential set:
    # {(url, username, password): (deadline, catalog)}. Adapter instances
    # are per-request, so the cache lives on the class.
    _catalog_cache: dict[tuple, tuple[float, dict]] = {}

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------

    def __init__(self, provider: models.BaseModel) -> None:
        """Store the owning provider record and start with no vendor key."""
        super().__init__(provider)
        self._vendor_key = ''

    # ----------------------------------------------------------
    # Config
    # ----------------------------------------------------------

    @property
    def server_url(self) -> str:
        """Return the base URL of the OpenCode server, without trailing slash."""
        return (self.provider.sudo().opencode_server_url or '').strip().rstrip('/')

    @property
    def api_url(self) -> str:
        """Return the server URL — the catalog and credential source.

        Inference itself targets each vendor's own endpoint, resolved per
        request from the server catalog; the server URL is what the
        endpoint display and the readiness check care about.
        """
        return self.server_url

    @property
    def authenticated(self) -> bool:
        """Return whether the OpenCode server is configured.

        The vendor API keys live on the server, so no key is configured
        here: a server URL is all this adapter needs to serve.
        """
        return bool(self.server_url)

    def headers(self) -> dict:
        """Return the vendor inference headers with the resolved bearer key."""
        return {
            'Authorization': f'Bearer {self._vendor_key}',
            'Content-Type': 'application/json',
        }

    def _server_headers(self) -> dict:
        """Return the basic-auth headers for the OpenCode server API."""
        provider = self.provider.sudo()
        username = provider.opencode_server_username or ''
        password = provider.opencode_server_password or ''
        if not username and not password:
            return {}
        token = base64.b64encode(f'{username}:{password}'.encode()).decode()
        return {'Authorization': f'Basic {token}'}

    # ----------------------------------------------------------
    # Server API
    # ----------------------------------------------------------

    def _server_get(self, path: str) -> dict:
        """GET a JSON document from the OpenCode server.

        :raise UserError: on HTTP, transport, or non-JSON errors.
        """
        try:
            response = self._http_session().get(
                f'{self.server_url}{path}',
                headers=self._server_headers(),
                timeout=self.request_timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as error:
            self._raise(getattr(error.response, 'text', '') or str(error))
        except requests.RequestException as error:
            self._raise(error)
        try:
            return response.json()
        except ValueError as error:
            self._raise(error)

    def fetch_catalog(self) -> dict:
        """Return the resolved server catalog, TTL-cached per credential set.

        The catalog maps each vendor id to its resolved API key and model
        entries, plus the server's per-vendor default models.
        """
        provider = self.provider.sudo()
        cache_key = (
            self.server_url,
            provider.opencode_server_username or '',
            provider.opencode_server_password or '',
        )
        cached = self._catalog_cache.get(cache_key)
        if cached and cached[0] > time.monotonic():
            return cached[1]
        payload = self._server_get('/config/providers')
        vendors = {}
        for entry in payload.get('providers') or []:
            vendor_id = entry.get('id')
            if not vendor_id:
                continue
            vendors[vendor_id] = {
                'id': vendor_id,
                'name': entry.get('name') or vendor_id,
                'key': entry.get('key') or '',
                'models': entry.get('models') or {},
            }
        catalog = {'vendors': vendors, 'defaults': payload.get('default') or {}}
        self._catalog_cache[cache_key] = (time.monotonic() + CATALOG_TTL, catalog)
        return catalog

    @classmethod
    def clear_catalog(cls) -> None:
        """Drop every cached catalog, e.g. after a manual sync."""
        cls._catalog_cache.clear()

    def _resolve_model(self, model: str) -> tuple[str, dict]:
        """Resolve a ``vendor/model`` technical name against the server catalog.

        :return: ``(vendor_url, model_entry)``; the vendor key is kept on
            the adapter for :meth:`headers`
        :raise UserError: when the name is unprefixed, the vendor is
            unknown, carries no API key, or exposes no endpoint URL
        """
        vendor_id, separator, model_id = model.partition('/')
        if not separator or not vendor_id or not model_id:
            self._raise(
                f'"{model}" is not a vendor-prefixed model — the OpenCode '
                'server catalogues models as "vendor/model", e.g. '
                '"nvidia/deepseek-ai/deepseek-r1".'
            )
        catalog = self.fetch_catalog()
        vendor = catalog['vendors'].get(vendor_id)
        if vendor is None:
            self._raise(
                f'the OpenCode server does not resolve vendor "{vendor_id}"'
            )
        if not vendor['key']:
            self._raise(
                f'vendor "{vendor_id}" has no API key configured on the '
                'OpenCode server'
            )
        entry = vendor['models'].get(model_id) or {}
        url = (entry.get('api') or {}).get('url') or ''
        if not url:
            url = next(
                (
                    (model_entry.get('api') or {}).get('url') or ''
                    for model_entry in vendor['models'].values()
                    if (model_entry.get('api') or {}).get('url')
                ),
                '',
            )
        if not url:
            self._raise(f'vendor "{vendor_id}" exposes no endpoint URL')
        self._vendor_key = vendor['key']
        return url, entry

    # ----------------------------------------------------------
    # Contract
    # ----------------------------------------------------------

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
        """Run the request on the vendor endpoint the server resolves."""
        model = self.model_for(model)
        if not model:
            self._raise(
                'no model configured — synchronize the catalog on the '
                'provider record and pick a default chat model'
            )
        if not self.server_url:
            raise UserError(
                self.env._(
                    '%(provider)s server URL is not configured.',
                    provider=self.label,
                )
            )
        url, entry = self._resolve_model(model)
        body = {
            'model': model.partition('/')[2],
            'messages': self._chat_messages(inputs),
        }
        max_tokens = self.max_tokens
        if (entry.get('limit') or {}).get('output'):
            max_tokens = min(max_tokens, entry['limit']['output'])
        if max_tokens:
            body['max_tokens'] = max_tokens
        if effort := (extra or {}).get('reasoning_effort'):
            body['reasoning_effort'] = effort
        if text_schema:
            body['response_format'] = {
                'type': 'json_schema',
                'json_schema': {
                    'name': text_schema.get('name', 'response'),
                    'schema': text_schema['schema'],
                    'strict': True,
                },
            }
        tools = self._chat_tools(tools_schema)
        if tools:
            body['tools'] = tools
        return self._invoke_with_reasoning_retry(
            model,
            lambda callback: self._invoke(url, body, callback),
            on_delta,
            ((body, ('reasoning_effort',)),),
        )

    def test_connection(self) -> bool:
        """Ping the OpenCode server health endpoint.

        :raise UserError: when the server URL is missing or unhealthy
        """
        if not self.server_url:
            raise UserError(
                self.env._(
                    '%(provider)s server URL is not configured.',
                    provider=self.label,
                )
            )
        health = self._server_get('/global/health')
        if not health.get('healthy'):
            self._raise(f'the server reports an unhealthy state: {health}')
        return True

    # ----------------------------------------------------------
    # Sync
    # ----------------------------------------------------------

    def model_values(self, entry: dict) -> dict | None:
        """Map a server model entry to ``muk_ai.model`` values, or None to skip.

        Models that cannot take text in and produce text out —
        transcription, image rendering, … — are no chat models and get
        skipped.
        """
        capabilities = entry.get('capabilities') or {}
        accepts = capabilities.get('input') or {}
        produces = capabilities.get('output') or {}
        if not (accepts.get('text') and produces.get('text')):
            return None
        limit = entry.get('limit') or {}
        cost = entry.get('cost') or {}
        cache = cost.get('cache') or {}
        context_window = limit.get('context') or 0
        notes = ''
        if not context_window:
            context_window = FALLBACK_CONTEXT_WINDOW
            notes = (
                'Context window unknown: the OpenCode server reported '
                'no limit for this model.'
            )
        return {
            'name': entry.get('name') or entry.get('id') or '?',
            'modality': 'chat',
            'context_window': context_window,
            'input_rate': cost.get('input') or 0.0,
            'output_rate': cost.get('output') or 0.0,
            'cache_read_rate': cache.get('read') or 0.0,
            'cache_write_rate': cache.get('write') or 0.0,
            'currency': 'USD',
            'reasoning_efforts': (
                list(REASONING_EFFORTS) if capabilities.get('reasoning') else []
            ),
            'notes': notes,
        }

    # ----------------------------------------------------------
    # Wire conversion
    # ----------------------------------------------------------

    @classmethod
    def _chat_messages(cls, inputs) -> list[dict]:
        """Convert canonical input items into chat/completions messages.

        A run of function-call items collapses into one assistant message
        carrying their tool calls, so the tool answers that follow find
        the calls they answer in the immediately preceding message.
        """
        messages: list[dict] = []
        pending_calls: list[dict] = []

        def flush_calls() -> None:
            if pending_calls:
                messages.append(
                    {
                        'role': 'assistant',
                        'content': '',
                        'tool_calls': list(pending_calls),
                    }
                )
                pending_calls.clear()

        for item in cls._wire_items(inputs):
            if not isinstance(item, dict):
                continue
            item_type = item.get('type')
            if item_type == 'function_call':
                pending_calls.append(
                    {
                        'id': item.get('call_id'),
                        'type': 'function',
                        'function': {
                            'name': item.get('name'),
                            'arguments': item.get('arguments') or '{}',
                        },
                    }
                )
                continue
            flush_calls()
            if item_type == 'function_call_output':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': item.get('call_id'),
                        'content': item.get('output') or '',
                    }
                )
                continue
            role = item.get('role')
            if role not in ('system', 'user', 'assistant', 'tool'):
                continue
            content = item.get('content')
            if isinstance(content, str):
                messages.append({'role': role, 'content': content})
            elif isinstance(content, list):
                if role == 'assistant':
                    messages.append(
                        {
                            'role': role,
                            'content': ''.join(
                                block.get('text') or ''
                                for block in cls._chat_content(content)
                                if block.get('type') == 'text'
                            ),
                        }
                    )
                else:
                    messages.append(
                        {'role': role, 'content': cls._chat_content(content)}
                    )
        flush_calls()
        return messages

    @classmethod
    def _chat_content(cls, blocks: list) -> list[dict]:
        """Convert canonical content blocks into chat content parts."""
        parts = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get('type') or ''
            if block_type in ('input_text', 'output_text', 'text'):
                parts.append({'type': 'text', 'text': block.get('text') or ''})
            elif block_type == 'input_image':
                url = block.get('image_url') or ''
                if url:
                    parts.append({'type': 'image_url', 'image_url': {'url': url}})
            elif block_type == 'muk_ai_attachment':
                if part := cls._attachment_part(block):
                    parts.append(part)
        return parts

    @staticmethod
    def _attachment_part(block: dict) -> dict | None:
        """Convert a materialized attachment block into an image or text part.

        Images ride as data URIs; every other attachment degrades to its
        inline text, since the OpenAI ``file`` part is not part of the
        wire every vendor serves.
        """
        if block.get('strategy') == 'image':
            mimetype = block.get('mimetype') or 'application/octet-stream'
            return {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:{mimetype};base64,{block.get("data_b64", "")}'
                },
            }
        text = block.get('inline_text') or ''
        if not text:
            return None
        filename = block.get('filename') or 'attachment'
        mimetype = block.get('mimetype') or 'application/octet-stream'
        prefix = f'--- File: {filename} ({mimetype}) ---\n'
        if block.get('truncated'):
            text += '\n[truncated]'
        return {'type': 'text', 'text': prefix + text}

    @staticmethod
    def _chat_tools(tools_schema) -> list[dict]:
        """Convert flattened Responses tool schemas to chat function tools."""
        return [
            {
                'type': 'function',
                'function': {
                    key: tool[key]
                    for key in ('name', 'description', 'parameters')
                    if tool.get(key) is not None
                },
            }
            for tool in tools_schema or []
            if isinstance(tool, dict) and tool.get('type') == 'function'
        ]

    # ----------------------------------------------------------
    # Invoke
    # ----------------------------------------------------------

    def _invoke(self, url: str, body: dict, on_delta: Callable | None) -> dict:
        """Dispatch the request to the streaming or non-streaming path."""
        endpoint = f'{url}/chat/completions'
        if callable(on_delta):
            return self._stream(endpoint, body, on_delta)
        return self._parse_response(
            self._post_json('/chat/completions', body, url=endpoint)
        )

    def _parse_response(self, payload: dict) -> dict:
        """Parse a non-streaming chat/completions response."""
        choices = payload.get('choices') or []
        choice = choices[0] if choices else {}
        message = choice.get('message') or {}
        calls = [
            {
                'call_id': call.get('id'),
                'name': (call.get('function') or {}).get('name'),
                'arguments': (call.get('function') or {}).get('arguments') or '{}',
            }
            for call in message.get('tool_calls') or []
        ]
        return self._assemble(
            message.get('content') or '',
            calls,
            payload.get('usage') or {},
            choice.get('finish_reason'),
        )

    def _stream(self, endpoint: str, body: dict, on_delta: Callable) -> dict:
        """Stream a chat/completions request, emitting deltas as they land."""
        body = {**body, 'stream': True, 'stream_options': {'include_usage': True}}
        text_parts: list[str] = []
        calls_by_index: dict[int, dict] = {}
        announced: set[int] = set()
        usage: dict = {}
        finish_reason = None
        for event in self._post_stream('/chat/completions', body, url=endpoint):
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices') or []:
                delta = choice.get('delta') or {}
                reasoning = (
                    delta.get('reasoning_content') or delta.get('reasoning') or ''
                )
                if reasoning:
                    self._call_on_delta(on_delta, 'reasoning', {'delta': reasoning})
                if content := delta.get('content'):
                    text_parts.append(content)
                    self._call_on_delta(on_delta, 'text', {'delta': content})
                for call in delta.get('tool_calls') or []:
                    index = call.get('index') or 0
                    entry = calls_by_index.setdefault(
                        index, {'call_id': '', 'name': '', 'arguments': ''}
                    )
                    if call.get('id'):
                        entry['call_id'] = call['id']
                    function = call.get('function') or {}
                    if function.get('name'):
                        entry['name'] = function['name']
                    if (
                        entry['call_id']
                        and entry['name']
                        and index not in announced
                    ):
                        announced.add(index)
                        self._call_on_delta(
                            on_delta,
                            'tool_start',
                            {'call_id': entry['call_id'], 'name': entry['name']},
                        )
                    if args := function.get('arguments'):
                        entry['arguments'] += args
                        self._call_on_delta(
                            on_delta,
                            'tool_args',
                            {'call_id': entry['call_id'], 'delta': args},
                        )
                if choice.get('finish_reason'):
                    finish_reason = choice['finish_reason']
        calls = [
            {
                'call_id': entry['call_id'] or f'call_{index}',
                'name': entry['name'],
                'arguments': entry['arguments'],
            }
            for index, entry in sorted(calls_by_index.items())
        ]
        return self._assemble(
            ''.join(text_parts), calls, usage, finish_reason, on_delta
        )

    def _assemble(
        self,
        text: str,
        calls: list[dict],
        usage: dict,
        finish_reason: str | None,
        on_delta: Callable | None = None,
    ) -> dict:
        """Build the canonical result from raw text, tool calls, and usage."""
        tool_calls = []
        carry_inputs = []
        for entry in calls:
            arguments, parse_error = self._parse_tool_arguments(entry['arguments'])
            tool_calls.append(
                {
                    'call_id': entry['call_id'],
                    'name': entry['name'],
                    'arguments': arguments,
                    '_parse_error': parse_error,
                }
            )
            carry_inputs.append(
                {
                    'type': 'function_call',
                    'call_id': entry['call_id'],
                    'name': entry['name'],
                    'arguments': entry['arguments'] or '{}',
                }
            )
        result = {
            'text': text.strip(),
            'tool_calls': tool_calls,
            'carry_inputs': carry_inputs,
            'usage': self._usage(
                input_tokens=usage.get('prompt_tokens'),
                output_tokens=usage.get('completion_tokens'),
                cache_read_tokens=(usage.get('prompt_tokens_details') or {}).get(
                    'cached_tokens'
                ),
            ),
        }
        if finish_reason == 'length':
            self._apply_truncation(result, on_delta, limit=self.max_tokens)
        return result
