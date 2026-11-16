from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import requests

from odoo.exceptions import UserError

from odoo.addons.muk_ai.providers import REGISTRY
from odoo.addons.muk_ai.providers.opencode_server import OpenCodeServerProvider
from odoo.addons.muk_ai.tests.common import AITestCommon

CATALOG = {
    'providers': [
        {
            'id': 'nvidia',
            'name': 'NVIDIA',
            'key': 'nvapi-test',
            'models': {
                'deepseek-ai/deepseek-r1': {
                    'id': 'deepseek-ai/deepseek-r1',
                    'name': 'DeepSeek R1',
                    'api': {'url': 'https://integrate.api.nvidia.com/v1'},
                    'capabilities': {
                        'reasoning': True,
                        'input': {'text': True, 'image': True},
                        'output': {'text': True},
                    },
                    'limit': {'context': 128000, 'output': 8192},
                    'cost': {
                        'input': 0.5,
                        'output': 1.5,
                        'cache': {'read': 0.1, 'write': 0.2},
                    },
                },
                'nvidia/unknown-context': {
                    'id': 'nvidia/unknown-context',
                    'name': 'Unknown Context',
                    'api': {'url': 'https://integrate.api.nvidia.com/v1'},
                    'capabilities': {
                        'reasoning': False,
                        'input': {'text': True},
                        'output': {'text': True},
                    },
                    'limit': {'context': 0, 'output': 0},
                    'cost': {'input': 0, 'output': 0, 'cache': {}},
                },
                'openai/whisper-large-v3': {
                    'id': 'openai/whisper-large-v3',
                    'name': 'Whisper Large v3',
                    'api': {'url': 'https://integrate.api.nvidia.com/v1'},
                    'capabilities': {
                        'reasoning': False,
                        'input': {'text': False},
                        'output': {'text': False},
                    },
                    'limit': {'context': 0, 'output': 0},
                    'cost': {'input': 0, 'output': 0, 'cache': {}},
                },
                'qwen/qwen-image-edit': {
                    'id': 'qwen/qwen-image-edit',
                    'name': 'Qwen Image Edit',
                    'api': {'url': 'https://integrate.api.nvidia.com/v1'},
                    'capabilities': {
                        'reasoning': False,
                        'input': {'text': True},
                        'output': {'text': False, 'image': True},
                    },
                    'limit': {'context': 0, 'output': 0},
                    'cost': {'input': 0, 'output': 0, 'cache': {}},
                },
            },
        },
        {
            'id': 'openrouter',
            'name': 'OpenRouter',
            'key': 'sk-or-test',
            'models': {
                'z-ai/glm-5.3-flash': {
                    'id': 'z-ai/glm-5.3-flash',
                    'name': 'GLM 5.3 Flash',
                    'api': {'url': 'https://openrouter.ai/api/v1'},
                    'capabilities': {
                        'reasoning': False,
                        'input': {'text': True},
                        'output': {'text': True},
                    },
                    'limit': {'context': 200000, 'output': 0},
                    'cost': {
                        'input': 0.2,
                        'output': 1.0,
                        'cache': {'read': 0.02, 'write': 0.03},
                    },
                },
            },
        },
        {
            'id': 'opencode',
            'name': 'OpenCode',
            'key': None,
            'models': {
                'big-pickle': {
                    'id': 'big-pickle',
                    'name': 'Big Pickle',
                    'api': {'url': 'https://opencode.ai/inference/openai/v1'},
                    'capabilities': {
                        'reasoning': False,
                        'input': {'text': True},
                        'output': {'text': True},
                    },
                    'limit': {'context': 128000, 'output': 0},
                    'cost': {'input': 0, 'output': 0, 'cache': {}},
                },
            },
        },
    ],
    'default': {'nvidia': 'deepseek-ai/deepseek-r1', 'opencode': 'big-pickle'},
}

COMPLETION = {
    'choices': [
        {
            'message': {
                'content': 'hello',
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'function',
                        'function': {
                            'name': 'search_read',
                            'arguments': '{"model": "res.partner"}',
                        },
                    }
                ],
            },
            'finish_reason': 'tool_calls',
        }
    ],
    'usage': {
        'prompt_tokens': 100,
        'completion_tokens': 20,
        'prompt_tokens_details': {'cached_tokens': 40},
    },
}


class TestAiOpenCodeServerProvider(AITestCommon):
    """Verify the OpenCode server adapter: broker, sync, and chat wire."""

    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.provider_server = cls.env.ref('muk_ai.provider_opencode_server')
        cls.provider_server.sudo().write(
            {
                'opencode_server_url': 'http://opencode:4096/',
                'opencode_server_username': 'opencode',
                'opencode_server_password': 'secret',
            }
        )

    def setUp(self) -> None:
        super().setUp()
        OpenCodeServerProvider.clear_catalog()

    # ----------------------------------------------------------
    # Helper
    # ----------------------------------------------------------

    def _catalog_model(self, technical_name: str, **values):
        """Create a catalogued chat model on the OpenCode server provider."""
        return self.env['muk_ai.model'].create(
            {
                'name': technical_name,
                'provider_id': self.provider_server.id,
                'technical_name': technical_name,
                'context_window': 128000,
                'input_rate': 0.5,
                'output_rate': 1.5,
                **values,
            }
        )

    def _capture_http(
        self,
        captured: dict,
        catalog: dict | None = None,
        completion: dict | None = None,
    ) -> object:
        """Patch the pooled HTTP, answering the server GET and vendor POST."""

        def fake_get(url, **kwargs):
            captured['get_url'] = url
            captured['get_kwargs'] = kwargs
            return self._mock_http_response(catalog if catalog is not None else CATALOG)

        def fake_post(url, **kwargs):
            captured['post_url'] = url
            captured['post_kwargs'] = kwargs
            return self._mock_http_response(
                completion if completion is not None else COMPLETION
            )

        return patch.object(
            requests.Session, 'get', side_effect=fake_get
        ), patch.object(requests.Session, 'post', side_effect=fake_post)

    def _mock_stream_response(self, events: Sequence[dict]) -> MagicMock:
        """Build a mocked streaming HTTP response replaying the given events."""
        lines = []
        for event in events:
            lines.append('data: ' + json.dumps(event))
            lines.append('')
        response = MagicMock()
        response.iter_lines.return_value = iter(lines)
        response.raise_for_status.return_value = None
        return response

    @contextmanager
    def _capture_stream(self, captured: dict, events: Sequence[dict]):
        """Patch the pooled HTTP for a streamed vendor request."""
        stream = self._mock_stream_response(events)

        def fake_get(url, **kwargs):
            captured['get_url'] = url
            return self._mock_http_response(CATALOG)

        def fake_post(url, **kwargs):
            captured['post_url'] = url
            captured['post_kwargs'] = kwargs
            return stream

        with (
            patch.object(requests.Session, 'get', side_effect=fake_get),
            patch.object(requests.Session, 'post', side_effect=fake_post),
        ):
            yield

    # ----------------------------------------------------------
    # Tests
    # ----------------------------------------------------------

    def test_provider_registered_and_seeded(self):
        self.assertIs(REGISTRY.get('opencode_server'), OpenCodeServerProvider)
        self.assertEqual(self.provider_server.code, 'default')
        self.assertEqual(self.provider_server.api_region, 'custom')
        self.assertEqual(
            self.provider_server.api_endpoint, 'http://opencode:4096'
        )

    def test_capabilities(self):
        self.assertFalse(self.provider_server.supports_web_search)
        self.assertFalse(self.provider_server.supports_code_interpreter)
        self.assertTrue(self.provider_server.supports_vision)
        self.assertTrue(self.provider_server._can_serve())
        self.provider_server.sudo().opencode_server_url = False
        self.assertFalse(self.provider_server._can_serve())

    def test_connection_pings_health_with_basic_auth(self):
        captured = {}
        get_patch, _ = self._capture_http(
            captured, catalog={'healthy': True, 'version': '1.19.5'}
        )
        with get_patch:
            self.assertTrue(self.provider_server._get_client().test_connection())
        self.assertEqual(captured['get_url'], 'http://opencode:4096/global/health')
        expected = base64.b64encode(b'opencode:secret').decode()
        self.assertEqual(
            captured['get_kwargs']['headers']['Authorization'],
            f'Basic {expected}',
        )

    def test_connection_raises_when_unhealthy(self):
        captured = {}
        get_patch, _ = self._capture_http(captured, catalog={'healthy': False})
        with get_patch:
            with self.assertRaises(UserError):
                self.provider_server._get_client().test_connection()

    def test_connection_requires_server_url(self):
        self.provider_server.sudo().opencode_server_url = False
        with self.assertRaisesRegex(UserError, 'not configured'):
            self.provider_server._get_client().test_connection()

    def test_sync_creates_chat_models_and_default(self):
        captured = {}
        get_patch, _ = self._capture_http(captured)
        with get_patch:
            self.provider_server.action_sync_opencode_server()
        Model = self.env['muk_ai.model']
        deepseek = Model.search(
            [
                ('provider_id', '=', self.provider_server.id),
                ('technical_name', '=', 'nvidia/deepseek-ai/deepseek-r1'),
            ]
        )
        self.assertEqual(len(deepseek), 1)
        self.assertEqual(deepseek.name, 'DeepSeek R1')
        self.assertEqual(deepseek.context_window, 128000)
        self.assertEqual(deepseek.input_rate, 0.5)
        self.assertEqual(deepseek.output_rate, 1.5)
        self.assertEqual(deepseek.cache_read_rate, 0.1)
        self.assertEqual(deepseek.cache_write_rate, 0.2)
        self.assertEqual(deepseek.reasoning_efforts, ['low', 'medium', 'high'])
        glm = Model.search(
            [
                ('provider_id', '=', self.provider_server.id),
                ('technical_name', '=', 'openrouter/z-ai/glm-5.3-flash'),
            ]
        )
        self.assertEqual(len(glm), 1)
        self.assertFalse(glm.reasoning_efforts)
        unknown = Model.search(
            [
                ('provider_id', '=', self.provider_server.id),
                ('technical_name', '=', 'nvidia/nvidia/unknown-context'),
            ]
        )
        self.assertEqual(unknown.context_window, 8192)
        self.assertIn('unknown', unknown.notes)
        self.assertFalse(
            Model.search(
                [
                    ('provider_id', '=', self.provider_server.id),
                    ('technical_name', 'in', [
                        'nvidia/openai/whisper-large-v3',
                        'nvidia/qwen/qwen-image-edit',
                        'opencode/big-pickle',
                    ]),
                ]
            )
        )
        self.assertEqual(
            self.provider_server.default_chat_model_id,
            deepseek,
        )

    def test_sync_is_idempotent_and_keeps_archived(self):
        captured = {}
        get_patch, _ = self._capture_http(captured)
        with get_patch:
            self.provider_server.action_sync_opencode_server()
            deepseek = self.env['muk_ai.model'].search(
                [('technical_name', '=', 'nvidia/deepseek-ai/deepseek-r1')]
            )
            deepseek.active = False
            self.provider_server.action_sync_opencode_server()
        self.assertEqual(deepseek.active, False)
        self.assertEqual(
            self.env['muk_ai.model'].with_context(active_test=False).search_count(
                [('provider_id', '=', self.provider_server.id)]
            ),
            3,
        )

    def test_sync_rejected_on_other_providers(self):
        with self.assertRaises(UserError):
            self.provider.action_sync_opencode_server()

    def test_request_posts_to_vendor_chat_completions(self):
        captured = {}
        get_patch, post_patch = self._capture_http(captured)
        with get_patch, post_patch:
            result = self.provider_server._request_responses(
                inputs=[
                    {
                        'role': 'user',
                        'content': [{'type': 'input_text', 'text': 'hi'}],
                    }
                ],
                model='nvidia/deepseek-ai/deepseek-r1',
            )
        self.assertEqual(
            captured['post_url'],
            'https://integrate.api.nvidia.com/v1/chat/completions',
        )
        self.assertEqual(
            captured['post_kwargs']['headers']['Authorization'],
            'Bearer nvapi-test',
        )
        body = captured['post_kwargs']['json']
        self.assertEqual(body['model'], 'deepseek-ai/deepseek-r1')
        self.assertEqual(
            body['messages'],
            [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}],
        )
        self.assertEqual(body['max_tokens'], 8192)
        self.assertNotIn('tools', body)
        self.assertEqual(result['text'], 'hello')
        self.assertEqual(result['tool_calls'][0]['name'], 'search_read')
        self.assertEqual(result['usage']['input_tokens'], 100)
        self.assertEqual(result['usage']['cache_read_tokens'], 40)

    def test_request_maps_tool_history(self):
        captured = {}
        get_patch, post_patch = self._capture_http(captured)
        with get_patch, post_patch:
            self.provider_server._request_responses(
                inputs=[
                    {
                        'role': 'system',
                        'content': [{'type': 'input_text', 'text': 'Terse.'}],
                    },
                    {
                        'type': 'function_call',
                        'call_id': 'call_9',
                        'name': 'list_modules',
                        'arguments': '{"installed_only": true}',
                    },
                    {
                        'type': 'function_call',
                        'call_id': 'call_10',
                        'name': 'read',
                        'arguments': '{"id": 1}',
                    },
                    {
                        'type': 'function_call_output',
                        'call_id': 'call_9',
                        'output': '{"ok": true}',
                    },
                    {
                        'type': 'function_call_output',
                        'call_id': 'call_10',
                        'output': '{"name": "x"}',
                    },
                    {'role': 'user', 'content': [{'type': 'input_text', 'text': 'go'}]},
                ],
                model='nvidia/deepseek-ai/deepseek-r1',
            )
        messages = captured['post_kwargs']['json']['messages']
        self.assertEqual(
            messages[0], {'role': 'system', 'content': [{'type': 'text', 'text': 'Terse.'}]}
        )
        self.assertEqual(messages[1]['role'], 'assistant')
        self.assertEqual(
            [call['id'] for call in messages[1]['tool_calls']],
            ['call_9', 'call_10'],
        )
        self.assertEqual(messages[2], {
            'role': 'tool',
            'tool_call_id': 'call_9',
            'content': '{"ok": true}',
        })
        self.assertEqual(messages[3]['tool_call_id'], 'call_10')
        self.assertEqual(messages[4]['content'], [{'type': 'text', 'text': 'go'}])

    def test_request_carries_canonical_function_calls_back(self):
        captured = {}
        get_patch, post_patch = self._capture_http(captured)
        with get_patch, post_patch:
            result = self.provider_server._request_responses(
                inputs=[], model='nvidia/deepseek-ai/deepseek-r1'
            )
        carry = result['carry_inputs'][0]
        self.assertEqual(carry['type'], 'function_call')
        self.assertEqual(carry['call_id'], 'call_1')
        self.assertEqual(carry['name'], 'search_read')
        self.assertIsInstance(carry['arguments'], str)

    def test_request_converts_tool_schemas(self):
        captured = {}
        get_patch, post_patch = self._capture_http(captured)
        with get_patch, post_patch:
            self.provider_server._request_responses(
                inputs=[],
                model='nvidia/deepseek-ai/deepseek-r1',
                tools_schema=[
                    {
                        'type': 'function',
                        'name': 'x',
                        'description': 'd',
                        'parameters': {'type': 'object'},
                        'strict': True,
                    }
                ],
            )
        body = captured['post_kwargs']['json']
        self.assertEqual(
            body['tools'],
            [
                {
                    'type': 'function',
                    'function': {
                        'name': 'x',
                        'description': 'd',
                        'parameters': {'type': 'object'},
                    },
                }
            ],
        )
        self.assertNotIn('parallel_tool_calls', body)

    def test_request_sends_structured_output_schema(self):
        captured = {}
        get_patch, post_patch = self._capture_http(captured)
        with get_patch, post_patch:
            self.provider_server._request_responses(
                inputs=[],
                model='nvidia/deepseek-ai/deepseek-r1',
                text_schema={'name': 'plan', 'schema': {'type': 'object'}},
            )
        body = captured['post_kwargs']['json']
        self.assertEqual(body['response_format']['type'], 'json_schema')
        self.assertEqual(body['response_format']['json_schema']['name'], 'plan')

    def test_request_sends_reasoning_effort_of_catalogued_model(self):
        self._catalog_model(
            'nvidia/deepseek-ai/deepseek-r1',
            reasoning_efforts=['low', 'medium', 'high'],
        )
        captured = {}
        get_patch, post_patch = self._capture_http(captured)
        with get_patch, post_patch:
            self.provider_server._request_responses(
                inputs=[],
                model='nvidia/deepseek-ai/deepseek-r1',
                reasoning_effort='high',
            )
        self.assertEqual(
            captured['post_kwargs']['json']['reasoning_effort'], 'high'
        )

    def test_request_retries_without_rejected_effort(self):
        self._catalog_model(
            'nvidia/deepseek-ai/deepseek-r1',
            reasoning_efforts=['low', 'medium', 'high'],
        )
        bodies = []

        def fake_get(url, **kwargs):
            return self._mock_http_response(CATALOG)

        def fake_post(url, **kwargs):
            bodies.append(dict(kwargs.get('json') or {}))
            if len(bodies) == 1:
                response = self._mock_http_response({}, status_code=400)
                response.text = 'reasoning_effort is not supported by this model'
                response.raise_for_status.side_effect = requests.HTTPError(
                    'boom', response=response
                )
                return response
            return self._mock_http_response(COMPLETION)

        with (
            patch.object(requests.Session, 'get', side_effect=fake_get),
            patch.object(requests.Session, 'post', side_effect=fake_post),
        ):
            result = self.provider_server._request_responses(
                inputs=[],
                model='nvidia/deepseek-ai/deepseek-r1',
                reasoning_effort='high',
            )
        self.assertEqual(result['text'], 'hello')
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0]['reasoning_effort'], 'high')
        self.assertNotIn('reasoning_effort', bodies[1])

    def test_request_without_effort_support_sends_none(self):
        self._catalog_model(
            'openrouter/z-ai/glm-5.3-flash',
            reasoning_efforts=[],
        )
        captured = {}
        get_patch, post_patch = self._capture_http(captured)
        with get_patch, post_patch:
            self.provider_server._request_responses(
                inputs=[],
                model='openrouter/z-ai/glm-5.3-flash',
                reasoning_effort='high',
            )
        self.assertNotIn('reasoning_effort', captured['post_kwargs']['json'])
        self.assertEqual(captured['post_kwargs']['json']['max_tokens'], 16384)

    def test_request_caches_the_catalog_between_calls(self):
        gets = []

        def fake_get(url, **kwargs):
            gets.append(url)
            return self._mock_http_response(CATALOG)

        def fake_post(url, **kwargs):
            return self._mock_http_response(COMPLETION)

        with (
            patch.object(requests.Session, 'get', side_effect=fake_get),
            patch.object(requests.Session, 'post', side_effect=fake_post),
        ):
            for _ in range(2):
                self.provider_server._request_responses(
                    inputs=[], model='nvidia/deepseek-ai/deepseek-r1'
                )
        self.assertEqual(len(gets), 1)

    def test_request_rejects_unknown_or_keyless_vendors(self):
        captured = {}
        get_patch, post_patch = self._capture_http(captured)
        with get_patch, post_patch:
            with self.assertRaisesRegex(UserError, 'does not resolve'):
                self.provider_server._request_responses(
                    inputs=[], model='mistral/mistral-large'
                )
            with self.assertRaisesRegex(UserError, 'no API key'):
                self.provider_server._request_responses(
                    inputs=[], model='opencode/big-pickle'
                )

    def test_request_requires_vendor_prefixed_model(self):
        captured = {}
        get_patch, post_patch = self._capture_http(captured)
        with get_patch, post_patch:
            with self.assertRaisesRegex(UserError, 'vendor-prefixed'):
                self.provider_server._request_responses(
                    inputs=[], model='glm-5.3-flash'
                )

    def test_request_without_model_raises(self):
        self.provider_server.sudo().default_chat_model_id = False
        with self.assertRaises(UserError):
            self.provider_server._request_responses(inputs=[])

    def test_stream_emits_text_reasoning_and_tool_deltas(self):
        captured = {}
        events = [
            {
                'choices': [
                    {'delta': {'reasoning_content': 'thinking'}, 'index': 0}
                ]
            },
            {'choices': [{'delta': {'content': 'He'}, 'index': 0}]},
            {'choices': [{'delta': {'content': 'llo'}, 'index': 0}]},
            {
                'choices': [
                    {
                        'delta': {
                            'tool_calls': [
                                {
                                    'index': 0,
                                    'id': 'call_1',
                                    'type': 'function',
                                    'function': {
                                        'name': 'do_it',
                                        'arguments': '{"a":',
                                    },
                                }
                            ]
                        },
                        'index': 0,
                    }
                ]
            },
            {
                'choices': [
                    {
                        'delta': {
                            'tool_calls': [
                                {
                                    'index': 0,
                                    'function': {'arguments': '1}'},
                                }
                            ]
                        },
                        'index': 0,
                    }
                ]
            },
            {
                'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'tool_calls'}],
                'usage': {'prompt_tokens': 7, 'completion_tokens': 3},
            },
        ]
        deltas = []
        with self._capture_stream(captured, events):
            result = self.provider_server._request_responses(
                inputs=[],
                model='nvidia/deepseek-ai/deepseek-r1',
                on_delta=lambda k, p: deltas.append((k, p)),
            )
        self.assertEqual(
            captured['post_url'],
            'https://integrate.api.nvidia.com/v1/chat/completions',
        )
        self.assertTrue(captured['post_kwargs']['json']['stream'])
        self.assertEqual(
            captured['post_kwargs']['json']['stream_options'],
            {'include_usage': True},
        )
        self.assertEqual(deltas[0], ('reasoning', {'delta': 'thinking'}))
        self.assertEqual(deltas[1], ('text', {'delta': 'He'}))
        self.assertIn(('tool_start', {'call_id': 'call_1', 'name': 'do_it'}), deltas)
        self.assertIn(('tool_args', {'call_id': 'call_1', 'delta': '1}'}), deltas)
        self.assertEqual(result['text'], 'Hello')
        self.assertEqual(result['tool_calls'][0]['arguments'], {'a': 1})
        self.assertEqual(result['usage']['output_tokens'], 3)

    def test_stream_applies_truncation_notice(self):
        captured = {}
        events = [
            {'choices': [{'delta': {'content': 'partial'}, 'index': 0}]},
            {
                'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'length'}],
                'usage': {'prompt_tokens': 7, 'completion_tokens': 8192},
            },
        ]
        with self._capture_stream(captured, events):
            result = self.provider_server._request_responses(
                inputs=[],
                model='nvidia/deepseek-ai/deepseek-r1',
                on_delta=lambda k, p: None,
            )
        self.assertTrue(result['usage']['truncated'])
        self.assertIn('truncated', result['text'])
