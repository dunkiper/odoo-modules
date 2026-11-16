from __future__ import annotations

from unittest.mock import patch

import requests

from odoo.exceptions import UserError

from odoo.addons.muk_ai.providers import REGISTRY
from odoo.addons.muk_ai.providers.opencode import OpenCodeProvider
from odoo.addons.muk_ai.tests.common import AITestCommon


class TestAiOpenCodeProvider(AITestCommon):
    """Verify the OpenCode provider wiring, request body, and capabilities."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.provider_opencode = cls.env.ref('muk_ai.provider_opencode')
        cls.provider_opencode.sudo().api_key = 'oc-test-key'

    # ----------------------------------------------------------
    # Helper
    # ----------------------------------------------------------

    def _capture_post(self, captured: dict, payload: dict) -> object:
        """Return a patch of the pooled POST answering ``payload``.

        :param captured: filled with the request kwargs and its ``url``
        """

        def fake_post(url, **kwargs):
            captured.update(kwargs, url=url)
            return self._mock_http_response(payload)

        return patch.object(requests.Session, 'post', side_effect=fake_post)

    # ----------------------------------------------------------
    # Tests
    # ----------------------------------------------------------

    def test_provider_registered_and_seeded(self):
        self.assertIs(REGISTRY.get('opencode'), OpenCodeProvider)
        self.assertEqual(self.provider_opencode.code, 'default')
        self.assertEqual(
            self.provider_opencode.api_endpoint, OpenCodeProvider.default_url
        )

    def test_default_model_is_gateway_model(self):
        self.assertEqual(
            self.provider_opencode.default_chat_model_id.technical_name,
            'glm-5.3-flash',
        )

    def test_capability_flags(self):
        self.assertFalse(self.provider_opencode.supports_web_search)
        self.assertFalse(self.provider_opencode.supports_code_interpreter)
        self.assertTrue(self.provider_opencode.supports_vision)

    def test_request_posts_to_gateway_responses_endpoint(self):
        captured = {}
        patcher = self._capture_post(
            captured,
            {
                'output': [{'type': 'message', 'content': [{'text': 'ok'}]}],
                'usage': {'input_tokens': 5, 'output_tokens': 2},
            },
        )
        with patcher:
            result = self.provider_opencode._request_responses(
                inputs=[
                    {'role': 'user', 'content': [{'type': 'input_text', 'text': 'hi'}]}
                ],
            )
        self.assertEqual(
            captured['url'], 'https://opencode.ai/inference/openai/v1/responses'
        )
        self.assertEqual(captured['headers']['Authorization'], 'Bearer oc-test-key')
        self.assertEqual(captured['json']['model'], 'glm-5.3-flash')
        self.assertEqual(result['text'], 'ok')

    def test_request_strips_openai_only_parts(self):
        captured = {}
        patcher = self._capture_post(
            captured,
            {'output': [], 'usage': {'input_tokens': 1, 'output_tokens': 1}},
        )
        with patcher:
            self.provider_opencode._request_responses(
                inputs=[],
                enable_web_search=True,
                enable_code_interpreter=True,
                cache_key='muk_ai.session:42',
            )
        body = captured['json']
        self.assertNotIn('prompt_cache_key', body)
        self.assertNotIn('tools', body)

    def test_missing_api_key_raises_labelled_error(self):
        self.provider_opencode.sudo().api_key = False
        with self.assertRaisesRegex(UserError, 'OpenCode'):
            self.provider_opencode._request_responses(inputs=[])

    def test_catalog_seeded_with_unique_positive_models(self):
        models = self.provider_opencode.model_ids.filtered(
            lambda record: record.modality == 'chat'
        )
        self.assertGreaterEqual(len(models), 20)
        self.assertTrue(all(record.context_window > 0 for record in models))
        technical_names = models.mapped('technical_name')
        self.assertEqual(len(technical_names), len(set(technical_names)))
        self.assertIn('glm-5.3-flash-free', technical_names)
