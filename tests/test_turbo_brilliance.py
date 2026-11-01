"""Unit tests for the DavidAU Qwen3.8 Turbo-Brilliance integration in JARVS."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvs import client, config, turbo_brilliance  # noqa: E402
from jarvs.chat import JarvsSession, dispatch_command  # noqa: E402


class _SilentVoice:
    enabled = False

    def speak(self, text): pass
    def speak_synced(self, text, reveal): pass
    def set_enabled(self, on): pass
    def stop(self): pass
    def status(self): return "voice: unavailable"


class TurboBrillianceJarvsTests(unittest.TestCase):
    def setUp(self):
        self._saved_state = client.saved_reason_state()
        self._saved_thinking = config.THINKING_ENABLED
        client.REASON_TAGS_ON = True
        client.REASON_TAGS_SOURCE = "test"
        client.REASON_TAG_MAIN = "{REASON:low}"
        client.REASON_TAG_BACKGROUND = "{REASON:medium}"
        client.REASON_TAG_CRITIC = "{REASON:ioff}"
        client.REASON_TAG_DREAM = "{REASON:ioff}"
        client.REASON_TAG_CLAUDE = "{REASON:ioff}"
        client.REASON_TAG_ROUTER = "{REASON:ioff}"
        client.REASON_TAG_OFF = "{REASON:ioff}"
        client.CACHE_PROMPT_MODE = "auto"
        client.REASON_PANEL_DETAIL = False
        client.ACTIVE_CONSTRUCT = ""
        client.CONSTRUCT_BLEND = True
        client.CONSTRUCT_INSTRUCT = False
        client.HOLODECK_ON = False
        client.HOLODECK_INSTRUCT = False
        config.THINKING_ENABLED = True

    def tearDown(self):
        config.THINKING_ENABLED = self._saved_thinking
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tmp_path = tf.name
        try:
            client.save_reason_state(tmp_path)
            # Restore original state
            for k, v in self._saved_state.items():
                pass
            client.REASON_TAGS_ON = self._saved_state["enabled"]
            client.REASON_TAG_MAIN = self._saved_state["main"]
            client.REASON_TAG_BACKGROUND = self._saved_state["background"]
            client.REASON_TAG_CRITIC = self._saved_state["critic"]
            client.REASON_TAG_DREAM = self._saved_state["dream"]
            client.REASON_TAG_CLAUDE = self._saved_state["claude"]
            client.REASON_TAG_ROUTER = self._saved_state["router"]
            client.REASON_TAG_OFF = self._saved_state["off"]
            client.CACHE_PROMPT_MODE = self._saved_state["cache_prompt"]
            client.REASON_PANEL_DETAIL = self._saved_state["panel_detail"]
            client.ACTIVE_CONSTRUCT = self._saved_state["construct"]
            client.CONSTRUCT_BLEND = self._saved_state["construct_blend"]
            client.CONSTRUCT_INSTRUCT = self._saved_state["construct_instruct"]
            client.HOLODECK_ON = self._saved_state["holodeck"]
            client.HOLODECK_INSTRUCT = self._saved_state["holodeck_instruct"]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_template_inspection_and_mode_normalization(self):
        fake_template = (
            "{%- set system_instruction_teacher_DAU = 'SUPER_TEACHER_PROMPT' -%}"
            "{%- if '{REASON:' in message.content -%}"
            "{%- set reasoning_map = {'omni': system_instruction_omni_DAU, "
            "'iomni': system_instruction_omni_DAU, 'stark': system_instruction_stark_DAU, "
            "'istark': system_instruction_stark_DAU, 'teacher': system_instruction__DAU} -%}"
        )
        info = turbo_brilliance.inspect_chat_template(fake_template)
        self.assertTrue(info["detected"])
        self.assertEqual(info["teacher_instruction"], "SUPER_TEACHER_PROMPT")
        self.assertIn("stark", info["modes"])

        self.assertEqual(turbo_brilliance.normalize_instruct_reason_tag("off"), "{REASON:ioff}")
        self.assertEqual(turbo_brilliance.normalize_instruct_reason_tag("stark"), "{REASON:istark}")
        self.assertEqual(turbo_brilliance.normalize_instruct_reason_tag("icoder"), "{REASON:icoder}")

    def test_task_routing_and_presets(self):
        self.assertEqual(client.reason_tag_for_task("main", reason=True), "{REASON:low}")
        self.assertEqual(client.reason_tag_for_task("main", reason=False), "{REASON:ioff}")
        self.assertEqual(client.reason_tag_for_task("background", reason=True), "{REASON:medium}")
        self.assertEqual(client.reason_tag_for_task("critic", reason=False), "{REASON:ioff}")
        self.assertEqual(client.reason_tag_for_task("dream", reason=False), "{REASON:ioff}")
        self.assertEqual(client.reason_tag_for_task("claude", reason=False), "{REASON:ioff}")
        self.assertEqual(client.reason_tag_for_task("router", reason=False), "{REASON:ioff}")

        with mock.patch.object(client, "save_reason_state"):
            out = dispatch_command(mock.MagicMock(), "/reason preset jarvis")
            self.assertIn("jarvis", out)
            self.assertEqual(client.active_reason_preset(), "jarvis")
            self.assertEqual(client.REASON_TAG_MAIN, "{REASON:stark}")
            self.assertEqual(client.REASON_TAG_BACKGROUND, "{REASON:spoon}")
            self.assertEqual(client.REASON_TAG_CRITIC, "{REASON:ilogic}")
            self.assertEqual(client.REASON_TAG_DREAM, "{REASON:ideeptree}")
            self.assertEqual(client.REASON_TAG_CLAUDE, "{REASON:icoder}")
            self.assertEqual(client.REASON_TAG_ROUTER, "{REASON:ioff}")
            self.assertEqual(client.REASON_TAG_OFF, "{REASON:istark}")

    def test_inline_tag_hoisting_and_clean_history(self):
        session = JarvsSession(speak=False, voice=_SilentVoice())
        captured = {}

        def fake_gen_with_history(system, messages, **kw):
            captured["system"] = system
            captured["messages"] = messages
            captured["kw"] = kw
            return "◤H◥ It's not lupus. [STATUS: ONLINE]"

        with mock.patch.object(client, "generate_with_history", side_effect=fake_gen_with_history):
            reply = session.reply("{REASON:ihouse} Check these symptoms.", learn=False)
            self.assertEqual(captured["kw"].get("reason_tag"), "{REASON:ihouse}")
            # History records cleaned user text without {REASON:ihouse}
            self.assertEqual(session.history[0]["content"], "Check these symptoms.")
            # System prompt includes Dr. Gregory House voice framing
            self.assertIn("Dr. Gregory House", captured["system"])
            self.assertIn("[VOICE & COGNITIVE LENS", captured["system"])
            self.assertTrue(reply)

    def test_cache_prompt_and_enable_thinking_kwargs(self):
        base = [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Check"}]
        client.REASON_TAG_CRITIC = "{REASON:ilogic}"
        tagged_critic = client._with_reason_tag(base, reason=False, task="critic")
        self.assertEqual(tagged_critic[-1]["content"], "{REASON:ilogic}\nCheck")
        self.assertTrue(client._should_disable_thinking_kwarg(base, tagged_critic, reason=False))
        self.assertEqual(client._resolve_cache_prompt_kwarg(base, tagged_critic, task="critic"), False)

        # Router uses {REASON:ioff} (pure depth tag -> does not mutate system prompt at token 0)
        tagged_router = client._with_reason_tag(base, reason=False, task="router")
        self.assertEqual(tagged_router[-1]["content"], "{REASON:ioff}\nCheck")
        self.assertIsNone(client._resolve_cache_prompt_kwarg(base, tagged_router, task="router"))

    def test_commands_and_state_persistence(self):
        orig_save = client.save_reason_state
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tmp_path = tf.name
        try:
            with mock.patch.object(client, "save_reason_state", side_effect=lambda: orig_save(tmp_path)):
                dispatch_command(mock.MagicMock(), "/reason xdeeptree")
                self.assertEqual(client.REASON_TAG_MAIN, "{REASON:xdeeptree}")

                dispatch_command(mock.MagicMock(), "/construct pkd blend instruct")
                self.assertEqual(client.ACTIVE_CONSTRUCT, "pkd")
                self.assertTrue(client.CONSTRUCT_BLEND)
                self.assertTrue(client.CONSTRUCT_INSTRUCT)
                self.assertEqual(client.reason_tag_for_task("main"), "{REASON:ipkd}")

                dispatch_command(mock.MagicMock(), "/holodeck on")
                self.assertTrue(client.HOLODECK_ON)
                self.assertEqual(client.reason_tag_for_task("main"), "{REASON:holodeck}")

            # Reset and reload from tmp_path
            client.REASON_TAG_MAIN = "{REASON:low}"
            client.ACTIVE_CONSTRUCT = ""
            client.HOLODECK_ON = False
            self.assertTrue(client.load_reason_state(tmp_path))
            self.assertEqual(client.REASON_TAG_MAIN, "{REASON:xdeeptree}")
            self.assertEqual(client.ACTIVE_CONSTRUCT, "pkd")
            self.assertTrue(client.HOLODECK_ON)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
