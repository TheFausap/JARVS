"""Reply-path web-search decision tests — should_search / looks_time_sensitive. The
standing 'verify online' rule lowers the bar so version/availability questions trigger a
search even without the explicit markers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jarvs.web_access import (  # noqa: E402
    needs_web_search, looks_time_sensitive, should_search,
    is_search_followup, last_searchable,
)
from jarvs.chat import distill_search_query  # noqa: E402


class DecisionTests(unittest.TestCase):
    def test_explicit_markers_always_search(self):
        for t in ["check the internet for the latest Gemma model",
                  "what is the current price of bitcoin?",
                  "search for recent news on this"]:
            self.assertTrue(should_search(t, lower_bar=False), t)

    def test_version_question_needs_the_rule(self):
        q = "is Gemma 4 available yet?"
        # No explicit web marker, so the default bar does not trigger...
        self.assertFalse(needs_web_search(q))
        self.assertFalse(should_search(q, lower_bar=False))
        # ...but it is clearly time-sensitive, so the verify rule lowers the bar.
        self.assertTrue(looks_time_sensitive(q))
        self.assertTrue(should_search(q, lower_bar=True))

    def test_bare_version_and_existence_questions_search_under_rule(self):
        # The phrasings that previously slipped through and made JARVS confabulate.
        for t in ["is there a Gemma 3 or Gemma 4?",
                  "has Llama 3 been released?",
                  "did GPT 5 come out?",
                  "what's new with Google's models?"]:
            self.assertTrue(should_search(t, lower_bar=True), t)

    def test_meta_question_about_jarvs_does_not_search(self):
        # Follow-ups about the assistant/conversation must not trigger a (useless) search.
        for t in ["did you find something?", "are you sure?", "can you check again?"]:
            self.assertFalse(should_search(t, lower_bar=True), t)

    def test_generic_factual_question_searches_under_rule_only(self):
        q = "who is the current prime minister of Italy?"
        self.assertTrue(should_search(q, lower_bar=True))

    def test_non_time_sensitive_never_searches(self):
        for t in ["tell me a joke", "write a haiku about autumn"]:
            self.assertFalse(should_search(t, lower_bar=False), t)
            self.assertFalse(should_search(t, lower_bar=True), t)


class QueryDistillTests(unittest.TestCase):
    def test_uses_model_query(self):
        # The model strips the chatter down to key terms.
        got = distill_search_query(
            "I admire your stubbornness, but can you look online for Gemma 3 and Gemma 4?",
            lambda s, u, r: "Gemma 3 Gemma 4 release")
        self.assertEqual(got, "Gemma 3 Gemma 4 release")

    def test_strips_quotes_and_whitespace(self):
        self.assertEqual(
            distill_search_query("find X", lambda s, u, r: '  "latest X model"  '),
            "latest X model")

    def test_falls_back_to_cleaned_text_when_model_empty(self):
        self.assertEqual(distill_search_query("look up Gemma 4", lambda s, u, r: ""),
                         "look up Gemma 4")

    def test_falls_back_on_model_error(self):
        def boom(s, u, r):
            raise RuntimeError("down")
        self.assertEqual(distill_search_query("check Gemma 4  now", boom),
                         "check Gemma 4 now")


class FollowupBridgeTests(unittest.TestCase):
    def test_detects_followups(self):
        for t in ["did you find something?", "any luck?", "what did you find?",
                  "and?", "found anything?"]:
            self.assertTrue(is_search_followup(t), t)

    def test_ignores_non_followups(self):
        for t in ["tell me a joke", "what is your name?"]:
            self.assertFalse(is_search_followup(t), t)

    def test_last_searchable_recovers_prior_question(self):
        history = [
            {"role": "user", "content": "please look for Gemma 3 and Gemma 4"},
            {"role": "assistant", "content": "I am searching now, stand by."},
            {"role": "user", "content": "did you find something?"},
        ]
        # The follow-up is not itself searchable, but the prior question was.
        self.assertFalse(should_search("did you find something?", lower_bar=True))
        self.assertEqual(last_searchable(history),
                         "please look for Gemma 3 and Gemma 4")

    def test_last_searchable_empty_when_nothing_prior(self):
        history = [{"role": "user", "content": "hello there"},
                   {"role": "assistant", "content": "Hello."}]
        self.assertEqual(last_searchable(history), "")


if __name__ == "__main__":
    unittest.main()
