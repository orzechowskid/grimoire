"""Tests for observer/filter.py: urgency, deadline, principle detection."""

import time
from memory_lib.observer.filter import (
    detect_urgency,
    parse_deadline,
    detect_principle,
    deterministic_filter,
)


# ── detect_urgency ──────────────────────────────────────────────────────────

class TestDetectUrgency:
    """Tests for detect_urgency(text)."""

    # --- deadline_h signals ---
    def test_detects_in_1_hour(self):
        urgency, deadline_ts = detect_urgency("Finish this in 1 hour")
        assert urgency == "deadline_h"
        assert deadline_ts is not None
        assert deadline_ts > int(time.time())

    def test_detects_asap(self):
        urgency, deadline_ts = detect_urgency("Do this ASAP")
        assert urgency == "deadline_h"
        assert deadline_ts is not None

    def test_detects_right_now(self):
        urgency, deadline_ts = detect_urgency("I need this right now")
        assert urgency == "deadline_h"
        assert deadline_ts is not None

    def test_detects_in_2_hours(self):
        urgency, deadline_ts = detect_urgency("Complete in 2 hours")
        assert urgency == "deadline_h"

    def test_detects_within_an_hour(self):
        urgency, deadline_ts = detect_urgency("Review within an hour")
        assert urgency == "deadline_h"

    # --- deadline_d signals ---
    def test_detects_by_tomorrow(self):
        urgency, deadline_ts = detect_urgency("Submit by tomorrow")
        assert urgency == "deadline_d"
        assert deadline_ts is not None

    def test_detects_due_today(self):
        urgency, deadline_ts = detect_urgency("This is due today")
        assert urgency == "deadline_d"

    def test_detects_deadline_today(self):
        urgency, deadline_ts = detect_urgency("Deadline today at 5pm")
        assert urgency == "deadline_d"

    def test_detects_by_eod(self):
        urgency, deadline_ts = detect_urgency("Need it by eod")
        assert urgency == "deadline_d"

    # --- deadline_w signals ---
    def test_detects_this_week(self):
        urgency, deadline_ts = detect_urgency("Finish this week")
        assert urgency == "deadline_w"

    def test_detects_by_friday(self):
        urgency, deadline_ts = detect_urgency("By friday please")
        assert urgency == "deadline_w"

    def test_detects_in_a_week(self):
        urgency, deadline_ts = detect_urgency("Complete in a week")
        assert urgency == "deadline_w"

    def test_detects_end_of_week(self):
        urgency, deadline_ts = detect_urgency("Due end of week")
        assert urgency == "deadline_w"

    def test_detects_by_eow(self):
        urgency, deadline_ts = detect_urgency("Submit by eow")
        assert urgency == "deadline_w"

    # --- no urgency ---
    def test_no_urgency_plain_text(self):
        urgency, deadline_ts = detect_urgency("Just a regular note")
        assert urgency == "none"
        assert deadline_ts is None

    def test_no_urgency_empty(self):
        urgency, deadline_ts = detect_urgency("")
        assert urgency == "none"
        assert deadline_ts is None

    def test_no_urgency_without_signals(self):
        urgency, deadline_ts = detect_urgency("We should consider this later")
        assert urgency == "none"
        assert deadline_ts is None

    # --- due next week (no specific urgency signal, falls back to none) ---
    def test_due_next_week_no_signal(self):
        urgency, deadline_ts = detect_urgency("due next week")
        assert urgency == "none"
        assert deadline_ts is None

    # --- precedence: more urgent levels win ---
    def test_urgency_h_takes_precedence_over_d(self):
        # "today" is deadline_d, "in 1 hour" is deadline_h — h should win
        urgency, _ = detect_urgency("in 1 hour by tomorrow")
        assert urgency == "deadline_h"

    def test_urgency_d_takes_precedence_over_w(self):
        urgency, _ = detect_urgency("today this week")
        assert urgency == "deadline_d"

    # --- deadline pattern fallback (date patterns trigger deadline_d) ---
    def test_date_pattern_triggers_deadline_d(self):
        urgency, deadline_ts = detect_urgency("Meeting on 3/15/2025")
        assert urgency == "deadline_d"
        assert deadline_ts is not None

    def test_short_date_pattern_triggers_deadline_d(self):
        urgency, deadline_ts = detect_urgency("Due 12/25")
        assert urgency == "deadline_d"
        assert deadline_ts is not None


# ── parse_deadline ──────────────────────────────────────────────────────────

class TestParseDeadline:
    """Tests for parse_deadline(text, level)."""

    def test_returns_future_timestamp(self):
        now = int(time.time())
        ts = parse_deadline("some text", "deadline_h")
        assert ts is not None
        assert ts > now

    def test_returns_non_none(self):
        ts = parse_deadline("hello world", "deadline_d")
        assert ts is not None

    def test_deadline_h_offset(self):
        now = int(time.time())
        ts = parse_deadline("in 1 hour", "deadline_h")
        assert now + 3500 < ts <= now + 3700, (
            f"Expected ~{now + 3600}, got {ts}"
        )

    def test_deadline_d_offset(self):
        now = int(time.time())
        ts = parse_deadline("due today", "deadline_d")
        assert now + 85000 < ts <= now + 86500, (
            f"Expected ~{now + 86400}, got {ts}"
        )

    def test_deadline_w_offset(self):
        now = int(time.time())
        ts = parse_deadline("this week", "deadline_w")
        assert now + 604000 < ts <= now + 605000, (
            f"Expected ~{now + 604800}, got {ts}"
        )

    def test_unknown_level_defaults_to_1_day(self):
        now = int(time.time())
        ts = parse_deadline("something", "unknown_level")
        assert now + 85000 < ts <= now + 86500, (
            f"Expected ~{now + 86400}, got {ts}"
        )

    def test_text_with_date_pattern(self):
        # Date pattern match also returns now + 86400
        now = int(time.time())
        ts = parse_deadline("on 1/15/2025", "deadline_h")
        assert ts == now + 86400

    def test_empty_text(self):
        now = int(time.time())
        ts = parse_deadline("", "deadline_w")
        assert ts == now + 604800

    def test_level_parameter_affects_offset(self):
        ts_h = parse_deadline("text", "deadline_h")
        ts_d = parse_deadline("text", "deadline_d")
        ts_w = parse_deadline("text", "deadline_w")
        assert ts_h < ts_d < ts_w


# ── detect_principle ────────────────────────────────────────────────────────

class TestDetectPrinciple:
    """Tests for detect_principle(text)."""

    # --- principle signals ---
    def test_detects_always(self):
        assert detect_principle("Always do X") is True

    def test_detects_never(self):
        assert detect_principle("Never do Y") is True

    def test_detects_remember_this(self):
        assert detect_principle("Remember this: clean code") is True

    def test_detects_non_negotiable(self):
        assert detect_principle("This is non-negotiable") is True

    def test_detects_project_rule(self):
        assert detect_principle("Project rule: no global state") is True

    def test_detects_architectural_principle(self):
        assert detect_principle("Architectural principle: loose coupling") is True

    def test_detects_this_is_a_rule(self):
        assert detect_principle("This is a rule: always test") is True

    def test_detects_hard_rule(self):
        assert detect_principle("Hard rule: no direct DB access") is True

    def test_detects_must_always(self):
        # "must" alone is IMPORTANT keyword; "always" makes it principle
        assert detect_principle("We must always validate input") is True

    # --- case-insensitive ---
    def test_case_insensitive_always(self):
        assert detect_principle("ALWAYS DO THIS") is True

    def test_case_insensitive_never(self):
        assert detect_principle("Never Ever Do That") is True

    def test_case_insensitive_project_rule(self):
        assert detect_principle("PROJECT RULE: strict typing") is True

    # --- non-principle text ---
    def test_no_principle_plain_text(self):
        assert detect_principle("Just a regular note") is False

    def test_no_principle_empty(self):
        assert detect_principle("") is False

    def test_no_principle_critical_text(self):
        # "must not" is in CRITICAL, but not in PRINCIPLE_SIGNALS
        assert detect_principle("This must not be done") is False

    def test_no_principle_important_text(self):
        assert detect_principle("This is an important artifact") is False

    def test_no_principle_without_signals(self):
        assert detect_principle("We decided on approach A") is False

    # --- signals inside longer text ---
    def test_principle_mid_sentence(self):
        assert detect_principle(
            "The team agreed: never bypass authentication"
        ) is True

    def test_multiple_principle_signals(self):
        assert detect_principle(
            "Always remember this: hard rule on testing"
        ) is True


# ── deterministic_filter ────────────────────────────────────────────────────

class TestDeterministicFilter:
    """Tests for deterministic_filter(text)."""

    def _check_required_keys(self, result):
        """Assert that all required keys are present."""
        required = {
            "importance",
            "conflict",
            "precision_items",
            "needs_ner",
            "urgency",
            "deadline_val",
            "discard",
        }
        assert set(result.keys()) == required, (
            f"Missing keys: {required - set(result.keys())}, "
            f"Extra keys: {set(result.keys()) - required}"
        )

    # --- returns dict with correct keys ---
    def test_returns_dict_with_all_keys(self):
        result = deterministic_filter("some text")
        self._check_required_keys(result)

    # --- handles empty text ---
    def test_empty_text(self):
        result = deterministic_filter("")
        assert result["importance"] == "background"
        assert result["conflict"] is False
        assert result["precision_items"] == []
        assert result["urgency"] == "none"
        assert result["deadline_val"] is None

    # --- principle detection ---
    def test_principle_importance(self):
        result = deterministic_filter("Always do this")
        assert result["importance"] == "principle"

    def test_principle_overrides_critical(self):
        # "never" is a principle signal; even though "must not" is critical,
        # principle takes priority
        result = deterministic_filter("never do this")
        assert result["importance"] == "principle"

    # --- critical detection ---
    def test_critical_importance(self):
        result = deterministic_filter("This is critical")
        assert result["importance"] == "critical"

    def test_critical_decided(self):
        result = deterministic_filter("The decision is decided")
        assert result["importance"] == "critical"

    def test_critical_forbidden(self):
        result = deterministic_filter("Access is forbidden")
        assert result["importance"] == "critical"

    def test_critical_final(self):
        result = deterministic_filter("This is the final version")
        assert result["importance"] == "critical"

    # --- important detection ---
    def test_important_importance_keyword(self):
        result = deterministic_filter("This is important")
        assert result["importance"] == "important"

    def test_important_must_keyword(self):
        result = deterministic_filter("We must deploy")
        assert result["importance"] == "important"

    def test_important_essential(self):
        result = deterministic_filter("An essential requirement")
        assert result["importance"] == "important"

    def test_important_key(self):
        result = deterministic_filter("Key dependency found")
        assert result["importance"] == "important"

    # --- important via structural importance (code) ---
    def test_structural_importance_code_block(self):
        result = deterministic_filter("```python\nprint('hello')\n```")
        assert result["importance"] == "important"

    def test_structural_importance_function_def(self):
        result = deterministic_filter("def my_function(): pass")
        assert result["importance"] == "important"

    def test_structural_importance_class_def(self):
        result = deterministic_filter("class MyClass: pass")
        assert result["importance"] == "important"

    def test_structural_importance_import(self):
        result = deterministic_filter("import os")
        assert result["importance"] == "important"

    def test_structural_importance_method_call(self):
        result = deterministic_filter("foo.bar() is important")
        assert result["importance"] == "important"

    def test_structural_importance_backtick(self):
        result = deterministic_filter("Use `some_code` here")
        assert result["importance"] == "important"

    # --- important via structural importance (long text) ---
    def test_long_text_importance(self):
        long_text = "a" * 301
        result = deterministic_filter(long_text)
        assert result["importance"] == "important"

    def test_short_text_no_structural_importance(self):
        result = deterministic_filter("short")
        assert result["importance"] == "background"

    # --- background detection ---
    def test_background_importance(self):
        result = deterministic_filter("just some random text")
        assert result["importance"] == "background"

    # --- conflict detection ---
    def test_conflict_true(self):
        result = deterministic_filter("But we changed the plan")
        assert result["conflict"] is True

    def test_conflict_false(self):
        result = deterministic_filter("Everything is fine")
        assert result["conflict"] is False

    def test_conflict_with_contradicts(self):
        result = deterministic_filter("This contradicts the spec")
        assert result["conflict"] is True

    def test_conflict_with_overrides(self):
        result = deterministic_filter("This overrides the previous rule")
        assert result["conflict"] is True

    # --- urgency detection ---
    def test_urgency_with_asap(self):
        result = deterministic_filter("Do this ASAP")
        assert result["urgency"] == "deadline_h"
        assert result["deadline_val"] is not None

    def test_urgency_with_tomorrow(self):
        result = deterministic_filter("Submit by tomorrow")
        assert result["urgency"] == "deadline_d"
        assert result["deadline_val"] is not None

    def test_urgency_with_this_week(self):
        result = deterministic_filter("Finish this week")
        assert result["urgency"] == "deadline_w"
        assert result["deadline_val"] is not None

    def test_urgency_none(self):
        result = deterministic_filter("No urgency here")
        assert result["urgency"] == "none"
        assert result["deadline_val"] is None

    # --- precision items detection ---
    def test_precision_items_with_url(self):
        result = deterministic_filter("See https://example.com")
        assert len(result["precision_items"]) > 0
        types = [item["type"] for item in result["precision_items"]]
        assert "link" in types

    def test_precision_items_with_email(self):
        result = deterministic_filter("Contact user@example.com")
        types = [item["type"] for item in result["precision_items"]]
        assert "email" in types

    def test_precision_items_empty(self):
        result = deterministic_filter("no special items here")
        assert result["precision_items"] == []

    def test_precision_items_type_value(self):
        result = deterministic_filter("See https://foo.com")
        item = result["precision_items"][0]
        assert "type" in item
        assert "value" in item
        assert item["type"] == "link"
        assert "https://foo.com" in item["value"]

    # --- needs_ner logic ---
    def test_needs_ner_for_background(self):
        result = deterministic_filter("a simple note")
        assert result["needs_ner"] is True

    def test_needs_ner_false_for_critical_with_precision(self):
        result = deterministic_filter(
            "This is critical. See https://example.com"
        )
        assert result["needs_ner"] is False

    def test_needs_ner_true_for_critical_without_precision(self):
        result = deterministic_filter("This is critical but no links")
        assert result["needs_ner"] is True

    def test_needs_ner_true_for_principle_without_precision(self):
        result = deterministic_filter("Always remember this")
        assert result["needs_ner"] is True

    # --- complete pipeline integration ---
    def test_principle_with_urgency(self):
        result = deterministic_filter(
            "Always do X by tomorrow"
        )
        assert result["importance"] == "principle"
        assert result["urgency"] == "deadline_d"
        assert result["deadline_val"] is not None

    def test_critical_with_conflict(self):
        result = deterministic_filter(
            "This is final but contradicts the old plan"
        )
        assert result["importance"] == "critical"
        assert result["conflict"] is True

    def test_comprehensive_text(self):
        result = deterministic_filter(
            "We must always validate input. This is critical "
            "but contradicts the previous decision. "
            "See https://docs.example.com for more."
        )
        assert result["importance"] == "principle"
        assert result["conflict"] is True
        assert len(result["precision_items"]) > 0
        assert result["urgency"] == "none"

    def test_important_with_urgency_and_code(self):
        result = deterministic_filter(
            "def validate(x): important by tomorrow"
        )
        assert result["importance"] == "important"
        assert result["urgency"] == "deadline_d"
        assert result["deadline_val"] is not None


class TestAgentMonologueDetection:
    """Tests for is_agent_monologue() and discard flag."""

    def test_monologue_detected_let_me(self):
        result = deterministic_filter("Let me check the code")
        assert result["discard"] is True

    def test_monologue_detected_now_exploring(self):
        result = deterministic_filter("Now exploring the file structure")
        assert result["discard"] is True

    def test_monologue_with_url_not_discarded(self):
        result = deterministic_filter("Let me check https://example.com")
        assert result["discard"] is False  # has precision item (URL)

    def test_monologue_with_code_not_discarded(self):
        result = deterministic_filter("Let me check def foo(): pass")
        assert result["discard"] is False  # has code pattern

    def test_monologue_long_text_not_discarded(self):
        long_text = "Let me check " + "a" * 100
        result = deterministic_filter(long_text)
        assert result["discard"] is False  # > 100 chars

    def test_non_monologue_not_discarded(self):
        result = deterministic_filter("Replaced axios with fetch")
        assert result["discard"] is False

    def test_i_will_monologue(self):
        result = deterministic_filter("I will now examine the code")
        assert result["discard"] is True

    def test_analyzing_monologue(self):
        result = deterministic_filter("Analyzing the test results")
        assert result["discard"] is True

    def test_substantive_text_not_discarded(self):
        result = deterministic_filter("The build failed with exit code 1")
        assert result["discard"] is False

    def test_empty_text_not_discarded(self):
        result = deterministic_filter("")
        assert result["discard"] is False
