"""Tests for observer/marker.py: temporal, emotion, keyword classification."""

import pytest

from memory_lib.observer.entities import (
    Entity,
    EmotionCharge,
    EntityType,
    Explicitness,
    TemporalMarker,
    TemporalRelations,
    TimeRef,
)
from memory_lib.observer.marker import (
    _build_t_rel,
    _detect_emotion_charge,
    _importance_from_label,
    _keyword_classify,
    infer_temporal,
    structural_prefilter,
)


# ---------------------------------------------------------------------------
# infer_temporal
# ---------------------------------------------------------------------------


class TestInferTemporal:
    def test_past_tense_simple(self):
        result = infer_temporal("the code was broken", [])
        assert result.gram_time == TimeRef.PAST
        assert result.ref_time == TimeRef.PAST
        assert result.explicitness == Explicitness.EXPLICIT
        assert result.confidence == 1.0

    def test_past_tense_multiple_patterns(self):
        for word in ["was", "were", "had", "been", "yesterday", "previously", "before"]:
            result = infer_temporal(f"something {word} happened", [])
            assert result.gram_time == TimeRef.PAST
            assert result.ref_time == TimeRef.PAST
            assert result.explicitness == Explicitness.EXPLICIT
            assert result.confidence == 1.0

    def test_future_tense_simple(self):
        result = infer_temporal("we will fix this tomorrow", [])
        assert result.gram_time == TimeRef.FUTURE
        assert result.ref_time == TimeRef.FUTURE
        assert result.explicitness == Explicitness.EXPLICIT
        assert result.confidence == 1.0

    def test_future_tense_patterns(self):
        for pattern in ["will", "shall", "going to", "tomorrow", "next week", "soon"]:
            result = infer_temporal(f"we {pattern} do it", [])
            assert result.gram_time == TimeRef.FUTURE
            assert result.ref_time == TimeRef.FUTURE
            assert result.explicitness == Explicitness.EXPLICIT
            assert result.confidence == 1.0

    def test_present_with_chain(self):
        chain = [Entity.create("previous fact", EntityType.FACT, "user")]
        result = infer_temporal("something interesting", chain)
        assert result.gram_time == TimeRef.PRESENT
        assert result.ref_time == TimeRef.PAST
        assert result.explicitness == Explicitness.INFERRED
        assert result.confidence == 0.8

    def test_unknown_no_chain(self):
        result = infer_temporal("some text without temporal cues", [])
        assert result.gram_time == TimeRef.UNKNOWN
        assert result.ref_time == TimeRef.UNKNOWN
        assert result.explicitness == Explicitness.LOST
        assert result.confidence == 0.3

    def test_past_tense_overrides_chain(self):
        """Past tense should return PAST even when chain is present."""
        chain = [Entity.create("previous fact", EntityType.FACT, "user")]
        result = infer_temporal("it was done before", chain)
        assert result.gram_time == TimeRef.PAST
        assert result.ref_time == TimeRef.PAST
        assert result.explicitness == Explicitness.EXPLICIT
        assert result.confidence == 1.0

    def test_case_insensitive_past(self):
        result = infer_temporal("The Code Was Broken", [])
        assert result.gram_time == TimeRef.PAST
        assert result.explicitness == Explicitness.EXPLICIT

    def test_case_insensitive_future(self):
        result = infer_temporal("We Will Fix This Tomorrow", [])
        assert result.gram_time == TimeRef.FUTURE
        assert result.explicitness == Explicitness.EXPLICIT


# ---------------------------------------------------------------------------
# _build_t_rel
# ---------------------------------------------------------------------------


class TestBuildTRel:
    def test_empty_chain(self):
        temp = TemporalMarker()
        result = _build_t_rel(temp, [])
        assert result.is_empty()

    def test_inferred_past(self):
        temp = TemporalMarker(
            TimeRef.PAST, TimeRef.PAST, Explicitness.INFERRED, 0.8
        )
        chain = [Entity.create("first", EntityType.FACT, "user")]
        result = _build_t_rel(temp, chain)
        assert not result.is_empty()
        assert result.after == [chain[0].id]
        assert result.before == []

    def test_explicit_past(self):
        temp = TemporalMarker(
            TimeRef.PAST, TimeRef.PAST, Explicitness.EXPLICIT, 1.0
        )
        chain = [Entity.create("first", EntityType.FACT, "user")]
        result = _build_t_rel(temp, chain)
        assert result.after == [chain[0].id]
        assert result.before == []

    def test_explicit_future(self):
        temp = TemporalMarker(
            TimeRef.FUTURE, TimeRef.FUTURE, Explicitness.EXPLICIT, 1.0
        )
        chain = [Entity.create("first", EntityType.FACT, "user")]
        result = _build_t_rel(temp, chain)
        assert result.after == []
        assert result.before == [chain[0].id]

    def test_multiple_entities_uses_last(self):
        temp = TemporalMarker(
            TimeRef.PAST, TimeRef.PAST, Explicitness.INFERRED, 0.8
        )
        chain = [
            Entity.create("first", EntityType.FACT, "user"),
            Entity.create("second", EntityType.FACT, "user"),
            Entity.create("third", EntityType.FACT, "user"),
        ]
        result = _build_t_rel(temp, chain)
        assert result.after == [chain[-1].id]

    def test_explicit_present_no_relation(self):
        """Explicit present-time should return empty relations."""
        temp = TemporalMarker(
            TimeRef.PRESENT, TimeRef.PRESENT, Explicitness.EXPLICIT, 1.0
        )
        chain = [Entity.create("first", EntityType.FACT, "user")]
        result = _build_t_rel(temp, chain)
        assert result.is_empty()


# ---------------------------------------------------------------------------
# structural_prefilter
# ---------------------------------------------------------------------------


class TestStructuralPrefilter:
    def test_short_text(self):
        assert structural_prefilter("hi") is False
        assert structural_prefilter("a") is False
        assert structural_prefilter("ab") is False
        assert structural_prefilter("abc") is False
        assert structural_prefilter("abcd") is False

    def test_short_text_with_padding(self):
        # Less than 5 chars after strip
        assert structural_prefilter(" hi ") is False

    def test_text_with_word(self):
        assert structural_prefilter("hello world") is True
        assert structural_prefilter("a bb cc") is True

    def test_no_word_pattern(self):
        # Has >= 5 chars but no word of length >= 2
        assert structural_prefilter("a a a a a") is False
        assert structural_prefilter("!@# $%^ &*(") is False

    def test_whitespace_only(self):
        assert structural_prefilter("     ") is False

    def test_exact_boundary_length(self):
        # Exactly 5 chars, needs \w{2,}
        assert structural_prefilter("a b c") is False  # 5 chars, no word >= 2
        assert structural_prefilter("ab cd") is True  # 5 chars, words >= 2

    def test_empty_string(self):
        assert structural_prefilter("") is False


# ---------------------------------------------------------------------------
# _detect_emotion_charge
# ---------------------------------------------------------------------------


class TestDetectEmotionCharge:
    def test_positive_single(self):
        charge, intensity = _detect_emotion_charge("this is great")
        assert charge == EmotionCharge.POSITIVE
        assert intensity == pytest.approx(0.6, abs=1e-9)  # 0.4 + 1 * 0.2 = 0.6

    def test_positive_multiple(self):
        charge, intensity = _detect_emotion_charge("great works finally")
        assert charge == EmotionCharge.POSITIVE
        assert intensity == 1.0  # 0.4 + 3 * 0.2 = 1.0, capped at 1.0

    def test_positive_two_matches(self):
        charge, intensity = _detect_emotion_charge("great works")
        assert charge == EmotionCharge.POSITIVE
        assert intensity == 0.8  # 0.4 + 2 * 0.2 = 0.8

    def test_negative_single(self):
        charge, intensity = _detect_emotion_charge("this is terrible")
        assert charge == EmotionCharge.NEGATIVE
        assert intensity == pytest.approx(0.6, abs=1e-9)  # 0.4 + 1 * 0.2 = 0.6

    def test_negative_multiple(self):
        charge, intensity = _detect_emotion_charge("terrible broken failed")
        assert charge == EmotionCharge.NEGATIVE
        assert intensity == 1.0  # 0.4 + 3 * 0.2 = 1.0, capped at 1.0

    def test_equal_positive_negative(self):
        """When pos_count == neg_count and both > 0, returns UNCERTAIN."""
        charge, intensity = _detect_emotion_charge("great terrible")
        assert charge == EmotionCharge.UNCERTAIN
        assert intensity == 0.5

    def test_equal_positive_negative_multiple(self):
        charge, intensity = _detect_emotion_charge("great works terrible broken")
        assert charge == EmotionCharge.UNCERTAIN
        assert intensity == 0.5

    def test_neutral_no_keywords(self):
        charge, intensity = _detect_emotion_charge("the code compiles fine")
        assert charge == EmotionCharge.NEUTRAL
        assert intensity == 0.2

    def test_case_insensitive(self):
        charge, intensity = _detect_emotion_charge("This Is Great")
        assert charge == EmotionCharge.POSITIVE

    def test_negative_words_list(self):
        """All negative keywords that should match."""
        for word in ["terrible", "broken", "failed", "error", "bug", "crash"]:
            charge, intensity = _detect_emotion_charge(f"this is {word}")
            assert charge == EmotionCharge.NEGATIVE

    def test_positive_words_list(self):
        """All positive keywords that should match."""
        for word in ["great", "works", "finally", "success", "fixed", "perfect"]:
            charge, intensity = _detect_emotion_charge(f"this is {word}")
            assert charge == EmotionCharge.POSITIVE


# ---------------------------------------------------------------------------
# _keyword_classify
# ---------------------------------------------------------------------------


class TestKeywordClassify:
    def test_decision_keywords(self):
        for word in ["decided", "chosen", "rejected", "forbidden"]:
            result = _keyword_classify(f"I {word} to proceed")
            assert result == EntityType.DECISION.value

    def test_code_keywords(self):
        for pattern in ["def foo(", "class Bar:", "import os", "async def"]:
            result = _keyword_classify(f"{pattern} ...")
            assert result == EntityType.CODE.value

    def test_question_question_mark(self):
        result = _keyword_classify("Is this correct?")
        assert result == EntityType.QUESTION.value

    def test_question_wh_words(self):
        for word in ["why", "how", "what", "when", "where"]:
            result = _keyword_classify(f"{word} is this happening?")
            assert result == EntityType.QUESTION.value

    def test_question_boundary_matching(self):
        """'why' should not match as substring of other words."""
        # "mywhys" should not match \bwhy\b
        result = _keyword_classify("mywhys are many")
        assert result != EntityType.QUESTION.value

        # "whatever" should not match \bwhat\b
        result = _keyword_classify("whatever happens")
        assert result != EntityType.QUESTION.value

        # "whereas" should not match \bwhere\b
        result = _keyword_classify("whereas others do")
        assert result != EntityType.QUESTION.value

        # "whenever" should not match \bwhen\b
        result = _keyword_classify("whenever possible")
        assert result != EntityType.QUESTION.value

    def test_question_word_boundary_positive(self):
        """'why' as standalone word should match."""
        result = _keyword_classify("why did this happen")
        assert result == EntityType.QUESTION.value

    def test_result_keywords(self):
        for word in ["done", "complete", "finished", "failed"]:
            result = _keyword_classify(f"it is {word}")
            assert result == EntityType.RESULT.value

    def test_emotion_keywords(self):
        for word in ["great", "terrible", "wrong", "works", "finally"]:
            result = _keyword_classify(f"this feels {word}")
            assert result == "emotion"

    def test_default_fact(self):
        result = _keyword_classify("the project is ongoing")
        assert result == EntityType.FACT.value

    def test_first_match_wins(self):
        """Decision keyword should match before fact even if fact keywords appear."""
        result = _keyword_classify("I decided a critical important fact")
        assert result == EntityType.DECISION.value

    def test_case_insensitive_classification(self):
        result = _keyword_classify("The Decided to proceed")
        assert result == EntityType.DECISION.value

    def test_code_with_function_keyword(self):
        result = _keyword_classify("def process_data():")
        assert result == EntityType.CODE.value

    def test_code_with_import_keyword(self):
        result = _keyword_classify("import numpy as np")
        assert result == EntityType.CODE.value

    def test_code_with_async_keyword(self):
        result = _keyword_classify("async def handler():")
        assert result == EntityType.CODE.value


# ---------------------------------------------------------------------------
# _importance_from_label
# ---------------------------------------------------------------------------


class TestImportanceFromLabel:
    def test_decision(self):
        assert _importance_from_label(EntityType.DECISION.value) == 0.9

    def test_principle(self):
        assert _importance_from_label("principle") == 1.0

    def test_urgency(self):
        assert _importance_from_label("urgency") == 0.85

    def test_fact(self):
        assert _importance_from_label(EntityType.FACT.value) == 0.6

    def test_code(self):
        assert _importance_from_label(EntityType.CODE.value) == 0.7

    def test_event(self):
        assert _importance_from_label(EntityType.EVENT.value) == 0.65

    def test_question(self):
        assert _importance_from_label(EntityType.QUESTION.value) == 0.5

    def test_result(self):
        assert _importance_from_label(EntityType.RESULT.value) == 0.75

    def test_unknown_label(self):
        assert _importance_from_label("unknown_label") == 0.5

    def test_random_string(self):
        assert _importance_from_label("xyz") == 0.5

    def test_empty_string(self):
        assert _importance_from_label("") == 0.5

    def test_all_known_labels_return_correct_values(self):
        expected = {
            EntityType.DECISION.value: 0.9,
            "principle": 1.0,
            "urgency": 0.85,
            EntityType.FACT.value: 0.6,
            EntityType.CODE.value: 0.7,
            EntityType.EVENT.value: 0.65,
            EntityType.QUESTION.value: 0.5,
            EntityType.RESULT.value: 0.75,
        }
        for label, importance in expected.items():
            assert _importance_from_label(label) == importance, f"Failed for label: {label}"
