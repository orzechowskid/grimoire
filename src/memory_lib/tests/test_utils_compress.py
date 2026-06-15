"""Tests for observer/utils.py: compress_text"""

from memory_lib.observer.utils import compress_text


# ---------------------------------------------------------------------------
# Entity prefix replacement — person
# ---------------------------------------------------------------------------
class TestPersonEntities:
    def test_person_replaced_with_per_prefix(self):
        text = "John Doe went to the store."
        entities = [{"type": "person", "value": "John Doe"}]
        brief, tags = compress_text(text, entities)
        assert "per:john doe" in tags

    def test_multiple_persons_same_tag(self):
        text = "Alice and Bob talked."
        entities = [
            {"type": "person", "value": "Alice"},
            {"type": "person", "value": "Bob"},
        ]
        brief, tags = compress_text(text, entities)
        assert "per:alice" in tags
        assert "per:bob" in tags
        assert tags.count("per:alice") == 1  # deduped even if same person listed twice

    def test_person_deduplication(self):
        text = "Alice and Alice met."
        entities = [
            {"type": "person", "value": "Alice"},
            {"type": "person", "value": "Alice"},
        ]
        brief, tags = compress_text(text, entities)
        assert tags.count("per:alice") == 1

    def test_person_value_lowercased(self):
        text = "JOHN DOE spoke."
        entities = [{"type": "person", "value": "JOHN DOE"}]
        brief, tags = compress_text(text, entities)
        assert "per:john doe" in tags


# ---------------------------------------------------------------------------
# Entity prefix replacement — organization
# ---------------------------------------------------------------------------
class TestOrganizationEntities:
    def test_org_replaced_with_org_prefix(self):
        text = "Google announced new products."
        entities = [{"type": "organization", "value": "Google"}]
        brief, tags = compress_text(text, entities)
        assert "org:google" in tags

    def test_multiple_organizations(self):
        text = "Microsoft and Apple competed."
        entities = [
            {"type": "organization", "value": "Microsoft"},
            {"type": "organization", "value": "Apple"},
        ]
        brief, tags = compress_text(text, entities)
        assert "org:microsoft" in tags
        assert "org:apple" in tags

    def test_org_deduplication(self):
        text = "Google and Google reported."
        entities = [
            {"type": "organization", "value": "Google"},
            {"type": "organization", "value": "Google"},
        ]
        brief, tags = compress_text(text, entities)
        assert tags.count("org:google") == 1

    def test_org_value_lowercased(self):
        text = "GOOGLE released updates."
        entities = [{"type": "organization", "value": "GOOGLE"}]
        brief, tags = compress_text(text, entities)
        assert "org:google" in tags


# ---------------------------------------------------------------------------
# Entity prefix replacement — address (location)
# ---------------------------------------------------------------------------
class TestAddressEntities:
    def test_address_replaced_with_loc_prefix(self):
        text = "Paris is beautiful."
        entities = [{"type": "address", "value": "Paris"}]
        brief, tags = compress_text(text, entities)
        assert "loc:paris" in tags

    def test_address_value_lowercased(self):
        text = "NEW YORK is big."
        entities = [{"type": "address", "value": "NEW YORK"}]
        brief, tags = compress_text(text, entities)
        assert "loc:new york" in tags


# ---------------------------------------------------------------------------
# Entity prefix replacement — date
# ---------------------------------------------------------------------------
class TestDateEntities:
    def test_date_replaced_with_date_prefix(self):
        text = "On January 1st, something happened."
        entities = [{"type": "date", "value": "January 1st"}]
        brief, tags = compress_text(text, entities)
        assert "date:january 1st" in tags

    def test_multiple_dates(self):
        text = "January and February are early months."
        entities = [
            {"type": "date", "value": "January"},
            {"type": "date", "value": "February"},
        ]
        brief, tags = compress_text(text, entities)
        assert "date:january" in tags
        assert "date:february" in tags


# ---------------------------------------------------------------------------
# Entity prefix replacement — technology
# ---------------------------------------------------------------------------
class TestTechnologyEntities:
    def test_tech_replaced_with_tech_prefix(self):
        text = "Python is great for ML."
        entities = [{"type": "technology", "value": "Python"}]
        brief, tags = compress_text(text, entities)
        # Technology tags now use the format "tech:<value>"
        assert "tech:python" in tags

    def test_multiple_technologies(self):
        text = "Python and Rust are fast."
        entities = [
            {"type": "technology", "value": "Python"},
            {"type": "technology", "value": "Rust"},
        ]
        brief, tags = compress_text(text, entities)
        # Each technology now gets its own tag from its value
        assert "tech:python" in tags
        assert "tech:rust" in tags


# ---------------------------------------------------------------------------
# Entity prefix replacement — code
# ---------------------------------------------------------------------------
class TestCodeEntities:
    def test_code_replaced_with_code_prefix(self):
        text = "The function returned null."
        entities = [{"type": "code", "value": "null"}]
        brief, tags = compress_text(text, entities)
        # Code tags now use the format "code:<value>"
        assert "code:null" in tags

    def test_multiple_codes(self):
        text = "The variables x and y differ."
        entities = [
            {"type": "code", "value": "x"},
            {"type": "code", "value": "y"},
        ]
        brief, tags = compress_text(text, entities)
        # Each code entity now gets its own tag from its value
        assert "code:x" in tags
        assert "code:y" in tags


# ---------------------------------------------------------------------------
# No entities
# ---------------------------------------------------------------------------
class TestNoEntities:
    def test_no_entities_unchanged_tags(self):
        text = "Simple text with no entities."
        brief, tags = compress_text(text, [])
        assert "ctx:simple" in tags

    def test_none_entities_unchanged_tags(self):
        text = "Simple text with no entities."
        brief, tags = compress_text(text, None)
        assert "ctx:simple" in tags

    def test_no_entities_brief_equals_text(self):
        text = "Short text."
        brief, tags = compress_text(text, [])
        assert brief == "Short text."


# ---------------------------------------------------------------------------
# Empty text
# ---------------------------------------------------------------------------
class TestEmptyText:
    def test_empty_text(self):
        text = ""
        brief, tags = compress_text(text, [])
        assert brief == ""
        assert tags == []

    def test_empty_text_with_entities(self):
        text = ""
        entities = [{"type": "person", "value": "Alice"}]
        brief, tags = compress_text(text, entities)
        assert brief == ""
        assert "per:alice" in tags


# ---------------------------------------------------------------------------
# Multiple entities of different types
# ---------------------------------------------------------------------------
class TestMultipleEntityTypes:
    def test_person_and_org(self):
        text = "Google hired John."
        entities = [
            {"type": "organization", "value": "Google"},
            {"type": "person", "value": "John"},
        ]
        brief, tags = compress_text(text, entities)
        assert "org:google" in tags
        assert "per:john" in tags

    def test_all_entity_types(self):
        text = "Google hired John on 2024-01-01 to build Python on Linux."
        entities = [
            {"type": "organization", "value": "Google"},
            {"type": "person", "value": "John"},
            {"type": "date", "value": "2024-01-01"},
            {"type": "technology", "value": "Python"},
            {"type": "address", "value": "Linux"},
        ]
        brief, tags = compress_text(text, entities)
        assert "org:google" in tags
        assert "per:john" in tags
        assert "date:2024-01-01" in tags
        assert "tech:python" in tags
        assert "loc:linux" in tags


# ---------------------------------------------------------------------------
# Overlapping / duplicate entities
# ---------------------------------------------------------------------------
class TestOverlappingEntities:
    def test_same_value_different_types(self):
        # "Linux" appears as both address and technology — dedup is by tag,
        # so "loc:linux" and "tech:linux" are different tags, both kept.
        text = "Linux runs on Linux."
        entities = [
            {"type": "address", "value": "Linux"},
            {"type": "technology", "value": "Linux"},
        ]
        brief, tags = compress_text(text, entities)
        assert "loc:linux" in tags
        assert "tech:linux" in tags

    def test_duplicate_person_different_values(self):
        text = "Alice met Bob, then Alice met Charlie."
        entities = [
            {"type": "person", "value": "Alice"},
            {"type": "person", "value": "Bob"},
            {"type": "person", "value": "Charlie"},
            {"type": "person", "value": "Alice"},
        ]
        brief, tags = compress_text(text, entities)
        assert tags.count("per:alice") == 1
        assert "per:bob" in tags
        assert "per:charlie" in tags

    def test_overlapping_text_same_type(self):
        text = "New York City is big."
        entities = [
            {"type": "address", "value": "New York"},
            {"type": "address", "value": "New York City"},
        ]
        brief, tags = compress_text(text, entities)
        assert "loc:new york" in tags
        assert "loc:new york city" in tags


# ---------------------------------------------------------------------------
# Brief generation (last sentence, max_length truncation)
# ---------------------------------------------------------------------------
class TestBriefGeneration:
    def test_short_text_unchanged(self):
        text = "Hello world."
        brief, _ = compress_text(text, [])
        assert brief == "Hello world."

    def test_text_over_max_length_truncated(self):
        text = "A" * 600
        brief, _ = compress_text(text, [])
        assert len(brief) <= 500
        # Should cut at a space boundary
        assert brief.endswith(" ") is False

    def test_text_under_max_length_unchanged(self):
        text = "Short sentence here."
        brief, _ = compress_text(text, [])
        assert brief == "Short sentence here."

    def test_text_exactly_max_length(self):
        text = "1234567890" * 50  # exactly 500 chars
        brief, _ = compress_text(text, [])
        assert brief == "1234567890" * 50

    def test_last_sentence_selected(self):
        text = "Hi. This is the first short sentence. This is the last and longest sentence that captures the actual outcome of the work."
        brief, _ = compress_text(text, [])
        assert "last and longest" in brief
        assert "captures the actual outcome" in brief

    def test_no_period_truncated(self):
        text = "This text has no period but is definitely very very long indeed " * 25
        brief, _ = compress_text(text, [])
        assert len(brief) <= 500

    def test_cut_at_space_not_char(self):
        # Ensure truncation prefers space boundary over exact max_length chars
        text = "A B C D E F G H I J " * 55  # about 605 chars
        brief, _ = compress_text(text, [])
        assert len(brief) <= 500

    def test_first_substantive_sentence_selected(self):
        text = "Hi. " + "padding " * 75 + ". This is a second sentence that is very long and exceeds a hundred and fifty characters in total length and more and goes on even further."
        brief, _ = compress_text(text, [])
        # First non-trivial sentence is "padding padding..." (>= 15 chars)
        assert "padding" in brief
        # Multi-sentence mode: second substantive sentence is appended if space allows
        assert "second sentence" in brief
        # Brief should contain ellipsis separator between sentences
        assert "..." in brief

    def test_cut_threshold_80_chars(self):
        # When rfind(" ") returns < 80, fallback to first max_length chars
        text = "ABCDE12345" * 50 + "EXTRA"  # about 650 chars
        brief, _ = compress_text(text, [])
        assert len(brief) <= 500

    def test_short_first_not_truncated(self):
        text = "Hi. " + "padding word " * 40 + ". The important decision was to replace the HTTP client with a native alternative."
        brief, _ = compress_text(text, [])
        # First non-trivial sentence is "padding word padding word..." (>= 15 chars)
        assert "padding word" in brief

    # ---------------------------------------------------------------------------
    # Tests for first-substantive-sentence preference
    # ---------------------------------------------------------------------------
    def test_first_substantive_sentence_after_trivial(self):
        text = "Hi. This is a much longer and more informative sentence that captures the actual outcome of the work session and describes what was done."
        brief, _ = compress_text(text, [])
        # First non-trivial sentence is the long one after "Hi."
        assert "captures the actual outcome" in brief

    def test_first_substantive_sentence_after_trivial(self):
        text = "Ok. " + "padding " * 75 + ". The important decision was to replace the HTTP client library with a native alternative for better performance."
        brief, _ = compress_text(text, [])
        # First non-trivial sentence is "padding padding..." (>= 15 chars)
        assert "padding" in brief

    def test_custom_max_length(self):
        text = "A" * 1000
        brief, _ = compress_text(text, [], max_length=200)
        assert len(brief) <= 200

    def test_default_max_length_is_500(self):
        # 400-char text should pass through untruncated with default max_length=500
        text = ("word " * 79 + "word").strip()  # 400 chars, no trailing whitespace
        brief, _ = compress_text(text, [])
        assert brief == text



# ---------------------------------------------------------------------------
# Precision items appended to brief
# ---------------------------------------------------------------------------
class TestPrecisionItems:
    def test_precision_items_appended_to_brief(self):
        text = "Replaced axios with fetch"
        precision_items = [
            {"type": "link", "value": "http.ts"},
            {"type": "version", "value": "v2.0"},
        ]
        brief, _ = compress_text(text, [], precision_items=precision_items)
        assert "http.ts" in brief
