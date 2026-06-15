# SPDX-License-Identifier: MIT
"""Tests for src/memory_lib/domain/types.py

Covers the Result pattern (ok/err), exception hierarchy, and type alias behavior.
"""
import pytest
from typing import get_args, get_origin

from memory_lib.domain.types import (
    EmbedderError,
    Err,
    NotFoundError,
    Ok,
    Result,
    StorageError,
    ToolError,
    err,
    ok,
)


# ── ok() ─────────────────────────────────────────────────────────────

def test_ok_returns_tuple_with_value_and_none():
    """ok() wraps a value into an Ok tuple with None as the second element."""
    value = "hello"
    result = ok(value)
    assert result == (value, None)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_ok_preserves_various_types():
    """ok() works with integers, floats, lists, dicts, etc."""
    assert ok(42) == (42, None)
    assert ok(3.14) == (3.14, None)
    assert ok([1, 2, 3]) == ([1, 2, 3], None)
    assert ok({"key": "val"}) == ({"key": "val"}, None)
    assert ok(None) == (None, None)


# ── err() ─────────────────────────────────────────────────────────────

def test_err_returns_tuple_with_exception_and_none():
    """err() wraps an exception into an Err tuple with None as the first element."""
    exc = ValueError("boom")
    result = err(exc)
    assert result == (None, exc)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_err_preserves_various_exception_types():
    """err() works with different exception subclasses."""
    r1 = err(RuntimeError("oops"))
    assert r1[0] is None
    assert isinstance(r1[1], RuntimeError)
    assert str(r1[1]) == "oops"

    r2 = err(StorageError("persist"))
    assert r2[0] is None
    assert isinstance(r2[1], StorageError)
    assert str(r2[1]) == "persist"

    r3 = err(NotFoundError("missing"))
    assert r3[0] is None
    assert isinstance(r3[1], NotFoundError)
    assert str(r3[1]) == "missing"


# ── StorageError ──────────────────────────────────────────────────────

def test_storage_error_is_exception():
    """StorageError is a subclass of Exception."""
    assert issubclass(StorageError, Exception)


def test_storage_error_can_be_raised_and_caught():
    """StorageError instances can be raised and caught."""
    with pytest.raises(StorageError, match="storage failed"):
        raise StorageError("storage failed")


# ── NotFoundError ─────────────────────────────────────────────────────

def test_not_found_error_subclass_of_storage_error():
    """NotFoundError is a subclass of StorageError."""
    assert issubclass(NotFoundError, StorageError)


def test_not_found_error_subclass_of_exception():
    """NotFoundError is also a subclass of Exception (via StorageError)."""
    assert issubclass(NotFoundError, Exception)


def test_not_found_error_can_be_raised_and_caught():
    """NotFoundError instances can be raised and caught."""
    with pytest.raises(NotFoundError, match="entity not found"):
        raise NotFoundError("entity not found")


# ── EmbedderError ─────────────────────────────────────────────────────

def test_embedder_error_is_exception():
    """EmbedderError is a subclass of Exception."""
    assert issubclass(EmbedderError, Exception)


# ── ToolError ─────────────────────────────────────────────────────────

def test_tool_error_is_exception():
    """ToolError is a subclass of Exception."""
    assert issubclass(ToolError, Exception)


# ── Type alias behavior ───────────────────────────────────────────────

def test_ok_type_alias_is_tuple():
    """ok() returns a 2-tuple whose second element is None (Ok semantics)."""
    result = ok(42)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == 42
    assert result[1] is None


def test_err_type_alias_is_tuple():
    """err() returns a 2-tuple whose first element is None (Err semantics)."""
    exc = ValueError("boom")
    result = err(exc)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] is None
    assert result[1] is exc


def test_result_type_alias_is_union():
    """Result[T, E] resolves to a union of Ok and Err."""
    # Result[T, E] = Ok[T] | Err[E]
    assert get_origin(Result[int, ValueError]) is not None


def test_ok_value_has_none_at_second_index():
    """The second element of an Ok tuple is always None."""
    result = ok(100)
    assert result[1] is None


def test_err_value_has_none_at_first_index():
    """The first element of an Err tuple is always None."""
    result = err(ValueError("bad"))
    assert result[0] is None
