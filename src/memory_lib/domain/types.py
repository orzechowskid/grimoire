# SPDX-License-Identifier: MIT
"""Core domain types and the Result pattern.

Provides algebraic Result type (Ok/Err), type aliases, and standard
exception classes used across the memory library.
"""
from typing import TypeVar

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


type Ok[T] = tuple[T, None]
type Err[E: Exception] = tuple[None, E]
type Result[T, E: Exception] = Ok[T] | Err[E]


def ok[T](value: T) -> Ok[T]:
    return (value, None)


def err[E: Exception](error: E) -> Err[E]:
    return (None, error)


class StorageError(Exception):
    """Base class for all persistence-related errors."""

    pass


class NotFoundError(StorageError):
    """Raised when a requested entity is not found in storage."""

    pass


class EmbedderError(Exception):
    """Raised when the embedding engine fails."""

    pass


class ToolError(Exception):
    """Base class for errors occurring during tool execution."""

    pass
