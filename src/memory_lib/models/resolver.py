# SPDX-License-Identifier: MIT
"""Unified model resolver.

Resolves model identifiers (HuggingFace repo IDs or local paths) to
local file paths for both ONNX and GGUF models.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

# Constants
ONNX_TYPES = {"embedding", "ner", "reranker"}
GGUF_TYPES = {"observer"}

logger = logging.getLogger(__name__)


def _is_path(value: str) -> bool:
    """Return True if value looks like a local filesystem path."""
    if not value:
        return False
    return value.startswith(("/", "~", "./")) or Path(value).is_absolute() or (
        "/" not in value and not value.startswith("hf://")
    )


def _resolve_local_path(model_id: str, model_type: str) -> dict[str, str]:
    """Resolve a local filesystem path to model file paths."""
    path = Path(model_id).expanduser()

    if path.is_file():
        # Single file — return it directly
        if model_type in GGUF_TYPES:
            return {"model_path": str(path.resolve())}
        # For ONNX types, a single file doesn't make sense (need model + tokenizer)
        raise ValueError(f"Single file path not supported for {model_type} model type")

    if path.is_dir():
        # Search directory for model files
        if model_type in GGUF_TYPES:
            gguf_files = list(path.rglob("*.gguf"))
            if gguf_files:
                return {"model_path": str(gguf_files[0].resolve())}
            raise FileNotFoundError(f"No .gguf files found in {path}")
        else:
            # ONNX model: find model.onnx and tokenizer.json
            # Check root first, then onnx/ subdirectory
            model_candidates = [path / "model.onnx", path / "onnx" / "model.onnx"]
            model_path = None
            for c in model_candidates:
                if c.exists():
                    model_path = str(c.resolve())
                    break
            if not model_path:
                raise FileNotFoundError(f"No model.onnx found in {path}")

            tokenizer_path = str((path / "tokenizer.json").resolve())
            if not Path(tokenizer_path).exists():
                raise FileNotFoundError(f"No tokenizer.json found in {path}")

            return {"model_path": model_path, "tokenizer_path": tokenizer_path}

    raise FileNotFoundError(f"Path not found: {path}")


def _resolve_gguf_model(repo_id: str, quant: str | None, force: bool = False) -> dict[str, str]:
    """Download a GGUF model from HuggingFace Hub.

    Tries Pattern A (multi-quant repo with single file) then Pattern B (single-quant repo).
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    # Parse repo_id for quant suffix if not explicitly provided
    if ":" in repo_id and quant is None:
        repo_base, quant = repo_id.rsplit(":", 1)
    elif quant is None:
        repo_base = repo_id

    # If no quant and repo looks like a single-quant GGUF repo (contains -gguf)
    if quant is None and "gguf" in repo_base.lower():
        logger.info("GGUF resolve: snapshot_download %s", repo_base)
        snapshot_path = snapshot_download(repo_id=repo_base, local_dir=None, force_download=force)
        snapshot_path = Path(snapshot_path).resolve()
        gguf_files = list(snapshot_path.rglob("*.gguf"))
        if gguf_files:
            return {"model_path": str(gguf_files[0].resolve())}
        raise FileNotFoundError(f"No .gguf files in snapshot for {repo_base}")

    # Quant provided — try Pattern A first, then Pattern B
    repo_base_name = Path(repo_base).name  # e.g., "Phi-4-mini-instruct"

    # Pattern A: repo_base-gguf / filename-{quant}.gguf
    pattern_a_repo = f"{repo_base}-gguf"
    pattern_a_file = f"{repo_base_name}-{quant}.gguf"
    try:
        logger.info("GGUF Pattern A: hf_hub_download %s / %s", pattern_a_repo, pattern_a_file)
        local_path = hf_hub_download(
            repo_id=pattern_a_repo,
            filename=pattern_a_file,
            force_download=force,
        )
        return {"model_path": str(Path(local_path).resolve())}
    except Exception as e:
        logger.debug("GGUF Pattern A failed: %s", e)

    # Pattern B: repo_base-{quant}-GGUF (full repo)
    pattern_b_repo = f"{repo_base}-{quant}-GGUF"
    try:
        logger.info("GGUF Pattern B: snapshot_download %s", pattern_b_repo)
        snapshot_path = snapshot_download(repo_id=pattern_b_repo, local_dir=None, force_download=force)
        snapshot_path = Path(snapshot_path).resolve()
        gguf_files = list(snapshot_path.rglob("*.gguf"))
        if gguf_files:
            return {"model_path": str(gguf_files[0].resolve())}
        raise FileNotFoundError(f"No .gguf files in snapshot for {pattern_b_repo}")
    except Exception as e:
        logger.debug("GGUF Pattern B failed: %s", e)

    raise FileNotFoundError(
        f"Could not resolve GGUF model for {repo_id}:{quant}. "
        f"Tried Pattern A ({pattern_a_repo}/{pattern_a_file}) and Pattern B ({pattern_b_repo})."
    )


def _resolve_onnx_model(repo_id: str, force: bool = False) -> dict[str, str]:
    """Download an ONNX model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    logger.info("ONNX resolve: snapshot_download %s", repo_id)
    snapshot_path = snapshot_download(repo_id=repo_id, local_dir=None, force_download=force)
    snapshot_path = Path(snapshot_path).resolve()

    # Find model.onnx
    model_candidates = [snapshot_path / "model.onnx", snapshot_path / "onnx" / "model.onnx"]
    model_path = None
    for c in model_candidates:
        if c.exists():
            model_path = str(c.resolve())
            break
    if not model_path:
        raise FileNotFoundError(f"No model.onnx found in snapshot {snapshot_path}")

    # Find tokenizer.json
    tokenizer_path = str((snapshot_path / "tokenizer.json").resolve())
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"No tokenizer.json found in snapshot {snapshot_path}")

    return {"model_path": model_path, "tokenizer_path": tokenizer_path}


def resolve_model(model_id: str, model_type: str, force: bool = False) -> dict[str, str]:
    """Resolve a model identifier to local file paths.

    Args:
        model_id: HuggingFace repo ID (e.g., "user/repo" or "user/repo:Q4_K_M") or local path.
        model_type: One of "embedding", "ner", "reranker", "observer".
        force: If True, force re-download from HuggingFace Hub.

    Returns:
        Dict with "model_path" (and "tokenizer_path" for ONNX models).
    """
    if not model_id:
        raise ValueError("model_id is empty")

    if _is_path(model_id):
        return _resolve_local_path(model_id, model_type)

    if model_type in GGUF_TYPES:
        return _resolve_gguf_model(model_id, quant=None, force=force)

    return _resolve_onnx_model(model_id, force=force)


async def aresolve_model(model_id: str, model_type: str, force: bool = False) -> dict[str, str]:
    """Async version of resolve_model.

    Runs the sync resolver in a thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, resolve_model, model_id, model_type, force
    )
