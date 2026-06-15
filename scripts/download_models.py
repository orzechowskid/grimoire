#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Download models from HuggingFace Hub and generate config.json.

Usage:
    python scripts/download_models.py          # Download all models
    python scripts/download_models.py --check   # Verify models present, don't re-download
    python scripts/download_models.py --force   # Force re-download all models
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent of src/ to path so we can import memory_lib modules
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

CONFIG_OUTPUT_PATH = PROJECT_ROOT / "config.json"

# Default model definitions — used as fallback when config.json doesn't specify them
DEFAULT_MODELS = [
    ("embedding", "embedding", "Sprylab/paraphrase-multilingual-MiniLM-L12-v2-onnx-quantized"),
    ("ner", "ner", "onnx-community/distilbert-NER-ONNX"),
    ("reranker", "reranker", "cross-encoder/ms-marco-TinyBERT-L2-v2"),
    ("observer", "observer", "google/gemma-3-1b-it-qat-q4_0-gguf"),
]

# Extra config metadata per model key
MODEL_METADATA = {
    "embedding": {"dim": 384, "max_length": 512},
    "ner": {},
    "reranker": {},
    "observer": {},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download models from HuggingFace Hub and generate config.json.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify models are present without re-downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of all models.",
    )
    return parser.parse_args()


def _load_model_ids_from_config() -> dict[str, str]:
    """Load model IDs from config.json, returning empty strings for missing models.
    
    Uses the existing Settings.from_json_file() infrastructure rather than
    parsing JSON directly.
    """
    from memory_lib.config import Settings
    
    if not CONFIG_OUTPUT_PATH.exists():
        return {}
    
    try:
        settings = Settings.from_json_file(str(CONFIG_OUTPUT_PATH))
        return {
            "embedding": settings.models.embedding_id,
            "ner": settings.models.ner_id,
            "reranker": settings.models.reranker_id,
            "observer": settings.summarizer.model_id,
        }
    except Exception as e:
        print(f"  [warn] Could not load config.json: {e}")
        print(f"  [warn] Falling back to default models.")
        return {}


def _build_model_list() -> list[tuple[str, str, str]]:
    """Build the effective model list: prefer config.json, fall back to defaults.
    
    Returns list of (model_key, model_type, model_id) tuples.
    """
    config_ids = _load_model_ids_from_config()
    models = []
    
    for model_key, model_type, default_id in DEFAULT_MODELS:
        actual_id = config_ids.get(model_key, "") or default_id
        source = "config.json" if (config_ids.get(model_key, "") and config_ids[model_key] == actual_id) else "default"
        models.append((model_key, model_type, actual_id))
    
    return models


def download_and_check(force: bool = False) -> None:
    """Download all models and write config.json."""
    from memory_lib.models.resolver import resolve_model

    paths: dict[str, dict] = {}

    config_ids = _load_model_ids_from_config()
    models = _build_model_list()
    
    for model_key, model_type, model_id in models:
        source = config_ids.get(model_key, "") or ""
        source_label = f" (from config.json)" if source == model_id else " (default)"
        print(f"\n{'─' * 60}")
        print(f"[{model_key}] {model_id}{source_label}")
        print(f"{'─' * 60}")

        resolved = resolve_model(model_id, model_type, force=force)
        model_path = resolved["model_path"]
        print(f"  Resolved to  : {model_path}")
        print(f"  ✓ OK")

        paths[model_key] = resolved

    # Build or merge config
    # Load existing config first to preserve non-model settings
    existing_config: dict = {}
    if CONFIG_OUTPUT_PATH.exists():
        try:
            existing_config = json.loads(CONFIG_OUTPUT_PATH.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    
    # Start with existing config (preserves storage, search, server, observer, dreamer, etc.)
    config = existing_config.copy() if existing_config else {
        "storage": {"db_path": "~/.grimoire/memory.db"},
        "search": {
            "embedding_dim": 384,
            "pipeline_width": 4,
            "half_life_days": 30,
        },
        "server": {"host": "127.0.0.1", "port": 8766},
        "observer": {"brief_max_length": 500},
    }
    
    # Update models section with current model IDs
    config["models"] = {}
    for model_key, model_type, model_id in models:
        entry: dict = {"repo_id": model_id}
        metadata = MODEL_METADATA.get(model_key, {})
        for k, v in metadata.items():
            entry[k] = v
        config["models"][model_key] = entry

    # Write config
    CONFIG_OUTPUT_PATH.write_text(json.dumps(config, indent=2) + "\n")
    print(f"\n✓ Config written to {CONFIG_OUTPUT_PATH}")
    print("Done.")


def check_models() -> None:
    """Verify all required models exist."""
    from memory_lib.models.resolver import resolve_model

    models = _build_model_list()
    all_ok = True
    for model_key, model_type, model_id in models:
        print(f"\nChecking {model_key}: {model_id}")
        try:
            resolved = resolve_model(model_id, model_type, force=False)
            model_path = resolved["model_path"]
            print(f"  ✓ {model_path}")
        except Exception as exc:
            print(f"  ✗ {exc}", file=sys.stderr)
            all_ok = False

    print()
    if all_ok:
        print("✓ All models verified successfully.")
    else:
        print("✗ Some models are missing or invalid.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()

    if args.check:
        check_models()
    else:
        download_and_check(force=args.force)


if __name__ == "__main__":
    main()
