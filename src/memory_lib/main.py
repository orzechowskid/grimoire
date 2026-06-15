# SPDX-License-Identifier: MIT
"""Memory Library — FastAPI entry point."""
import uvicorn


def main() -> None:
    uvicorn.run(
        "memory_lib.api.routes:app",
        host="127.0.0.1",
        port=8766,
        log_level="info",
    )


if __name__ == "__main__":
    main()
