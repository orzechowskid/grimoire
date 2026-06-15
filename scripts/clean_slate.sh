#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# clean_slate.sh — Stop memory_lib and delete all database files.
#
# Usage:
#   ./scripts/clean_slate.sh
#
# This script is safe by design: it never deletes without an explicit
# user confirmation.

set -euo pipefail

# ─── colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

info()    { printf "${CYAN}ℹ  %s${NC}\n" "$*"; }
success() { printf "${GREEN}✔  %s${NC}\n" "$*"; }
warn()    { printf "${YELLOW}⚠  %s${NC}\n" "$*"; }
error()   { printf "${RED}✘  %s${NC}\n" "$*" >&2; }
heading() { printf "\n${BOLD}${NC}─── %s ───${NC}\n" "$*"; }

# ─── paths & defaults ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/config.json"

DB_PATH="${HOME}/.grimoire/memory.db"
DB_DIR="${HOME}/.grimoire"

# Try to read the actual db_path from config.json (falls back to default)
if [[ -f "${CONFIG_FILE}" ]]; then
    # Use python one-liner to safely parse JSON and expand ~
    CONFIGURED_PATH=$(python3 -c "
import json, os
with open('${CONFIG_FILE}') as f:
    cfg = json.load(f)
path = cfg.get('storage', {}).get('db_path', '~/.grimoire/memory.db')
print(os.path.expanduser(path))
" 2>/dev/null || true)
    if [[ -n "${CONFIGURED_PATH}" ]]; then
        DB_PATH="${CONFIGURED_PATH}"
    fi
fi

DB_WAL="${DB_PATH}-wal"
DB_SHM="${DB_PATH}-shm"
DB_DIR=$(dirname "${DB_PATH}")

SERVER_PORT=8766
if [[ -f "${CONFIG_FILE}" ]]; then
    CONFIGURED_PORT=$(python3 -c "
import json
with open('${CONFIG_FILE}') as f:
    cfg = json.load(f)
print(cfg.get('server', {}).get('port', 8766))
" 2>/dev/null || true)
    if [[ -n "${CONFIGURED_PORT}" ]]; then
        SERVER_PORT="${CONFIGURED_PORT}"
    fi
fi

# ─── helper: check if a port is in use ────────────────────────────────────────
port_in_use() {
    ss -tlnp 2>/dev/null | grep -q ":${SERVER_PORT} " || \
    lsof -i ":${SERVER_PORT}" >/dev/null 2>&1
}

# ─── helper: count sessions in SQLite ─────────────────────────────────────────
count_sessions() {
    local db="$1"
    if [[ -f "${db}" ]]; then
        python3 -c "
import sqlite3, sys
try:
    conn = sqlite3.connect('${db}')
    cur = conn.execute(\"SELECT COUNT(*) FROM sessions\")
    print(cur.fetchone()[0])
    conn.close()
except Exception:
    print(0)
" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Show current state
# ═══════════════════════════════════════════════════════════════════════════════
heading "CURRENT STATE"

# Memory lib process
if port_in_use; then
    success "memory_lib is RUNNING on port ${SERVER_PORT}"
else
    warn "memory_lib is NOT running on port ${SERVER_PORT}"
fi

# Database session count
SESSION_COUNT=$(count_sessions "${DB_PATH}")
info "Session count in ${DB_PATH}: ${SESSION_COUNT}"

# Database file sizes
info "Database file sizes:"
for f in "${DB_PATH}" "${DB_WAL}" "${DB_SHM}"; do
    if [[ -f "${f}" ]]; then
        SIZE=$(du -h "${f}" 2>/dev/null | cut -f1)
        info "  ${f} — ${SIZE}"
    else
        info "  ${f} — (not found)"
    fi
done

# Config file location
info "Config file: ${CONFIG_FILE}"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Stop memory_lib (if running)
# ═══════════════════════════════════════════════════════════════════════════════
heading "STOPPING MEMORY_LIB"

if port_in_use; then
    warn "Attempting graceful shutdown via HTTP POST /shutdown ..."

    # Try graceful shutdown via HTTP
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "http://127.0.0.1:${SERVER_PORT}/shutdown" \
        --connect-timeout 3 --max-time 5 2>/dev/null || echo "000")

    if [[ "${HTTP_CODE}" != "000" ]]; then
        success "HTTP shutdown request returned status ${HTTP_CODE}"
    else
        warn "Could not reach the shutdown endpoint (HTTP ${HTTP_CODE})"
    fi

    # Small delay to let the server finish
    sleep 1

    # Find and SIGTERM the uvicorn process on the port
    PIDS=""
    if command -v lsof >/dev/null 2>&1; then
        PIDS=$(lsof -ti ":${SERVER_PORT}" 2>/dev/null || true)
    elif command -v ss >/dev/null 2>&1; then
        PIDS=$(ss -tlnp "sport = :${SERVER_PORT}" 2>/dev/null | \
               grep -oP 'pid=\K[0-9]+' | sort -u || true)
    fi

    if [[ -n "${PIDS}" ]]; then
        warn "Sending SIGTERM to processes: ${PIDS}"
        echo "${PIDS}" | xargs kill -TERM 2>/dev/null || true
        # Wait briefly for processes to exit
        sleep 1

        # Force kill if still alive
        STILL_ALIVE=$(echo "${PIDS}" | xargs -r kill -0 2>/dev/null && echo "yes" || echo "no")
        if [[ "${STILL_ALIVE}" == "yes" ]]; then
            warn "Processes still running — sending SIGKILL ..."
            echo "${PIDS}" | xargs kill -KILL 2>/dev/null || true
            sleep 1
        fi
        success "All server processes terminated"
    else
        info "No server process detected on port ${SERVER_PORT} (it may have already exited)"
    fi
else
    info "Server is not running — nothing to stop"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Delete database files (after confirmation)
# ═══════════════════════════════════════════════════════════════════════════════
heading "DELETING DATABASE FILES"

info "The following files will be DELETED:"
for f in "${DB_PATH}" "${DB_WAL}" "${DB_SHM}"; do
    if [[ -f "${f}" ]]; then
        SIZE=$(du -h "${f}" 2>/dev/null | cut -f1)
        info "  ${f} (${SIZE})"
    else
        info "  ${f} (does not exist)"
    fi
done

read -p "$(printf "${YELLOW}Are you sure you want to permanently delete these files? [y/N]${NC} ")" CONFIRM

if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
    success "Aborted — no files were deleted."
    exit 0
fi

# Perform deletions
DELETED=0
for f in "${DB_PATH}" "${DB_WAL}" "${DB_SHM}"; do
    if [[ -f "${f}" ]]; then
        rm -f "${f}"
        success "Deleted ${f}"
        ((DELETED++)) || true
    else
        info "Skipped ${f} (does not exist)"
    fi
done

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Confirmation
# ═══════════════════════════════════════════════════════════════════════════════
heading "VERIFICATION"

# Confirm deletion
REMAINING=0
for f in "${DB_PATH}" "${DB_WAL}" "${DB_SHM}"; do
    if [[ -f "${f}" ]]; then
        warn "File still exists: ${f}"
        ((REMAINING++)) || true
    fi
done

if [[ "${REMAINING}" -eq 0 ]]; then
    success "All database files have been successfully deleted (${DELETED} file(s) removed)"
else
    error "${REMAINING} file(s) could not be deleted — please check permissions"
fi

# Check db directory
if [[ -d "${DB_DIR}" ]]; then
    FILE_COUNT=$(find "${DB_DIR}" -maxdepth 1 -type f 2>/dev/null | wc -l)
    info "Directory ${DB_DIR} now contains ${FILE_COUNT} file(s)"
else
    warn "Directory ${DB_DIR} does not exist"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Next steps
# ═══════════════════════════════════════════════════════════════════════════════
heading "NEXT STEPS"

info "To start memory_lib fresh, run one of:"
info ""
info "  # If using the project's entry-point script:"
info "  python3 -m memory_lib.server   # or however you start it"
info ""
info "  # If you have a run script:"
info "  bash scripts/start_memory_lib.sh"
info ""
info "The database will be created automatically on first request."
info ""
info "If you want to wipe everything including the directory:"
warn "  rm -rf ${DB_DIR}"

heading "DONE"
success "Clean-slate operation complete."
