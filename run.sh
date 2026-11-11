#!/usr/bin/env bash
# run.sh - one-command setup and launch for this recipe on one NVIDIA GB10.
#
#   git clone https://github.com/sxuff/nemotron-3.5-bolt-gb10 && cd nemotron-3.5-bolt-gb10 && ./run.sh
#
# Every pin and flag below is copied from this repo's README, manifests or
# receipts; the comment next to each value names its source. Nothing here
# tunes, "improves" or changes a measured setting.
set -euo pipefail
cd "$(dirname "$0")"
REPO_DIR="$(pwd)"

# ===================== recipe config (per repo) =============================
RECIPE_NAME="nemotron-3.5-bolt-gb10"
PORT="${PORT:-8090}"                             # source: serve.sh:11, README.md:63
MODELS_DIR="${MODELS_DIR:-$HOME/models}"         # source: serve.sh:6 (MODELS default)
SRC_DIR="${SRC_DIR:-$HOME/src}"                  # README.md:45 clones under ~/src
STATE_DIR="${STATE_DIR:-$HOME/.local/state/nemotron-3.5-bolt-gb10}"
NEEDS_TOOLS="git curl sha256sum cmake"           # cmake for the llama.cpp build (README.md:47-49)
NEED_MEM_BYTES=30000000000                       # "About 30 GB of GPU memory" (unified); README.md:38
ENGINE_DISK_BYTES=0                              # llama.cpp source+build size: not stated in this repo
ENGINE_KIND=process                              # llama-server process (serve.sh:13)
CONTAINER_NAME=""                                # docker only
READY_URL="http://127.0.0.1:${PORT}/health"      # llama-server health endpoint; polled until HTTP 200
READY_TIMEOUT=1200                               # seconds; README.md:63 says ~8 s warm; margin for a cold
                                                 # first load of the ~22 GB gguf from disk
EXPECTED_LINE="80.4 tok/s (greedy decode, single stream, 512 forced tokens; mean of code/prose/math/summary)"  # README.md:19,25
RECEIPT_PATH="results.json"

# One line per file:  <sha256> <bytes> <hf_repo> <revision> <path-in-repo> <dest-relative-to-MODELS_DIR>
# SHA-256 values: README.md:115-117 (also results.json .sha256).
# Revisions: the README's pinned revisions 188282901e3b6f8169aff9de69022a78350259b0 and
# f640c389aa24ddbe13150ef8753495446a488858 (README.md:111-112) no longer exist on the Hub -
# NVIDIA re-created both model repos on 2026-10-15 (Hub check 2026-11-11). The revisions below
# are the current commits, which serve byte-identical files: same SHA-256 as README.md:115-117,
# so the download is still verified against the repo's original pins.
# Destinations match serve.sh:8-9 (TARGET_DIR / DRAFT_DIR under $MODELS).
WEIGHTS="
91d63363e782c8a3d5f4be42816eba8e0d42dd2530bd07a7d0d2dc440599af75 22018326528 nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246 NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
0ee891eadc7bf0cae6289b249c74eb1ef420534351f6ef8b64d49f0ca6a926b0 1733692736 nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246 mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
080e1ea41c9562a7804dbf30c6fbd4d17e216c6b8ff666f88e6edb1816010194 998242560 nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF 182d2987939e12f458298b5d5f6ef98912f4ad6c NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf
"

LLAMA_SHA=1268edac579110aeb1407f7928f53193ed4e28d4   # README.md:46, README.md:113, results.json .runtime
LLAMA_DIR="$SRC_DIR/llama.cpp-${LLAMA_SHA:0:8}"      # dir owned by run.sh (never moves a user checkout)

engine_step() {   # llama.cpp at the pinned commit, built per README.md:45-49; skipped when done
  if [ -x "$LLAMA_DIR/build/bin/llama-server" ] && [ -d "$LLAMA_DIR/.git" ] \
     && [ "$(git -C "$LLAMA_DIR" rev-parse HEAD)" = "$LLAMA_SHA" ]; then
    ok "llama-server already built at ${LLAMA_SHA:0:12}: $LLAMA_DIR"; return
  fi
  git_pin https://github.com/ggml-org/llama.cpp "$LLAMA_SHA" "$LLAMA_DIR"   # README.md:45-46
  local cuda_args=()
  if ! command -v nvcc >/dev/null && [ -x /usr/local/cuda/bin/nvcc ]; then
    # Not in the README: on boxes where nvcc is not on PATH (CUDA at /usr/local/cuda),
    # tell cmake where it is. Changes nothing when nvcc is already on PATH.
    cuda_args=(-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc)
    ok "nvcc not on PATH; adding -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"
  fi
  # cmake flags verbatim from README.md:47-48
  doit cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DGGML_NATIVE=ON \
    -DCMAKE_CUDA_ARCHITECTURES=121a-real -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF \
    ${cuda_args[@]+"${cuda_args[@]}"}
  doit cmake --build "$LLAMA_DIR/build" -j --target llama-server              # README.md:49
}

serve_step() {    # the repo's own serve.sh (NVIDIA's published command; serve.sh:13-31)
  if [ "$CHECK" = 1 ]; then
    plan "start ./serve.sh in the background (LLAMA=$LLAMA_DIR MODELS=$MODELS_DIR HOST=127.0.0.1 PORT=$PORT)"
    return
  fi
  LLAMA="$LLAMA_DIR" MODELS="$MODELS_DIR" HOST=127.0.0.1 PORT="$PORT" \
    nohup ./serve.sh >"$LOGFILE" 2>&1 & echo $! >"$PIDFILE"
  # serve.sh exec's llama-server (serve.sh:13), so the pid above is the server itself.
}

curl_test() {     # the README's own example request (README.md:66-68)
  cat <<EOF
curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Compute 17*23. Reply with only the integer."}],
  "max_tokens": 4096, "temperature": 0}'
EOF
}
# ===================== end of recipe config =================================

# ---------------------------------------------------------------------------
# Shared machinery (identical in every sxuff recipe repo).
# ---------------------------------------------------------------------------
CHECK=0; ACTION=run
usage() {
  cat <<EOF
Usage: ./run.sh [--check] [--stop] [--help]

Sets up and launches ${RECIPE_NAME} on one NVIDIA GB10 (SM121).

  (no flag)  check the machine, build or pull the pinned engine, download the
             pinned weights, verify SHA-256, start the server, print a test.
             Safe to re-run: finished steps are skipped.
  --check    print what would happen; change nothing.
  --stop     stop the server this script started.
  --help     show this text.

Environment:
  HF_TOKEN     Hugging Face token, read from the environment only, never saved.
  MODELS_DIR   weights        (default: ${MODELS_DIR})
  SRC_DIR      engine source  (default: ${SRC_DIR})
  STATE_DIR    pid and log    (default: ${STATE_DIR})
  PORT         server port    (default: ${PORT})

Expected: ${EXPECTED_LINE}
Receipt:  ${RECEIPT_PATH}
EOF
}
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --stop)  ACTION=stop ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $a (see ./run.sh --help)" >&2; exit 2 ;;
  esac
done

say()  { printf '\n==> %s\n' "$*"; }
ok()   { printf '    ok    %s\n' "$*"; }
plan() { printf '    would %s\n' "$*"; }
die()  { printf '\nerror: %s\n' "$*" >&2; exit 1; }
CHECK_FAILED=0   # under --check, a failed pre-flight check is reported and the plan continues
fail() { if [ "$CHECK" = 1 ]; then printf '    FAIL  %s\n' "$*"; CHECK_FAILED=1; else die "$*"; fi; }
human(){ numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 B"; }
doit() { if [ "$CHECK" = 1 ]; then plan "$*"; else "$@"; fi; }   # mutate, or describe under --check

PIDFILE="${STATE_DIR}/server.pid"
LOGFILE="${STATE_DIR}/server.log"
STOP_CMD="cd ${REPO_DIR} && ./run.sh --stop"

server_running() {
  if [ "$ENGINE_KIND" = docker ]; then
    [ -n "$(docker ps -q --filter "name=^${CONTAINER_NAME}\$" 2>/dev/null)" ]
  else
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
  fi
}

stop_step() {   # may be overridden in the config block (e.g. to call scripts/stop.sh)
  if ! server_running; then ok "nothing started by this script is running"; return; fi
  if [ "$ENGINE_KIND" = docker ]; then doit docker stop "$CONTAINER_NAME"
  else
    local pid; pid="$(cat "$PIDFILE")"; doit kill "$pid"
    if [ "$CHECK" = 0 ]; then
      for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid"; rm -f "$PIDFILE"
    fi
  fi
  ok "stopped"
}
declare -f stop_override >/dev/null && stop_step() { stop_override; }

if [ "$ACTION" = stop ]; then say "Stopping ${RECIPE_NAME}"; stop_step; exit 0; fi
[ "$CHECK" = 1 ] && say "--check: nothing will be changed"

# git checkout of an exact commit, idempotent. Usage: git_pin <url> <sha> <dir>
git_pin() {
  local url="$1" sha="$2" dir="$3"
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ]; then ok "source at ${sha:0:12}: $dir"; return; fi
  if [ ! -d "$dir/.git" ]; then doit git clone --quiet "$url" "$dir"; fi
  if [ "$CHECK" = 1 ]; then plan "check out $sha in $dir"; return; fi
  [ -z "$(git -C "$dir" status --porcelain)" ] || die "$dir has local changes; refusing to move it to $sha"
  git -C "$dir" fetch --quiet origin "$sha" 2>/dev/null || git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet --detach "$sha"
  [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || die "could not check out $sha in $dir"
  ok "source at ${sha:0:12}: $dir"
}

# docker image by digest, idempotent. Usage: image_pin <repo@sha256:...>
image_pin() {
  if docker image inspect "$1" >/dev/null 2>&1; then ok "image present: $1"; else doit docker pull "$1"; fi
}

# --- 1. machine ------------------------------------------------------------
say "Machine"
[ "$(uname -m)" = aarch64 ] || die "expected aarch64 (GB10), got $(uname -m)"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
GPU="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
case "$GPU" in *GB10*", 12.1") ok "GPU: $GPU" ;; *) die "measured on an NVIDIA GB10 (SM121); found: ${GPU:-none}" ;; esac
MISSING=""; for t in $NEEDS_TOOLS; do command -v "$t" >/dev/null || MISSING="$MISSING $t"; done
if [ -n "$MISSING" ]; then fail "missing tool(s):$MISSING"; else ok "tools: $NEEDS_TOOLS"; fi
ALREADY=0; server_running && ALREADY=1
MEM_AVAIL=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) * 1024 ))
if [ "$ALREADY" = 1 ]; then ok "server already running; memory check skipped"
elif [ "$MEM_AVAIL" -lt "$NEED_MEM_BYTES" ]; then
  fail "need $(human "$NEED_MEM_BYTES") available memory, have $(human "$MEM_AVAIL"). Stop other models first (nvidia-smi, docker ps)."
else ok "memory: $(human "$MEM_AVAIL") available, need $(human "$NEED_MEM_BYTES")"; fi
if [ "$ALREADY" = 1 ]; then ok "port ${PORT}: served by this recipe"
else
  # The server binds 127.0.0.1 (or the host network). A listener on 127.0.0.1, a wildcard or ::1
  # blocks it; one bound to another interface address does not, and is only reported.
  PORT_ALL="$(ss -ltnH "sport = :${PORT}" 2>/dev/null || true)"
  PORT_CLASH="$(printf '%s\n' "$PORT_ALL" | awk '{print $4}' | grep -E '^(127\.0\.0\.1|0\.0\.0\.0|\*|\[::\]|\[::1\]|\[::ffff:127\.0\.0\.1\]):' || true)"
  PORT_DOCKER="$(docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null | grep -E "(127\.0\.0\.1|0\.0\.0\.0|\[::\]|:::?):${PORT}->" || true)"
  if [ -n "$PORT_CLASH$PORT_DOCKER" ]; then
    printf '    port %s is taken:\n%s\n%s\n' "$PORT" "$PORT_CLASH" "$PORT_DOCKER"
    fail "free port ${PORT} or run: PORT=<other> ./run.sh"
  elif [ -n "$PORT_ALL" ]; then
    ok "port ${PORT}: free on 127.0.0.1 (also bound elsewhere: $(printf '%s' "$PORT_ALL" | awk '{print $4}' | tr '\n' ' '))"
  else ok "port ${PORT}: free"; fi
fi

# --- 2. disk ---------------------------------------------------------------
say "Disk"
nearest_dir() { local d="$1"; while [ ! -d "$d" ]; do d="$(dirname "$d")"; done; echo "$d"; }
NEED_DISK=$ENGINE_DISK_BYTES
while read -r sha size repo rev rpath dest; do
  [ -z "${sha:-}" ] && continue
  have=0; [ -f "$MODELS_DIR/$dest" ] && have=$(stat -c %s "$MODELS_DIR/$dest")
  [ "$have" -lt "$size" ] && NEED_DISK=$(( NEED_DISK + size - have ))
done <<< "$WEIGHTS"
FREE_DISK=$(df -B1 --output=avail "$(nearest_dir "$MODELS_DIR")" | tail -1 | tr -d ' ')
if [ "$FREE_DISK" -ge "$NEED_DISK" ]; then ok "need $(human "$NEED_DISK") more, have $(human "$FREE_DISK")"
else fail "need $(human "$NEED_DISK") more disk under $MODELS_DIR, have $(human "$FREE_DISK"). Nothing was deleted."; fi

# --- 3. engine -------------------------------------------------------------
say "Engine"
engine_step

# --- 4. weights: pinned revision + SHA-256 -----------------------------------
hf_fetch() {   # repo rev path dest. The token reaches curl on stdin: never in argv or a file.
  { [ -n "${HF_TOKEN:-}" ] && printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN"; true; } |
    curl --fail --location --retry 5 --retry-delay 5 --continue-at - --config - \
         --output "$4.part" "https://huggingface.co/$1/resolve/$2/$3"
  mv "$4.part" "$4"
}
default_weights_step() {
  while read -r sha size repo rev rpath dest; do
    [ -z "${sha:-}" ] && continue
    local f="$MODELS_DIR/$dest" mark="$MODELS_DIR/$dest.sha256-ok"
    if [ -f "$mark" ] && [ "$(cat "$mark")" = "$sha $size" ] && [ "$(stat -c %s "$f" 2>/dev/null)" = "$size" ]; then
      ok "verified earlier: $dest"; continue
    fi
    if [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" != "$size" ]; then
      if [ "$CHECK" = 1 ]; then plan "download $repo@${rev:0:12} $rpath ($(human "$size"))"; continue; fi
      mkdir -p "$(dirname "$f")"; printf '    get   %s (%s)\n' "$rpath" "$(human "$size")"
      hf_fetch "$repo" "$rev" "$rpath" "$f"
    elif [ "$CHECK" = 1 ]; then plan "verify SHA-256 of $dest"; continue; fi
    local got; got="$(sha256sum "$f" | cut -d' ' -f1)"
    if [ "$got" != "$sha" ]; then mv "$f" "$f.sha256-mismatch"; die "SHA-256 mismatch for $dest: expected $sha, got $got (kept as $f.sha256-mismatch)"; fi
    echo "$sha $size" > "$mark"; ok "sha256 ok: $dest"
  done <<< "$WEIGHTS"
}
declare -f weights_step >/dev/null || weights_step() { default_weights_step; }
say "Weights"
weights_step

# --- 5. serve --------------------------------------------------------------
say "Server"
if [ "$ALREADY" = 1 ]; then ok "already running"
elif [ "$CHECK" = 1 ]; then serve_step
else
  mkdir -p "$STATE_DIR"; serve_step
  printf '    wait  for %s (up to %ss; log: %s)\n' "$READY_URL" "$READY_TIMEOUT" "$LOGFILE"
  t=0
  until curl -fsS -o /dev/null "$READY_URL" 2>/dev/null; do
    server_running || die "server exited during start-up; see $LOGFILE"
    [ "$t" -ge "$READY_TIMEOUT" ] && die "not ready after ${READY_TIMEOUT}s; see $LOGFILE"
    sleep 5; t=$((t + 5))
  done
  ok "ready after ${t}s"
fi

# --- 6. done ---------------------------------------------------------------
if [ "$CHECK" = 1 ]; then
  if [ "$CHECK_FAILED" = 1 ]; then say "Check finished with FAIL lines above; nothing was changed. ./run.sh would stop at the first one."; exit 1; fi
  say "Check finished; nothing was changed. Run ./run.sh to do it."; exit 0
fi
say "Test it"
curl_test
say "Stop it"
echo "$STOP_CMD"
echo
echo "Expected: ${EXPECTED_LINE} - receipt: ${RECEIPT_PATH}"
