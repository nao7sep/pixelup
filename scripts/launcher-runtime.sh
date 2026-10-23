#!/usr/bin/env bash

LAUNCHER_RUNTIME_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

launcher_runtime_transaction() {
  /bin/zsh "$LAUNCHER_RUNTIME_HELPER_DIR/launcher-runtime-lock.zsh" "$@"
}

claim_launcher_runtime() {
  launcher_runtime_transaction "$2" claim "$1"
}

is_launcher_runtime_owner() {
  launcher_runtime_transaction "$2" owns "$1"
}

release_launcher_runtime() {
  launcher_runtime_transaction "$2" release "$1"
}

stop_owned_runtime() {
  local label="$1" repo_dir="$2" token="$3" expected_receipt_token="${4:-}"
  local receipt receipt_token pid identity
  if receipt="$(launcher_runtime_transaction "$repo_dir" snapshot "$token" "$expected_receipt_token")"; then
    :
  else
    local snapshot_status=$?
    [[ "$snapshot_status" -eq 2 ]] && return 0
    return "$snapshot_status"
  fi
  [[ -n "$receipt" ]] || return 0
  {
    IFS= read -r receipt_token || true
    IFS= read -r pid || true
    IFS= read -r identity || true
  } <<< "$receipt"
  if is_captured_runtime_alive "$pid" "$identity"; then
    echo "Stopping the existing $label runtime (pid $pid)."
    stop_captured_runtime "$pid" "$identity" || return $?
  fi
  launcher_runtime_transaction "$repo_dir" clear "$token" "$receipt_token" "$pid" "$identity"
}

start_owned_runtime() {
  local repo_dir="$1" token="$2"
  shift 2
  is_launcher_runtime_owner "$token" "$repo_dir" || return $?
  "$@" &
  LAUNCHED_RUNTIME_PID=$!
  LAUNCHED_RUNTIME_IDENTITY="$(runtime_creation_identity "$LAUNCHED_RUNTIME_PID" || true)"
  if [[ -z "$LAUNCHED_RUNTIME_IDENTITY" ]]; then
    echo "The runtime exited before it could be registered." >&2
    wait "$LAUNCHED_RUNTIME_PID" || true
    return 1
  fi
  if launcher_runtime_transaction "$repo_dir" register "$token" "$LAUNCHED_RUNTIME_PID" "$LAUNCHED_RUNTIME_IDENTITY"; then
    return 0
  else
    local registration_status=$?
    stop_captured_runtime "$LAUNCHED_RUNTIME_PID" "$LAUNCHED_RUNTIME_IDENTITY" || return 1
    wait "$LAUNCHED_RUNTIME_PID" || true
    [[ "$registration_status" -eq 2 ]] && return 2
    return 1
  fi
}

# This identity comes only from a PID captured by the launcher's direct child
# launch. Process command text never establishes ownership.
runtime_creation_identity() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  LC_ALL=C ps -p "$pid" -o lstart= 2>/dev/null |
    sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

is_captured_runtime_alive() {
  local pid="$1" identity="$2" current state
  [[ -n "$identity" ]] || return 1
  current="$(runtime_creation_identity "$pid")" || return 1
  [[ "$current" == "$identity" ]] || return 1
  state="$(ps -p "$pid" -o stat= 2>/dev/null)" || return 1
  [[ "$state" != *Z* && -n "$state" ]]
}

stop_captured_runtime() {
  local pid="$1" identity="$2" deadline
  is_captured_runtime_alive "$pid" "$identity" || return 0
  kill -TERM "$pid" 2>/dev/null || true
  deadline=$((SECONDS + 5))
  while (( SECONDS < deadline )); do
    is_captured_runtime_alive "$pid" "$identity" || return 0
    sleep 0.1
  done
  if is_captured_runtime_alive "$pid" "$identity"; then
    kill -KILL "$pid" 2>/dev/null || true
    deadline=$((SECONDS + 5))
    while is_captured_runtime_alive "$pid" "$identity"; do
      if (( SECONDS >= deadline )); then
        echo "The recorded runtime (pid $pid) did not stop." >&2
        return 1
      fi
      sleep 0.1
    done
  fi
}

wait_for_captured_runtime() {
  local label="$1" pid="$2" identity="$3" timeout_seconds="$4" deadline
  deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if is_captured_runtime_alive "$pid" "$identity"; then
      return 0
    fi
    # A direct child that has exited cannot become ready later.
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  echo "$label did not start within ${timeout_seconds}s." >&2
  return 1
}
