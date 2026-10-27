#!/usr/bin/env bash
# Shared by the macOS launchers (run-dev, run-built, rebuild): one PixelUp runtime
# per repository. The launcher that starts PixelUp records its process ID and
# executable under .git; the next launcher asks that process to quit only while it
# still runs that executable, and stops with a message rather than force-closing
# it when it stays open. PixelUp started any other way is left to its single-writer
# admission, which refuses a second copy on the same data folder.

LAUNCHER_RECORD_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.git/launcher-runtime-process"

# The executable a process runs, or nothing when no such process exists.
executable_of() {
  { ps -p "$1" -o comm= 2>/dev/null || true; } | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# The recorded process ID, printed only while that process still runs the recorded executable.
recorded_runtime_pid() {
  local pid="" executable=""
  [[ -f "$LAUNCHER_RECORD_FILE" ]] || return 0
  { IFS= read -r pid || true; IFS= read -r executable || true; } < "$LAUNCHER_RECORD_FILE"
  [[ "$pid" =~ ^[0-9]+$ && -n "$executable" ]] || return 0
  if [[ "$(executable_of "$pid")" == "$executable" ]]; then
    printf '%s\n' "$pid"
  fi
}

# Ask the recorded PixelUp to quit as a user would, so it can still ask about
# running work. Given a process ID, only while that is the recorded one.
stop_recorded_runtime() {
  local only_pid="${1:-}" pid deadline
  pid="$(recorded_runtime_pid)"
  [[ -n "$pid" ]] || return 0
  [[ -z "$only_pid" || "$pid" == "$only_pid" ]] || return 0
  echo "Asking the running PixelUp (pid $pid) to quit."
  /usr/bin/osascript -l JavaScript -e "ObjC.import('AppKit');
function run(args) {
  var application = \$.NSRunningApplication.runningApplicationWithProcessIdentifier(Number(args[0]));
  if (application) application.terminate;
}" "$pid" >/dev/null
  deadline=$((SECONDS + 5))
  while (( SECONDS < deadline )); do
    [[ "$(recorded_runtime_pid)" == "$pid" ]] || return 0
    sleep 0.1
  done
  echo "PixelUp is still open, perhaps finishing or asking about running work. Close it, then run this launcher again." >&2
  return 1
}

# Start the command as this launcher's child and record it once it runs the
# expected executable. Sets RUNTIME_PID.
start_recorded_runtime() {
  local expected="$1" deadline
  shift
  "$@" &
  RUNTIME_PID=$!
  deadline=$((SECONDS + 30))
  until [[ "$(executable_of "$RUNTIME_PID")" == "$expected" ]]; do
    if ! kill -0 "$RUNTIME_PID" 2>/dev/null || (( SECONDS >= deadline )); then
      echo "PixelUp did not start." >&2
      return 1
    fi
    sleep 0.1
  done
  printf '%s\n%s\n' "$RUNTIME_PID" "$expected" > "$LAUNCHER_RECORD_FILE"
}
