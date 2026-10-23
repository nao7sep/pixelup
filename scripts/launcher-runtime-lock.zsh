#!/bin/zsh
# Short metadata transactions only. The OS releases flock even if a launcher dies.
set -eu
repo_dir="$1"
operation="$2"
shift 2
owner_file="$repo_dir/.git/launcher-runtime-owner"
receipt_file="$repo_dir/.git/launcher-runtime-process"
lock_file="$repo_dir/.git/launcher-runtime-lock"
: >> "$lock_file"
zmodload zsh/system
zsystem flock -f lock_fd "$lock_file"

owns_runtime() {
  [[ -f "$owner_file" ]] || return 2
  local current_owner
  current_owner="$(cat "$owner_file")" || return 1
  [[ "$current_owner" == "$1" ]] || return 2
}

case "$operation" in
  claim)
    temporary_file="$owner_file.$$"
    print -r -- "$1" > "$temporary_file"
    mv -f "$temporary_file" "$owner_file"
    ;;
  owns)
    owns_runtime "$1"
    ;;
  release)
    if owns_runtime "$1"; then
      rm -f "$owner_file"
    else
      ownership_status=$?
      [[ "$ownership_status" -eq 2 ]] || exit "$ownership_status"
    fi
    ;;
  register)
    owns_runtime "$1" || exit $?
    temporary_file="$receipt_file.$$"
    printf '%s\n%s\n%s\n' "$1" "$2" "$3" > "$temporary_file"
    mv -f "$temporary_file" "$receipt_file"
    ;;
  snapshot)
    owns_runtime "$1" || exit $?
    if [[ -f "$receipt_file" ]]; then
      receipt="$(cat "$receipt_file")" || exit 1
      if [[ -z "$2" || "${receipt%%$'\n'*}" == "$2" ]]; then
        print -r -- "$receipt"
      fi
    fi
    ;;
  clear)
    if owns_runtime "$1"; then
      if [[ -f "$receipt_file" ]]; then
        receipt="$(cat "$receipt_file")" || exit 1
        if [[ "$receipt" == "$(printf '%s\n%s\n%s' "$2" "$3" "$4")" ]]; then
          rm -f "$receipt_file"
        fi
      fi
    else
      ownership_status=$?
      [[ "$ownership_status" -eq 2 ]] || exit "$ownership_status"
    fi
    ;;
  *) exit 1 ;;
esac
