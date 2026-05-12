#!/usr/bin/env bash
set -Eeuo pipefail

# Auto-commit Gomoku model artifacts and push the current Git branch.
# Defaults can be overridden by environment variables, including the systemd
# environment file at /etc/default/gomoku-auto-commit-models.
#
# MODEL_PATHS accepts a space-separated list of Git pathspecs. If unset, the
# legacy single MODEL_PATHSPEC value is used for backward compatibility.
# REQUIRED_PATH, when set, must exist before any commit is attempted.

PROJECT_DIR="${PROJECT_DIR:-/Users/nangsontay/gomoku}"
MODEL_PATHSPEC="${MODEL_PATHSPEC:-*.model}"
MODEL_PATHS="${MODEL_PATHS:-}"
REQUIRED_PATH="${REQUIRED_PATH:-}"
GIT_BIN="${GIT_BIN:-git}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
COMMIT_MESSAGE_PREFIX="${COMMIT_MESSAGE_PREFIX:-chore(models): auto-commit model files}"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERROR: required command not found: $1"
    exit 127
  fi
}

main() {
  require_command "${GIT_BIN}"

  local model_pathspecs=()
  if [[ -n "${MODEL_PATHS}" ]]; then
    read -r -a model_pathspecs <<< "${MODEL_PATHS}"
  else
    model_pathspecs=("${MODEL_PATHSPEC}")
  fi

  if [[ "${#model_pathspecs[@]}" -eq 0 ]]; then
    log "ERROR: no model pathspecs configured. Set MODEL_PATHS or MODEL_PATHSPEC."
    exit 1
  fi

  local model_pathspec_label
  model_pathspec_label="${model_pathspecs[*]}"

  if [[ ! -d "${PROJECT_DIR}" ]]; then
    log "ERROR: PROJECT_DIR does not exist: ${PROJECT_DIR}"
    exit 1
  fi

  cd "${PROJECT_DIR}"

  if ! "${GIT_BIN}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "ERROR: PROJECT_DIR is not inside a Git work tree: ${PROJECT_DIR}"
    exit 1
  fi

  if [[ -n "${REQUIRED_PATH}" && ! -e "${REQUIRED_PATH}" ]]; then
    log "Required checkpoint path '${REQUIRED_PATH}' does not exist; skipping commit and push."
    return 0
  fi

  local branch
  branch="$("${GIT_BIN}" branch --show-current)"
  if [[ -z "${branch}" ]]; then
    log "ERROR: cannot push from a detached HEAD; checkout a branch first."
    exit 1
  fi

  local changed_records=() changed_paths=() record path
  mapfile -d '' -t changed_records < <(
    "${GIT_BIN}" status --porcelain=v1 -z --untracked-files=all -- "${model_pathspecs[@]}"
  )

  if [[ "${#changed_records[@]}" -eq 0 ]]; then
    log "No model changes found for pathspec(s) '${model_pathspec_label}'; skipping commit and push."
    return 0
  fi

  for record in "${changed_records[@]}"; do
    path="${record:3}"
    if [[ -e "${path}" ]]; then
      changed_paths+=("${path}")
    else
      log "Skipping missing model path '${path}'."
    fi
  done

  if [[ "${#changed_paths[@]}" -eq 0 ]]; then
    log "No existing changed model files found for pathspec(s) '${model_pathspec_label}'; skipping commit and push."
    return 0
  fi

  "${GIT_BIN}" add -- "${changed_paths[@]}"

  if "${GIT_BIN}" diff --cached --quiet -- "${changed_paths[@]}"; then
    log "No staged model changes found after git add; skipping commit and push."
    return 0
  fi

  local commit_timestamp commit_message upstream
  commit_timestamp="$(date '+%Y-%m-%d %H:%M:%S %z')"
  commit_message="${COMMIT_MESSAGE_PREFIX} (${commit_timestamp})"

  log "Committing model changes with message: ${commit_message}"
  "${GIT_BIN}" commit --only -m "${commit_message}" -- "${changed_paths[@]}"

  upstream="$("${GIT_BIN}" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  if [[ -n "${upstream}" ]]; then
    log "Pushing current branch to configured upstream: ${upstream}"
    "${GIT_BIN}" push
  else
    log "No upstream configured; pushing ${branch} to ${GIT_REMOTE}/${branch}"
    "${GIT_BIN}" push "${GIT_REMOTE}" "HEAD:${branch}"
  fi

  log "Model auto-commit and push completed."
}

main "$@"
