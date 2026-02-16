#!/usr/bin/env bash
set -euo pipefail

RUN_TMUX_META_DIR="${RUN_TMUX_META_DIR:-runs/.tmux_meta}"
mkdir -p "${RUN_TMUX_META_DIR}"

_extract_run_id_from_command() {
  local args=("$@")
  local i
  for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--run-id" && $((i + 1)) -lt ${#args[@]} ]]; then
      printf "%s" "${args[$((i + 1))]}"
      return 0
    fi
    if [[ "${args[$i]}" =~ ^--run-id=(.+)$ ]]; then
      printf "%s" "${BASH_REMATCH[1]}"
      return 0
    fi
  done
  return 1
}

_meta_path() {
  local session_name="$1"
  printf "%s/%s.meta" "${RUN_TMUX_META_DIR}" "${session_name}"
}

run_tmux_job() {
  if [[ $# -lt 3 ]]; then
    echo "Usage: run_tmux_job <session_name> <conda_env> <command...>" >&2
    return 2
  fi

  local session_name="$1"
  shift
  local conda_env="$1"
  shift

  if tmux has-session -t "${session_name}" 2>/dev/null; then
    tmux kill-session -t "${session_name}"
  fi

  local run_id
  run_id=""
  if run_id="$(_extract_run_id_from_command "$@")"; then
    :
  else
    run_id="${session_name}"
  fi

  local run_dir="runs/${run_id}"
  local log_file="${run_dir}/run_console.log"
  local exit_file="${run_dir}/.tmux_exit_code"
  mkdir -p "${run_dir}"
  rm -f "${exit_file}"

  local cmd_str
  cmd_str="$(printf '%q ' "$@")"
  local wrapped
  wrapped="cd $(printf '%q' "$(pwd)"); set -o pipefail; ${cmd_str} > $(printf '%q' "${log_file}") 2>&1; status=\$?; echo \$status > $(printf '%q' "${exit_file}"); exit \$status"

  tmux new-session -d -s "${session_name}" \
    "conda run --no-capture-output -n $(printf '%q' "${conda_env}") bash -lc $(printf '%q' "${wrapped}")"

  local meta
  meta="$(_meta_path "${session_name}")"
  cat > "${meta}" <<META
run_id=${run_id}
run_dir=${run_dir}
log_file=${log_file}
exit_file=${exit_file}
META

  echo "session=${session_name} run_id=${run_id} log=${log_file}"
}

wait_tmux_job() {
  if [[ $# -ne 1 ]]; then
    echo "Usage: wait_tmux_job <session_name>" >&2
    return 2
  fi

  local session_name="$1"
  local meta
  meta="$(_meta_path "${session_name}")"
  if [[ ! -f "${meta}" ]]; then
    echo "Missing tmux meta for session ${session_name}: ${meta}" >&2
    return 2
  fi

  # shellcheck disable=SC1090
  source "${meta}"

  while tmux has-session -t "${session_name}" 2>/dev/null; do
    sleep 2
  done

  local status
  status=1
  if [[ -f "${exit_file}" ]]; then
    status="$(cat "${exit_file}" | tr -d '[:space:]')"
    if [[ -z "${status}" ]]; then
      status=1
    fi
  fi

  if [[ "${status}" != "0" ]]; then
    local tail_log="${run_dir}/tail.log"
    if [[ -f "${log_file}" ]]; then
      tail -n 200 "${log_file}" > "${tail_log}" || true
    else
      printf "log file not found: %s\n" "${log_file}" > "${tail_log}"
    fi
    echo "session=${session_name} status=${status} tail=${tail_log}" >&2
    return "${status}"
  fi

  echo "session=${session_name} status=0"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run_tmux_job|wait_tmux_job> ..." >&2
    exit 2
  fi
  subcmd="$1"
  shift
  case "${subcmd}" in
    run_tmux_job)
      run_tmux_job "$@"
      ;;
    wait_tmux_job)
      wait_tmux_job "$@"
      ;;
    *)
      echo "Unknown subcommand: ${subcmd}" >&2
      exit 2
      ;;
  esac
fi
