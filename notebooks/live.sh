#!/usr/bin/env bash
# Serves the analysis notebooks live from this computer, for a browser and an agent on another computer.
#
#   notebooks/live.sh install            the Python environment, the Tailscale programs and the access token
#   notebooks/live.sh start [--burst D]  start (or restart) the notebook server and put it on the tailnet
#   notebooks/live.sh status             what runs, and the address to open
#   notebooks/live.sh stop [--all]       stop the notebook server (--all: take this computer off the tailnet too)
#
# The notebook server (marimo) listens on this computer only, behind a token. Tailscale makes it reachable
# from the devices of one Tailscale account, wherever they are, and from nothing else: nothing listens on the
# local network or the internet. Tailscale runs as this user, without root (userspace networking). Adding this
# computer to the tailnet takes a sign-in to the Tailscale account; `start` prints the link for it.
# notebooks/LIVE.md has the steps for the other computer.
set -euo pipefail

NOTEBOOKS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NOTEBOOK=$NOTEBOOKS_DIR/foundation.py
VENV=$NOTEBOOKS_DIR/.venv
PORT=2718
TAILNET_NAME=limbus  # this computer's name on the tailnet
CONFIG_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/limbus-live
STATE_DIR=${XDG_STATE_HOME:-$HOME/.local/state}/limbus-live
TAILSCALE_DIR=${XDG_DATA_HOME:-$HOME/.local/share}/limbus-live/tailscale
TAILSCALE=$TAILSCALE_DIR/tailscale
TOKEN_FILE=$CONFIG_DIR/token
SOCKET=$STATE_DIR/tailscaled.sock
# where the notebook writes what it makes (previews/, vessel_pictures/, neural_cache/): outside the repository
WORK_DIR=$HOME/limbus-work
# the notebook server's share of this computer, which runs other jobs too
MEMORY_MAX=${LIMBUS_LIVE_MEMORY_MAX:-12G}
THREADS=${LIMBUS_LIVE_THREADS:-6}

# Tailscale's static build for Linux, pinned and checked (https://pkgs.tailscale.com/stable/)
TAILSCALE_VERSION=1.104.7
TAILSCALE_SHA256=e56693ff74a64f2591b7e819c280635a100aae1ddfb0ef058fcd052b7cd8d16c
TAILSCALE_TARBALL=tailscale_${TAILSCALE_VERSION}_amd64.tgz

MARIMO_UNIT=limbus-live-marimo
TAILSCALED_UNIT=limbus-live-tailscaled
JOIN_UNIT=limbus-live-join

say() { printf '%s\n' "$*"; }
die() { printf 'live.sh: %s\n' "$*" >&2; exit 1; }

need_systemd() {
    systemctl --user show-environment >/dev/null 2>&1 \
        || die "systemd user services are needed (systemctl --user does not answer); in WSL, set systemd=true under [boot] in /etc/wsl.conf and restart WSL"
}

# User services stop when this user's last login session ends, unless lingering is on for the user.
warn_linger() {
    local user
    user=$(id -un)
    [ "$(loginctl show-user "$user" --property=Linger --value 2>/dev/null)" = yes ] && return 0
    say "" >&2
    say "Warning: the notebook server and Tailscale stop when $user's last login session on this computer ends." >&2
    say "To keep them running after that, run once:  loginctl enable-linger $user" >&2
}

ts() { "$TAILSCALE" --socket="$SOCKET" "$@"; }

# One field of `tailscale status --json`, such as BackendState, AuthURL or Self.DNSName ("" if absent).
ts_field() {
    ts status --json 2>/dev/null | python3 -c '
import json, sys
try:
    value = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
for key in sys.argv[1].split("."):
    value = value.get(key) if isinstance(value, dict) else None
print((value or "").rstrip("."))' "$1" || true
}

running() { systemctl --user is-active --quiet "$1"; }  # running UNIT

# Start a program as a user service, logging to LOG. Extra systemd properties come in the array UNIT_PROPERTIES.
launch() {  # launch UNIT LOG -- command...
    local unit=$1 log=$2
    shift 3
    systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
    systemd-run --user --quiet --unit="$unit" --collect \
        -p StandardOutput="append:$log" -p StandardError="append:$log" \
        ${UNIT_PROPERTIES[@]+"${UNIT_PROPERTIES[@]}"} "$@"
}

halt() { systemctl --user stop "$1" >/dev/null 2>&1 || true; }  # halt UNIT

write_url() {
    local name
    name=$(ts_field Self.DNSName)
    [ -n "$name" ] || return 0
    (umask 077; echo "https://$name/?access_token=$(cat "$TOKEN_FILE")" >"$STATE_DIR/url")
}

cmd_install() {
    mkdir -p "$CONFIG_DIR" "$STATE_DIR" "$TAILSCALE_DIR" "$WORK_DIR"
    chmod 700 "$CONFIG_DIR" "$STATE_DIR"

    say "Python environment: $VENV"
    if command -v uv >/dev/null; then
        [ -x "$VENV/bin/python" ] || uv venv --quiet --python 3.12 "$VENV"
        VIRTUAL_ENV=$VENV uv pip install --quiet -r "$NOTEBOOKS_DIR/requirements.txt"
    else
        [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
        "$VENV/bin/pip" install --quiet -r "$NOTEBOOKS_DIR/requirements.txt"
    fi
    say "  marimo $("$VENV/bin/marimo" --version)"

    if [ -x "$TAILSCALE_DIR/tailscaled" ]; then
        say "Tailscale $("$TAILSCALE" version | head -n 1): already in $TAILSCALE_DIR"
    else
        say "Tailscale $TAILSCALE_VERSION -> $TAILSCALE_DIR"
        local download
        download=$(mktemp -d "$TAILSCALE_DIR/.download.XXXXXX")
        # a 40 MB download: try again when the connection breaks part way
        curl -fsSL --retry 5 --retry-all-errors -o "$download/$TAILSCALE_TARBALL" \
            "https://pkgs.tailscale.com/stable/$TAILSCALE_TARBALL" \
            || { rm -rf "$download"; die "could not download Tailscale; run notebooks/live.sh install again"; }
        echo "$TAILSCALE_SHA256  $download/$TAILSCALE_TARBALL" | sha256sum --check --quiet \
            || { rm -rf "$download"; die "the Tailscale download does not match its checksum"; }
        tar -xzf "$download/$TAILSCALE_TARBALL" -C "$download"
        install -m 755 "$download/tailscale_${TAILSCALE_VERSION}_amd64/tailscale" \
            "$download/tailscale_${TAILSCALE_VERSION}_amd64/tailscaled" "$TAILSCALE_DIR/"
        rm -rf "$download"
    fi

    if [ -s "$TOKEN_FILE" ]; then
        say "Access token: $TOKEN_FILE (kept)"
    else
        (umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(24))' >"$TOKEN_FILE")
        say "Access token: $TOKEN_FILE (new)"
    fi
    say "Installed. Next: notebooks/live.sh start"
}

start_marimo() {  # start_marimo BURST
    local burst=$1
    halt "$MARIMO_UNIT"
    UNIT_PROPERTIES=(-p "MemoryMax=$MEMORY_MAX" -p MemorySwapMax=0 -p "CPUQuota=${THREADS}00%"
                     -p "WorkingDirectory=$WORK_DIR")
    local environment=("OMP_NUM_THREADS=1" "OPENBLAS_NUM_THREADS=1" "MKL_NUM_THREADS=1" "OPENCV_FOR_THREADS_NUM=1")
    [ -n "$burst" ] && environment+=("LIMBUS_BURST=$burst")
    # on this computer only, behind the token; --watch takes in changes made to the file on disk
    launch "$MARIMO_UNIT" "$STATE_DIR/marimo.log" -- env "${environment[@]}" \
        "$VENV/bin/marimo" -q edit "$NOTEBOOK" --host 127.0.0.1 --port "$PORT" --headless --watch \
        --skip-update-check --token-password-file "$TOKEN_FILE"
    for _ in $(seq 150); do
        curl -fs -o /dev/null "http://127.0.0.1:$PORT/health" && return 0
        sleep 0.2
    done
    die "the notebook server did not start; see $STATE_DIR/marimo.log"
}

start_tailscaled() {
    [ -x "$TAILSCALE_DIR/tailscaled" ] || die "Tailscale is not installed: run notebooks/live.sh install"
    if ! running "$TAILSCALED_UNIT"; then
        UNIT_PROPERTIES=(-p Restart=on-failure)
        launch "$TAILSCALED_UNIT" "$STATE_DIR/tailscaled.log" -- "$TAILSCALE_DIR/tailscaled" \
            --tun=userspace-networking --statedir="$STATE_DIR/tailscale" --socket="$SOCKET" --port=0
    fi
    for _ in $(seq 50); do
        ts status --json >/dev/null 2>&1 && return 0
        sleep 0.2
    done
    die "tailscaled did not start; see $STATE_DIR/tailscaled.log"
}

# HTTPS on the tailnet (port 443), to the notebook server. The first time, Tailscale may print a link to
# enable HTTPS for the tailnet, and waits until it is enabled.
SERVE=(serve --bg --yes --https=443)

cmd_start() {
    local burst=${LIMBUS_BURST:-} state url
    while [ $# -gt 0 ]; do
        case $1 in
            --burst) burst=$(cd "$2" && pwd); shift 2 ;;
            *) die "unknown option $1" ;;
        esac
    done
    [ -x "$VENV/bin/marimo" ] && [ -s "$TOKEN_FILE" ] || die "not installed yet: run notebooks/live.sh install"
    mkdir -p "$STATE_DIR" "$WORK_DIR"

    start_marimo "$burst"
    say "Notebook server: http://127.0.0.1:$PORT (this computer only)${burst:+, burst $burst}"

    start_tailscaled
    # a tailscaled that has just started takes a moment to say whether it is signed in
    for _ in $(seq 150); do
        state=$(ts_field BackendState)
        case $state in ''|NoState|Starting) sleep 0.2 ;; *) break ;; esac
    done

    if [ "$state" != Running ]; then
        # Not on the tailnet. In the background, `tailscale up` joins it, waiting for the sign-in if one is needed,
        # and then the notebook is served on the tailnet, so nothing more has to be done on this computer.
        if ! running "$JOIN_UNIT"; then
            UNIT_PROPERTIES=()
            # shellcheck disable=SC2016  # the arguments expand in the inner shell
            launch "$JOIN_UNIT" "$STATE_DIR/join.log" -- bash -c '"$0" --socket="$1" up --hostname="$2" &&
                "$0" --socket="$1" serve --bg --yes --https=443 "http://127.0.0.1:$3"' \
                "$TAILSCALE" "$SOCKET" "$TAILNET_NAME" "$PORT"
        fi
        url=
        for _ in $(seq 150); do
            state=$(ts_field BackendState)
            [ "$state" = Running ] && break
            url=$(ts_field AuthURL)
            [ -n "$url" ] && break
            sleep 0.2
        done
    fi

    if [ "$state" = Running ]; then
        halt "$JOIN_UNIT"  # it joined without a sign-in: the notebook is served from here
        timeout 120 "$TAILSCALE" --socket="$SOCKET" "${SERVE[@]}" "http://127.0.0.1:$PORT" \
            >"$STATE_DIR/serve.log" 2>&1 \
            || { cat "$STATE_DIR/serve.log" >&2; die "tailscale serve did not finish: see the message above"; }
        write_url
        say "On the tailnet: https://$(ts_field Self.DNSName)/"
        say "The address with the token (keep it private) is in $STATE_DIR/url"
        return 0
    fi

    [ -n "$url" ] || die "Tailscale neither joined the tailnet nor gave a sign-in link; see $STATE_DIR/join.log"
    (umask 077; echo "$url" >"$STATE_DIR/signin-url")
    say ""
    say "This computer is not on the tailnet yet. To add it, open this link on any device and sign in to the"
    say "Tailscale account the other computer uses:"
    say "  $url"
    say "After the sign-in it joins by itself and serves the notebook at https://$TAILNET_NAME.<tailnet>.ts.net/"
    say "(notebooks/live.sh status shows the address; if Tailscale asks for HTTPS to be enabled for the tailnet,"
    say "its link is in $STATE_DIR/join.log)."
}

cmd_status() {
    if running "$MARIMO_UNIT"; then say "notebook server: running, http://127.0.0.1:$PORT"
    else say "notebook server: stopped"; fi
    if [ -x "$TAILSCALE" ] && ts status --json >/dev/null 2>&1; then
        local state
        state=$(ts_field BackendState)
        say "tailscale: $state $(ts_field Self.DNSName)"
        if [ "$state" = Running ]; then
            ts serve status 2>/dev/null | sed 's/^/  /' || true
            write_url
        elif [ -n "$(ts_field AuthURL)" ]; then
            say "  waiting for the sign-in: $(ts_field AuthURL)"
        fi
        [ -f "$STATE_DIR/join.log" ] && grep -o 'https://login.tailscale.com/[^ ]*' "$STATE_DIR/join.log" \
            | sort -u | sed 's/^/  link from Tailscale: /' || true
    else
        say "tailscale: not running"
    fi
    if [ -f "$STATE_DIR/url" ]; then say "address with the token: $STATE_DIR/url"; fi
}

cmd_stop() {
    halt "$MARIMO_UNIT"
    if [ -x "$TAILSCALE" ] && ts status --json >/dev/null 2>&1; then
        ts serve reset >/dev/null 2>&1 || true
        if [ "${1:-}" = --all ]; then
            halt "$JOIN_UNIT"
            ts down >/dev/null 2>&1 || true
            halt "$TAILSCALED_UNIT"
        fi
    fi
    rm -f "$STATE_DIR/url"
    say "stopped"
}

case ${1:-} in
    install|start) need_systemd; action=$1; shift; "cmd_$action" "$@"; warn_linger ;;
    status|stop) need_systemd; action=$1; shift; "cmd_$action" "$@" ;;
    *) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
