# The live notebook, from another computer

The analysis notebook runs on the lab PC, where the bursts are. From a laptop anywhere you open it in
a browser, watch edits made on the PC as they happen, run cells on the PC's kernel, and let an agent on
the laptop work on the same running notebook.

```
 laptop (anywhere)                          lab PC
 ┌────────────────────────┐   Tailscale    ┌────────────────────────────────────────────────┐
 │ browser ───────────────┼── (encrypted, ─┼─► tailscale serve :443 ─► marimo :2718 (local) │
 │ agent: marimo pair ────┼── account's   ─┼─►   one session, one kernel, the bursts on D: │
 │                        │   devices only)│      notebooks/foundation.py  ◄── git, agents │
 └────────────────────────┘                └────────────────────────────────────────────────┘
```

- **Who can reach it.** Only the devices signed in to the same Tailscale account (the *tailnet*), and
  only with the notebook's access token. The server listens on the PC itself; nothing is opened on the
  local network or the internet.
- **One session.** Every browser that opens the notebook joins the same session and kernel: cells
  edited in one show up in the others, and outputs are shared. Add `?kiosk=true` to the address to
  watch without editing.
- **Edits to the file.** When `notebooks/foundation.py` changes on disk (an agent on the PC, a
  `git pull`), the change appears in the open browsers within a second. Changed cells are marked
  stale; they run when someone runs them.
- **Nothing runs by itself.** Opening the notebook does not run it (`notebooks/pyproject.toml`). Press
  *Run all* (or run cells one by one). Sections 0–16 take about 3 minutes on a whole burst; the full
  notebook needs roughly 8–11 GB (see [README.md](README.md#memory-and-time)).

## Once, on the PC

From a checkout of this repository on the PC (WSL):

```bash
notebooks/live.sh install
notebooks/live.sh start --burst "/mnt/d/Conjunctiva Code/limbus/recordings/burst_2026-09-16_15-50-52"
```

`install` makes the Python environment (`notebooks/.venv`), fetches Tailscale (a pinned, checksummed
build that runs without administrator rights) and makes the access token. `start` starts the notebook
server and Tailscale. The first time it prints a **sign-in link**: the PC is not on a tailnet yet.

`notebooks/live.sh status` shows what runs and where; `notebooks/live.sh stop` stops the server.
After a restart of the PC, run the same `notebooks/live.sh start --burst …` again: the burst is not
remembered. The server and Tailscale run as systemd user services, so WSL needs systemd on
(`systemd=true` under `[boot]` in `/etc/wsl.conf`).

User services stop when your last login session on the PC ends (the last terminal is closed), unless
*lingering* is on for your user. `install` and `start` print a warning while it is off. To keep the
server and Tailscale running without a session, run once on the PC: `loginctl enable-linger $USER`.

## Once, on the laptop

1. **A Tailscale account.** Open the sign-in link that `start` printed (it works from any device) and
   sign in, with an existing Tailscale account or a new free one. The PC then joins the tailnet as
   `limbus` and publishes the notebook by itself.
2. **HTTPS for the tailnet.** In the Tailscale admin console, under
   [DNS](https://login.tailscale.com/admin/dns), keep *MagicDNS* on and enable *HTTPS Certificates*.
   (If Tailscale asks for this with a link of its own, `notebooks/live.sh status` on the PC shows it.)
3. **Tailscale on the laptop.** Install it from <https://tailscale.com/download> and sign in with the
   same account.
4. **The address.** On the PC, `~/.local/state/limbus-live/url` holds the notebook's address with its
   token, `https://limbus.<your-tailnet>.ts.net/?access_token=…`. Get it from whoever runs the PC, or
   from the agent working there. Treat it like a password. Open it once in the laptop's browser; the
   browser then stays signed in, and `https://limbus.<your-tailnet>.ts.net/` is enough.

## Working with an agent on the laptop

**Live, on the running notebook (recommended).** `marimo pair` lets an agent read, edit and run the
cells of the running session. You see its edits in your browser as it makes them, and the PC saves
them to `notebooks/foundation.py`. On the laptop:

```bash
uv tool install "marimo>=0.25.2"            # or: pip install "marimo>=0.25.2"
# the token is the access_token part of the address; keep it in a file only you can read
printf '%s' 'TOKEN' > ~/.limbus-token && chmod 600 ~/.limbus-token
marimo pair notebook list --url https://limbus.<your-tailnet>.ts.net --token-file ~/.limbus-token
```

For Claude Code, add marimo's pairing skill once:

```bash
claude plugin marketplace add marimo-team/marimo-pair
claude plugin install marimo-pair@marimo-pair
```

then ask it to pair with the notebook at `https://limbus.<your-tailnet>.ts.net`, with the token in
`~/.limbus-token`. It drives the notebook with `marimo pair execute --url … --session … --token-file …`.

**Through git.** The notebook is a plain Python file in this repository, so an agent can also work on
a clone of it: edit `notebooks/foundation.py` on a branch and push. On the PC, `git pull` in the
served checkout brings the change into the open browsers. Edits made in the browser are saved to the
PC's checkout, and are committed from there.

## If something does not work

| What you see | What to do |
|---|---|
| The address does not load at all | Is Tailscale on and signed in on the laptop? On the PC, `notebooks/live.sh status` should say `tailscale: Running`. |
| A login page asks for a password | Paste the token (the `access_token` part of the address). |
| "kernel disconnected" after *Run all* | The kernel ran out of the memory the server is given (12 GB by default). Restart the kernel from the notebook's menu; run fewer sections, or start the server with more, e.g. `LIMBUS_LIVE_MEMORY_MAX=16G notebooks/live.sh start --burst …`, when the PC has it free. |
| The browser shows an old version of a cell | Reload the page. |
