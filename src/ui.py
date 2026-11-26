"""A local page for trying the helpdesk and watching the three memories.

One browser tab is one session. The page mints a session id when it loads
and keeps it in sessionStorage, which is not shared with other tabs. Opening
a second tab therefore starts a fresh Session: slots start empty and only
the files in memory/ carry over. Switching user does the same. The page
shows, per turn, the reply, the intent, the slots, what was recalled, and
the rows now on disk.

Nothing here is imported by the agent. It is a way to exercise it.

    .venv/bin/python -m src.ui
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.agent import Session
from src.config import ABLATION_CONFIGS, DB_PATH
from src.history import STRATEGIES
from src.episodic_memory import EPISODIC_PATH
from src.semantic_memory import SEMANTIC_PATH
from src.db import connect

HOST = "127.0.0.1"
PORT = 8002

# Saved runs written by src.run, one folder per configuration.
RUNS_ROOT = Path(__file__).resolve().parent.parent / "results" / "runs"

# One Session per (user, session). A switch drops the old one, so its slots
# do not leak, which is the same rule the isolation tests check.
_sessions: dict[tuple[str, str], Session] = {}
_lock = threading.Lock()


def get_session(user: str, session: str, strategy: str = "full") -> Session:
    # The page sends the id it minted for this tab. A missing one still has
    # to name something, but a real tab always sends its own. The strategy is
    # fixed when the session is created: changing it in the page starts a new
    # conversation, so a chat is never half full and half trim.
    chosen = strategy if strategy in STRATEGIES else "full"
    key = (user.strip() or "U1", session.strip() or "tab")
    with _lock:
        found = _sessions.get(key)
        if found is None:
            found = Session(*key, strategy=chosen)
            found.run_id = f"stm-{chosen}"
            _sessions[key] = found
        return found


def drop_session(user: str, session: str) -> None:
    with _lock:
        _sessions.pop((user, session), None)


def _clean(value):
    """JSON-ready copy with embeddings removed. They are long and not facts."""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items() if key != "embedding"}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def snapshot(user: str) -> dict:
    """What is on disk for this user, for the side panels.

    The files hold every user. The page is showing one of them, so a row
    that belongs to anyone else is left out. Switching user reloads this.
    """
    who = user.strip() or "U1"
    semantic = [row for row in _read_jsonl(SEMANTIC_PATH) if row.get("user") == who]
    episodic = [
        _clean(row) for row in _read_jsonl(EPISODIC_PATH) if row.get("user") == who
    ]
    return {
        "facts": semantic,
        "episodes": episodic,
        "tickets": [row for row in _table("tickets") if row.get("owner_user") == who],
        "bookings": [row for row in _table("slots") if row.get("owner_user") == who],
    }


def list_runs() -> list[str]:
    """Configurations that have a saved transcript, in the ablation order."""
    return [
        name for name in ABLATION_CONFIGS if (RUNS_ROOT / name / "transcript.json").exists()
    ]


def load_run(config: str) -> dict | None:
    """One saved transcript, or None. The name is checked against the known
    configurations, so the query string cannot point at another path."""
    if config not in ABLATION_CONFIGS:
        return None
    path = RUNS_ROOT / config / "transcript.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _table(name: str) -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = connect(DB_PATH)
    try:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {name}").fetchall()]
    finally:
        conn.close()


def _changed_slot(decision: dict) -> int | None:
    """The slot this turn wrote, when the turn updated a visit rather than inserting one.

    A reschedule keeps the same slot_id. The page uses it to mark that row,
    so the updated day is not read as a second booking beside the chat history.
    """
    if decision.get("name") != "update_slot":
        return None
    result = decision.get("result")
    if not isinstance(result, dict):
        return None
    slot_id = result.get("slot_id")
    return slot_id if isinstance(slot_id, int) else None


def turn(user: str, session: str, text: str, strategy: str = "full") -> dict:
    """One user message through the real agent. Recall is captured before write."""
    current = get_session(user, session, strategy)
    labelled = current.ingest(text)
    decision = current.decide(text)
    said = current.reply(text, decision)
    recalled = labelled.pop("recalled")
    action = decision.get("action")
    return {
        "reply": said,
        "intent": labelled["intent"],
        "source": labelled["source"],
        "turn": labelled["turn"],
        "action": action,
        "tool": decision.get("name"),
        "slots": labelled["slots"],
        "recalled": {
            "facts": [
                {"key": row["key"], "value": row["value"], "score": round(row.get("score", 0), 3)}
                for row in recalled["facts"]
            ],
            "episodes": [
                {"text": row["text"], "type": row["type"], "score": round(row.get("score", 0), 3)}
                for row in recalled["episodes"]
            ],
        },
        "state": snapshot(user),
        "changed_slot": _changed_slot(decision),
        "strategy": current.strategy,
        "window": current._history_block(),
    }


def forget(user: str, session: str, value: str) -> dict:
    current = get_session(user, session)
    removed = current.memory.forget(value)
    return {"removed": removed, "state": snapshot(user)}


def clear_user(user: str, session: str) -> dict:
    """Wipe this user and drop every live session of theirs.

    The page button is per user, not per tab. Facts, episodes, transcripts,
    tickets and bookings go. The other user is left as they are. The live
    Session objects go too, so the next message starts with empty slots.
    """
    who = user.strip() or "U1"
    current = get_session(who, session)
    removed = current.memory.clear_user()
    conn = connect(DB_PATH)
    try:
        for table in ("tickets", "slots", "assets"):
            conn.execute(f"DELETE FROM {table} WHERE owner_user = ?", (who,))
        conn.commit()
    finally:
        conn.close()
    with _lock:
        for key in [key for key in _sessions if key[0] == who]:
            _sessions.pop(key, None)
    return {"removed": removed, "state": snapshot(who)}


def close_session(user: str, session: str, reflection: str) -> dict:
    current = get_session(user, session)
    # An empty box means the session writes its own note. A typed box is
    # stored as given, so a test can still plant a known reflection.
    typed = reflection.strip()
    written = current.end(typed or None)
    drop_session(user, session)
    return {
        "reflection": {"id": written["id"], "text": written["text"]},
        "state": snapshot(user),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/runs":
            self._send_json({"runs": list_runs()})
            return
        if path == "/api/run":
            found = load_run(self._query().get("config", ""))
            if found is None:
                self._send_json({"error": "no saved run for that configuration"}, status=404)
                return
            self._send_json(found)
            return
        if path == "/api/state":
            user = self._query().get("user", "U1")
            self._send_json({"state": snapshot(user)})
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "that was not JSON"}, status=400)
            return
        user = str(body.get("user") or "U1")
        session = str(body.get("session") or "A")
        try:
            if path == "/api/turn":
                text = str(body.get("text") or "").strip()
                if not text:
                    self._send_json({"error": "type a message first"}, status=400)
                    return
                strategy = str(body.get("strategy") or "full")
                self._send_json(turn(user, session, text, strategy))
            elif path == "/api/forget":
                self._send_json(forget(user, session, str(body.get("value") or "")))
            elif path == "/api/end":
                self._send_json(close_session(user, session, str(body.get("reflection") or "")))
            elif path == "/api/reset":
                drop_session(user, session)
                self._send_json({"ok": True, "state": snapshot(user)})
            elif path == "/api/clear":
                self._send_json(clear_user(user, session))
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:  # shown in the page rather than as a stack trace
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, fmt: str, *args) -> None:
        # The call log already records what matters. Skip the access log.
        return

    def _query(self) -> dict:
        raw = urlparse(self.path).query
        found = {}
        for part in raw.split("&"):
            if "=" in part:
                key, value = part.split("=", 1)
                found[key] = value
        return found

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        self._send(status, json.dumps(payload, default=str).encode("utf-8"), "application/json")


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RecallBot</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.45 "Iowan Old Style", Palatino, Georgia, serif;
         background: #f4f1ea; color: #231f1a; }
  header { padding: 16px 22px; background: #231f1a; color: #f4f1ea;
           display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
  header h1 { font-size: 18px; margin: 0; font-weight: 600; letter-spacing: .02em; }
  header label { font: 13px/1 sans-serif; color: #c9c1b4; }
  header select, header input { font: 14px sans-serif; padding: 4px 8px;
           background: #3a342c; color: #f4f1ea; border: 1px solid #5b5348; border-radius: 4px; }
  #session-label { font: 12px ui-monospace, monospace; color: #c9c1b4; }
  main { display: grid; grid-template-columns: 1.3fr .9fr .9fr; gap: 14px; padding: 14px; }
  section { background: #fffdf8; border: 1px solid #e2d9cc; border-radius: 8px; padding: 14px; }
  h2 { font: 600 13px sans-serif; letter-spacing: .08em; text-transform: uppercase;
       color: #7a7064; margin: 0 0 10px; }
  #log { min-height: 360px; max-height: 62vh; overflow-y: auto; display: flex;
         flex-direction: column; gap: 10px; }
  .bubble { padding: 8px 12px; border-radius: 8px; max-width: 85%; white-space: pre-wrap; }
  .me { align-self: flex-end; background: #231f1a; color: #f4f1ea; }
  .bot { align-self: flex-start; background: #efe8dc; }
  .meta { align-self: stretch; font: 12px/1.4 ui-monospace, monospace; color: #5c564d;
          background: #f7f3eb; border: 1px dashed #e2d9cc; border-radius: 6px; padding: 8px 10px; }
  .meta b { color: #231f1a; }
  form { display: flex; gap: 8px; margin-top: 12px; }
  textarea { flex: 1; font: 15px/1.4 Georgia, serif; padding: 8px; resize: vertical;
             border: 1px solid #d9d0c3; border-radius: 6px; min-height: 46px; }
  button { font: 14px sans-serif; background: #231f1a; color: #f4f1ea; border: 0;
           border-radius: 6px; padding: 8px 14px; cursor: pointer; }
  button.ghost { background: transparent; color: #231f1a; border: 1px solid #c9bfb0; }
  header button.ghost { color: #f4f1ea; border-color: #5b5348; padding: 4px 10px; }
  .row { display: flex; gap: 8px; margin-top: 8px; }
  .row input { flex: 1; font: 14px sans-serif; padding: 6px 8px;
               border: 1px solid #d9d0c3; border-radius: 6px; }
  .card { border-top: 1px solid #eee6da; padding: 8px 0; font-size: 14px; }
  .card .tag { font: 11px sans-serif; letter-spacing: .04em; text-transform: uppercase;
               padding: 1px 6px; border-radius: 8px; background: #efe8dc; color: #5c564d; }
  .card .tag.off { background: #f3d6d0; }
  .empty { color: #9a9084; font-style: italic; }
  .error { color: #8d2d22; }
  table { width: 100%; border-collapse: collapse; font: 12px/1.35 ui-monospace, monospace; }
  td { padding: 4px 6px; border-bottom: 1px solid #f0e9de; vertical-align: top; }
  tr.changed td { background: #f4eccf; }
  td.when { white-space: nowrap; }
  [hidden] { display: none !important; }
  #replay { max-height: 72vh; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; }
  .session-head { font: 600 13px sans-serif; letter-spacing: .06em; text-transform: uppercase;
                  background: #231f1a; color: #f4f1ea; border-radius: 6px; padding: 6px 10px;
                  margin-top: 10px; }
  .turn-tag { font: 11px ui-monospace, monospace; color: #7a7064; margin-bottom: 2px; }
  .probe { border: 1px solid #e2d9cc; border-radius: 6px; padding: 8px 10px; background: #f7f3eb;
           font-size: 14px; }
  .probe .q { font-weight: 600; }
  .probe .exp { font: 12px ui-monospace, monospace; color: #5c564d; }
  .stepper { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
  #step-label { font: 12px ui-monospace, monospace; color: #5c564d; }
  .now { outline: 2px solid #c9a227; outline-offset: 3px; }
  .closed { border: 1px dashed #c9bfb0; border-radius: 6px; padding: 8px 10px;
            font-size: 14px; background: #f7f3eb; }
  .ok { color: #2d6a3a; }
  .bad { color: #8d2d22; }
</style>
</head>
<body>
<header>
  <h1>RecallBot · helpdesk</h1>
  <label>User
    <select id="user"><option>U1</option><option>U2</option></select>
  </label>
  <label>Memory
    <select id="strategy">
      <option value="full">full</option>
      <option value="trim">trim</option>
      <option value="summary">summary</option>
    </select>
  </label>
  <button type="button" class="ghost" id="do-clear">Clear this user</button>
  <label>Saved run
    <select id="run-config"></select>
  </label>
  <button type="button" class="ghost" id="do-load">Load run</button>
  <button type="button" class="ghost" id="do-live" hidden>Back to live</button>
  <span id="session-label"></span>
  <span style="font:12px sans-serif;color:#9a9084">Each tab is its own session. Facts in memory/ carry over; this tab's slots do not.</span>
</header>
<main>
  <section>
    <h2 id="convo-title">Conversation</h2>
    <div id="stepper" class="stepper" hidden>
      <button type="button" class="ghost" id="step-restart">Restart</button>
      <button type="button" class="ghost" id="step-back">Back</button>
      <button type="button" id="step-next">Next step</button>
      <button type="button" class="ghost" id="step-session">Skip to end of session</button>
      <span id="step-label"></span>
    </div>
    <div id="replay" hidden></div>
    <div id="live">
    <div id="log"></div>
    <form id="composer">
      <textarea id="text" placeholder="I sit at the Pune office and I use LAP-T14G3." autofocus></textarea>
      <button type="submit">Send</button>
    </form>
    <div class="row">
      <input id="forget" placeholder="forget this value, e.g. Hyderabad">
      <button type="button" class="ghost" id="do-forget">Forget</button>
    </div>
    <div class="row">
      <input id="reflection" placeholder="reflection written when the session ends">
      <button type="button" class="ghost" id="do-end">End session</button>
    </div>
    </div>
  </section>
  <section>
    <h2>Recalled this turn</h2>
    <div id="recalled" class="empty">Nothing recalled yet.</div>
    <h2 style="margin-top:18px">Semantic facts on disk</h2>
    <div id="facts"></div>
  </section>
  <section>
    <h2>Episodes on disk</h2>
    <div id="episodes"></div>
    <h2 style="margin-top:18px">Tickets and bookings</h2>
    <div id="rows"></div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const log = $("log");

// sessionStorage is per tab. A second tab does not see this key, so it
// mints a different id and the server builds a new Session for it.
let sessionId = sessionStorage.getItem("recallbot-session");
if (!sessionId) {
  const stamp = new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14);
  const rand = Math.random().toString(36).slice(2, 6);
  sessionId = "S-" + stamp + "-" + rand;
  sessionStorage.setItem("recallbot-session", sessionId);
}
$("session-label").textContent = "Session " + sessionId;

function who() {
  return { user: $("user").value, session: sessionId, strategy: $("strategy").value };
}

function add(cls, text) {
  const div = document.createElement("div");
  div.className = "bubble " + cls;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function note(html) {
  const div = document.createElement("div");
  div.className = "meta";
  div.innerHTML = html;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

let changedSlot = null;

function renderState(state) {
  const facts = $("facts");
  facts.innerHTML = "";
  if (!state.facts.length) facts.innerHTML = '<div class="empty">No facts stored.</div>';
  for (const row of state.facts) {
    const div = document.createElement("div");
    div.className = "card";
    div.innerHTML = '<span class="tag' + (row.status === "active" ? "" : " off") + '">' +
      row.status + "</span> " + row.user + " · " + row.key + " = " + row.value;
    facts.appendChild(div);
  }
  const episodes = $("episodes");
  episodes.innerHTML = "";
  if (!state.episodes.length) episodes.innerHTML = '<div class="empty">No episodes stored.</div>';
  for (const row of state.episodes) {
    const div = document.createElement("div");
    div.className = "card";
    div.innerHTML = '<span class="tag">' + row.type + "</span> " + row.user +
      " · " + row.text;
    episodes.appendChild(div);
  }
  const rows = $("rows");
  const tickets = state.tickets.map(r => ({
    kind: "ticket",
    id: r.ticket_id,
    detail: r.issue_type + " · " + r.status,
    changed: false,
  }));
  const bookings = state.bookings.map(r => ({
    kind: "booking",
    id: r.slot_id,
    detail: (r.date || "day not set") + " " + (r.time || "") + " · " + (r.location || ""),
    changed: changedSlot !== null && r.slot_id === changedSlot,
  }));
  const all = tickets.concat(bookings);
  if (!all.length) { rows.innerHTML = '<div class="empty">No rows.</div>'; return; }
  rows.innerHTML = "<table>" + all.map(row =>
    '<tr class="' + (row.changed ? "changed" : "") + '">' +
    "<td>" + row.kind + (row.changed ? " · updated" : "") + "</td>" +
    "<td>" + row.id + "</td>" +
    "<td>" + row.detail.trim() + "</td></tr>").join("") + "</table>";
}

function renderRecalled(recalled, slots) {
  const box = $("recalled");
  const facts = recalled.facts.map(r => r.key + " = " + r.value + " (" + r.score + ")");
  const episodes = recalled.episodes.map(r => r.type + ": " + r.text + " (" + r.score + ")");
  const filled = Object.entries(slots).filter(([, v]) => v);
  box.innerHTML = "";
  box.className = "";
  const lines = [
    ["Facts", facts],
    ["Episodes", episodes],
    ["Slots carried", filled.map(([k, v]) => k + " = " + v)],
  ];
  for (const [title, items] of lines) {
    const div = document.createElement("div");
    div.className = "card";
    div.innerHTML = "<b>" + title + ".</b> " + (items.length ? items.join("; ") : "none");
    box.appendChild(div);
  }
}

async function post(path, extra) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...who(), ...extra }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

$("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = $("text").value.trim();
  if (!text) return;
  $("text").value = "";
  add("me", text);
  try {
    const data = await post("/api/turn", { text });
    add("bot", data.reply);
    note("<b>" + data.intent + "</b> via " + data.source +
      " · turn " + data.turn + " · " + data.strategy +
      " · action " + data.action +
      (data.tool ? " (" + data.tool + ")" : "") +
      (data.window ? "<br><b>window sent:</b><br>" +
        data.window.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/\\n/g, "<br>")
        : "<br><b>window sent:</b> none"));
    renderRecalled(data.recalled, data.slots);
    changedSlot = data.changed_slot;
    renderState(data.state);
  } catch (err) {
    note('<span class="error">' + err.message + "</span>");
  }
});

$("do-forget").addEventListener("click", async () => {
  const value = $("forget").value.trim();
  if (!value) return;
  const data = await post("/api/forget", { value });
  note("Forgot \\"" + value + "\\": " + data.removed.facts + " fact(s), " +
    data.removed.episodes + " episode(s).");
  renderState(data.state);
  $("forget").value = "";
});

$("do-end").addEventListener("click", async () => {
  const data = await post("/api/end", { reflection: $("reflection").value });
  note("Session closed. Reflection " + data.reflection.id + ": " + data.reflection.text);
  renderState(data.state);
  $("reflection").value = "";
});

function freshSession() {
  const stamp = new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14);
  const rand = Math.random().toString(36).slice(2, 6);
  sessionId = "S-" + stamp + "-" + rand;
  sessionStorage.setItem("recallbot-session", sessionId);
  $("session-label").textContent = "Session " + sessionId;
  log.innerHTML = "";
  $("recalled").className = "empty";
  $("recalled").textContent = "Nothing recalled yet.";
}

$("do-clear").addEventListener("click", async () => {
  const name = $("user").value;
  if (!confirm("Clear everything stored for " + name + "? The other user is kept.")) return;
  try {
    const data = await post("/api/clear", {});
    freshSession();
    renderState(data.state);
    note("Cleared " + name + ": " + data.removed.facts + " fact(s), " +
      data.removed.episodes + " episode(s), " + data.removed.sessions + " conversation(s).");
  } catch (err) {
    note('<span class="error">' + err.message + "</span>");
  }
});

$("strategy").addEventListener("change", () => {
  // The window is chosen when the session starts. A new choice needs a new
  // session, otherwise the turns already sent stay under the old strategy.
  freshSession();
});

$("user").addEventListener("change", async () => {
  // A different user in this tab is a different conversation. Mint a new
  // session so the previous user's slots are not reused.
  const stamp = new Date().toISOString().replace(/[-:T.Z]/g, "").slice(0, 14);
  const rand = Math.random().toString(36).slice(2, 6);
  sessionId = "S-" + stamp + "-" + rand;
  sessionStorage.setItem("recallbot-session", sessionId);
  $("session-label").textContent = "Session " + sessionId;
  log.innerHTML = "";
  $("recalled").className = "empty";
  $("recalled").textContent = "Nothing recalled yet.";
  const res = await fetch("/api/state?user=" + $("user").value);
  renderState((await res.json()).state);
});

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

// A saved run is flattened into steps. Step 0 is the empty start. Then one
// step per turn, one for the session closing (where the reflection is
// written), and one per probe. Each step carries the state on disk after it.
let steps = [];
let at = 0;
let runInfo = null;

function buildSteps(data) {
  const out = [{
    kind: "start", session: "", user: "U1",
    state: data.initial_state, recalled: {facts: [], episodes: []}, slots: {},
  }];
  for (const played of data.sessions) {
    for (const t of played.turns || []) {
      out.push({kind: "turn", session: played.session, user: t.user, row: t,
        state: t.state, recalled: t.recalled, slots: t.slots || {}});
    }
    const last = (played.turns || []).slice(-1)[0];
    out.push({kind: "close", session: played.session, user: last ? last.user : "U1",
      state: played.closed_state, recalled: {facts: [], episodes: []}, slots: {}});
    for (const p of played.probes || []) {
      out.push({kind: "probe", session: played.session, user: p.user, row: p,
        state: p.state, recalled: p.recalled, slots: {}});
    }
  }
  return out;
}

function mine(state, user) {
  // The files hold every user. Show the one speaking at this step.
  return {
    facts: state.facts.filter(r => r.user === user),
    episodes: state.episodes.filter(r => r.user === user),
    tickets: state.tickets.filter(r => r.owner_user === user),
    bookings: state.bookings.filter(r => r.owner_user === user),
  };
}

function describe(step, index) {
  if (step.kind === "start") return "Start - nothing stored yet";
  if (step.kind === "turn") return "Session " + step.session + " - turn " + step.row.turn + " (" + step.user + ")";
  if (step.kind === "close") return "Session " + step.session + " closes";
  return "Probe " + step.row.id + " (" + step.user + ")";
}

function showStep() {
  const box = $("replay");
  box.innerHTML = "";
  let shown = null;
  let lastSession = null;
  for (let i = 0; i <= at; i++) {
    const step = steps[i];
    if (step.kind === "start") continue;
    if (step.session !== lastSession) {
      box.appendChild(el("div", "session-head", "Session " + step.session + " - " + runInfo.config));
      lastSession = step.session;
    }
    let node;
    if (step.kind === "turn") {
      node = el("div", "");
      node.appendChild(el("div", "turn-tag", step.user + " - turn " + step.row.turn +
        " - " + step.row.intent + " - action " + step.row.action +
        (step.row.tool ? " (" + step.row.tool + ")" : "")));
      node.appendChild(el("div", "bubble me", step.row.text));
      node.appendChild(el("div", "bubble bot", step.row.bot));
    } else if (step.kind === "close") {
      node = el("div", "closed", "Session " + step.session + " ended. The reflection and the summary episode were written to memory.");
    } else {
      node = el("div", "probe");
      node.appendChild(el("div", "q", step.row.id + " (" + step.row.category + ", " + step.row.user + ") " + step.row.question));
      node.appendChild(el("div", "exp", "expected: " + step.row.expected +
        (step.row.evidence_turn ? " - evidence turn " + step.row.evidence_turn : "")));
      const reply = el("div", "", "reply: " + step.row.reply);
      if (step.row.passed !== undefined) reply.className = step.row.passed ? "ok" : "bad";
      node.appendChild(reply);
    }
    if (i === at) { node.classList.add("now"); shown = node; }
    box.appendChild(node);
  }
  if (shown) shown.scrollIntoView({block: "nearest"});
  const step = steps[at];
  $("step-label").textContent = "Step " + at + " of " + (steps.length - 1) + " - " + describe(step) +
    " - memory of " + step.user;
  $("step-back").disabled = at === 0;
  $("step-restart").disabled = at === 0;
  $("step-next").disabled = at === steps.length - 1;
  $("step-session").disabled = at === steps.length - 1;
  changedSlot = null;
  renderState(mine(step.state, step.user));
  renderRecalled(step.recalled || {facts: [], episodes: []}, step.slots || {});
}

function renderRun(data) {
  if (!data.initial_state) {
    note('<span class="error">This run has no saved state. Run it again: ' +
      'uv run python -m src.run --config ' + data.config + ' --raw</span>');
    return;
  }
  runInfo = data;
  steps = buildSteps(data);
  at = 0;
  $("convo-title").textContent = "Saved run - " + data.config + " - " + data.run_id;
  $("live").hidden = true;
  $("replay").hidden = false;
  $("stepper").hidden = false;
  $("do-live").hidden = false;
  showStep();
}

function go(to) {
  at = Math.max(0, Math.min(steps.length - 1, to));
  showStep();
}

function nextSessionEnd() {
  // The next step that closes a session, or the very end.
  for (let i = at + 1; i < steps.length; i++) {
    if (steps[i].kind === "close") return i;
  }
  return steps.length - 1;
}

$("step-next").addEventListener("click", () => go(at + 1));
$("step-back").addEventListener("click", () => go(at - 1));
$("step-restart").addEventListener("click", () => go(0));
$("step-session").addEventListener("click", () => go(nextSessionEnd()));

document.addEventListener("keydown", (event) => {
  if ($("stepper").hidden) return;
  const tag = (event.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;
  if (event.key === "ArrowRight") go(at + 1);
  if (event.key === "ArrowLeft") go(at - 1);
});

async function loadRunList() {
  const res = await fetch("/api/runs");
  const names = (await res.json()).runs;
  const select = $("run-config");
  select.innerHTML = "";
  for (const name of names) {
    const option = document.createElement("option");
    option.textContent = name;
    select.appendChild(option);
  }
  if (!names.length) {
    const option = document.createElement("option");
    option.textContent = "no saved runs";
    option.disabled = true;
    select.appendChild(option);
  }
}

$("do-load").addEventListener("click", async () => {
  const config = $("run-config").value;
  if (!config || config === "no saved runs") return;
  const res = await fetch("/api/run?config=" + encodeURIComponent(config));
  const data = await res.json();
  if (!res.ok) { note('<span class="error">' + (data.error || res.statusText) + "</span>"); return; }
  renderRun(data);
});

$("do-live").addEventListener("click", async () => {
  $("replay").hidden = true;
  $("stepper").hidden = true;
  $("live").hidden = false;
  $("do-live").hidden = true;
  $("convo-title").textContent = "Conversation";
  $("recalled").className = "empty";
  $("recalled").textContent = "Nothing recalled yet.";
  const res = await fetch("/api/state?user=" + $("user").value);
  renderState((await res.json()).state);
});

loadRunList();
fetch("/api/state?user=U1").then(r => r.json()).then(d => renderState(d.state));
</script>
</body>
</html>
"""


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"open http://{HOST}:{PORT}")
    print("ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
