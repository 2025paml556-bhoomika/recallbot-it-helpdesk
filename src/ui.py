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
from urllib.parse import urlparse

from src.agent import Session
from src.config import DB_PATH
from src.episodic_memory import EPISODIC_PATH
from src.semantic_memory import SEMANTIC_PATH
from src.db import connect

HOST = "127.0.0.1"
PORT = 8000

# One Session per (user, session). A switch drops the old one, so its slots
# do not leak, which is the same rule the isolation tests check.
_sessions: dict[tuple[str, str], Session] = {}
_lock = threading.Lock()


def get_session(user: str, session: str) -> Session:
    # The page sends the id it minted for this tab. A missing one still has
    # to name something, but a real tab always sends its own.
    key = (user.strip() or "U1", session.strip() or "tab")
    with _lock:
        found = _sessions.get(key)
        if found is None:
            found = Session(*key)
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
    """What is on disk right now, for the side panels."""
    semantic = _read_jsonl(SEMANTIC_PATH)
    episodic = [_clean(row) for row in _read_jsonl(EPISODIC_PATH)]
    return {
        "facts": semantic,
        "episodes": episodic,
        "tickets": _table("tickets"),
        "bookings": _table("slots"),
    }


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


def turn(user: str, session: str, text: str) -> dict:
    """One user message through the real agent. Recall is captured before write."""
    current = get_session(user, session)
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
    }


def forget(user: str, session: str, value: str) -> dict:
    current = get_session(user, session)
    removed = current.memory.forget(value)
    return {"removed": removed, "state": snapshot(user)}


def close_session(user: str, session: str, reflection: str) -> dict:
    current = get_session(user, session)
    written = current.end(reflection.strip() or f"{user} ended session {session}.")
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
                self._send_json(turn(user, session, text))
            elif path == "/api/forget":
                self._send_json(forget(user, session, str(body.get("value") or "")))
            elif path == "/api/end":
                self._send_json(close_session(user, session, str(body.get("reflection") or "")))
            elif path == "/api/reset":
                drop_session(user, session)
                self._send_json({"ok": True, "state": snapshot(user)})
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
  .row { display: flex; gap: 8px; margin-top: 8px; }
  .row input { flex: 1; font: 14px sans-serif; padding: 6px 8px;
               border: 1px solid #d9d0c3; border-radius: 6px; }
  .card { border-top: 1px solid #eee6da; padding: 8px 0; font-size: 14px; }
  .card .tag { font: 11px sans-serif; letter-spacing: .04em; text-transform: uppercase;
               padding: 1px 6px; border-radius: 8px; background: #efe8dc; color: #5c564d; }
  .card .tag.off { background: #f3d6d0; }
  .empty { color: #9a9084; font-style: italic; }
  .error { color: #8d2d22; }
  table { width: 100%; border-collapse: collapse; font: 12px/1.3 ui-monospace, monospace; }
  td { padding: 3px 4px; border-bottom: 1px solid #f0e9de; vertical-align: top; }
</style>
</head>
<body>
<header>
  <h1>RecallBot · helpdesk</h1>
  <label>User
    <select id="user"><option>U1</option><option>U2</option></select>
  </label>
  <span id="session-label"></span>
  <span style="font:12px sans-serif;color:#9a9084">Each tab is its own session. Facts in memory/ carry over; this tab's slots do not.</span>
</header>
<main>
  <section>
    <h2>Conversation</h2>
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

function who() { return { user: $("user").value, session: sessionId }; }

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
  const all = state.tickets.map(r => ["ticket", r.ticket_id, r.owner_user, r.issue_type])
    .concat(state.bookings.map(r => ["booking", r.slot_id, r.owner_user, r.location + " " + r.date]));
  if (!all.length) { rows.innerHTML = '<div class="empty">No rows.</div>'; return; }
  rows.innerHTML = "<table>" + all.map(cells =>
    "<tr>" + cells.map(c => "<td>" + c + "</td>").join("") + "</tr>").join("") + "</table>";
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
      " · turn " + data.turn + " · action " + data.action +
      (data.tool ? " (" + data.tool + ")" : ""));
    renderRecalled(data.recalled, data.slots);
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
