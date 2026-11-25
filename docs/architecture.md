# Dialogue pipeline

```mermaid
flowchart TD
    userMsg[User message] --> ingest[Session.ingest]
    ingest --> understand[understand]
    understand --> merge[Session._merge]
    merge --> record[Memory.record_turn]
    record --> decide[Session.decide]
    decide -->|slot missing| ask[Session._ask]
    decide -->|knowledge question| search[search_kb]
    decide -->|own ticket| ticket[get_ticket]
    decide -->|yes, slots full| book[book_it_slot]
    decide -->|nothing to do| none[ACTION_NONE]
    ask --> reply[Session.reply]
    search --> reply
    ticket --> reply
    book --> reply
    none --> reply
    reply -->|ask| fixed[return the question, no model call]
    reply -->|tool result| task[chat at temperature 0]
    reply -->|open turn| open[chat at temperature 0.7, top-p 0.9]
    task --> skills[load_skill reads skills/]
    open --> skills
    book --> db[(helpdesk.sqlite)]
    record --> mem[(memory/semantic.jsonl and episodic.jsonl)]
```

`understand` labels the turn. `Session.decide` either asks for the next missing slot or calls one tool, and `book_it_slot` runs only after the user confirms. `Session.reply` reads `skills/system_prompt_v1.md` from disk, and adds `skills/book_slot_steps_v1.md` only when the turn is about a visit.
