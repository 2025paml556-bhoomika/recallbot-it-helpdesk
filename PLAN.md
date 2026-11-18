# RecallBot V06: Implementation Plan

**Student:** Bhoomika Bathla · **Variant:** V06
**Domain:** Corporate IT helpdesk · **Twist:** A. Multi-user isolation
**Window k:** 4 turns · **Context window:** 4,096 tokens · **Scale:** 100,000 daily users
**Deadline:** 28/11/2026, 11:59 pm (LMS) · **Zip:** `<BITS_ID>_V06_MP1.zip`

**Proposed stack:**
- Python 3.10+
- Chat model: Qwen2.5-7B through Ollama, or Groq's free tier
- Tokenizer: the matching Hugging Face `AutoTokenizer`
- Embeddings: `all-MiniLM-L6-v2` (384 dimensions, mean pooling)
- Index: FAISS
- Database: SQLite
- Tests: pytest

---

## Build order at a glance
Sections are listed in the order to **build** them, following the PDF's "Suggested plan". Each step needs the ones before it. The report still follows the Task 1 → 6 numbering.

| Step | Section | Rubric task | Depends on |
|---|---|---|---|
| 1 | Project setup | Supports all tasks | – |
| 2 | Knowledge base | Task 1 | 1 |
| 3 | Test data and probes | Task 1 | 2 |
| 4 | Retrieval | Task 3(a), 3(b) | 2, 3 |
| 5 | Dialogue pipeline | Task 2(a) | 4 |
| 6 | Short-term memory | Task 2(b), 2(c) | 5 |
| 7 | Semantic memory | Task 3(c) | 5 |
| 8 | Episodic memory | Task 3(d) | 7 |
| 9 | Context budget and twist | Task 3(e), 3(f) | 6, 7, 8 |
| 10 | Evaluation | Task 4 | 9 |
| 11 | Cost and scale | Task 5 | 10 |
| 12 | Report, video, submission | Task 6 | All |

**Validation rule:** each step ends with a **✅ Validate** line. Before reporting a step as done, Cursor runs those checks itself and shows the command output as evidence. If any check fails, the step is not complete. Anything only you can do (your own ratings, recording the video) is marked **(manual)**, and Cursor checks only that the file exists and has the right format.

---

## 1. Project setup (18 Nov) · supports all tasks
- [ ] Create folders: `src/`, `skills/`, `tests/`, `data/kb/`, `db/`, `index/`, `memory/`, `logs/`, `results/plots/`
- [ ] Fill in `requirements.txt` (sentence-transformers, faiss-cpu, numpy, pandas, matplotlib, transformers, pytest, requests, ollama/groq)
- [ ] Choose the chat model and tokenizer, and use the same model for every configuration
- [ ] Write `src/config.py`: model names, k=4, CONTEXT_WINDOW=4096, file paths, RUN_ID
- [ ] Write `src/logger.py` with `log_call()`, which appends to `logs/calls.jsonl` and never edits it
- [ ] Write `src/llm.py`: one wrapper for chat calls, with tokens taken from the API's usage field and latency measured
- [ ] Start `README.md` with setup and run commands
- [ ] Set up `.gitignore` and commit

**✅ Validate:** `pip install -r requirements.txt` succeeds, all folders exist, one test chat call through `src/llm.py` returns a reply, and `logs/calls.jsonl` gains exactly one line with all the required fields (run_id, call_type, model, input/output tokens, latency).

## 2. Knowledge base · Task 1 · depends on 1
- [ ] Write 15+ IT documents in `data/kb/`, 4,000–8,000 words in total. Topics:
  - VPN, password and MFA policy, laptop refresh, software requests
  - email, Wi-Fi, printers, onboarding and offboarding
  - backup, acceptable use, incident severity, SLA, known outages, hardware catalogue
- [ ] Include 10+ codes or IDs (`ERR-VPN-809`, `SLA-P1`, `SW-REQ-112`, `LAP-T14G3`, ...)
- [ ] Write 2+ old/new document pairs (Password Policy 2025 vs 2026, Laptop Refresh 2025 vs 2026), with `version`, `valid_from` and `status` in each file's header
- [ ] Check that the documents don't contradict each other
- [ ] Count the words with a script and record the total

**✅ Validate:** `python -m src.kb_stats` reports 15+ files, a total of 4,000–8,000 words, 10+ unique codes (found by regex) and 2+ `doc_id`s with both a 2025 and a 2026 version; every file header has `version`, `valid_from` and `status`.

## 3. Test data and probes · Task 1 (2 marks) · depends on 2
- [ ] `data/test_conversation.json`: 40+ turns, Session A = turns 1–25, Session B = 26+, with turns from both **U1 and U2**
  - [ ] 8+ user facts, each tagged with its turn (OS, laptop, office, department, manager, contact time, VPN client, phone)
  - [ ] 2+ updates (e.g. Pune → Hyderabad office) and 1+ forget request (e.g. personal phone number)
  - [ ] 3+ notable events (ticket raised, complaint, upgrade decision) and 5+ small-talk turns
  - [ ] 1 complete slot-filling task that ends with a confirmed database write
  - [ ] 2+ follow-ups that depend on the conversation state ("And for Friday?") and 1+ turn that needs a clarifying question
  - [ ] 5+ knowledge-base questions spread across both sessions
- [ ] `data/kb_questions.json`: 25+ questions, each with its gold answer span and `doc_id`
  - [ ] 5+ containing a code, 5+ reworded with no keywords shared with the answer, 2+ answered only by the newer version
- [ ] `data/probes.json`: 14+ probes covering all 7 types (recalled fact, updated fact, forgotten fact, past event, document answer, never said, follow-up)
  - [ ] 2+ **isolation probes**: U2 asks about something only U1 said, and the expected answer is "I don't know"

**✅ Validate:** `python -m src.validate_data` checks every minimum count above (turns, facts, updates, forgets, events, small talk, follow-ups, users U1 and U2, the 25+ questions with their subtypes, 14+ probes across all 7 types, 2+ isolation probes), and confirms that each gold answer span appears word for word in its `doc_id` file.

## 4. Retrieval: chunking, embeddings, BM25, RRF · Task 3(a), 3(b) (2.5 marks) · depends on 2, 3
- [ ] `src/chunker.py`: split by token count at 2 sizes (128 and 512) with about 15% overlap, using the chat model's tokenizer
- [ ] Store metadata with each chunk: `doc_id`, `section`, `version`, `valid_from`, `status`
- [ ] `src/embedder.py`: MiniLM with mean pooling and normalised vectors; log every embedding call
- [ ] `src/vector_index.py`: a flat FAISS index (justify it: a small corpus needs exact search), with metadata saved to `index/`
- [ ] `src/bm25.py`: my own BM25 (k1, b, IDF)
- [ ] `src/rrf.py`: my own RRF with k=60
- [ ] `tests/test_bm25.py`: reproduce the CS4 example score of **4.160**
- [ ] `tests/test_rrf.py`: reproduce the CS4 ranking A, B, C, D, then E and F tied
- [ ] `src/search_kb.py`: dense, BM25 or hybrid search, chosen by a parameter
- [ ] `src/eval_retrieval.py`: write `results/hit_rate.csv`
  - [ ] Hit rate in the top 1, 3 and 5, for 3 methods × 2 chunk sizes
  - [ ] Separate rows for code questions and reworded questions
- [ ] Choose 1 question hybrid fixed and 1 it didn't, and save both ranked lists
- [ ] `python -m pytest tests/` passes

**✅ Validate:** `pytest tests/test_bm25.py tests/test_rrf.py` passes (4.160 and A, B, C, D, then E and F tied); `grep` finds no `rank_bm25`, LangChain or LlamaIndex imports in `src/`; chunk overlap measures 10–20%; `results/hit_rate.csv` has 6 method × size rows plus the code and reworded rows, with hit rate never falling as the cut-off grows (top 1 ≤ top 3 ≤ top 5); `logs/calls.jsonl` contains embedding calls.

## 5. Dialogue pipeline and procedural memory · Task 2(a) (1.5 marks) · depends on 4
- [ ] `db/helpdesk.sqlite`: tables `tickets`, `slots`, `assets`, each with an `owner_user` column
- [ ] `src/tools.py`:
  - `search_kb(query)`
  - `get_ticket(ticket_id, user)` checks the owner, for isolation
  - `book_it_slot(...)` runs only after the user confirms
  - Every tool call is logged with 0 tokens
- [ ] `src/nlu.py`: hybrid intent detection (rules first, then an LLM returning JSON). Entities: ticket ID, asset ID, error codes, date, location
- [ ] `data/nlu_labelled.json`: 20+ labelled messages, plus a script that measures accuracy
- [ ] `src/state_tracker.py`: fields `user`, `intent`, `slots{asset_id, issue_type, location, date, time}`, `pending_confirmation`, `last_entities`, `turn`
- [ ] `src/dialogue_manager.py`: the main loop. It fills missing slots, asks a clarifying question when needed, and confirms before any write
- [ ] `src/nlg.py`: fixed templates for confirmations and slot questions, the LLM for open answers
- [ ] `skills/system_prompt_v1.md` (persona, scope, error handling, output format) and `skills/book_slot_steps_v1.md`, both loaded at run time
- [ ] Decoding settings: temperature 0 for task turns; temperature 0.7 and top-p 0.9 for open turns. Write down why
- [ ] Draw the architecture diagram (mermaid), labelled with real function names

**✅ Validate:** `pytest tests/test_pipeline.py` passes the scripted booking flow: the slot questions are asked, there is **no** database row before "yes" and exactly one row after, `get_ticket` as U2 on U1's ticket returns "not found", and the NLU script reports its accuracy on 20+ labelled messages; the log shows `tool` calls with 0 tokens, and the skill files are loaded from disk.

## 6. Short-term memory strategies · Task 2(b), 2(c) (2.5 marks) · depends on 5
- [ ] `src/history.py` with three strategies:
  - `full`: whole history
  - `trim`: last k=4 turns
  - `summary`: a rolling summary written by an LLM call, plus the last 2 turns
- [ ] Run Session A once per strategy, each with its own run_id, keeping the state tracker on
- [ ] `src/plots.py`: plot input tokens per turn for the 3 strategies from `calls.jsonl`, with totals that include the summary calls
- [ ] Score the Session A probes per strategy and write `results/probe_results.csv`
- [ ] For each strategy, record 1 failed probe and the turn where the information was lost
- [ ] Record the first turn whose facts trim can no longer recall
- [ ] For each follow-up and clarification, record the state or question that resolved it
- [ ] Show log evidence that the write ran only after confirmation

**✅ Validate:** the log has 3 distinct run_ids (one per strategy); the plot PNG exists and its totals match a recount from `calls.jsonl`; `full` input tokens grow every turn while `trim` levels off; `summary` totals include the `summarise` calls; `probe_results.csv` has one row per probe per strategy; in the log, the database-write timestamp comes after the confirmation turn.

## 7. Semantic memory · Task 3(c) (1 mark) · depends on 5
- [ ] `src/semantic_memory.py`: records `{user, key, value, source_turn, date, status}` in `memory/semantic.jsonl`, plus a vector index
- [ ] An LLM extraction step, logged as `extract`. Small-talk turns produce no record
- [ ] Update: the old record is marked `superseded`, so only `active` records are retrieved
- [ ] Forget: delete from the JSONL file **and** the vector index
- [ ] Measure extraction precision and recall against the facts planted in Task 1
- [ ] **Filter by `user` on every read**, for the twist

**✅ Validate:** `pytest tests/test_semantic.py` passes: an update leaves exactly 1 `active` and 1 `superseded` record; after a forget and a **fresh process**, the fact is in neither the JSONL file nor the vector index; small talk adds 0 records; a U2 query never returns U1's records; extraction precision and recall are printed from a script, not entered by hand.

## 8. Episodic memory and reflection · Task 3(d) (1 mark) · depends on 7
- [ ] `src/episodic_memory.py`: a stream in `memory/episodic.jsonl` with `{user, ts, text, importance 1–10, type}`
- [ ] Importance from a rule or an LLM (logged as `importance`). Explain the choice
- [ ] Score = α·recency (exponential decay, e.g. 0.995 per hour) + β·importance + γ·cosine similarity, written myself
- [ ] Compare 2 weightings on the past-event probes and justify the one I keep
- [ ] Write a reflection at the end of Session A (logged as `reflect`) and show Session B using it
- [ ] Forget also removes or redacts matching episodes and reflections. Verify after a restart
- [ ] Filter by `user` on every read

**✅ Validate:** `pytest tests/test_episodic.py` passes: a hand-calculated score for one episode matches `score()`; the two weightings give different top-3 results on at least 1 probe; a `reflect` line exists in the log, and the reflection's ID appears in a Session B prompt; after a forget and a restart, no episode or reflection contains the forgotten value; U1's episodes are never returned for U2.

## 9. Context engineering and twist · Task 3(e), 3(f) (1.5 marks) · depends on 6, 7, 8
- [ ] `src/context_builder.py`: a token budget for 4,096 tokens
  - system prompt → tools → long-term memory → history → document chunks → current task → about 500 tokens kept for the reply
- [ ] Context-assembly table for one Session B turn: tokens, memory type, and whether each part lives in the prompt or an external store
- [ ] Grounding rules, cited IDs (`[doc_id#chunk]`, `[mem:id]`) and the exact fallback sentence when nothing relevant is found
- [ ] Twist A: user filter in `semantic_memory`, `episodic_memory` and `get_ticket`; the knowledge base stays shared
- [ ] Run the 2+ isolation probes (pass or fail) and explain the results
- [ ] Session A and Session B run as **separate processes** (`python -m src.run --session A` / `--session B`)

**✅ Validate:** the assembled prompt never goes over 4,096 − 500 tokens (asserted on every turn of one run); the budget table's numbers match a recount using the tokenizer; a deliberately unanswerable question returns the exact fallback sentence; replies contain citations that point to IDs which actually exist; the isolation probes are run and their pass/fail is written to CSV; Sessions A and B show different process IDs in the log.

## 10. Evaluation and error analysis · Task 4 (4 marks) · depends on 9
- [ ] `src/run.py --config {no_rag, hybrid, hybrid_sem, hybrid_mem}`, running Sessions A and B for each
- [ ] Ablation table: accuracy per probe type, mean input tokens per turn, median latency, total tokens, all computed by `src/analyze_logs.py`
- [ ] Rate 15 Session B replies per configuration (1–5 on coherence, relevance, groundedness, safety)
- [ ] LLM judge: check it against my own ratings on 10–20 replies and report the agreement rate
- [ ] Error analysis: 3 failed `hybrid_mem` probes, each traced to its stage with evidence (ranked list, prompt excerpt, state) and a fix at that stage
- [ ] **24 Nov:** adapt the 5 surprise probes into `data/surprise_probes.json`, run them on `hybrid_mem`, and report raw results, plus before/after if I change the system

**✅ Validate:** the log has run_ids for all 4 configs × 2 sessions; running `analyze_logs.py` again reproduces the ablation table exactly; the ratings CSV has 15 rows × 4 configs **(manual: your ratings)**; the judge agreement figure is computed by a script; each of the 3 error traces links to a real log line or ranked list; surprise-probe results exist with their own run_id.

## 11. Cost and scale · Task 5 (2 marks) · depends on 10
- [ ] Monthly cost for 100,000 users/day × the turns in my test conversation, for `trim` and `hybrid_mem`, with and without prompt caching, using measured tokens per turn and a named CS3 price row
- [ ] Check whether the fixed prefix reaches the minimum cacheable length, and what to change if not
- [ ] Vector count after 1 year (chunks + 100,000 users' facts and episodes) and memory for flat float32 vs HNSW with M=16, using the CS3 formula
- [ ] `src/bench_index.py`: flat vs HNSW latency on 100,000+ vectors (padded with random vectors), and HNSW recall against flat on kb_questions
- [ ] Recommend an index with its M and ef_search

**✅ Validate:** `python -m src.cost` reads the tokens per turn from `calls.jsonl` (not hard-coded), and recalculating it by hand for one row matches; the memory formula gives the same figure in code and report; `bench_index.py` finishes on 100,000+ vectors and writes its latency and recall to CSV; HNSW recall is ≤ 1.0, and flat recall equals 1.0.

## 12. Report, video and submission · Task 6 (2 marks) · depends on all
- [ ] `report.pdf`, at most 8 pages:
  - variant and setup
  - diagram
  - Tasks 1–5 with sub-part labels
  - limitations
  - Appendix A: AI-use declaration
  - Appendix B: all prompts
- [ ] AI-use declaration with one specific wrong AI suggestion and how I fixed it
- [ ] Video, at most 5 minutes, in my own voice, showing:
  - a confirmed booking
  - a restart
  - Session B recalling a fact and a past event
  - a forget request honoured after restart
  - a cited document answer
  - one failure
  - a walkthrough of my RRF or episodic scoring code
- [ ] `video_link.txt` (viewable by anyone with the link)
- [ ] README reproduces every run from a fresh environment
- [ ] Final check: tests pass, logs unedited, CSVs match the report, `memory/` reflects the state after `hybrid_mem` Session B
- [ ] Zip as `<BITS_ID>_V06_MP1.zip` and upload to the LMS before **28/11/2026, 11:59 pm**

**✅ Validate:** in a fresh virtual environment, following the README steps reruns the tests and the main runs without errors; the PDF has ≤ 8 pages; every number in the report matches a CSV or the log; `video_link.txt` contains a URL **(manual: video viewable, ≤ 5 minutes, your voice)**; the zip has the right name and contains `src/`, `tests/`, `data/`, `logs/`, `results/`, `memory/`, `skills/`, `report.pdf`, `README.md`, and no `.venv` or secrets.

---

## Timeline (catch-up, starting 18 Nov)
| Dates | Sections |
|---|---|
| 18–19 Nov | 1, 2, 3 |
| 19–20 Nov | 4 |
| 20–22 Nov | 5, 6 |
| 22–24 Nov | 7, 8, 9 |
| 24–26 Nov | 10 (surprise probes on 24 Nov) |
| 26–28 Nov | 11, 12 |
