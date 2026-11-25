# PLAN.md vs requirements.pdf

Source of truth: `requirements.pdf` (PCA MZG517 Mini Project 1).
Compared against: `PLAN.md` (RecallBot V06 build plan).

Variant check is fine. V06 is Bhoomika Bathla, corporate IT helpdesk, twist A (multi-user isolation), window k = 4, context window = 4,096 tokens, scale = 100,000 daily users. Tasks 1–6, marks (2+4+6+4+2+2), build order, logging vocabulary, own BM25/RRF, banned libraries, and Session A/B as separate processes all match the PDF.

The items below are places the plan is missing, softer, or inconsistent with the PDF. `PLAN.md` is a build plan, not a restatement of the spec, so checklist dates and “still open until a later step” notes are not gaps.

Gaps 1–8 compare the plan text with the PDF. Gaps 9–16 came from a second pass that also checked the repo (`.gitignore`, `logs/calls.jsonl`, `results/`, `src/`), because some plan items are marked done but the files they produce break a PDF rule.

## Gaps

### 1. Chat model contradicts itself

- PDF: one chat model for every configuration, so comparisons stay fair. Any allowed API or a small local model (Ollama, llama.cpp, Hugging Face) is fine.
- Plan header: “Qwen2.5-7B through Ollama, or Groq's free tier”.
- Plan Step 1: Qwen2.5-3B via Ollama (`qwen2.5:3b`), with tokenizer `Qwen/Qwen2.5-3B`.
- Fix: pick one model and use it for every config (`full`, `trim`, `summary`, `no_rag`, `hybrid`, `hybrid_sem`, `hybrid_mem`). The header and Step 1 must name the same model.
- Repo check: `src/config.py` sets `CHAT_MODEL = "qwen2.5:3b"`, and every chat call in `logs/calls.jsonl` used it. Only the plan header is wrong; change it to Qwen2.5-3B via Ollama.

### 2. Log schema drops three required fields

- PDF: every line in `logs/calls.jsonl` includes `ts`, `run_id`, `config`, `user`, `session`, `turn`, `call_type`, `model`, `input_tokens`, `output_tokens`, `retrieved_tokens`, `latency_ms`.
- Plan Step 1 validate list: run_id, call_type, model, input/output tokens, latency. It does not require `user`, `session`, or `turn`.
- `config` must be one of `full`, `trim`, `summary`, `no_rag`, `hybrid`, `hybrid_sem`, `hybrid_mem`.
- `call_type` must be one of `nlu`, `chat`, `summary`, `extract`, `importance`, `reflect`, `embed`, `tool`, `judge`. Tool calls log 0 tokens and measured latency.
- `retrieved_tokens` is tokens of KB chunks, semantic memories, and episodic memories placed in the prompt (0 when nothing is retrieved).
- Never edit or delete lines. A bad run gets a new `run_id`.

### 3. `hit_rate.csv` grain is wrong

- PDF required columns: `method`, `chunk_size`, `k`, `question_id`, `hit`.
- `method` is `dense`, `bm25`, or `hybrid`. `hit` is 1 or 0. One row per question at each method, chunk size, and k.
- Plan Step 4: aggregated rows only (6 method × size rows, plus code and reworded rows) in `results/hit_rate.csv`.
- Repo check: the current file's header is `method,chunk_size,slice,n,hit_at_1,hit_at_3,hit_at_5`, so it already has the wrong grain. Regenerate it, and keep the aggregates in a separate file if they are needed.
- Aggregates can be computed from the per-question file. The file the zip must contain is the per-question one. Report still needs hit rate@1, @3, and @5 for 3 methods × 2 chunk sizes, with separate code and paraphrase breakdowns, plus one hybrid-fixed and one hybrid-unfixed case with ranked lists.

### 4. LLM-judge check is under-counted

- PDF Task 4(a): if an LLM judge is used, check it against your own ratings on at least 10 replies and report agreement.
- PDF FAQ 4: check the judge by hand on at least 20 judgements and report the agreement rate. `judged_by` is `manual` or `llm` on every `probe_results.csv` row.
- Plan Step 10: “check it against my own ratings on 10–20 replies”.
- These are two separate checks, not one number to pick between:
  - **Rubric ratings (Task 4(a)):** 15 Session B replies × 4 configs, scored 1–5 on coherence, relevance, groundedness, and safety. If an LLM judge gives these ratings, check it against your own ratings on at least 10 replies and report the agreement.
  - **Probe marking (FAQ 4):** if an LLM judge sets `correct` in `probe_results.csv`, check it by hand on at least 20 judgements and report the agreement rate.
- If the judge is used for both, do both checks, each with its own agreement figure computed by a script. Log every judge call as `judge`.

### 5. “Session A probes” is not defined

- PDF Task 2(c): probe accuracy for each short-term strategy (`full`, `trim`, `summary`) on the Session A probes. For each strategy, name one failed probe and the turn where the information was lost or distorted. State the first turn whose facts `trim` can no longer recall.
- Plan Step 6 says “Score the Session A probes” but does not say which probe IDs belong to Session A (turns 1–25 only) versus Session B.
- Repo check: `data/probes.json` already has a `session` field, but only 4 probes are Session A (P03 R, P09 K, P13 S, P14 S). P03 is the only fact-recall probe, and there are no Session A U, F or E probes. That is too few to give one failed probe per strategy and the first turn `trim` (k = 4) can no longer recall.
- Fix: add Session A R and U probes that ask about facts from early turns (well over 4 turns back), and list the Session A probe IDs in Step 6.
- Task 1 still needs at least 14 probes covering R, U, F, E, K, N, S, plus at least 2 twist probes. Twist A: U2 asks about something only U1 said, and the expected answer is that the bot does not know.

### 6. Architecture diagram is narrower than the report requirement

- PDF Task 2(a) and the report structure: one diagram of NLU, state tracker, dialogue manager, tools, retrieval, the four memory types (short-term, semantic, episodic, procedural), and NLG, labelled with real function names.
- Plan Step 5: mermaid of `understand`, `Session.ingest`, `Session.decide`, `Session.reply`, `search_kb`, `book_it_slot`. Retrieval and the four memory types are not required in that item.
- Procedural memory stays versioned files in `skills/`, loaded at run time. The prompt must cover persona, scope, error handling, and output format. Decoding: temperature 0 for task turns; temperature 0.7 and top-p 0.9 for open turns, with reasons.

### 7. Report step is cut off

- PDF: main body at most 8 pages.
  1. Variant, LLM, embedding model and pooling, index type, hardware (half a page).
  2. Architecture diagram (1 page).
  3. Tasks 1–5 in order, with sub-part labels such as 3(b) (about 6 pages).
  4. Limitations and what you would change with one more week (half a page).
  5. Appendix A: AI-use declaration (tools, what for, one specific wrong suggestion, how you found it, what you changed).
  6. Appendix B: prompts (system, RAG, extraction, importance, reflection, summary, judge).
- Plan Step 12 stops at “limitations”. Appendices A and B are missing. Task 6(b) is the declaration; the plan never lists it as its own item.
- `PLAN.md` ends mid-list at “- limitations”. Step 12 also has no video item (Task 6(a), 1.5 marks), no zip-build item, no README reproduction check, no pass through the PDF submission checklist, and no **✅ Validate** line.

### 8. Zip layout is incomplete

- PDF zip name: `<BITS_ID>_V06_MP1.zip`.
- Inner folder: `<BITS_ID>_V06_MP1/` containing `README.md`, `requirements.txt`, `src/`, `skills/`, `tests/`, `data/kb/`, `data/test_conversation.json`, `data/kb_questions.json`, `data/probes.json`, `data/surprise_probes.json`, `db/`, `index/`, `memory/` (after Session B of `hybrid_mem`), `logs/calls.jsonl` (unedited), `results/probe_results.csv`, `results/hit_rate.csv`, `results/plots/`, `report.pdf`, `video_link.txt`.
- Plan names the zip but not the inner folder. `data/surprise_probes.json` appears only under Step 10, not in a deliverable tree. README must give the exact commands to reproduce every run from a fresh environment.

### 9. `.gitignore` excludes required deliverables

- PDF: the zip must contain `logs/calls.jsonl` (untouched), `results/probe_results.csv`, `results/hit_rate.csv`, `results/plots/`, and `memory/` as it stands after Session B of `hybrid_mem`.
- Repo: `.gitignore` ignores `logs/calls.jsonl` (last line), `results/*` and `memory/*`. The last line contradicts the file's own comment, “logs/calls.jsonl is not ignored … Do not add it back.”
- Risk: a zip built from a git checkout or `git archive` would be missing the log and both result files. That triggers G1 and fails the submission checklist.
- Fix: remove the `logs/calls.jsonl` line, and either un-ignore the required `results/` and `memory/` files or build the zip from the working tree. Add a Step 12 check that every deliverable path is present inside the zip.

### 10. Most log lines have no valid `config`

- PDF: `config` is one of `full`, `trim`, `summary`, `no_rag`, `hybrid`, `hybrid_sem`, `hybrid_mem`. Every configuration in the report must have log lines.
- Repo: 8,434 of 9,837 lines in `logs/calls.jsonl` have `config: null` (development, tests, index builds).
- The log cannot be edited, so the fix has to be a rule going forward: every call made during a reported run sets `config`. The report must say which `run_id`s it uses, and that lines with a null `config` are build or test lines it does not use.

### 11. One `run_id` covers most of the log

- PDF: “change for every run”. A bad run gets a new `run_id`, and the report says which runs it used.
- Repo: `r01` covers 9,527 of 9,837 lines, across many separate runs. Plan Step 1 treats `RUN_ID` as a fixed value in `src/config.py`.
- Fix: give every reported run its own `run_id`, set per run (by CLI flag or generated), and keep a run list for the report: run_id → config, session, date.

### 12. Old `call_type` names in the log and in the plan

- PDF: `call_type` is one of `nlu`, `chat`, `summary`, `extract`, `importance`, `reflect`, `embed`, `tool`, `judge`.
- Repo: the log already contains `embedding` (3,677 lines), `reply` (224) and `summarise` (7). The code now refuses these names (`src/logger.py`), and `src/plots.py` maps them to `embed`, `chat` and `summary`.
- Plan Step 6's Validate line still says “`summary` totals include the `summarise` calls”. Change it to `summary`.
- The report must disclose the mapping, or the reported runs must contain only the allowed names. Otherwise an evaluator's recount will not match the report (G1).

### 13. Step 9's separate-process check cannot pass as written

- PDF: Session A and Session B are separate processes (gate G3). Anything known in Session B must come from disk.
- Plan Step 9 Validate: “Sessions A and B show different process IDs in the log.” No log line records a process ID.
- Fix: add a `pid` field to every log line from now on (extra fields are already used, so this is fine), or change the check to one that can be run: separate `python -m src.run --session A/B` commands, Session B's first line timestamped after Session A's last, and Session B memory read only from `memory/`.

### 14. The 4,096-token window is not enforced at the model

- PDF: context window 4,096 tokens (V06). The budget must fit it with room for the reply.
- Repo: `src/llm.py` sends only `temperature` and `top_p` to Ollama, with no `num_ctx`. The model then runs with Ollama's default window, not the variant's.
- Fix: pass `num_ctx = 4096` on every chat call in Step 9, and assert in the context builder that prompt + reply reserve ≤ 4,096.

### 15. Smaller PDF items the plan omits

- **Reflections each session:** Table 3 says reflections are written at the end of *each* session. Plan Step 8 only writes one after Session A. Also write one after Session B (it ends up in `memory/`).
- **Memory-type contribution:** the 4(a) full-marks rubric needs a discussion of what each memory type adds, not just the ablation table. Add it to Step 10.
- **Same settings in Task 2:** all three strategies must use the same system prompt, model and decoding settings. Step 6 says the state tracker stays on but not this.
- **Few-shot prefix in 5(a):** the PDF defines the cacheable prefix as the system prompt plus few-shot examples. The repo has no few-shot examples. Either say so in 5(a) and measure the system prompt alone, or add examples before the minimum-length check.
- **Surprise-probe timing:** the PDF says to run them as soon as they are released (24/11), before the ablation. Step 10 lists them last, and that date has already passed, so do them next.

### 16. The plan's Validate lines name test files that do not exist

- Plan Steps 5, 7 and 8 refer to `tests/test_pipeline.py`, `tests/test_semantic.py` and `tests/test_episodic.py`. The repo has `tests/test_agent.py` and `tests/test_memory.py` instead.
- This is not a PDF rule, but those Validate lines cannot be run as written. Point them at the real files, or create the named ones.

## Do not treat as optional

These are in the PDF and only partly reflected in the plan. They are restated here so a later step cannot drop them.

- `results/probe_results.csv` columns: `run_id`, `config`, `probe_id`, `category`, `expected`, `answer`, `correct` (1 or 0), `judged_by` (`manual` or `llm`).
- Four ablation configs, each adding one memory type: `no_rag` (short-term only), `hybrid` (plus KB), `hybrid_sem` (plus semantic), `hybrid_mem` (plus episodic). Sessions A and B are separate processes. Ablation numbers (accuracy per probe category, mean input tokens per turn, median latency, total tokens) must be recomputable from `logs/calls.jsonl`.
- Task 5(a): monthly cost of `trim` and `hybrid_mem`, with and without prompt caching of the fixed prefix, from measured tokens per turn, for 100,000 users/day × the turn count of `test_conversation.json`. Name the CS3 price-table row. Say whether the prefix meets the minimum cacheable length, and what would change if it does not.
- Task 5(b): one-year vector count (KB chunks + all users’ semantic and episodic memories) and memory for flat float32 vs HNSW with M = 16, using the CS3 per-vector formula. Measure flat vs HNSW latency on at least 100,000 vectors, and HNSW recall against flat on `kb_questions`. Recommend an index with M and `ef_search`.
- Surprise probes: five probes published on the LMS on 24/11/2026. Adapt them, save `data/surprise_probes.json`, run on `hybrid_mem`, report raw results. If the system changes afterwards, report before and after separately.
- Video (≤ 5 minutes, own voice): Session A with a confirmed booking, a process restart, Session B recalling a semantic fact and an episode, a forget honoured after restart, one KB answer with its cited chunk, one failure, and a walkthrough of RRF or the episodic score.
- Integrity gates: G1 (missing or edited log / `hit_rate.csv`), G2 (banned library does core retrieval, fusion, or memory), G3 (Session B not a new process), G4 (wrong variant), G5 (cannot explain a task in the viva).
