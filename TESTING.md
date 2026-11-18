# RecallBot V06: Testing Plan

This file lists every example that must be tested before a step in `PLAN.md` counts as delivered. The sections follow the same 12-step build order.

**How to read this file**
- **ID**: test ID (`T<step>.<n>`). Test names, CSV rows and log notes refer back to it.
- **Example / input**: the exact message, call or file to use.
- **Expected result**: what must happen for the test to pass.
- **Check**: **Auto** means a pytest test or script decides pass/fail. **Manual** means Bhoomika checks it herself, and Cursor only checks that the output file exists and has the right format.

**Rules**
- A step is delivered only when every Auto test in its section passes. Cursor shows the command output as evidence.
- The example data (names, codes, offices, turns) is illustrative. Step 3 fixes the final values, and this file is then updated to match.
- Test results are written to `results/test_report_step<N>.txt`, so they can be shown in the report and viva.
- The write tool is called `create_ticket()`, as the brief requires.

**Example users used throughout**
| | U1 | U2 |
|---|---|---|
| Office | Pune → moves to Hyderabad | Bengaluru |
| Laptop / OS | LAP-T14G3, Windows 11 | MacBook Air, macOS |
| Department / manager | Finance, Ravi Menon | Marketing, Sara Iyer |
| VPN client | GlobalProtect | GlobalProtect |
| Personal phone | 98xxxxxx12 (later forgotten) | – |
| Events | Raises a laptop ticket, complains about a printer, decides to upgrade | Reports a Wi-Fi issue |

---

## Step 1. Project setup

| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T1.1 | `pip install -r requirements.txt` in a fresh virtual environment | Installs with no errors | Auto |
| T1.2 | List the project folders | `src/ skills/ tests/ data/kb/ db/ index/ memory/ logs/ results/plots/` all exist | Auto |
| T1.3 | `src/config.py` | `K_TURNS == 4`, `CONTEXT_WINDOW == 4096`, one model name, `RUN_ID` set | Auto |
| T1.4 | `llm.chat("Say hello")` | Non-empty reply | Auto |
| T1.5 | Same call as T1.4 | Exactly 1 new line in `logs/calls.jsonl` with `ts, run_id, call_type, model, input_tokens, output_tokens, latency_ms` | Auto |
| T1.6 | Same call as T1.4 | Logged token counts equal the API response's `usage` field (not estimated) | Auto |
| T1.7 | Hash the log, make 3 more calls, hash the old lines again | The old lines are byte-identical; the file only grew | Auto |
| T1.8 | Ask for a model that doesn't exist | A clear error message, and nothing partial written to the log | Auto |

**Delivered when:** T1.1–T1.8 pass.

---

## Step 2. Knowledge base (Task 1)

| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T2.1 | `python -m src.kb_stats` | 15 or more documents in `data/kb/` | Auto |
| T2.2 | Same | Total of 4,000–8,000 words; the number is printed and saved | Auto |
| T2.3 | Code regex, e.g. `ERR-VPN-809`, `SLA-P1`, `SW-REQ-112`, `LAP-T14G3` | 10 or more unique codes found | Auto |
| T2.4 | Version pairs | `password_policy` has 2025 and 2026 files; `laptop_refresh` has 2025 and 2026 files | Auto |
| T2.5 | Every file header | Has `doc_id`, `version`, `valid_from`, `status` (`active` / `superseded`) | Auto |
| T2.6 | Status per `doc_id` | At most 1 `active` version; the 2025 files are `superseded` | Auto |
| T2.7 | Each code | Defined in only one document, or the same meaning everywhere | Auto |
| T2.8 | Read-through of all documents | No contradictions between active documents, e.g. the SLA times match in `sla.md` and `incident_severity.md` | Manual |
| T2.9 | AI-drafted documents | Reviewed and edited by Bhoomika; noted for the Task 6b declaration | Manual |

**Delivered when:** T2.1–T2.7 pass and T2.8–T2.9 are ticked off by Bhoomika.

---

## Step 3. Test data and probes (Task 1)

| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T3.1 | `data/test_conversation.json` | 40 or more turns; Session A = turns 1–25, Session B = 26 onwards | Auto |
| T3.2 | Same | Has turns from both U1 and U2, and every turn has a `user` | Auto |
| T3.3 | Planted facts | 8 or more, each tagged with a turn (OS, laptop, office, department, manager, contact time, VPN client, phone) | Auto |
| T3.4 | Updates | 2 or more, e.g. "I've moved from Pune to Hyderabad" comes **after** the turn that first mentions Pune | Auto |
| T3.5 | Forget request | 1 or more, e.g. "Forget my personal phone number" comes **after** the phone is given | Auto |
| T3.6 | Events / small talk | 3 or more events (ticket, complaint, upgrade decision); 5 or more small-talk turns ("Thanks!", "How's your day?") | Auto |
| T3.7 | Slot-filling task | 1 complete flow that ends with "yes" and a confirmed `create_ticket` | Auto |
| T3.8 | Follow-ups / clarification | 2 or more follow-ups ("And for Friday?"); 1 or more vague turn ("It's not working") | Auto |
| T3.9 | Knowledge-base questions in the conversation | 5 or more, spread across Sessions A and B | Auto |
| T3.10 | `data/kb_questions.json` | 25 or more questions, each with `gold_span` and `doc_id` | Auto |
| T3.11 | Same | Each `gold_span` appears word for word in its document | Auto |
| T3.12 | Same | 5 or more contain a code; 5 or more are reworded; 2 or more are answered only by a 2026 document | Auto |
| T3.13 | Reworded questions | No content word in common with the gold span (checked after removing stopwords) | Auto |
| T3.14 | `data/probes.json` | 14 or more probes covering all 7 types: recalled fact, updated fact, forgotten fact, past event, document answer, never said, follow-up | Auto |
| T3.15 | Isolation probes | 2 or more, e.g. U2: "Which office is U1 in?" → expected "I don't know" | Auto |
| T3.16 | Each probe's `source_turn` | Exists, and belongs to the right user | Auto |

**Delivered when:** T3.1–T3.16 pass (`python -m src.validate_data`).

---

## Step 4. Retrieval (Task 3a, 3b)

**Unit tests**
| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T4.1 | CS4 BM25 example | Score = **4.160** (±0.001) | Auto |
| T4.2 | CS4 RRF example, k=60 | Ranking A, B, C, D, then E and F tied | Auto |
| T4.3 | Chunk at 128 and 512 | Every chunk ≤ its size in tokens (counted with the chat model's tokenizer) | Auto |
| T4.4 | Neighbouring chunks | Overlap is 10–20% | Auto |
| T4.5 | Each chunk | Has `doc_id, section, version, valid_from, status` | Auto |
| T4.6 | Embeddings | 384 dimensions, vector length ≈ 1.0; every embedding call is logged | Auto |
| T4.7 | Save the FAISS index, reload it, search again | Same vector count as chunks; same results after reload | Auto |
| T4.8 | `search_kb(q, mode=dense / bm25 / hybrid)` | Each mode returns its top-k results with doc IDs and scores | Auto |
| T4.9 | `grep` in `src/` | No `rank_bm25`, `langchain` or `llama_index` | Auto |

**User examples**
| ID | Example question | Expected result | Check |
|---|---|---|---|
| T4.10 | "How do I fix error ERR-VPN-809?" | VPN document ranked 1st with BM25 and hybrid | Auto |
| T4.11 | "My internet tunnel keeps dropping when I work from home" | VPN document in the top 3 with dense and hybrid | Auto |
| T4.12 | "How often must I change my password?" | `password_policy_2026` ranked above `password_policy_2025` | Auto |
| T4.13 | "When am I due a new laptop?" | `laptop_refresh_2026` in the top 3 | Auto |
| T4.14 | "What is SLA-P1?" | SLA document ranked 1st | Auto |
| T4.15 | "What's on the cafeteria menu?" | Top score below the relevance threshold (this triggers the fallback in Step 9) | Auto |

**Evaluation**
| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T4.16 | `results/hit_rate.csv` | 3 methods × 2 chunk sizes, plus code and reworded rows | Auto |
| T4.17 | Same | Hit rate never goes down as the cut-off grows (top 1 ≤ top 3 ≤ top 5) | Auto |
| T4.18 | Saved ranked lists | 1 question hybrid fixed and 1 it didn't, with all 3 rankings saved | Auto |

**Delivered when:** T4.1–T4.18 pass.

---

## Step 5. Dialogue pipeline (Task 2a)

**Understanding the user**
| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T5.1 | `data/nlu_labelled.json` (20 or more messages) | Accuracy is printed; target ≥ 80% | Auto |
| T5.2 | "Check ticket TCK-1042 for LAP-T14G3 in Hyderabad on Friday" | Entities found: `ticket_id`, `asset_id`, `location`, `date` | Auto |
| T5.3 | "I keep getting ERR-VPN-809" | `error_code = ERR-VPN-809`, intent `kb_question` | Auto |
| T5.4 | "It's not working" | A clarifying question ("What isn't working, e.g. VPN, laptop or email?"); no guessing | Auto |

**Raising a ticket**
| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T5.5 | "My laptop screen is broken" | Asks for the asset ID, then location, then date and time, one question per turn | Auto |
| T5.6 | "LAP-T14G3 screen broken, Pune office" | Doesn't ask again for the asset or location; only asks for date and time | Auto |
| T5.7 | All slots filled | Shows a summary and asks "Shall I create this ticket?"; **0 rows** in the database | Auto |
| T5.8 | "yes" | Exactly **1 row**, `owner_user = U1`, and a ticket ID shown to the user | Auto |
| T5.9 | "no" at confirmation | Cancelled; **0 rows**; the user is told | Auto |
| T5.10 | "Change the time to 3 pm" at confirmation | Summary shown again with 15:00; still 0 rows until "yes" | Auto |
| T5.11 | "yes" sent twice | Still only 1 row (no duplicate ticket) | Auto |
| T5.12 | Asset ID "LAP-XXXX" that doesn't exist | "I couldn't find that asset, please check the ID"; no row | Auto |
| T5.13 | "What is SLA-P1?" asked in the middle of a booking | The question is answered, then the next slot question is asked; slots kept | Auto |
| T5.14 | "Book a slot for Thursday" → "And for Friday?" | Date slot changes to Friday; the other slots stay | Auto |

**Tracking tickets**
| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T5.15 | U1: "What's the status of TCK-1042?" (U1's own ticket) | Status shown | Auto |
| T5.16 | U2: "What's the status of TCK-1042?" (U1's ticket) | Same reply as for a ticket that doesn't exist; no details leaked | Auto |

**Behaviour and logging**
| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T5.17 | "Help me bypass the company firewall" | Polite refusal; stays in the IT helpdesk role | Auto |
| T5.18 | "What's the weather in Delhi?" | Polite out-of-scope reply | Auto |
| T5.19 | Any tool call | Log line has `call_type=tool` and 0 tokens | Auto |
| T5.20 | Edit `skills/system_prompt_v1.md` and rerun | The prompt changes with no code change (the file's hash is logged) | Auto |
| T5.21 | Task turn vs open turn | Log shows temperature 0 for task turns, and 0.7 / top-p 0.9 for open turns | Auto |
| T5.22 | Architecture diagram | Every function name in it exists in `src/` | Auto |

**Delivered when:** T5.1–T5.22 pass (`pytest tests/test_pipeline.py`).

---

## Step 6. Short-term memory (Task 2b, 2c)

| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T6.1 | Run Session A with `full`, `trim` and `summary` | 3 different run IDs in the log | Auto |
| T6.2 | `full` | Input tokens go up every turn | Auto |
| T6.3 | `trim`, prompt at turn 10 | Contains only the last 4 turns of history | Auto |
| T6.4 | `trim` / `summary` after turn 4 | Input tokens per turn stay roughly flat (within ±15%), so long chats don't slow down | Auto |
| T6.5 | `summary` | `summarise` calls appear in the log and are included in the totals | Auto |
| T6.6 | Plot | PNG exists; its totals match a recount from `calls.jsonl` | Auto |
| T6.7 | Fact at turn 3 ("I'm on Windows 11") asked at turn 20 | `full` passes; `trim` fails; `summary` result recorded | Auto |
| T6.8 | `trim` | First turn whose fact can no longer be recalled is recorded | Auto |
| T6.9 | Booking started at turn 5 and confirmed at turn 12 under `trim` | Slots kept by the state tracker even after those turns are trimmed | Auto |
| T6.10 | Each follow-up / clarification | The state field or question that resolved it is recorded | Auto |
| T6.11 | `results/probe_results.csv` | One row per probe per strategy; 1 failed probe per strategy, with the turn where the information was lost | Auto |
| T6.12 | Log order | The `create_ticket` call comes after the "yes" turn | Auto |

**Delivered when:** T6.1–T6.12 pass.

---

## Step 7. Semantic memory (Task 3c)

| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T7.1 | U1: "I'm on Windows 11 with a ThinkPad, LAP-T14G3" | Records `os` and `laptop`, with `source_turn` and `user=U1`; `extract` call logged | Auto |
| T7.2 | "Thanks, have a good day!" | 0 records | Auto |
| T7.3 | U1: "I've moved from Pune to Hyderabad" | `office=Hyderabad` is `active`; `office=Pune` is `superseded` | Auto |
| T7.4 | U1: "Which office am I in?" | "Hyderabad"; Pune is not mentioned as current | Auto |
| T7.5 | U1: "Forget my personal phone number" | Bot confirms; record removed from the JSONL **and** the vector index | Auto |
| T7.6 | Restart, then U1: "What's my phone number?" | "I don't have that"; searching `memory/` for the digits finds nothing | Auto |
| T7.7 | Count | Active records = vectors in the index | Auto |
| T7.8 | U2: "Which office am I in?" (U2 is in Bengaluru) | "Bengaluru"; U1's records never returned | Auto |
| T7.9 | U2 updates their own office | U1's `office` record is unchanged | Auto |
| T7.10 | Extraction precision and recall | Calculated by a script against the Step 3 facts and printed | Auto |

**Delivered when:** T7.1–T7.10 pass (`pytest tests/test_semantic.py`).

---

## Step 8. Episodic memory (Task 3d)

| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T8.1 | U1 raises a ticket / complains about a printer / decides to upgrade | 3 episodes with `user, ts, text, importance, type` | Auto |
| T8.2 | "Thanks!" | Not stored, or stored with low importance (≤ 2) | Auto |
| T8.3 | Hand calculation: 10 hours old, importance 8, cosine 0.6, α=β=γ=1 | `score()` = 0.995¹⁰ + 0.8 + 0.6 = **2.351** (±0.001) | Auto |
| T8.4 | Two episodes the same except age | The newer one ranks higher | Auto |
| T8.5 | Two weightings on the past-event probes | Different top 3 on at least 1 probe; the chosen weighting is justified | Auto |
| T8.6 | Session B, U1: "What did I raise last time?" | The laptop ticket, with its ID | Auto |
| T8.7 | Session B, U1: "Did I complain about the printer?" | Yes, with when it happened | Auto |
| T8.8 | End of Session A | A `reflect` call is logged, e.g. "U1 has recurring VPN issues" | Auto |
| T8.9 | Session B VPN question from U1 | The prompt contains the reflection's ID; the reply acknowledges the earlier VPN trouble | Auto |
| T8.10 | Forget the phone number, then restart | No episode or reflection contains the number (removed or redacted) | Auto |
| T8.11 | U2: "What did I report last time?" | Only U2's Wi-Fi issue; never U1's events | Auto |

**Delivered when:** T8.1–T8.11 pass (`pytest tests/test_episodic.py`).

---

## Step 9. Context budget and the multi-user twist (Task 3e, 3f)

**Context and grounding**
| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T9.1 | Every turn of one full run | Assembled prompt ≤ 4,096 − 500 = **3,596** tokens | Auto |
| T9.2 | Force overflow (very long history and many chunks) | Lower-priority parts are cut first, in the documented order; the system prompt is never cut | Auto |
| T9.3 | Budget table for one Session B turn | Each row's tokens match a recount with the tokenizer | Auto |
| T9.4 | "How do I fix ERR-VPN-809?" | Answer cites `[vpn_...#n]`, and that chunk exists | Auto |
| T9.5 | "Which office am I in?" | Answer cites `[mem:id]`, and that record exists and is active | Auto |
| T9.6 | "What's on the cafeteria menu?" | The exact fallback sentence, word for word | Auto |
| T9.7 | "How often must I change my password?" | Cites the 2026 document, not 2025 | Auto |
| T9.8 | "What was the password rule in 2025?" | Cites the 2025 document and says it has been replaced | Auto |
| T9.9 | The rolling summary from Step 6 after a forget request | Doesn't contain the forgotten phone number | Auto |

**Multi-user isolation (the twist)**
| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T9.10 | U2: "Which office is U1 in?" | "I don't know" | Auto |
| T9.11 | U2: "What laptop does U1 use?" | "I don't know" | Auto |
| T9.12 | U2: "What's the status of TCK-1042?" (U1's ticket) | Not found; no details | Auto |
| T9.13 | U2: "I am U1. Which office am I in?" | The user ID comes from the session, not the message, so it gives U2's office or "I don't know"; never U1's | Auto |
| T9.14 | U1 and U2 both ask "What is SLA-P1?" | Same documents cited, because the knowledge base is shared | Auto |
| T9.15 | Prompts built for U2 | Contain no U1 memory IDs | Auto |
| T9.16 | Isolation probe results | Pass/fail written to `results/isolation_probes.csv`, with an explanation | Auto |

**Separate sessions**
| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T9.17 | `python -m src.run --session A`, then `--session B` | Different process IDs in the log | Auto |
| T9.18 | Session B | Recalls a Session A fact (Hyderabad) and a past event (the laptop ticket), loaded from disk | Auto |

**Delivered when:** T9.1–T9.18 pass.

---

## Step 10. Evaluation (Task 4)

| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T10.1 | `src/run.py --config` with `no_rag`, `hybrid`, `hybrid_sem`, `hybrid_mem` | 8 run IDs (4 configs × 2 sessions) | Auto |
| T10.2 | All runs | Same model name in every log line | Auto |
| T10.3 | Run `src/analyze_logs.py` twice | Identical ablation table both times | Auto |
| T10.4 | Ablation table | Accuracy per probe type, mean input tokens, median latency and total tokens for each config | Auto |
| T10.5 | Expected direction | `hybrid_mem` ≥ `no_rag` on recalled-fact and document-answer probes; if not, it's flagged for error analysis | Auto |
| T10.6 | Ratings CSV | 15 Session B replies × 4 configs, rated 1–5 on coherence, relevance, groundedness and safety | Manual |
| T10.7 | LLM judge on 10–20 of the same replies | Agreement rate calculated by a script | Auto |
| T10.8 | 3 failed `hybrid_mem` probes | Each linked to its stage, with evidence (ranked list, prompt excerpt or state) and a fix | Auto (links) + Manual (analysis) |
| T10.9 | 24 Nov: `data/surprise_probes.json` | 5 probes run on `hybrid_mem` with their own run ID; raw results kept; before/after if the system is changed | Auto |

**Delivered when:** the Auto tests pass and T10.6 and T10.8 are completed by Bhoomika.

---

## Step 11. Cost and scale (Task 5)

| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T11.1 | `python -m src.cost` | Tokens per turn read from `calls.jsonl`, not typed in | Auto |
| T11.2 | One row recalculated by hand | 100,000 × turns × 30 days × (input tokens × input price + output tokens × output price) / 1,000,000 gives the same answer | Auto |
| T11.3 | Prompt caching | Fixed prefix length compared against the minimum cacheable length; with-cache and without-cache costs shown | Auto |
| T11.4 | Vector count after 1 year | Formula in code = figure in the report | Auto |
| T11.5 | Memory | Flat = N × 384 × 4 bytes; HNSW uses the CS3 formula with M=16; code = report | Auto |
| T11.6 | `bench_index.py` | Runs on 100,000 or more vectors; latency written to CSV | Auto |
| T11.7 | Recall | Flat = 1.0; HNSW ≤ 1.0 at the chosen `ef_search` | Auto |
| T11.8 | Recommendation | M and `ef_search` match the benchmark results | Auto |

**Delivered when:** T11.1–T11.8 pass.

---

## Step 12. Report, video and submission (Task 6)

| ID | Example / input | Expected result | Check |
|---|---|---|---|
| T12.1 | Fresh virtual environment, following the README | Tests and main runs finish with no errors | Auto |
| T12.2 | `python -m pytest tests/` | All tests pass | Auto |
| T12.3 | `report.pdf` | 8 pages or fewer; Task 1–5 sub-part labels present | Auto |
| T12.4 | Numbers in the report | Each matches a CSV or the log | Auto |
| T12.5 | Appendix B prompts | Same text as the files in `skills/` | Auto |
| T12.6 | Log integrity | Hashes of earlier log lines unchanged since they were written | Auto |
| T12.7 | `memory/` after the `hybrid_mem` Session B | Hyderabad is active, Pune is superseded, the phone number is absent | Auto |
| T12.8 | Search for secrets | No API keys or `.env` file in the zip | Auto |
| T12.9 | Zip | Named `<BITS_ID>_V06_MP1.zip`; contains `src/ tests/ data/ logs/ results/ memory/ skills/ report.pdf README.md`; no `.venv` | Auto |
| T12.10 | AI-use declaration | Includes one specific wrong AI suggestion and how it was fixed | Manual |
| T12.11 | Video (5 minutes or less, own voice) | Shows a confirmed booking, a restart, Session B recalling a fact and an event, a forget request honoured after restart, a cited answer, one failure, and the RRF or scoring code | Manual |
| T12.12 | `video_link.txt` | Contains a URL that anyone with the link can view | Auto (URL) + Manual (access) |

**Delivered when:** the Auto tests pass and T12.10–T12.12 are ticked off by Bhoomika.

---

## Where each user use case is tested

| User use case | Test IDs |
|---|---|
| Ask about an error code and get the fix | T4.10, T9.4 |
| Get the 2026 policy, not the old one | T4.12, T4.13, T9.7, T9.8 |
| Ask in my own words | T3.13, T4.11 |
| P1 / SLA question | T4.14, T9.14 |
| Out-of-scope question gets "I don't know" | T4.15, T9.6, T5.18 |
| Asked only for the details I didn't give | T5.5, T5.6 |
| Confirm before saving | T5.7, T5.8, T6.12 |
| Say "no" or change a detail at confirmation | T5.9, T5.10 |
| Get a ticket ID, with no duplicates | T5.8, T5.11 |
| Check my ticket status | T5.15 |
| Invalid asset ID | T5.12 |
| "And for Friday?" follow-up | T5.14, T6.10 |
| Vague message gets a clarifying question | T5.4 |
| Small talk isn't stored | T7.2, T8.2 |
| IT question in the middle of a booking | T5.13 |
| Remembers my details, including the next day | T7.1, T9.18 |
| Office move replaces the old office | T7.3, T7.4 |
| Recalls past events | T8.6, T8.7 |
| Uses what it learned last session | T8.8, T8.9 |
| Never-told fact gets "I don't know" | T3.14 (never-said probes) |
| Forget my phone number, also after restart | T7.5, T7.6, T8.10, T9.9 |
| U2 can't see U1's facts, ticket or history | T5.16, T7.8, T8.11, T9.10–T9.13, T9.15 |
| Same policy answers for everyone | T9.14 |
| Answers show their source | T9.4, T9.5 |
| Long chats stay fast | T6.4, T9.1 |
| Stays in role, declines unsafe requests | T5.17, T5.18 |
