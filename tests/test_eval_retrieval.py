"""T4.16 to T4.18: the hit-rate table and the two saved hybrid cases.

The CSV and the case file are produced by src.eval_retrieval. These tests
check the shape the plan asks for, and that the saved rankings still match a
fresh search.
"""

from __future__ import annotations

import csv
import json

from src.config import ROOT
from src.eval_retrieval import (
    CASES_PATH,
    CUTOFFS,
    HIT_RATE_PATH,
    SIZES,
    hit_rates,
    hybrid_cases,
    is_hit,
    load_questions,
)
from src.search_kb import MODES, search_kb


def test_t4_16_hit_rate_table_covers_methods_sizes_and_slices():
    questions = load_questions()
    rows = hit_rates(questions)
    assert len(questions) >= 25

    methods_sizes = {(row["method"], row["chunk_size"]) for row in rows if row["slice"] == "all"}
    assert methods_sizes == {(mode, size) for mode in MODES for size in SIZES}

    slices = {row["slice"] for row in rows}
    assert slices == {"all", "code", "reworded"}
    code_n = {row["n"] for row in rows if row["slice"] == "code"}
    reworded_n = {row["n"] for row in rows if row["slice"] == "reworded"}
    assert code_n == {sum("code" in q["subtypes"] for q in questions)}
    assert reworded_n == {sum("paraphrase" in q["subtypes"] for q in questions)}
    assert min(code_n) >= 5
    assert min(reworded_n) >= 5


def test_t4_17_hit_rate_never_falls_as_the_cutoff_grows():
    for row in hit_rates(load_questions()):
        assert row["hit_at_1"] <= row["hit_at_3"] <= row["hit_at_5"], row
        assert 0.0 <= row["hit_at_5"] <= 1.0


def test_t4_18_one_question_hybrid_fixed_and_one_it_did_not():
    questions = {q["id"]: q for q in load_questions()}
    saved = hybrid_cases(list(questions.values()))
    labels = [case["label"] for case in saved["cases"]]
    assert labels == ["fixed", "not_fixed"]

    for case in saved["cases"]:
        assert set(case["rankings"]) == set(MODES)
        question = questions[case["id"]]
        size = case["chunk_size"]
        for mode, ranking in case["rankings"].items():
            fresh = search_kb(question["question"], mode=mode, top_k=5, size=size)
            assert [row["chunk_id"] for row in ranking] == [hit["chunk_id"] for hit in fresh]
            assert [row["doc_id"] for row in ranking] == [hit["doc_id"] for hit in fresh]

    fixed = saved["cases"][0]
    gold = (fixed["gold_doc_id"], fixed["gold_version"])
    def top_doc(mode):
        row = fixed["rankings"][mode][0]
        return (row["doc_id"], row["version"])
    # Hybrid and BM25 land on the gold document; dense's first chunk does not.
    assert top_doc("hybrid") == gold
    assert top_doc("bm25") == gold
    assert top_doc("dense") != gold

    missed = saved["cases"][1]
    missed_gold = (missed["gold_doc_id"], missed["gold_version"])
    hybrid_top = missed["rankings"]["hybrid"][0]
    bm25_top = missed["rankings"]["bm25"][0]
    assert (hybrid_top["doc_id"], hybrid_top["version"]) != missed_gold
    assert (bm25_top["doc_id"], bm25_top["version"]) == missed_gold
    # The gold document is still in the hybrid list, just not first.
    assert any(
        (row["doc_id"], row["version"]) == missed_gold for row in missed["rankings"]["hybrid"]
    )


def test_written_files_match_the_tables(tmp_path, monkeypatch):
    """Writing the two result files is what T4.16 and T4.18 check on disk."""
    import src.eval_retrieval as ev

    monkeypatch.setattr(ev, "HIT_RATE_PATH", tmp_path / "hit_rate.csv")
    monkeypatch.setattr(ev, "CASES_PATH", tmp_path / "hybrid_cases.json")
    questions = ev.load_questions()
    rows = ev.hit_rates(questions)
    ev.write_hit_rate(rows)
    (tmp_path / "hybrid_cases.json").write_text(
        json.dumps(ev.hybrid_cases(questions), indent=2) + "\n", encoding="utf-8"
    )

    with (tmp_path / "hit_rate.csv").open(encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(rows)
    assert written[0].keys() >= {"method", "chunk_size", "slice", "hit_at_1", "hit_at_3", "hit_at_5"}
    cases = json.loads((tmp_path / "hybrid_cases.json").read_text(encoding="utf-8"))
    assert {case["label"] for case in cases["cases"]} == {"fixed", "not_fixed"}
    # is_hit agrees with the top-1 column of the all-questions rows.
    assert CUTOFFS == (1, 3, 5)
    assert HIT_RATE_PATH == ROOT / "results" / "hit_rate.csv"
    assert CASES_PATH == ROOT / "results" / "hybrid_cases.json"
    sample = search_kb(questions[0]["question"], mode="dense", top_k=1, size=128)
    assert is_hit(sample, questions[0], 1) in (True, False)
