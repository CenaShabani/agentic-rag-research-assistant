"""Run the evaluation harness and write the results the README reports.

Two stages, mirroring the two halves of evaluation.py:

  * Retrieval metrics — fully local, always run, never need a key.
  * Answer quality (naive vs agent, LLM-as-judge) — needs ANTHROPIC_API_KEY the
    first time to generate answers and grades; afterwards it replays from the
    committed cache. If no key is set and nothing is cached, this stage is
    skipped with a clear message and only the retrieval half is produced.

Results are written to outputs/results/ as JSON and the README figures are
regenerated. Committing the results + cache is what lets every published number
reproduce without a key.

    python run_eval.py
"""

from __future__ import annotations

import json

from src import config, evaluation, figures
from src.llm import LLMKeyMissing
from src.vectorstore import VectorStore


def main() -> None:
    config.ensure_dirs()
    store = VectorStore.load()
    questions = evaluation.load_eval_set()

    print(f"Loaded index ({store.size} chunks) and {len(questions)} eval questions.")

    print("Retrieval metrics (local) ...")
    retrieval_rows = evaluation.evaluate_retrieval(store, questions)
    with open(config.RESULTS_DIR / "retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_rows, f, indent=2)

    answer_rows: list[dict] = []
    try:
        print("Answer quality (naive vs agent, LLM-judge) ...")
        answer_rows = evaluation.evaluate_answers(store, questions)
        with open(config.RESULTS_DIR / "answer_quality.json", "w", encoding="utf-8") as f:
            json.dump(answer_rows, f, indent=2)
    except LLMKeyMissing as exc:
        print(f"  Skipped answer eval: {exc}")
        # Fall back to any previously-committed results so figures still render.
        prev = config.RESULTS_DIR / "answer_quality.json"
        if prev.exists():
            with open(prev, encoding="utf-8") as f:
                answer_rows = json.load(f)
            print("  Loaded previously-committed answer_quality.json for figures.")

    print("Regenerating figures ...")
    figures.all_figures(retrieval_rows, answer_rows)
    print(f"Done. Results in {config.RESULTS_DIR}, figures in {config.FIGURES_DIR}.")


if __name__ == "__main__":
    main()
