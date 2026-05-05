"""
Evaluates TitleBlockExtractor against a golden set.

Usage:
    python evaluate.py run
    python evaluate.py run --log-level DEBUG
    python evaluate.py summary path/to/results.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import sys

from extractor import TitleBlockExtractor
from preprocessing import CROP_SIZE, DPI, THUMB_SIZE, prepare_inputs

logger = logging.getLogger("evaluate")

FIELDS = ("plan_name", "date", "plan_no", "scale", "bkp_code")
PDFS_DIR = Path("data/drawings")
RESULTS_DIR = Path("data/results")
GOLDEN_PATH = Path("data/golden.json")


def score_field(golden, extracted) -> str:
    """Score one extracted field against the golden value.

    Scalar fields require an exact match, while list-valued golden labels count
    as a hit if any extracted value matches one allowed item.
    """
    if golden is None:
        return "fp" if extracted is not None else "hit"
    # golden has a value — extracted must be non-null to have a chance
    if extracted is None:
        return "fn"
    # normalise both sides to strings for comparison
    extracted_str = extracted if isinstance(extracted, str) else None
    extracted_list = extracted if isinstance(extracted, list) else [extracted]
    golden_list = golden if isinstance(golden, list) else [golden]
    if isinstance(golden, str):
        # exact match against the single golden string
        return "hit" if extracted_str == golden else "miss"
    # golden is a list — hit if extracted value appears anywhere in it
    if extracted_str is not None and extracted_str in golden_list:
        return "hit"
    if isinstance(extracted, list):
        return "hit" if any(v in golden_list for v in extracted) else "miss"
    return "miss"


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def cmd_run(args) -> None:
    """Run the extractor over the golden set and write a timestamped results file."""
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        golden_set: list[dict] = json.load(f)

    extractor = TitleBlockExtractor()
    total = len(golden_set)
    results: list[dict] = []

    for i, golden_entry in enumerate(golden_set, start=1):
        pdf_path = golden_entry["pdf_path"]
        stem = Path(pdf_path).stem
        pdf_path = PDFS_DIR / (stem + ".pdf")
        if not pdf_path.exists():
            pdf_path = PDFS_DIR / (stem + ".PDF")
        if not pdf_path.exists():
            logger.warning("[%d/%d] skip (no PDF) %s", i, total, pdf_path)
            continue
        logger.info("[%d/%d] processing %s", i, total, pdf_path)

        error = None
        extracted: dict = {}
        t0 = time.perf_counter()
        try:
            crop, thumbnail = prepare_inputs(str(pdf_path))
            extracted = extractor.extract(crop=crop, thumbnail=thumbnail)
        except Exception as e:
            error = str(e)
            logger.error("[%d/%d] error: %s", i, total, error)
        elapsed = time.perf_counter() - t0

        field_results: dict[str, dict] = {}
        for field in FIELDS:
            golden_val = golden_entry.get(field)
            extracted_val = extracted.get(field) if not error else None
            outcome = "miss" if error else score_field(golden_val, extracted_val)
            field_results[field] = {
                "golden": golden_val,
                "extracted": extracted_val,
                "outcome": outcome,
            }

        result_entry: dict = {
            "pdf_path": str(pdf_path),
            "elapsed_s": round(elapsed, 3),
            "fallback": extracted.get("fallback", False) if not error else False,
            "fields": field_results,
        }
        if error:
            result_entry["error"] = error
        results.append(result_entry)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(args.results) if args.results else RESULTS_DIR / f"results_{ts}.json"
    payload = {
        "config": {
            "model": extractor.model,
            "num_ctx": extractor.num_ctx,
            "temperature": extractor.temperature,
            "fallback": extractor.fallback,
            "dpi": DPI,
            "crop_size": CROP_SIZE,
            "thumb_size": THUMB_SIZE,
        },
        "results": results,
    }
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s", results_path)
    args.results = str(results_path)
    cmd_summary(args)


def cmd_summary(args) -> None:
    """Print aggregate accuracy and timing metrics from a saved results JSON file."""
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}")
        return

    with results_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        config = payload.get("config", {})
        results: list[dict] = payload.get("results", [])
    else:
        config = {}
        results = payload

    counts: dict[str, dict[str, int]] = {
        field: {"hit": 0, "fn": 0, "fp": 0, "miss": 0} for field in FIELDS
    }
    timings: list[float] = []
    fallback_count = 0

    for entry in results:
        timings.append(entry.get("elapsed_s", 0.0))
        if entry.get("fallback", False):
            fallback_count += 1
        for field in FIELDS:
            outcome = entry["fields"][field]["outcome"]
            counts[field][outcome] += 1

    mean_t = sum(timings) / len(timings) if timings else 0.0
    median_t = percentile(timings, 50)
    p95_t = percentile(timings, 95)

    # printout summary
    W = 62
    print()
    print("=" * W)
    print(f"RESULTS SUMMARY  ({results_path})")
    if config:
        items = list(config.items())
        mid = (len(items) + 1) // 2
        print("  " + "  ".join(f"{k}={v}" for k, v in items[:mid]))
        print("  " + "  ".join(f"{k}={v}" for k, v in items[mid:]))
    print("=" * W)
    print(f"{'field':<12} {'acc':>6} {'hits':>6} {'total':>6}  {'miss':>6}  {'fn':>6}  {'fp':>6}")
    print("-" * W)
    total_hits = total_possible = 0
    for field in FIELDS:
        c = counts[field]
        total_field = sum(c.values())
        acc = c["hit"] / total_field if total_field else 0.0
        miss_rate = c["miss"] / total_field if total_field else 0.0
        fn_rate = c["fn"] / total_field if total_field else 0.0
        fp_rate = c["fp"] / total_field if total_field else 0.0
        print(
            f"{field:<12} {acc:>6.1%} {c['hit']:>6} {total_field:>6}"
            f"  {miss_rate:>6.1%}  {fn_rate:>6.1%}  {fp_rate:>6.1%}"
        )
        total_hits += c["hit"]
        total_possible += total_field
    print("-" * W)
    overall = total_hits / total_possible if total_possible else 0.0
    print(f"{'overall':<12} {overall:>6.1%} {total_hits:>6} {total_possible:>6}")
    print("-" * W)
    total_entries = len(results)
    print(f"Fallback  {fallback_count}/{total_entries} ({fallback_count/total_entries:.1%})" if total_entries else "Fallback  n/a")
    print(f"Speed     mean={mean_t:.2f}s  median={median_t:.2f}s  p95={p95_t:.2f}s")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_sub = subparsers.add_parser("run")
    run_sub.add_argument("--results", default=None, help="path to results JSON (default: timestamped)")
    run_sub.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    summary_sub = subparsers.add_parser("summary")
    summary_sub.add_argument("results", help="path to results JSON")
    args = parser.parse_args()

    if args.command == "run":
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(args.log_level)
        cmd_run(args)
    else:
        cmd_summary(args)
