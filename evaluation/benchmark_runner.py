import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))
    
from graph import build_graph


SECTION_PATTERN = re.compile(r"\b\d{2,3}\b")


def _to_section_set(items: list[Any] | None) -> set[str]:
    if not items:
        return set()
    return {str(x).strip() for x in items if str(x).strip()}


def _extract_predicted_sections(result: dict[str, Any]) -> set[str]:
    predicted: set[str] = set()

    text = str(result.get("ipc_sections", ""))
    predicted.update(SECTION_PATTERN.findall(text))

    for doc in result.get("ipc_sources", []) or []:
        section = None
        if hasattr(doc, "metadata"):
            section = doc.metadata.get("section")
        elif isinstance(doc, dict):
            section = doc.get("section")
        if section is not None:
            predicted.add(str(section).strip())

    return {p for p in predicted if p}


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded(value: float | None, low: float = 0.0, high: float = 1.0) -> float | None:
    if value is None:
        return None
    return max(low, min(high, value))


def _prf1(predicted: set[str], gold: set[str]) -> tuple[float | None, float | None, float | None]:
    if not gold and not predicted:
        return 1.0, 1.0, 1.0
    if not gold:
        return 0.0, None, None
    if not predicted:
        return 0.0, 0.0, 0.0

    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _token_f1(prediction: str, reference: str) -> float | None:
    pred_tokens = re.findall(r"\w+", prediction.lower())
    ref_tokens = re.findall(r"\w+", reference.lower())
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_count: dict[str, int] = {}
    ref_count: dict[str, int] = {}
    for token in pred_tokens:
        pred_count[token] = pred_count.get(token, 0) + 1
    for token in ref_tokens:
        ref_count[token] = ref_count.get(token, 0) + 1

    overlap = 0
    for token, c in pred_count.items():
        overlap += min(c, ref_count.get(token, 0))

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _domain_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed.netloc.lower().replace("www.", "")


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return statistics.fmean(present)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return rows


def run_benchmark(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    app = build_graph()
    dataset = _load_jsonl(dataset_path)

    case_results: list[dict[str, Any]] = []
    failures = 0

    for idx, sample in enumerate(dataset):
        case_id = sample.get("id", f"case_{idx + 1}")
        query = sample.get("query")
        if not query:
            failures += 1
            case_results.append(
                {
                    "id": case_id,
                    "error": "Missing required field: query",
                    "success": False,
                }
            )
            continue

        gold_sections = _to_section_set(sample.get("gold_ipc_sections"))
        gold_domains = {d.lower().replace("www.", "") for d in sample.get("gold_precedent_domains", [])}
        reference_opinion = sample.get("reference_opinion", "")

        start = time.perf_counter()
        try:
            result = app.invoke({"query": query})
            success = True
            error = None
        except Exception as exc:
            result = {}
            success = False
            error = str(exc)
            failures += 1
        latency_s = time.perf_counter() - start

        predicted_sections = _extract_predicted_sections(result)
        ipc_precision, ipc_recall, ipc_f1 = _prf1(predicted_sections, gold_sections)
        ipc_exact_match = float(predicted_sections == gold_sections) if gold_sections else None

        precedent_domains = {
            _domain_from_url(src.get("url", ""))
            for src in (result.get("precedent_sources", []) or [])
            if isinstance(src, dict)
        }
        precedent_domains.discard("")
        domain_recall = None
        if gold_domains:
            domain_recall = len(precedent_domains & gold_domains) / len(gold_domains)

        opinion_text = str(result.get("final_opinion", ""))
        opinion_ref_f1 = _token_f1(opinion_text, reference_opinion) if reference_opinion else None

        row = {
            "id": case_id,
            "success": success,
            "error": error,
            "latency_seconds": round(latency_s, 4),
            "ipc_precision": _bounded(ipc_precision),
            "ipc_recall": _bounded(ipc_recall),
            "ipc_f1": _bounded(ipc_f1),
            "ipc_exact_match": ipc_exact_match,
            "precedent_domain_recall": _bounded(domain_recall),
            "precedent_confidence": _bounded(_safe_float(result.get("precedent_confidence"))),
            "ipc_confidence": _bounded(_safe_float(result.get("ipc_confidence"))),
            "opinion_confidence": _bounded(_safe_float(result.get("opinion_confidence"))),
            "overall_confidence": _bounded(_safe_float(result.get("overall_confidence"))),
            "opinion_reference_token_f1": _bounded(opinion_ref_f1),
            "predicted_ipc_sections": sorted(predicted_sections),
            "gold_ipc_sections": sorted(gold_sections),
        }
        case_results.append(row)

    summary = {
        "dataset_size": len(dataset),
        "success_rate": (len(dataset) - failures) / len(dataset) if dataset else 0.0,
        "mean_latency_seconds": _mean([r.get("latency_seconds") for r in case_results]),
        "mean_ipc_precision": _mean([r.get("ipc_precision") for r in case_results]),
        "mean_ipc_recall": _mean([r.get("ipc_recall") for r in case_results]),
        "mean_ipc_f1": _mean([r.get("ipc_f1") for r in case_results]),
        "mean_ipc_exact_match": _mean([r.get("ipc_exact_match") for r in case_results]),
        "mean_precedent_domain_recall": _mean([r.get("precedent_domain_recall") for r in case_results]),
        "mean_ipc_confidence": _mean([r.get("ipc_confidence") for r in case_results]),
        "mean_precedent_confidence": _mean([r.get("precedent_confidence") for r in case_results]),
        "mean_opinion_confidence": _mean([r.get("opinion_confidence") for r in case_results]),
        "mean_overall_confidence": _mean([r.get("overall_confidence") for r in case_results]),
        "mean_opinion_reference_token_f1": _mean([r.get("opinion_reference_token_f1") for r in case_results]),
    }

    report = {"summary": summary, "cases": case_results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the legal assistant on a JSONL dataset.")
    parser.add_argument("--dataset", required=True, help="Path to benchmark dataset JSONL file.")
    parser.add_argument(
        "--output",
        default="evaluation/benchmark_report.json",
        help="Path to save benchmark report JSON.",
    )
    args = parser.parse_args()

    report = run_benchmark(Path(args.dataset), Path(args.output))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
