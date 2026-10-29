"""Paired-input validation and descriptive Tiny/Base comparison, no significance claims."""
import csv
import json
from .evaluate import aggregate


def validate_pair(tiny, base):
    for key in ("manifest_sha256", "normalization", "decoding", "device", "compute_type"):
        if key not in tiny or tiny[key] != base.get(key):
            raise ValueError(f"comparison requires identical {key}")
    for run in (tiny, base):
        rows = run["items"]
        ids = [r["id"] for r in rows]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("nonempty unique sample IDs required")
    if [r["id"] for r in tiny["items"]] != [r["id"] for r in base["items"]]:
        raise ValueError("identical sample IDs and order required")
    for left, right in zip(tiny["items"], base["items"]):
        for key in ("language", "reference", "audio_sha256", "status"):
            if key not in left or left[key] != right.get(key):
                raise ValueError(f"paired sample {left['id']} differs in {key}")
        if not left["audio_sha256"]:
            raise ValueError("verified audio hashes required")


def compare(tiny, base):
    validate_pair(tiny, base)
    rows, differences = [], []
    languages = sorted({r["language"] for r in tiny["items"]})
    for language in [*languages, "all"]:
        scores = {}
        for name, run in [("tiny", tiny), ("base", base)]:
            selected = [r for r in run["items"] if language == "all" or r["language"] == language]
            scores[name] = aggregate(selected)
            rows.append(dict(model=name, language=language, device=run["device"],
                             compute_type=run["compute_type"], **scores[name]))
        diff = dict(language=language)
        for metric in ("wer", "cer"):
            before, after = scores["tiny"][metric], scores["base"][metric]
            diff[metric + "_difference_pp_base_minus_tiny"] = (
                100 * (after - before) if before is not None and after is not None else None)
            diff[metric + "_relative_error_reduction"] = (
                (before-after)/before if before and after is not None else None)
        before = scores["tiny"]["mean_inference_seconds"]
        after = scores["base"]["mean_inference_seconds"]
        diff["latency_difference_seconds_base_minus_tiny"] = (
            after-before if before is not None and after is not None else None)
        diff["latency_ratio_base_over_tiny"] = after/before if before and after is not None else None
        differences.append(diff)
    return dict(kind="descriptive_paired_asr_comparison", manifest_sha256=tiny["manifest_sha256"],
                normalization=tiny["normalization"], decoding=tiny["decoding"],
                sample_ids=[r["id"] for r in tiny["items"]], items=rows, differences=differences,
                runs={name: {k: run.get(k) for k in (
                    "model_provenance", "model_sha256", "device", "compute_type", "requested_device",
                    "fallback_reason", "evaluation_duration_seconds", "model_load_seconds",
                    "created_utc", "python", "versions", "hardware")}
                    for name, run in [("tiny", tiny), ("base", base)]},
                caveat="Descriptive convenience-sample comparison; no statistical significance test.")


def export_comparison(directory, result):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rows in [("comparison", result["items"]), ("differences", result["differences"])]:
        with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["| Model | Language | Samples (successful/total) | WER | CER | Mean latency | Device |",
             "|---|---|---:|---:|---:|---:|---|"]
    for row in result["items"]:
        def percent(value):
            return "n/a" if value is None else f"{100 * value:.2f}%"
        latency = row["mean_inference_seconds"]
        lines.append(f"| Whisper {row['model'].title()} | {row['language']} | "
                     f"{row['succeeded']}/{row['total']} | {percent(row['wer'])} | "
                     f"{percent(row['cer'])} | " + ("n/a" if latency is None else f"{latency:.3f} s") +
                     f" | {row['device']} ({row['compute_type']}) |")
    (directory / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
