"""Validate real study artifacts and a completed pytest JUnit report.

Run after the documented download/inference/compare commands. This is not an
inference replacement: it recomputes scores from saved hypotheses and verifies
their recorded inputs, artifacts and preservation checks.
"""
import argparse
import json
import platform
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from alexabench.compare import compare, validate_pair
from alexabench.dataset import DATA_REVISION, MODEL_REVISIONS, reference_lookup, verify_model
from alexabench.evaluate import aggregate, digest
from alexabench.text import NORMALIZATION_VERSION, score


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/fleurs_60/manifest.json")
    parser.add_argument("--results", default="results/comparison_60")
    parser.add_argument("--pytest-xml", default=".cache/pytest.xml")
    parser.add_argument("--out", default="results/validation/report.json")
    args = parser.parse_args()
    manifest_path, results_path = Path(args.manifest), Path(args.results)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["revision"] == DATA_REVISION, "unexpected dataset revision")
    require(manifest["split"] == "test", "test split required")
    items = manifest["items"]
    require(len(items) == len({x["id"] for x in items}) == 60, "60 unique recordings required")
    require(all(sum(x["language"] == lang for x in items) == 30 for lang in ("en", "hi")),
            "30 recordings per language required")
    for language, config in (("en", "en_us"), ("hi", "hi_in")):
        tsv = manifest_path.parent / f"{config}_test.tsv"
        require(digest(tsv) == manifest["reference_tsv_sha256"][config], "TSV checksum mismatch")
        refs = reference_lookup(tsv.read_text(encoding="utf-8"))
        for item in (x for x in items if x["language"] == language):
            reference = refs[Path(item["source_member"]).name]
            require(item["reference"] == reference[2], "reference differs from raw TSV")
            require(item["source_id"] == reference[0], "sentence ID differs from TSV")
            require(item["expected_samples"] == int(reference[5]), "TSV sample count mismatch")
            require(digest(manifest_path.parent / item["audio"]) == item["sha256"], "audio checksum mismatch")
    runs = {}
    for name in ("tiny", "base"):
        run = json.loads((results_path / name / "asr.json").read_text(encoding="utf-8"))
        runs[name] = run
        require(run["kind"] == "real_asr_experiment", "not a real-ASR result artifact")
        require(run["manifest_sha256"] == digest(manifest_path), "result manifest mismatch")
        require(run["normalization"] == NORMALIZATION_VERSION, "normalization mismatch")
        require(run["device"] == "cuda" and run["compute_type"] == "float16" and
                run["requested_device"] == "cuda" and run["fallback_reason"] is None,
                "this recorded study requires explicit CUDA float16 without fallback")
        provenance = verify_model(Path(run["model"]))
        require(provenance == run["model_provenance"], "model provenance mismatch")
        require(provenance["repository"] == f"Systran/faster-whisper-{name}" and
                provenance["revision"] == MODEL_REVISIONS[name], "wrong pinned model")
        require(run["model_sha256"] == provenance["sha256"]["model.bin"], "weight hash mismatch")
        require(len(run["items"]) == len(items), "result sample count mismatch")
        for item, row in zip(items, run["items"]):
            require(row["id"] == item["id"] and row["language"] == item["language"] and
                    row["reference"] == item["reference"], "result input differs from manifest")
            require(row["status"] == "ok", "incomplete inference")
            require(row["audio_sha256"] == item["sha256"] and
                    row["audio_samples"] == item["expected_samples"], "decoded audio mismatch")
            for key, value in score(row["reference"], row["hypothesis"]).items():
                require(row[key] == value, f"metric mismatch: {key}")
        for group, summary in run["summary"].items():
            selected = [r for r in run["items"] if group == "all" or r["language"] == group]
            require(summary == aggregate(selected), "aggregate mismatch")
    validate_pair(runs["tiny"], runs["base"])
    saved_comparison = json.loads((results_path / "comparison.json").read_text(encoding="utf-8"))
    require(compare(runs["tiny"], runs["base"]) == saved_comparison, "comparison artifact mismatch")
    historical = json.loads(Path("docs/historical_baseline_sha256.json").read_text(encoding="utf-8"))
    require(all(digest(path) == expected for path, expected in historical.items()),
            "historical evidence changed")
    root = ET.parse(args.pytest_xml).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = {key: sum(int(s.get(key, "0")) for s in suites)
             for key in ("tests", "failures", "errors", "skipped")}
    require(tests["tests"] >= 38 and not any(tests[k] for k in ("failures", "errors", "skipped")),
            "full passing test suite required")
    report = dict(status="passed", created_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), pytest=tests,
                  unique_recordings=60, model_recording_evaluations=120,
                  manifest_sha256=digest(manifest_path), historical_artifacts_unchanged=len(historical),
                  checks=["raw references and TSV hashes", "audio hashes and decoded sample counts",
                          "identical ordered inputs across models", "pinned model files verified",
                          "CUDA float16 without fallback", "per-item metrics independently recomputed",
                          "aggregates and comparison recomputed", "historical evidence unchanged",
                          "full pytest suite passed"],
                  source_sha256={p.as_posix(): digest(p) for p in sorted(Path("alexabench").glob("*.py"))},
                  result_sha256={p.as_posix(): digest(p) for p in sorted(results_path.rglob("*"))
                                 if p.is_file() and p.resolve() != Path(args.out).resolve()})
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validated {report['unique_recordings']} recordings, {report['model_recording_evaluations']} "
          f"model-recording evaluations, {tests['tests']} tests; report: {target}")


if __name__ == "__main__":
    main()
