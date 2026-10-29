"""ASR experiment runner. Failed items remain explicit and never become zero errors."""
import csv
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from .text import NORMALIZATION_VERSION, normalize, score
from .diagnostics import transcript_diagnostics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_results(directory, name, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = result["items"]
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with (directory / f"{name}_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        summary = [dict(group=k, **v) for k, v in result["summary"].items()]
        keys = list(dict.fromkeys(k for row in summary for k in row))
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary)


def aggregate(rows):
    good = [r for r in rows if r["status"] == "ok"]
    result = dict(total=len(rows), succeeded=len(good), failed=len(rows)-len(good),
                  coverage=len(good)/len(rows) if rows else 0)
    for key in ("word_errors", "reference_words", "character_errors", "reference_characters"):
        result[key] = sum(r[key] for r in good)
    for key in ("empty_transcript", "hindi_latin_only", "hindi_no_devanagari",
                "unexpected_english_candidate"):
        result[key + "_count"] = sum(bool(r.get(key)) for r in good)
    for label, numerator, denominator in [("wer", "word_errors", "reference_words"),
                                           ("cer", "character_errors", "reference_characters")]:
        count = sum(r[denominator] for r in good)
        result[label] = sum(r[numerator] for r in good) / count if count else None
    times = [r["inference_seconds"] for r in good]
    seconds = sum(r["audio_seconds"] for r in good)
    result.update(mean_inference_seconds=statistics.mean(times) if times else None,
                  median_inference_seconds=statistics.median(times) if times else None,
                  audio_seconds=seconds,
                  total_inference_seconds=sum(times),
                  attempted_inference_seconds=sum(r.get("inference_seconds") or 0 for r in rows),
                  total_item_evaluation_seconds=sum(r.get("item_evaluation_seconds") or 0 for r in rows),
                  real_time_factor=sum(times)/seconds if seconds else None)
    return result


def evaluate(manifest, backend):
    import av
    evaluation_start = perf_counter()
    manifest = Path(manifest)
    entries = json.loads(manifest.read_text(encoding="utf-8"))["items"]
    if not entries or len({x["id"] for x in entries}) != len(entries):
        raise ValueError("manifest must contain nonempty uniquely identified items")
    rows = []
    for entry in entries:
        item_start = perf_counter()
        row = {k: entry[k] for k in ("id", "language", "reference")}
        row.update(status="error", audio_seconds=None, inference_seconds=None,
                   language_identification="not_measured_language_forced")
        stage = "input_validation"
        try:
            if entry["language"] not in ("en", "hi"):
                raise ValueError("unsupported language")
            if not normalize(entry["reference"]):
                raise ValueError("empty reference after normalization")
            audio = manifest.parent / entry["audio"]
            if not audio.is_file():
                raise FileNotFoundError(str(audio))
            row["audio_sha256"] = digest(audio)
            if entry.get("sha256") and entry["sha256"] != row["audio_sha256"]:
                raise ValueError("audio checksum mismatch")
            stage = "audio_loading"
            with av.open(str(audio)) as container:
                stream = container.streams.audio[0]
                frames = [(frame.samples, frame.sample_rate) for frame in container.decode(stream)]
                row["audio_seconds"] = sum(samples / rate for samples, rate in frames)
                row["audio_samples"] = sum(samples for samples, _ in frames)
            if entry.get("expected_samples") and entry["expected_samples"] != row["audio_samples"]:
                raise ValueError("audio sample count differs from reference TSV")
            if row["audio_seconds"] <= 0:
                raise ValueError("empty audio")
            start = perf_counter()
            stage = "model_inference"
            try:
                hypothesis = backend.transcribe(str(audio), entry["language"])
            finally:
                row["inference_seconds"] = perf_counter() - start
            stage = "transcript_scoring"
            row.update(hypothesis=hypothesis, **score(entry["reference"], hypothesis), status="ok")
            row.update(transcript_diagnostics(hypothesis, entry["language"]))
            row.update(getattr(backend, "last_metadata", {}))
            row["real_time_factor"] = row["inference_seconds"] / row["audio_seconds"]
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["failure_stage"] = stage
        row["item_evaluation_seconds"] = perf_counter() - item_start
        rows.append(row)
    groups = {"all": aggregate(rows)}
    groups.update({lang: aggregate([r for r in rows if r["language"] == lang])
                   for lang in sorted({r["language"] for r in rows})})
    return dict(kind="asr_evaluation_unclassified_backend", created_utc=datetime.now(timezone.utc).isoformat(),
                python=sys.version, platform=platform.platform(),
                normalization=NORMALIZATION_VERSION, manifest_sha256=digest(manifest),
                evaluation_duration_seconds=perf_counter() - evaluation_start,
                items=rows, summary=groups)
