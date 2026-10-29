"""All backends/recordings here are test fixtures, never benchmark evidence."""
import copy
import csv
import io
import json
import sys
import tarfile
import wave
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from alexabench.compare import compare, export_comparison, validate_pair
from alexabench.dataset import download, reference_lookup
from alexabench.diagnostics import transcript_diagnostics
from alexabench.evaluate import aggregate, evaluate
from alexabench.text import score


def fixture_run(hypothesis):
    rows = [dict(id=lang, language=lang, reference="a b", audio_sha256="fixture-hash",
                 status="ok", inference_seconds=.1, audio_seconds=1,
                 **score("a b", hypothesis)) for lang in ("en", "hi")]
    return dict(kind="test_fixture", manifest_sha256="fixture-manifest",
                normalization="fixture", decoding={"beam_size": 1}, device="cuda",
                compute_type="float16", items=rows)


def test_paired_comparison_and_exports(tmp_path):
    tiny, base = fixture_run("a x"), fixture_run("a b")
    result = compare(tiny, base)
    assert result["sample_ids"] == ["en", "hi"]
    difference = result["differences"][0]
    assert difference["wer_difference_pp_base_minus_tiny"] == -50
    assert difference["wer_relative_error_reduction"] == 1
    assert result["items"][0]["reference_words"] == 2
    export_comparison(tmp_path, result)
    assert json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8")) == result
    with (tmp_path / "comparison.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert rows[0]["device"] == "cuda"
    assert rows[0]["reference_characters"] == "2"


@pytest.mark.parametrize("field,value", [("manifest_sha256", "other"),
    ("normalization", "other"), ("device", "cpu"), ("compute_type", "int8"),
    ("decoding", {"beam_size": 5})])
def test_reject_uncontrolled_comparisons(field, value):
    tiny, base = fixture_run("a b"), fixture_run("a b")
    base[field] = value
    with pytest.raises(ValueError, match=field):
        validate_pair(tiny, base)


@pytest.mark.parametrize("field,value", [("id", "missing"), ("reference", "different"),
    ("audio_sha256", "different"), ("language", "other"), ("status", "error")])
def test_reject_unpaired_inputs(field, value):
    tiny, base = fixture_run("a b"), fixture_run("a b")
    base["items"][0][field] = value
    with pytest.raises(ValueError):
        validate_pair(tiny, base)


def test_duplicate_ids_and_zero_baseline():
    tiny, base = fixture_run("a b"), fixture_run("a b")
    assert compare(tiny, base)["differences"][0]["wer_relative_error_reduction"] is None
    tiny["items"].append(copy.deepcopy(tiny["items"][0]))
    with pytest.raises(ValueError, match="unique"):
        validate_pair(tiny, base)


def test_reference_quotes_are_literal():
    row = '1\ta.wav\t"literal reference"\t"unbalanced normalized\tchars\t16\tFEMALE\n'
    assert reference_lookup(row)["a.wav"][2] == '"literal reference"'
    with pytest.raises(ValueError, match="duplicate"):
        reference_lookup(row + row)


def test_reproducible_30_per_language_selection(tmp_path, monkeypatch):
    from alexabench import dataset
    audio = io.BytesIO()
    with wave.open(audio, "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(b"\0\0" * 16)
    contents = io.BytesIO()
    with tarfile.open(fileobj=contents, mode="w:gz") as archive:
        for i in range(31):
            info = tarfile.TarInfo(f"test/{i:03}.wav")
            info.size = len(audio.getvalue())
            archive.addfile(info, io.BytesIO(audio.getvalue()))
    @contextmanager
    def fixture_remote(url):
        yield io.BytesIO(contents.getvalue())
    def fixture_fetch(url, target):
        ref = "हिंदी किताब" if "/hi_in/" in url else "English book"
        Path(target).write_text("".join(
            f"{i}\t{i:03}.wav\t{ref}\tunused\tunused\t16\tFEMALE\n"
            for i in range(31)), encoding="utf-8")
    monkeypatch.setattr(dataset, "remote", fixture_remote)
    monkeypatch.setattr(dataset, "fetch", fixture_fetch)
    download(tmp_path, 30)
    original = (tmp_path / "manifest.json").read_bytes()
    manifest = json.loads(original)
    assert len(manifest["items"]) == 60
    for lang in ("en", "hi"):
        selected = [r for r in manifest["items"] if r["language"] == lang]
        assert len(selected) == 30
        assert selected[0]["source_member"] == "test/000.wav"
        assert selected[-1]["source_member"] == "test/029.wav"
        assert all(r["expected_samples"] == 16 for r in selected)
    download(tmp_path, 30)
    assert (tmp_path / "manifest.json").read_bytes() == original


def test_script_diagnostics():
    assert transcript_diagnostics("यह हिंदी है", "hi")["devanagari_letters"] > 0
    assert not transcript_diagnostics("यह हिंदी है", "hi")["hindi_latin_only"]
    assert transcript_diagnostics("yeh hindi hai", "hi")["hindi_latin_only"]
    assert not transcript_diagnostics("yeh hindi hai", "hi")["unexpected_english_candidate"]
    assert transcript_diagnostics("This is the story of a book", "hi")["unexpected_english_candidate"]
    assert transcript_diagnostics("। ", "hi")["empty_transcript"]
    assert not transcript_diagnostics("१२३", "hi")["hindi_latin_only"]
    with pytest.raises(TypeError):
        transcript_diagnostics(None, "hi")


@pytest.mark.parametrize("device,compute_type", [("cpu", "int8"), ("cuda", "float16")])
def test_whisper_device_reporting_and_lazy_segments(monkeypatch, device, compute_type):
    from alexabench.asr import WhisperASR
    captured = {}
    class FixtureModel:
        def __init__(self, path, **kwargs):
            captured.update(kwargs)
        def transcribe(self, path, **kwargs):
            captured.update(kwargs)
            return iter([SimpleNamespace(text=" हिंदी ")]), SimpleNamespace(language="hi")
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FixtureModel))
    backend = WhisperASR("fixture", device)
    assert backend.transcribe("fixture.wav", "hi") == "हिंदी"
    assert backend.device == captured["device"] == device
    assert backend.compute_type == captured["compute_type"] == compute_type
    assert captured["language"] == "hi" and captured["task"] == "transcribe"
    assert backend.last_metadata["language_identification"] == "not_measured_language_forced"


def test_empty_prediction_is_scored(tmp_path):
    with wave.open(str(tmp_path / "fixture.wav"), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(b"\0\0" * 16)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [dict(id="fixture", language="hi", reference="किताब",
                                                  audio="fixture.wav", expected_samples=16)]}))
    backend = SimpleNamespace(transcribe=lambda path, language: "")
    result = evaluate(manifest, backend)
    assert result["items"][0]["status"] == "ok"
    summary = result["summary"]["hi"]
    assert summary["wer"] == summary["cer"] == 1
    assert summary["empty_transcript_count"] == 1
    assert summary["reference_words"] == 1 and summary["reference_characters"] == 5
    assert result["evaluation_duration_seconds"] > 0


@pytest.mark.parametrize("mode,requested,expected_device", [
    ("ok", "cpu", "cpu"), ("ok", "cuda", "cuda"),
    ("no_cuda", "auto", "cpu"), ("load_failure", "auto", "cpu"),
    ("inference_failure", "auto", "cpu")])
def test_cli_device_and_fallback_reporting(monkeypatch, tmp_path, mode, requested, expected_device):
    from alexabench import cli, asr, hardware
    captured = {}
    class FixtureBackend:
        def __init__(self, model, device):
            if device == "cuda" and mode == "load_failure":
                raise RuntimeError("fixture CUDA unavailable")
            self.device = device
            self.compute_type = "float16" if device == "cuda" else "int8"
            self.load_seconds = .1
    def fixture_evaluate(manifest, backend):
        failed = mode == "inference_failure" and backend.device == "cuda"
        return {"items": [{"error": "RuntimeError: fixture CUDA inference failure"}] if failed else [],
                "summary": {"all": {"failed": int(failed)}}}
    monkeypatch.setattr(asr, "WhisperASR", FixtureBackend)
    monkeypatch.setattr(cli, "evaluate", fixture_evaluate)
    monkeypatch.setattr(cli, "write_results", lambda directory, name, result: captured.update(result))
    monkeypatch.setattr(hardware, "hardware_info", lambda: {"kind": "test_fixture"})
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(
        get_cuda_device_count=lambda: 0 if mode == "no_cuda" else 1))
    monkeypatch.setattr(sys, "argv", ["alexabench", "asr", "--device", requested,
                                     "--model", str(tmp_path / "fixture"), "--out", str(tmp_path)])
    cli.main()
    assert captured["device"] == expected_device
    assert captured["requested_device"] == requested
    assert captured["compute_type"] == ("float16" if expected_device == "cuda" else "int8")
    assert bool(captured["fallback_reason"]) == (mode != "ok")


def test_explicit_cuda_never_silently_falls_back(monkeypatch):
    from alexabench import cli, asr
    def failure(*args):
        raise RuntimeError("fixture CUDA unavailable")
    monkeypatch.setattr(asr, "WhisperASR", failure)
    monkeypatch.setattr(sys, "argv", ["alexabench", "asr", "--device", "cuda"])
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        cli.main()


def test_model_integrity_and_cache(tmp_path, monkeypatch):
    from alexabench import dataset
    def fixture_fetch(url, target):
        Path(target).write_bytes(b"fixture-model-artifact")
    monkeypatch.setattr(dataset, "fetch", fixture_fetch)
    dataset.download_model(tmp_path, "base")
    assert dataset.verify_model(tmp_path)["revision"] == dataset.MODEL_REVISIONS["base"]
    monkeypatch.setattr(dataset, "fetch", lambda *args: pytest.fail("cache should be reused"))
    dataset.download_model(tmp_path, "base")
    (tmp_path / "model.bin").write_bytes(b"corrupted-fixture")
    with pytest.raises(ValueError, match="checksum mismatch"):
        dataset.verify_model(tmp_path)
