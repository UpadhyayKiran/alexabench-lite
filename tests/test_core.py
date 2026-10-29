import json
import wave
import pytest
from alexabench.text import normalize, distance, score
from alexabench.evaluate import evaluate, aggregate
from alexabench.assistant import grade, evaluate_assistant


def test_normalization():
    assert normalize("  HELLO,\tworld! ") == "hello world"
    assert normalize("हिन्दी। क़लम\u200d") == normalize("हिन्दी क़लम")
    assert normalize("cafe\u0301") == "café"
    assert normalize("१२ 12") == "१२ 12"
    assert normalize("किताब") == "किताब"  # vowel marks must survive


def test_metrics():
    r = score("one two three", "one too three four")
    assert r["wer"] == pytest.approx(2/3)
    assert score("ab cd", "ab xd")["cer"] == .25
    assert score("hello", "")["wer"] == 1
    assert score("one", "one two three")["wer"] == 2
    assert score("हिंदी", "हिंदी")["cer"] == 0
    assert distance("kitten", "sitting") == 3


@pytest.mark.parametrize("reference", ["", " \t", "। !"])
def test_empty_reference(reference):
    with pytest.raises(ValueError, match="empty reference"):
        score(reference, "hello")


class FakeBackend:
    def transcribe(self, path, language):
        raise RuntimeError("test fixture inference failure")


@pytest.mark.parametrize("mode,error", [("missing", "FileNotFoundError"),
    ("empty", "empty reference"), ("inference", "test fixture inference failure")])
def test_failures(tmp_path, mode, error):
    with wave.open(str(tmp_path / "sample.wav"), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(b"\0\0" * 1600)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [dict(id="fixture", language="en",
        audio="missing.wav" if mode == "missing" else "sample.wav",
        reference="" if mode == "empty" else "hello")]}))
    result = evaluate(manifest, FakeBackend())
    assert error in result["items"][0]["error"]
    assert result["summary"]["all"]["wer"] is None
    assert result["summary"]["all"]["failed"] == 1


def test_micro_aggregation():
    rows = [dict(status="ok", word_errors=1, reference_words=n, character_errors=1,
                 reference_characters=n, audio_seconds=1, inference_seconds=.1) for n in [1, 9]]
    assert aggregate(rows)["wer"] == .2


def test_assistant_contract():
    expected = {"action": "set_timer", "minutes": 5}
    assert grade(expected, json.dumps(expected))["task_completed"]
    assert not grade(expected, {"action": "set_timer", "minutes": "5"})["task_completed"]
    assert not grade(expected, "```json\n{}\n```")["instruction_following"]
    result = grade(expected, dict(expected, extra="text"))
    assert result["task_completed"] and not result["instruction_following"]
    assert not grade(expected, None)["task_completed"]
    assert not grade(expected, {})["task_completed"]


def test_scenarios():
    from pathlib import Path
    scenarios = json.loads(Path("scenarios/assistant.json").read_text(encoding="utf-8"))
    assert evaluate_assistant(scenarios)["summary"]["all"]["total"] == 10
    assert evaluate_assistant(scenarios, {})["summary"]["all"]["task_completion"] == 0


def test_exports(tmp_path):
    from alexabench.evaluate import write_results
    import csv
    result = {"items": [{"id": "hi", "reference": "हिंदी", "wer": .5}],
              "summary": {"all": {"wer": .5}, "hi": {"wer": .5}}}
    write_results(tmp_path, "test_fixture", result)
    assert json.loads((tmp_path / "test_fixture.json").read_text(encoding="utf-8")) == result
    with (tmp_path / "test_fixture.csv").open(encoding="utf-8-sig", newline="") as f:
        assert list(csv.DictReader(f))[0]["reference"] == "हिंदी"
    assert (tmp_path / "test_fixture_summary.csv").is_file()


def test_checksum_failure(tmp_path):
    manifest = tmp_path / "manifest.json"
    (tmp_path / "audio.wav").write_bytes(b"fixture")
    manifest.write_text(json.dumps({"items": [dict(id="fixture", language="en",
        audio="audio.wav", reference="hello", sha256="incorrect")]}))
    assert "checksum mismatch" in evaluate(manifest, FakeBackend())["items"][0]["error"]
