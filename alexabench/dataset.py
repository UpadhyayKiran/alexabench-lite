"""Bounded streaming FLEURS subset download; no datasets/Arrow dependency.

Select the first N regular WAV members in upstream archive order, then close the
connection. Selection is made before inference and never based on ASR quality.
"""
import csv
import io
import json
import shutil
import subprocess
import tarfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from .evaluate import digest

DATA_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
MODEL_REVISION = "d90ca5fe260221311c53c58e660288d3deb8d356"
MODEL_REVISIONS = {"tiny": MODEL_REVISION,
                   "base": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"}


def reference_lookup(metadata):
    """Read original FLEURS TSV: ID, filename, raw text, normalized text, ... ."""
    result = {}
    # FLEURS TSV has literal/unbalanced quotation marks; quotes are text, not CSV quoting.
    for row in csv.reader(io.StringIO(metadata), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) != 7 or row[1] in result or not row[2].strip():
            raise ValueError("invalid, duplicate, or empty FLEURS reference row")
        result[row[1]] = row
    return result


@contextmanager
def remote(url):
    # curl uses the OS TLS/network stack on Windows; urllib remains a fallback.
    if shutil.which("curl"):
        process = subprocess.Popen(["curl", "--fail", "--silent", "--show-error", "--location",
                                    "--connect-timeout", "20", "--max-time", "300", url],
                                   stdout=subprocess.PIPE)
        try:
            yield process.stdout
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait()
    else:
        with urllib.request.urlopen(url, timeout=60) as stream:
            yield stream


def fetch(url, target):
    target = Path(target)
    partial = target.with_suffix(target.suffix + ".partial")
    if shutil.which("curl"):
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                        "--connect-timeout", "20", "--max-time", "300", "--retry", "2",
                        "--output", str(partial), url], check=True)
    else:
        with remote(url) as stream, partial.open("wb") as output:
            shutil.copyfileobj(stream, output)
    if not partial.stat().st_size:
        raise RuntimeError(f"empty download: {url}")
    partial.replace(target)


def download(directory, count):
    if not 1 <= count <= 100:
        raise ValueError("per-language must be 1..100 for this small benchmark")
    directory.mkdir(parents=True, exist_ok=True)
    items = []
    for language, config in [("en", "en_us"), ("hi", "hi_in")]:
        base = f"https://huggingface.co/datasets/google/fleurs/resolve/{DATA_REVISION}/data/{config}"
        metadata = directory / f"{config}_test.tsv"
        fetch(f"{base}/test.tsv", metadata)
        lookup = reference_lookup(metadata.read_text(encoding="utf-8"))
        archive_url = f"{base}/audio/test.tar.gz"
        selected = 0
        with remote(archive_url) as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                name = Path(member.name).name
                if not member.isfile() or not name.endswith(".wav"):
                    continue
                row = lookup[name]
                if int(row[5]) <= 0:
                    raise ValueError("invalid reference audio sample count")
                filename = f"{config}_{name}"
                target = directory / filename
                # Never extract upstream paths, links or directory names.
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                items.append(dict(id=f"{config}_{name[:-4]}", language=language,
                                  audio=filename, reference=row[2], source_id=row[0],
                                  source_member=member.name, source_url=archive_url,
                                  sha256=digest(target), gender=row[6], expected_samples=int(row[5])))
                selected += 1
                if selected == count:
                    break
        if selected != count:
            raise RuntimeError("not enough audio members")
    manifest = dict(dataset="google/fleurs", revision=DATA_REVISION, split="test",
                    license="CC-BY-4.0", attribution="FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech, Conneau et al., 2022, Google",
                    selection=f"first {count} WAV archive members per language", items=items)
    manifest["reference_tsv_sha256"] = {config: digest(directory / f"{config}_test.tsv")
                                        for config in ("en_us", "hi_in")}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(items)} real recordings to {directory}")


def download_model(directory, size="tiny"):
    revision = MODEL_REVISIONS[size]
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "provenance.json").is_file():
        provenance = verify_model(directory)
        if (provenance["revision"] == revision and
                provenance["repository"] == f"Systran/faster-whisper-{size}"):
            print(f"Verified cached {size} model at {revision}")
            return
    hashes = {}
    for filename in ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]:
        target = directory / filename
        fetch(f"https://huggingface.co/Systran/faster-whisper-{size}/resolve/{revision}/{filename}", target)
        hashes[filename] = digest(target)
    (directory / "provenance.json").write_text(json.dumps(dict(
        repository=f"Systran/faster-whisper-{size}", revision=revision,
        license="MIT", sha256=hashes), indent=2), encoding="utf-8")


def verify_model(directory):
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    for filename in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        if digest(directory / filename) != provenance["sha256"][filename]:
            raise ValueError(f"model artifact checksum mismatch: {filename}")
    return provenance
