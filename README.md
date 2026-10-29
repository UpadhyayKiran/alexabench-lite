# AlexaBench Lite: English-Hindi Speech & Assistant Evaluation

A reproducible local comparison of pretrained multilingual Whisper Tiny and
Base on **60 FLEURS test recordings**, with WER/CER, CUDA latency, script diagnostics
and automated tests. This small research-oriented engineering study exposes
useful English results and severe Hindi failures. It includes no model training,
fine-tuning, paid API or production deployment.

## Capabilities and architecture

- English (`en_us`) and Hindi (`hi_in`) human speech with pinned raw references.
- Whisper Tiny/Base through faster-whisper; CUDA float16 or CPU int8.
- Unicode-aware WER/CER, pooled counts, coverage, failures, latency and RTF.
- Paired-input validation, JSON/CSV exports and descriptive model differences.
- Separate deterministic assistant evaluation with ten authored scenarios.

```mermaid
flowchart LR
    A[Pinned FLEURS subset] --> B[Audio loading and integrity checks]
    B --> C[Whisper Tiny / Base inference]
    C --> D[Transcript normalization]
    D --> E[WER / CER evaluation]
    E --> F[JSON / CSV result export]
    F --> G[Paired comparative analysis]
```

Modules in `alexabench/`: `dataset.py` acquires bounded subsets and pinned models;
`asr.py` adapts Whisper; `text.py` defines metrics; `evaluate.py` runs experiments;
`diagnostics.py` flags script issues; `compare.py` validates and compares runs;
`hardware.py` records hardware; `assistant.py` scores rule-based tasks; `cli.py`
provides commands. There is no frontend or extra LLM inference feature.

## Experimental setup

| Component | Recorded configuration |
|---|---|
| Dataset | [Google FLEURS](https://huggingface.co/datasets/google/fleurs), test split, CC BY 4.0 |
| Dataset revision | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` |
| Selection | First 30 regular WAV members in each language's original archive order |
| Evaluation set | 30 English + 30 Hindi recordings, 653.16 seconds total |
| Tiny | [Systran/faster-whisper-tiny](https://huggingface.co/Systran/faster-whisper-tiny), MIT, `d90ca5fe260221311c53c58e660288d3deb8d356` |
| Base | [Systran/faster-whisper-base](https://huggingface.co/Systran/faster-whisper-base), MIT, `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66` |
| Hardware | Intel i7-14700HX; NVIDIA RTX 4060 Laptop GPU, 8188 MiB; driver 581.86 |
| Device / compute | Both: `cuda` / `float16`, explicitly requested, no CPU fallback |
| Environment | Windows, Python 3.11.9, faster-whisper 1.2.1, CTranslate2 4.8.4, PyAV 18.1.0, NumPy 2.4.6 |
| Decoding | Supplied language, transcription task, beam size 1, temperature 0, no VAD, no previous-text conditioning; four CPU threads |
| Run date | 29 October 2026 |

The **same ordered manifest, raw references and verified audio hashes** are used
for both models. Pairing is across models, not sentence alignment across
languages. Each language has 29 distinct sentence IDs across 30 recordings.
This convenience sample is not random or speaker-balanced.

`data/fleurs_60/manifest.json` records stable IDs, source archives/members, raw
references, sentence IDs, expected audio sample counts, audio SHA-256 and TSV
hashes. TSV quotation marks are literal text, avoiding unbalanced-quote parsing
errors in the source. Decoded sample counts are checked against the TSV before
inference. The downloader streams only the needed beginning of each archive,
then closes the connection. No full archive, Parquet dataset or training split
is downloaded. Audio and full TSVs remain local (about 42 MB study audio);
public manifests and results are lightweight.

FLEURS attribution: **Conneau et al. (2022), FLEURS: Few-shot Learning Evaluation
of Universal Representations of Speech**, Google ([paper](https://arxiv.org/abs/2205.12446)),
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See attribution
in [data/fleurs_60](data/fleurs_60/ATTRIBUTION.md) and
[data/fleurs](data/fleurs/ATTRIBUTION.md). Source references in manifests/results
retain that license. Code and authored assistant scenarios are MIT; model
weights are separately MIT and are not committed.

## Normalization, scoring and timing

Both models use unchanged `nfc-casefold-punctuation-space-v1`: NFC, Unicode case
folding, removal of ZWJ/ZWNJ/BOM, Unicode punctuation replaced with spaces
(including Hindi danda), and whitespace collapsed. Hindi vowel signs, nukta,
virama and combining marks are preserved. Digits, symbols and accents remain.
There is no number expansion, digit-script conversion, translation or transliteration.

WER uses Levenshtein distance on whitespace tokens; CER uses Unicode code points
excluding spaces, **not grapheme clusters**. Error counts are divided by reference
length. Aggregates pool counts and denominators rather than averaging utterance
scores. Insertions can validly push scores above 100%. Empty normalized references
are errors; empty normalized predictions are valid deletion results.

Failures retain stage/error, are excluded from metric denominators, and make the
ASR command exit nonzero. Always compare scores with coverage. Missing audio,
integrity failures and inference exceptions are distinct from a successfully
returned but unusable transcript.

Per-recording inference latency includes faster-whisper audio decoding and full
consumption of lazy segments. Model loading, separate audio inspection, scoring,
downloads and export are excluded. `evaluation_duration_seconds` measures the
whole scoring pass, including inspection/scoring, excluding model load/export.
Per-language item evaluation sums, reference counts and total inference time
are exported. RTF = inference seconds / audio seconds.

Runs are sequential, Tiny then Base, on one GPU, with identical sample order,
one pass per model and no explicit warm-up or discarded first sample. Load,
caching, power and decoding length affect latency. Supplied decoder language
is recorded; independent language identification is **not measured**. Fixed
decoding settings do not promise identical outputs across hardware/library versions.

## Actual comparative results

Generated from completed inference by `compare`: **60 unique recordings and
120 model-recording evaluations**, all successful, no CPU fallback.

| Model | Language | Samples | WER | CER | Mean latency / recording | Device |
|---|---|---:|---:|---:|---:|---|
| Whisper Tiny | English | 30 | 17.61% | 8.12% | 0.092 s | CUDA float16 |
| Whisper Base | English | 30 | 12.24% | 5.94% | 0.157 s | CUDA float16 |
| Whisper Tiny | Hindi | 30 | 150.44% | 130.31% | 0.160 s | CUDA float16 |
| Whisper Base | Hindi | 30 | 141.43% | 119.46% | 0.323 s | CUDA float16 |

| Model / language | Success / failure | Reference words | Reference characters | Inference sum | Item evaluation sum |
|---|---:|---:|---:|---:|---:|
| Tiny / English | 30 / 0 | 670 | 3,215 | 2.754 s | 2.882 s |
| Tiny / Hindi | 30 / 0 | 799 | 3,243 | 4.793 s | 4.997 s |
| Base / English | 30 / 0 | 670 | 3,215 | 4.700 s | 4.871 s |
| Base / Hindi | 30 / 0 | 799 | 3,243 | 9.687 s | 9.858 s |

Whole evaluation passes took **7.908 s Tiny / 14.777 s Base**, plus model loads
of 1.776 s / 1.894 s. Pooled WER is 89.86% / 82.51%, pooled CER 69.48% / 62.95%.
Reference lengths differ across languages; pooled scores are not language macro
averages. The 653.16 seconds of unique audio were processed once per model.

| Base minus Tiny | English | Hindi |
|---|---:|---:|
| Absolute WER difference | -5.37 percentage points | -9.01 percentage points |
| Relative WER error reduction | 30.51% | 5.99% |
| Absolute CER difference | -2.18 percentage points | -10.85 percentage points |
| Relative CER error reduction | 26.82% | 8.33% |
| Mean latency difference | +0.065 s | +0.163 s |
| Mean latency ratio | 1.71x | 2.02x |

Relative error reduction is `(Tiny error - Base error) / Tiny error`, undefined
for a zero Tiny baseline. Differences are descriptive: no significance test or
general model superiority is claimed.

### Failure analysis

English errors include phonetic substitutions, omissions and names such as
`Dunlap broadsides`. Hindi outputs often contain Latin-script approximation,
English phrasing, Arabic-script text or repetition. Tiny has 21 Latin-only Hindi
outputs and all 30 lack Devanagari letters. Base has 16 Latin-only outputs,
26 nonempty outputs without Devanagari, and two punctuation-only outputs that
normalize to empty and remain scored as deletions. No scored outputs are
translated, repaired, filtered or substituted.

The English-cue heuristic nominates 7 Tiny and 2 Base Hindi outputs for review;
these are candidates, not confirmed language-ID errors. The supplied language
token is not a language prediction. See [traceable examples and diagnostic
definitions](docs/error_analysis.md). Severe Hindi errors and repetitions remain
the main limitation, despite Base's lower aggregate error.

### Results and historical evidence

- [Comparison JSON](results/comparison_60/comparison.json),
  [CSV](results/comparison_60/comparison.csv),
  [differences](results/comparison_60/differences.csv),
  [generated table](results/comparison_60/table.md).
- Per-recording/per-language exports:
  [Tiny](results/comparison_60/tiny/asr.json) /
  [Base](results/comparison_60/base/asr.json), with CSV siblings.
- Original four-recording results are untouched:
  [CPU Tiny](results/real_cpu_tiny/asr.json) /
  [CUDA Tiny](results/real_cuda_tiny/asr.json). These recordings overlap the
  expanded subset; they are not four additional unique samples.
- [Historical README](docs/historical_baseline.md) preserves the original study
  snapshot; [artifact hashes](docs/historical_baseline_sha256.json) verify preservation.

All reported ASR values come from real inference. Synthetic audio/fake backends
exist only in temporary test fixtures and never supply benchmark evidence.

## Reproduce on Windows PowerShell

From a fresh clone, with Python 3.11 installed:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install --no-deps -e .
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m alexabench.cli download --per-language 30 --out data/fleurs_60
.venv\Scripts\python -m alexabench.cli download-model --size tiny
.venv\Scripts\python -m alexabench.cli download-model --size base
.venv\Scripts\python -m alexabench.cli asr --manifest data/fleurs_60/manifest.json --model .cache/models/whisper-tiny --device cuda --out results/reproduce_60/tiny
.venv\Scripts\python -m alexabench.cli asr --manifest data/fleurs_60/manifest.json --model .cache/models/whisper-base --device cuda --out results/reproduce_60/base
.venv\Scripts\python -m alexabench.cli compare --tiny results/reproduce_60/tiny/asr.json --base results/reproduce_60/base/asr.json --out results/reproduce_60
.venv\Scripts\python -m alexabench.cli assistant --out results/reproduce_assistant
```

To verify the downloaded data, models, recomputed metrics and original evidence
against the recorded study, and record the full test result:

```powershell
.venv\Scripts\python -m pytest -q --junitxml=.cache/pytest.xml
.venv\Scripts\python scripts/validate_study.py
```

The validator defaults to the archived `results/comparison_60` evidence. Use
`--results results/reproduce_60 --out results/reproduce_60/validation.json` to
validate a new CUDA run; timings/hypotheses may differ and are never copied from
the archived run. Validation recomputes metrics from saved outputs; it does not
rerun inference.

Each command must exit successfully before continuing. New output directories
preserve original evidence. `compare` rejects different IDs/order, raw references,
audio hashes, manifests, normalization, decoding, devices, compute types or
per-item success status. Models are pinned; cached artifacts are verified and
reused. Checksums are checked before inference. To regenerate historical audio,
use `download --per-language 2 --out data/reproduce_historical` and compare item
content (extended manifests now contain additional provenance fields).

Explicit `--device cuda` surfaces CUDA errors. `--device auto` reports selection
and fallback reasons, retrying on CPU for CUDA runtime failures. For CPU runs
change both ASR commands to `--device cpu` with distinct output paths; CPU uses
int8 and must not be presented as GPU results. A CUDA driver plus compatible
CUDA/cuDNN runtimes are required; see [faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu).
The RTX 4060 worked without additional GPU-library installation in this session.

Runtime dependencies remain faster-whisper and its dependencies; pytest is
optional for development. The lock records tested versions. PyAV decodes without
a separate FFmpeg command. Curl is preferred for downloads, with urllib fallback.
Acquisition requires internet; local inference/scoring require no API or paid
service. Audio/TSVs, weights, environments, secrets and caches are ignored by Git.
Public reference text retains FLEURS attribution and CC BY 4.0.

## Tests and separate assistant baseline

**38 tests passed**, including the original 13. Coverage includes English/Hindi
normalization, edit metrics/pooling, empty references/predictions, missing audio,
checksum failures, inference exceptions, deterministic 30-per-language selection,
literal-quote TSV parsing, paired IDs/content, model integrity/cache reuse,
CPU/CUDA/auto fallback reporting, JSON/CSV exports and assistant contracts.
See [validation report](results/validation/report.json). Tests require no downloads.

The existing [rule-based assistant baseline](results/assistant_baseline/assistant.json)
passes 10/10 authored cases (5 per language) for instruction following and declared
task completion. These text tasks cover timers, lights and clarification, separate
from FLEURS/ASR. They are **not an LLM inference benchmark**, do not execute real
device actions, and do not demonstrate generalization. See [scenario contracts](scenarios/README.md).
Existing external-response scoring remains available; no additional LLM features
were added.

## Limitations and future work

The sample is small, selected by archive order, includes repeated sentence IDs,
and does not establish population accuracy, fairness or significance. FLEURS is
read speech rather than assistant conversation; pretraining overlap is unknown.
Known-language decoding does not test language identification. Code-point CER
and strict script matching are deliberate but incomplete measures of intelligibility.
No speech-model training/fine-tuning, production deployment or multilingual model
generalization is demonstrated. The assistant baseline is rule-based. Results
may vary with hardware/software; single-pass timing and fixed model order do not
establish reliable throughput or latency percentiles.

Future work could expand independent speaker samples, noise and code switching,
repeated/counterbalanced timing, sentence-aware statistical analysis and held-out
decoder configurations. None is claimed as completed. This is an independent
portfolio study, not an Amazon/Alexa product or published scientific paper.
