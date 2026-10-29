# AlexaBench Lite: English-Hindi Speech & Assistant Evaluation

A small Python 3.11 research benchmark for local multilingual Whisper ASR,
Unicode-aware error analysis, latency measurement and deterministic assistant
output evaluation. No paid APIs, custom speech-model training or frontend.

## Reproduce

From this directory, in PowerShell (Python 3.11 required):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install --no-deps -e .
.venv\Scripts\python -m pytest -q
# The four real recordings and manifest are included; this regenerates them:
.venv\Scripts\python -m alexabench.cli download --per-language 2
.venv\Scripts\python -m alexabench.cli download-model
.venv\Scripts\python -m alexabench.cli asr --device cpu --out results/reproduce_cpu
.venv\Scripts\python -m alexabench.cli asr --device cuda --out results/reproduce_cuda
.venv\Scripts\python -m alexabench.cli assistant --out results/reproduce_assistant
```

On Linux use `python3.11` and `.venv/bin/python`. The lock captures the tested
Windows environment; wheel availability can differ on other platforms. For a
minimal fresh install use `pip install -e ".[test]"` instead of the lock.
Runtime dependencies are faster-whisper and its dependencies; metrics,
downloading and exports use the standard library. PyAV bundles audio decoding,
so no separate FFmpeg command is required. Curl is used when installed, with
urllib as a fallback. Downloads require internet; inference on the downloaded
local model and scoring run offline. The tiny model download is about 75 MB;
only the beginning of each audio archive is streamed, then the connection closes.
No full dataset or Parquet files are downloaded.

`--device auto` selects CUDA when available and retries the experiment on CPU
if CUDA raises a runtime error; the reason and attempted device are exported.
Explicit `--device cuda` surfaces errors rather than changing the requested
experiment. CUDA requires compatible CTranslate2 CUDA/cuDNN runtime libraries;
see [faster-whisper's installation instructions](https://github.com/SYSTRAN/faster-whisper#gpu).
CPU int8 requires no CUDA installation. Tiny comfortably fits an RTX 4060 8 GB.
Other multilingual Whisper sizes can be passed through `--model small`, or as
a local CTranslate2 model directory; record/pin their revisions separately.
Do not use English-only `.en` models for the Hindi experiment.

## Data and model provenance

The included four **real human speech** recordings come from Google's
[FLEURS](https://huggingface.co/datasets/google/fleurs) test split (`en_us`,
`hi_in`), [Conneau et al., 2022](https://arxiv.org/abs/2205.12446), under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution and
source links are in `data/fleurs/ATTRIBUTION.md`. The pinned dataset revision is
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`.

The selection rule, chosen before inference, is the first two regular WAV
members in each original test archive, in archive order. This is a convenience
sample, not random sampling. References use the original raw transcription TSV
column (index 2); unchanged TSVs are included. `manifest.json` records the source
URL, member, sentence ID, language and SHA-256 of each recording. The four audio
files total approximately 2.7 MB; recordings plus both TSVs occupy about 3.5 MB.
No training data is used and no tuning is performed on these samples.

The model is [Systran/faster-whisper-tiny](https://huggingface.co/Systran/faster-whisper-tiny),
the multilingual Whisper tiny conversion, under MIT, pinned to
`d90ca5fe260221311c53c58e660288d3deb8d356`. `download-model` records file hashes
and provenance under `.cache/models/whisper-tiny/`; result JSON includes the
weight hash and provenance. Weights are downloaded, not included in the project.
Project code and original assistant scenarios use MIT (see `LICENSE`).

## Metric and timing protocol

Normalization version `nfc-casefold-punctuation-space-v1` applies NFC, Unicode
case folding, removes ZWJ/ZWNJ/BOM, replaces all Unicode punctuation (including
Hindi danda) with spaces and collapses whitespace. It preserves Hindi vowel
signs, virama, nukta, diacritics, digits and symbols. It does not transliterate,
convert digit scripts, expand numbers or equate spelling variants. Thus `25`
and `twenty five`, and Hindi versus Latin-script output, remain different.

WER is Levenshtein distance on whitespace-separated tokens divided by reference
word count. CER is distance on Unicode code points after removing spaces,
divided by reference code-point count; it is not grapheme-cluster CER. Hindi
combining marks therefore count separately. Insertions can make either score
exceed 100%. Empty normalized references are errors, while an empty hypothesis
against a nonempty reference is a valid all-deletion result.

Aggregates pool edit counts and reference lengths (micro averaging), rather
than averaging utterance rates. `all`, `en` and `hi` each include coverage and
failure counts. Failed audio/inference items retain their error and are excluded
from metric denominators; a run with any failed item exits nonzero. Compare
accuracy only alongside coverage. Initialization errors abort before scoring.

Each item measures wall-clock inference with `perf_counter`, including audio
decoding inside faster-whisper and full consumption of its lazy segments.
The separate audio-duration inspection, model loading, downloads and exports
are excluded. Model load time is exported separately. RTF is total successful
inference time / total successful audio duration. The first item is retained;
there is no explicit warm-up. A single sequential pass is recorded per output
directory. OS/CUDA caching, hardware load and quantization affect timing; these
small runs do not establish reliable throughput or latency percentiles.

Decoding uses supplied language labels, transcription task, greedy beam size 1,
temperature 0, no VAD, no previous-text conditioning and four CPU threads.
This evaluates known-language ASR, not language identification. Fixed decoding
does not guarantee bit-identical outputs across devices/library versions.

## Actual results

Recorded 29 October 2026 with Python 3.11.9 on Windows, Intel i7-14700HX,
RTX 4060 Laptop GPU 8 GB, NVIDIA driver 581.86. Both runs completed all four
recordings (42.24 seconds) without fallback or failures. Versions and exact
per-item references, hypotheses, errors and timings are in `results/`.

| Run | Language | Clips | WER | CER | Mean inference/clip | RTF |
|---|---|---:|---:|---:|---:|---:|
| CPU int8 | English | 2 | 15.00% | 4.14% | 0.287 s | 0.0297 |
| CPU int8 | Hindi | 2 | 100.00% | 102.78% | 0.291 s | 0.0254 |
| CPU int8 | All | 4 | 67.31% | 63.18% | 0.289 s | 0.0274 |
| CUDA float16 | English | 2 | 15.00% | 4.14% | 0.259 s | 0.0268 |
| CUDA float16 | Hindi | 2 | 100.00% | 103.17% | 0.124 s | 0.0108 |
| CUDA float16 | All | 4 | 67.31% | 63.42% | 0.191 s | 0.0181 |

These are genuine inference results, including the poor Hindi performance:
tiny produced Latin-script approximations and English text despite the supplied
Hindi language token. No transliteration repair was applied. CPU English has
6 edits / 40 reference words; Hindi has 64 / 64. Quantization changes some Hindi
characters, hence the CER difference. This is evidence about these samples,
not an estimate of Whisper's overall multilingual accuracy.

The deterministic **rule baseline**, evaluated separately on ten authored
assistant scenarios, scored 10/10 instruction following and 10/10 declared task
completion (5/5 each language). This is a grammar regression result, **not an
LLM benchmark result**. It does not measure speech-to-task performance. Scenario
contracts, expected JSON actions, scoring rules and external local-LLM response
evaluation instructions are in `scenarios/README.md`. No actual LLM inference
or LLM latency experiment is claimed. Baseline dispatch timings are exported;
external-response generation latency is unknown and exported as null.

Each run exports per-item CSV, aggregate/per-language summary CSV and complete
JSON. Recorded runs: `results/real_cpu_tiny`, `results/real_cuda_tiny` and
`results/assistant_baseline`. The 13 pytest tests cover normalization, edit
counts, pooling, empty references/hypotheses, missing audio, checksum mismatch,
failed inference, assistant contracts and Unicode-preserving exports. Synthetic
audio and fake backends exist only in temporary test fixtures; they are never
used for the reported experiments.

## Scope and limitations

Four clips cannot support significance claims, population confidence intervals,
fairness conclusions or model rankings. Archive-order selection can be biased.
FLEURS is read speech rather than conversational assistant audio; Whisper's
pretraining overlap is unknown. Hindi script errors are intentionally visible.
Assistant cases cover only timers, lights and clarification; declared JSON
actions do not prove a task was executed. More representative held-out speakers,
noise, accents, code switching, paraphrases, repeated timing passes and a pinned
local LLM are sensible subsequent experiments. No improvements from them are
claimed here. AlexaBench Lite is an independent educational/research project,
not an Amazon or Alexa product.

Modules: `dataset.py` (bounded acquisition), `asr.py` (model adapter), `text.py`
(normalization/metrics), `evaluate.py` (runner/export), `assistant.py` (contracts),
`cli.py` (commands). There is no frontend.
