# Error and failure analysis

This report describes the recorded CUDA float16 runs in `results/comparison_60`.
It is a qualitative inspection of selected outputs plus deterministic script
counts, not a calibrated language-ID study or causal explanation of model errors.
All 120 model-recording evaluations returned transcript strings without an
inference exception. A returned string can still be unusable ASR output.

## Quantitative diagnostics

| Hindi diagnostic | Tiny / 30 | Base / 30 |
|---|---:|---:|
| All output letters Latin | 21 | 16 |
| Nonempty normalized output without Devanagari letters | 30 | 26 |
| Empty after normalization | 0 | 2 |
| English-cue candidates for review | 7 | 2 |
| Model inference exceptions | 0 | 0 |

Counts overlap. Base's two remaining nonempty outputs contain Devanagari but
also show repetition or mixed script. Other outputs contain Arabic-script
letters, mixed Latin/Arabic text or, for some Tiny outputs, CJK repetition.
The two normalized-empty Base outputs are punctuation-only strings, not missing
audio, a missing return value or infrastructure failures. They receive 100%
WER/CER as deletions and remain in the 30-recording denominators.

`diagnostics.py` counts letters by Unicode name. A Latin-only flag does not
prove that the text is Hindi romanization or English. The English-cue heuristic
requires Latin-only letters and at least three distinct tokens from the small
documented function-word set in that module. It can flag romanized/mixed text
and miss English text; the 7/2 candidate counts are **not verified English-output
rates**. They nominate samples for inspection only. Normalization and scores
never use these flags to repair or discard output.

## Traceable examples

The following excerpts were inspected from saved hypotheses. Ellipses indicate
excerpts for readability, never changes to scored transcripts. Full source
references and outputs remain in the per-item JSON and CSV.

- **English names and phonetic substitutions:**
  `en_us_1038214857203833067` has reference `Through the night between 150 and
  200 copies were made, now known as "Dunlap broadsides".` Tiny ends with
  `the Net Broad types` (60.00% WER); Base ends with `the LAP Broad Sives`
  (33.33% WER). Base improves this case while retaining errors.
- **Latin-script approximation of Hindi:**
  `hi_in_10011266027513218401`, Tiny begins `Kochi Arangu mein Aarastir Kendra
  kota...` against a Devanagari reference beginning `कुछ अणुओं में अस्थिर केंद्रक`.
  Script mismatch alone prevents most exact matches; the output also contains
  transcription errors. No transliteration is applied.
- **Unexpected English-dominant output for Hindi:**
  `hi_in_10014385262415025488`, Tiny begins `Greenland was a big game when we
  were 20 years old.` Base begins `Greenland got very low place, North Sagaasam
  vkethi...`. The Hindi reference describes Greenland's sparse settlement and
  Erik the Red. These outputs mix English phrasing, names and errors; they are
  not faithful Hindi transcripts. Another clear English-dominant example is
  `hi_in_10972260827293009900`: Tiny begins `People can't believe that the
  government...`; Base begins `Lohia Numan cannot lie that the people...`.
- **Repetition and insertions:**
  `hi_in_10924000697500767031`, Tiny continues with many `nd` tokens and reaches
  446.67% WER. `hi_in_10135688561274120767`, Base repeats Arabic-script `ھ`
  and reaches 555.00% WER. Insertion counts make WER greater than 100% valid.
  These outputs are retained; no repetition-removal postprocessing is used.
- **Punctuation-only output:**
  `hi_in_10200643191519059428` and `hi_in_10924000697500767031`, Base returns
  repeated `۔` (Arabic full stop). Unicode punctuation normalization removes
  it, leaving an empty hypothesis. Both are scored as all deletions.

## Interpretation boundaries

The language argument is supplied from FLEURS for every recording. The exported
`decoder_language` is the requested decoder condition (`en` or `hi`), not an
independent language prediction. `language_identification` explicitly says
`not_measured_language_forced`. Thus we observe wrong-language/script output,
but cannot attribute it to a measured language-identification error.

We verified raw references against the pinned original TSV, sample counts
against decoded audio, and identical audio hashes/IDs/order across the two
models. Both model artifacts are pinned and hashed. No alternative beams,
prompts, scripts, sample substitutions or score-improving cleanup were tried.
No cause for the Hindi failure is proven by this experiment. A subsequent study
could investigate decoder settings, larger models, acoustic conditions and
token/script behavior on an independent held-out sample.

Archive order is a convenience selection with repeated sentence IDs: 29 unique
sentences in 30 recordings per language. Source gender labels are English
13 male / 17 female and Hindi 21 male / 9 female; no demographic-performance
claim is supported. The pooled score is also weighted by different reference
lengths (670 English versus 799 Hindi words), not equally by language.
