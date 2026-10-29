# FLEURS comparative subset

References in `manifest.json` and the downloaded human speech recordings are
from **FLEURS: Few-shot Learning Evaluation of Universal Representations of
Speech**, Conneau et al., 2022, released by Google.

- Dataset: https://huggingface.co/datasets/google/fleurs
- Paper: https://arxiv.org/abs/2205.12446
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- Dataset revision: `70bb2e84b976b7e960aa89f1c648e09c59f894dd`

Selection is the first 30 regular WAV members per language, in each original
test archive's order. English `en_us` and Hindi `hi_in` each have 30 recordings
and 29 distinct sentence IDs. This is a convenience subset, not 60 independent
sentences or a sentence-aligned English-Hindi translation test. The paired
comparison is across models on the same recordings.

The manifest records unmodified raw TSV reference text, archive member, source
sentence ID, original gender label, expected sample count and audio SHA-256.
TSV hashes are also recorded. Waveforms are not edited. Text normalization is
applied only when calculating metrics. Model hypotheses and normalized source
references in result files are evaluation derivatives; FLEURS attribution and
CC BY 4.0 remain applicable to the source material.

Audio and full TSVs are excluded from Git; recreate them with:

```powershell
.venv\Scripts\python -m alexabench.cli download --per-language 30 --out data/fleurs_60
```

The downloader stops after the selected members rather than retrieving entire
archives. Data remains local and public manifests/results are lightweight.
