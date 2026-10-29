Assistant contract
==================

The ten cases in `assistant.json` are original, MIT-licensed text scenarios,
five per language. They are independent text-input tasks, not FLEURS transcripts.
Each case starts with fresh state. There is no physical device control.

Give a candidate assistant/LLM this instruction and each scenario's `prompt`:

> Return exactly one JSON object, without Markdown or explanatory prose. Support
> timers with an explicit duration in minutes and switching a single light on or
> off. For timers return {"action":"set_timer","minutes":<integer>}; for lights
> return {"action":"light","state":"on"} or {"action":"light","state":"off"}.
> For ambiguous or unsupported requests return {"action":"clarify"}.

Save responses in a JSON object keyed by scenario ID. Values can be JSON objects
or raw response strings. Raw strings are preferable when testing a model's JSON
format adherence. Do not strip Markdown fences or repair invalid model output.
Missing IDs fail both checks. Extra response IDs are unused.

Task completion checks exact expected action and slot values with strict types;
extra keys are allowed for this score. Instruction following additionally
requires exactly the expected key set. These checks evaluate declared tool
requests, not execution against a real timer/light. Exact expected outputs are
in `assistant.json`, so these are transparent regression scenarios, not a hidden
generalization test. The rule baseline implements the same narrow grammar.

To evaluate a local LLM, generate responses outside this minimal package and run:

```powershell
.venv\Scripts\python -m alexabench.cli assistant --responses responses.json --out results/local_llm
```

Archive model revision, system instruction, decoding configuration, raw outputs,
hardware and measured generation latency alongside that run. External response
evaluation is offline scoring: it records no inferred model latency. No actual
LLM inference experiment is claimed by the included baseline results.
