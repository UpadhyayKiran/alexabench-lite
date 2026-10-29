"""Deterministic JSON contract evaluator for assistant or LLM responses.

Gold scenarios are independent of the rule baseline. No network APIs or LLM judge.
Instruction score requires exactly the prescribed keys; task score compares actions
and slots, permitting extra keys. Values are exact and types must match.
"""
import json
import re
from time import perf_counter
from .text import normalize


def baseline(prompt):
    text = normalize(prompt)
    timer = re.fullmatch(r"(?:set a timer for|टाइमर लगाओ) (\d+) (?:minutes|मिनट)", text)
    if timer:
        return {"action": "set_timer", "minutes": int(timer[1])}
    if text in ("turn on the light", "लाइट चालू करो"):
        return {"action": "light", "state": "on"}
    if text in ("turn off the light", "लाइट बंद करो"):
        return {"action": "light", "state": "off"}
    return {"action": "clarify"}


def grade(expected, response):
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except (ValueError, TypeError):
            return dict(instruction_following=False, task_completed=False)
    task = isinstance(response, dict) and all(
        key in response and type(response[key]) is type(value) and response[key] == value
        for key, value in expected.items())
    return dict(instruction_following=bool(task and set(response) == set(expected)),
                task_completed=bool(task))


def evaluate_assistant(scenarios, responses=None):
    rows = []
    for scenario in scenarios:
        start = perf_counter()
        response = baseline(scenario["prompt"]) if responses is None else responses.get(scenario["id"])
        seconds = perf_counter() - start if responses is None else None
        rows.append(dict(id=scenario["id"], language=scenario["language"],
                         prompt=scenario["prompt"], expected=scenario["expected"], response=response,
                         inference_seconds=seconds, **grade(scenario["expected"], response)))
    def summary(items):
        return dict(total=len(items),
                    instruction_following=sum(r["instruction_following"] for r in items)/len(items),
                    task_completion=sum(r["task_completed"] for r in items)/len(items))
    return dict(kind="rule_baseline" if responses is None else "external_responses_unverified_provenance",
                items=rows, summary={"all": summary(rows), **{
                    lang: summary([r for r in rows if r["language"] == lang])
                    for lang in sorted({r["language"] for r in rows})}})
