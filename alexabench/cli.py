import argparse
import json
from pathlib import Path
from .evaluate import evaluate, write_results


def main():
    parser = argparse.ArgumentParser(description="AlexaBench Lite")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("download")
    data.add_argument("--per-language", type=int, default=2)
    data.add_argument("--out", default="data/fleurs")
    model = sub.add_parser("download-model")
    model.add_argument("--size", choices=["tiny", "base"], default="tiny")
    model.add_argument("--out")
    asr = sub.add_parser("asr")
    asr.add_argument("--manifest", default="data/fleurs/manifest.json")
    asr.add_argument("--model", default=".cache/models/whisper-tiny")
    asr.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    asr.add_argument("--out", default="results/real_cpu_tiny")
    assistant = sub.add_parser("assistant")
    assistant.add_argument("--scenarios", default="scenarios/assistant.json")
    assistant.add_argument("--responses", help="JSON object mapping scenario IDs to JSON responses")
    assistant.add_argument("--out", default="results/assistant_baseline")
    comparison = sub.add_parser("compare", help="Validate paired runs and export descriptive differences")
    comparison.add_argument("--tiny", required=True)
    comparison.add_argument("--base", required=True)
    comparison.add_argument("--out", default="results/comparison_60")
    args = parser.parse_args()
    if args.command == "download":
        from .dataset import download
        download(Path(args.out), args.per_language)
    elif args.command == "download-model":
        from .dataset import download_model
        download_model(Path(args.out or f".cache/models/whisper-{args.size}"), args.size)
    elif args.command == "compare":
        from .compare import compare, export_comparison
        result = compare(json.loads(Path(args.tiny).read_text(encoding="utf-8")),
                         json.loads(Path(args.base).read_text(encoding="utf-8")))
        export_comparison(Path(args.out), result)
    elif args.command == "asr":
        from .asr import WhisperASR
        provenance_path = Path(args.model) / "provenance.json"
        if provenance_path.is_file():
            from .dataset import verify_model
            verify_model(Path(args.model))
        device = args.device
        if device == "auto":
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        attempted_device = device
        fallback_reason = None
        if args.device == "auto" and device == "cpu":
            fallback_reason = "CTranslate2 reports no available CUDA device"
        try:
            backend = WhisperASR(args.model, device)
            result = evaluate(args.manifest, backend)
            if args.device == "auto" and device == "cuda":
                errors = [r.get("error", "") for r in result["items"]]
                if any(e.startswith("RuntimeError:") for e in errors):
                    raise RuntimeError("CUDA inference failed: " + "; ".join(filter(None, errors)))
        except RuntimeError as exc:
            if args.device != "auto" or device != "cuda":
                raise
            device, fallback_reason = "cpu", str(exc)
            backend = WhisperASR(args.model, device)
            result = evaluate(args.manifest, backend)
        result.update(kind="real_asr_experiment", attempted_device=attempted_device,
                      requested_device=args.device, fallback_reason=fallback_reason)
        result.update(model=args.model, device=device, compute_type=backend.compute_type,
                      model_load_seconds=backend.load_seconds,
                      decoding=dict(beam_size=1, temperature=0, language="provided", vad_filter=False,
                                    condition_on_previous_text=False, cpu_threads=4))
        from .evaluate import digest
        local_weights = Path(args.model) / "model.bin"
        if local_weights.is_file():
            result["model_sha256"] = digest(local_weights)
        provenance = Path(args.model) / "provenance.json"
        if provenance.is_file():
            result["model_provenance"] = json.loads(provenance.read_text(encoding="utf-8"))
        import importlib.metadata
        result["versions"] = {name: importlib.metadata.version(name) for name in
                              ("faster-whisper", "ctranslate2", "av", "numpy")}
        from .hardware import hardware_info
        result["hardware"] = hardware_info()
        write_results(args.out, "asr", result)
        print(json.dumps(result["summary"], indent=2))
        if result["summary"]["all"]["failed"]:
            raise SystemExit(1)
    else:
        from .assistant import evaluate_assistant
        scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
        responses = json.loads(Path(args.responses).read_text(encoding="utf-8")) if args.responses else None
        result = evaluate_assistant(scenarios, responses)
        from .evaluate import digest
        result["scenarios_sha256"] = digest(args.scenarios)
        if args.responses:
            result["responses_sha256"] = digest(args.responses)
        write_results(args.out, "assistant", result)
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
