"""Lazy faster-whisper adapter; timers include consumption of lazy segments."""
from time import perf_counter


class WhisperASR:
    def __init__(self, model_path: str, device: str = "cpu"):
        from faster_whisper import WhisperModel
        self.device = device
        self.compute_type = "int8" if device == "cpu" else "float16"
        start = perf_counter()
        self.model = WhisperModel(model_path, device=device,
                                  compute_type=self.compute_type, cpu_threads=4)
        self.load_seconds = perf_counter() - start

    def transcribe(self, path: str, language: str) -> str:
        segments, info = self.model.transcribe(
            path, language=language, task="transcribe", beam_size=1,
            temperature=0, condition_on_previous_text=False, vad_filter=False)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        self.last_metadata = {"decoder_language": info.language,
                              "language_identification": "not_measured_language_forced"}
        return text
