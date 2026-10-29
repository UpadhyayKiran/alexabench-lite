"""Record hardware without adding NVML or deep-learning dependencies."""
import platform
import subprocess


def hardware_info():
    result = {"cpu": platform.processor(), "machine": platform.machine()}
    try:
        response = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10)
        result["nvidia_smi_inventory"] = response.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        result["nvidia_smi_unavailable"] = type(exc).__name__
    return result
