"""Report OS, interpreter, package imports and actual CUDA/MPS execution."""

import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[4] / "outputs" / ".matplotlib"))


def inspect_environment():
    """Check every required import and diagnose CUDA and Apple MPS support."""
    report = {"os": platform.platform(), "python": sys.version, "executable": sys.executable, "packages": {}}
    for name in ["torch", "torchvision", "timm", "sklearn", "pandas", "PIL", "matplotlib", "tqdm", "yaml"]:
        try:
            module = importlib.import_module(name)
            report["packages"][name] = str(getattr(module, "__version__", "import OK"))
        except Exception as exc:
            report["packages"][name] = f"ERROR: {exc}"
    if platform.system() != "Darwin":
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
            report["nvidia_smi"] = (result.stdout or result.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["nvidia_smi"] = str(exc)
    try:
        import torch
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if report["cuda_available"]:
            x = torch.ones(2, device="cuda")
            report["cuda_tensor_test"] = float((x + x).sum().item())
            report["gpu"] = torch.cuda.get_device_name(0)
        else:
            report["cuda_reason"] = "CPU-only PyTorch build" if torch.version.cuda is None else "CUDA build installed, but driver/device initialization unavailable; inspect nvidia-smi and torch.cuda.init()."
    except Exception as exc:
        report["cuda_error"] = str(exc)
    try:
        import torch
        backend = getattr(torch.backends, "mps", None)
        report["mps_built"] = backend is not None and backend.is_built()
        report["mps_available"] = backend is not None and backend.is_available()
        if report["mps_available"]:
            x = torch.ones(2, device="mps", requires_grad=True)
            loss = (x * x).sum()
            loss.backward()
            report["mps_tensor_test"] = float(loss.detach().cpu().item())
            report["mps_backward_test"] = x.grad.cpu().tolist()
        else:
            report["mps_reason"] = "MPS requires a supported Mac, macOS and an MPS-enabled PyTorch build."
        report["selected_device"] = "cuda" if report.get("cuda_available") else "mps" if report["mps_available"] else "cpu"
    except Exception as exc:
        report["mps_error"] = str(exc)
    return report


def main():
    """Print diagnostics and fail when required package imports fail."""
    report = inspect_environment()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(any(str(v).startswith("ERROR:") for v in report["packages"].values()) or "cuda_error" in report or "mps_error" in report)


if __name__ == "__main__":
    sys.exit(main())
