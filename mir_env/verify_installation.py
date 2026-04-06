#!/usr/bin/env python3
"""
Verify that the MIR conda environment is correctly set up.

Usage:
    python -m mir_env.verify_installation
"""

import importlib
import platform
import shutil
import sys

# Minimum versions that matter — derived from environment.yml constraints.
# Packages without a min version just need to import successfully.
CHECKS = {
    "Core Scientific": [
        ("numpy",             "numpy",             "1.23.0", "2.0"),
        ("scipy",             "scipy",             "1.9.0",  None),
        ("scikit-learn",      "sklearn",           "1.3.0",  None),
        ("pandas",            "pandas",            "1.5.0",  None),
        ("numba",             "numba",             "0.57.0", None),
    ],
    "Audio": [
        ("librosa",           "librosa",           "0.10.0", None),
        ("soundfile",         "soundfile",         "0.12.0", None),
        ("audioread",         "audioread",         "3.0.0",  None),
        ("resampy",           "resampy",           "0.4.2",  None),
        ("pydub",             "pydub",             "0.25.1", None),
        ("audiomentations",   "audiomentations",   "0.33.0", None),
    ],
    "Deep Learning": [
        ("PyTorch",           "torch",             "2.0.0",  None),
        ("torchaudio",        "torchaudio",        "2.0.0",  None),
        ("PyTorch Lightning", "pytorch_lightning",  "2.1.0",  None),
    ],
    "MIR": [
        ("BeatNet",           "BeatNet",           None,     None),
        ("madmom",            "madmom",            None,     None),
        ("mir_eval",          "mir_eval",          "0.7",    None),
        ("mirdata",           "mirdata",           "1.0.0",  None),
        ("pretty_midi",       "pretty_midi",       None,     None),
        ("mido",              "mido",              None,     None),
    ],
    "Visualisation": [
        ("matplotlib",        "matplotlib",        "3.7.0",  None),
        ("seaborn",           "seaborn",           "0.12.0", None),
        ("plotly",            "plotly",             "5.14.0", None),
        ("TensorBoard",       "tensorboard",       "2.14.0", None),
    ],
    "Experiment Tracking": [
        ("wandb",             "wandb",             "0.24.0", None),
    ],
    "Utilities": [
        ("tqdm",              "tqdm",              "4.65.0", None),
        ("joblib",            "joblib",            "1.3.0",  None),
        ("PyYAML",            "yaml",              None,     None),
        ("rich",              "rich",              None,     None),
        ("click",             "click",             None,     None),
        ("DVC",               "dvc",               None,     None),
    ],
}

SEP = "=" * 70
failures = 0


def _parse_version(v):
    """Turn '1.23.0' into a comparable tuple of ints."""
    parts = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            break
    return tuple(parts)


def check_package(label, module_name, min_ver, max_ver):
    global failures
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"  \u2717 {label:<25} MISSING ({e!r})")
        failures += 1
        return

    version = getattr(mod, "__version__", None)
    ver_str = version or "?"
    issues = []

    if version and min_ver:
        if _parse_version(version) < _parse_version(min_ver):
            issues.append(f"need >={min_ver}")
    if version and max_ver:
        if _parse_version(version) >= _parse_version(max_ver):
            issues.append(f"need <{max_ver}")

    if issues:
        print(f"  \u2717 {label:<25} {ver_str}  ({', '.join(issues)})")
        failures += 1
    else:
        print(f"  \u2713 {label:<25} {ver_str}")


def check_tool(name, command):
    global failures
    if shutil.which(command):
        print(f"  \u2713 {name:<25} found in PATH")
    else:
        print(f"  \u2717 {name:<25} NOT in PATH")
        failures += 1


def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            cuda_ver = torch.version.cuda
            print(f"  \u2713 CUDA                      {cuda_ver}")
            print(f"    Device:                    {dev}")
            print(f"    Capability:                {cap[0]}.{cap[1]}")
            print(f"    VRAM:                      {mem:.1f} GB")
        else:
            print("  - CUDA                      not available (CPU-only is OK for laptop)")
    except Exception:
        print("  - CUDA                      could not query (torch not installed)")


def functional_tests():
    global failures
    print(f"\n{SEP}")
    print("Functional Tests")
    print(SEP)

    # librosa: MFCC on synthetic signal
    try:
        import numpy as np
        import librosa
        sr = 22050
        y = 0.5 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, sr, endpoint=False))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        print(f"  \u2713 librosa MFCC               shape {mfcc.shape}")
    except Exception as e:
        print(f"  \u2717 librosa MFCC               {e!r}")
        failures += 1

    # PyTorch: basic tensor ops
    try:
        import torch
        z = (torch.randn(3, 3) @ torch.randn(3, 3)).det()
        print(f"  \u2713 PyTorch tensor ops         OK (det={z.item():.4f})")
    except Exception as e:
        print(f"  \u2717 PyTorch tensor ops         {e!r}")
        failures += 1

    # PyTorch CUDA: move tensor to GPU
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.randn(2, 2, device="cuda")
            _ = x @ x.T
            print(f"  \u2713 CUDA tensor ops           OK")
        else:
            print(f"  - CUDA tensor ops           skipped (no GPU)")
    except Exception as e:
        print(f"  \u2717 CUDA tensor ops           {e!r}")
        failures += 1

    # madmom: DBN processor instantiation
    try:
        from madmom.features import DBNDownBeatTrackingProcessor
        proc = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4], fps=50)
        print(f"  \u2713 madmom DBN processor       OK")
    except Exception as e:
        print(f"  \u2717 madmom DBN processor       {e!r}")
        failures += 1

    # BeatNet: model instantiation
    try:
        from BeatNet.BeatNet import BeatNet
        estimator = BeatNet(1, mode="offline", inference_model="DBN", plot=[], thread=False, device="cpu")
        print(f"  \u2713 BeatNet model load         OK (model 1)")
    except Exception as e:
        print(f"  \u2717 BeatNet model load         {e!r}")
        failures += 1

    # mirdata: list datasets
    try:
        import mirdata
        datasets = mirdata.list_datasets()
        print(f"  \u2713 mirdata                    {len(datasets)} datasets available")
    except Exception as e:
        print(f"  \u2717 mirdata                    {e!r}")
        failures += 1

    # DVC: version check
    try:
        import dvc.version
        print(f"  \u2713 DVC                        {dvc.version.__version__}")
    except Exception as e:
        print(f"  \u2717 DVC                        {e!r}")
        failures += 1


def main():
    global failures

    print(SEP)
    print("MIR Environment Verification")
    print(SEP)
    print(f"  Python:    {sys.version.split()[0]} ({sys.executable})")
    print(f"  Platform:  {platform.platform()}")
    conda_prefix = sys.prefix.split("/")[-1] if "envs" in sys.prefix else "(base or not conda)"
    print(f"  Conda env: {conda_prefix}")

    for section, packages in CHECKS.items():
        print(f"\n[{section}]")
        for label, module_name, min_ver, max_ver in packages:
            check_package(label, module_name, min_ver, max_ver)

    print(f"\n[CLI Tools]")
    check_tool("FFmpeg", "ffmpeg")
    check_tool("git", "git")
    check_tool("dvc", "dvc")

    print(f"\n[GPU]")
    check_cuda()

    functional_tests()

    print(f"\n{SEP}")
    if failures == 0:
        print("All checks passed.")
    else:
        print(f"{failures} check(s) failed.")
    print(SEP)

    return failures


if __name__ == "__main__":
    sys.exit(main())
