from __future__ import annotations

import os
import platform
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


_EXTENSION = None


def _require_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Native championship wymaga CUDA.")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise RuntimeError(
            f"Native championship jest zoptymalizowany pod SM120/RTX 50; wykryto compute capability {major}.{minor}."
        )
    if CUDA_HOME is None:
        raise RuntimeError(
            "Nie znaleziono CUDA Toolkit/NVCC. Zainstaluj CUDA Toolkit 12.8+; sam wheel PyTorch CUDA nie wystarcza do kompilacji rozszerzenia."
        )


def load_native_championship_extension(*, verbose: bool = True):
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    _require_environment()
    root = Path(__file__).resolve().parent
    sources = [
        str(root / "native_championship.cpp"),
        str(root / "native_championship_kernel.cu"),
    ]

    # PyTorch cache'uje wynik kompilacji, więc NVCC uruchamia się tylko po zmianie
    # źródeł/wersji. Wymuszamy SM120 zamiast generować zbędne warianty PTX/SASS.
    old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    try:
        cflags = ["/O2", "/std:c++17"] if platform.system() == "Windows" else ["-O3", "-std=c++17"]
        cuda_flags = [
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
            "-std=c++17",
            "-gencode=arch=compute_120,code=sm_120",
        ]
        _EXTENSION = load(
            name="connect6_cuda_championship_sm120_v3",
            sources=sources,
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_flags,
            with_cuda=True,
            verbose=verbose,
        )
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch

    return _EXTENSION
