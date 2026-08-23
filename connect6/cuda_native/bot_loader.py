from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

from .loader import _bootstrap_msvc_environment


_BOT_EXTENSION = None


def load_native_bot_extension(*, verbose: bool = False):
    """Build/load the tiny CUDA extension used by the tactical bot.

    Unlike the championship extension this kernel does not depend on SM120
    tensor-core instructions, so it is compiled for the currently selected
    CUDA device capability.
    """
    global _BOT_EXTENSION
    if _BOT_EXTENSION is not None:
        return _BOT_EXTENSION

    if not torch.cuda.is_available():
        raise RuntimeError("GPU Tactical Bot requires CUDA.")
    if CUDA_HOME is None:
        raise RuntimeError(
            "GPU Tactical Bot requires CUDA Toolkit/NVCC. The PyTorch CUDA wheel "
            "alone is not enough to compile the native kernel."
        )

    if platform.system() == "Windows":
        _bootstrap_msvc_environment()

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"{major}.{minor}"
    arch_digits = f"{major}{minor}"

    root = Path(__file__).resolve().parent
    sources = [
        str(root / "native_bot.cpp"),
        str(root / "native_bot_kernel.cu"),
    ]

    extension_name = f"connect6_cuda_tactical_bot_sm{arch_digits}_v1"
    local_app_data = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    build_directory = local_app_data / "connect6_native_build" / extension_name
    build_directory.mkdir(parents=True, exist_ok=True)

    old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        is_windows = platform.system() == "Windows"
        cflags = ["/O2", "/std:c++17"] if is_windows else ["-O3", "-std=c++17"]
        cuda_flags = [
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "-std=c++17",
            f"-gencode=arch=compute_{arch_digits},code=sm_{arch_digits}",
        ]
        ldflags: list[str] = []

        if is_windows:
            ccbin = os.environ.get("CONNECT6_NVCC_CCBIN")
            cuda_msvc_include = os.environ.get("CONNECT6_NVCC_MSVC_INCLUDE")
            runtime_lib = os.environ.get("CONNECT6_MSVC_LIB_X64")
            if not ccbin or not cuda_msvc_include or not runtime_lib:
                raise RuntimeError("Missing internal MSVC/CUDA toolchain configuration.")
            cuda_flags.extend(
                [
                    f"-ccbin={ccbin}",
                    f"-I{cuda_msvc_include}",
                ]
            )
            ldflags.append(f"/LIBPATH:{runtime_lib}")

        _BOT_EXTENSION = load(
            name=extension_name,
            sources=sources,
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_flags,
            extra_ldflags=ldflags,
            with_cuda=True,
            verbose=verbose,
            build_directory=str(build_directory),
        )
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch

    return _BOT_EXTENSION
