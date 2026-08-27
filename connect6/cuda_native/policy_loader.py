from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

from .loader import _bootstrap_msvc_environment


_POLICY_EXTENSION = None


def load_native_policy_extension(*, verbose: bool = False):
    """Build/load dense-board V6 policy inference used by the fast bot gauntlet."""
    global _POLICY_EXTENSION
    if _POLICY_EXTENSION is not None:
        return _POLICY_EXTENSION

    if not torch.cuda.is_available():
        raise RuntimeError("Native policy requires CUDA.")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError(
            f"Native policy is tuned for SM120; got {torch.cuda.get_device_capability()}"
        )
    if CUDA_HOME is None:
        raise RuntimeError("Native policy requires CUDA Toolkit/NVCC.")
    if platform.system() == "Windows":
        _bootstrap_msvc_environment()

    root = Path(__file__).resolve().parent
    sources = [
        str(root / "native_policy.cpp"),
        str(root / "native_policy_kernel.cu"),
    ]
    extension_name = "connect6_cuda_policy_sm120_v1"
    local_app_data = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    build_directory = local_app_data / "connect6_native_build" / extension_name
    build_directory.mkdir(parents=True, exist_ok=True)

    lock_file = build_directory / "lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
        except OSError:
            pass

    old_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    try:
        is_windows = platform.system() == "Windows"
        cflags = ["/O2", "/std:c++17"] if is_windows else ["-O3", "-std=c++17"]
        cuda_flags = [
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
            "-std=c++17",
            "-gencode=arch=compute_120,code=sm_120",
        ]
        ldflags: list[str] = []
        if is_windows:
            ccbin = os.environ.get("CONNECT6_NVCC_CCBIN")
            cuda_msvc_include = os.environ.get("CONNECT6_NVCC_MSVC_INCLUDE")
            runtime_lib = os.environ.get("CONNECT6_MSVC_LIB_X64")
            if not ccbin or not cuda_msvc_include or not runtime_lib:
                raise RuntimeError("Missing MSVC/CUDA toolchain configuration.")
            cuda_flags.extend([f"-ccbin={ccbin}", f"-I{cuda_msvc_include}"])
            ldflags.append(f"/LIBPATH:{runtime_lib}")

        print(f"[POLICY BUILD] extension: {extension_name}", flush=True)
        print(f"[POLICY BUILD] build dir: {build_directory}", flush=True)
        print("[POLICY BUILD] ENTER torch.utils.cpp_extension.load()", flush=True)
        _POLICY_EXTENSION = load(
            name=extension_name,
            sources=sources,
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_flags,
            extra_ldflags=ldflags,
            with_cuda=True,
            verbose=verbose,
            build_directory=str(build_directory),
        )
        print("[POLICY BUILD] EXIT torch.utils.cpp_extension.load()", flush=True)
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch

    return _POLICY_EXTENSION
