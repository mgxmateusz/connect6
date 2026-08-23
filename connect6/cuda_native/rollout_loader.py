from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

from .loader import _bootstrap_msvc_environment


_ROLLOUT_EXTENSION = None


def load_native_rollout_extension(*, verbose: bool = False):
    """Build/load the SM120 GPU-native training rollout engine."""
    global _ROLLOUT_EXTENSION
    if _ROLLOUT_EXTENSION is not None:
        return _ROLLOUT_EXTENSION

    if not torch.cuda.is_available():
        raise RuntimeError("GPU-native rollout wymaga CUDA.")
    if CUDA_HOME is None:
        raise RuntimeError(
            "GPU-native rollout wymaga CUDA Toolkit/NVCC; sam wheel PyTorch CUDA "
            "nie wystarcza do kompilacji rozszerzenia."
        )

    if platform.system() == "Windows":
        _bootstrap_msvc_environment()

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (12, 0):
        raise RuntimeError(
            "Native rollout V1 jest strojony pod RTX 50 / SM120; "
            f"wykryto compute capability {major}.{minor}."
        )

    root = Path(__file__).resolve().parent
    sources = [
        str(root / "native_rollout.cpp"),
        str(root / "native_rollout_kernel.cu"),
    ]

    extension_name = "connect6_cuda_rollout_sm120_v1"
    local_app_data = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    build_directory = local_app_data / "connect6_native_build" / extension_name
    build_directory.mkdir(parents=True, exist_ok=True)

    lock_file = build_directory / "lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
            print(f"[ROLLOUT BUILD] Removed stale build lock: {lock_file}", flush=True)
        except OSError as exc:
            raise RuntimeError(
                f"Nie można usunąć starego locka CUDA: {lock_file}\n{exc}"
            ) from exc

    print(f"[ROLLOUT BUILD] extension: {extension_name}", flush=True)
    print(f"[ROLLOUT BUILD] build dir: {build_directory}", flush=True)
    print(
        "[ROLLOUT BUILD] CUDA Graph conditional loop: CNN+GroupNorm+game+V1/V2 bez host sync per move.",
        flush=True,
    )

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
                raise RuntimeError("Brak wewnętrznej konfiguracji MSVC/CUDA.")
            cuda_flags.extend([f"-ccbin={ccbin}", f"-I{cuda_msvc_include}"])
            ldflags.append(f"/LIBPATH:{runtime_lib}")

        print("[ROLLOUT BUILD] ENTER torch.utils.cpp_extension.load()", flush=True)
        _ROLLOUT_EXTENSION = load(
            name=extension_name,
            sources=sources,
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_flags,
            extra_ldflags=ldflags,
            with_cuda=True,
            verbose=verbose,
            build_directory=str(build_directory),
        )
        print("[ROLLOUT BUILD] EXIT torch.utils.cpp_extension.load()", flush=True)
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch

    return _ROLLOUT_EXTENSION
