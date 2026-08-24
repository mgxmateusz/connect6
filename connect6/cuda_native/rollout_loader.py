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

    # CUDA 12.9 + the legacy v143 host compiler used for NVCC can hit an MSVC
    # namespace collision while parsing torch/extension.h from a .cu translation
    # unit (compiled_autograd.h vs CUDAStream.h).  Python bindings remain in the
    # pure C++ file; the CUDA TU needs only Tensor/C++ API types.  Build a tiny
    # generated compatibility copy using torch/types.h, which is the upstream
    # recommended Windows workaround for this class of NVCC header failures.
    extension_name = "connect6_cuda_rollout_sm120_v2"
    local_app_data = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
    build_directory = local_app_data / "connect6_native_build" / extension_name
    build_directory.mkdir(parents=True, exist_ok=True)

    kernel_src = root / "native_rollout_kernel.cu"
    kernel_text = kernel_src.read_text(encoding="utf-8")
    old_include = "#include <torch/extension.h>"
    new_include = "#include <torch/types.h>"
    if old_include not in kernel_text:
        raise RuntimeError(
            "Nie znaleziono oczekiwanego include torch/extension.h w native_rollout_kernel.cu"
        )
    compat_text = kernel_text.replace(old_include, new_include, 1)
    compat_kernel = build_directory / "native_rollout_kernel_compat.cu"
    if not compat_kernel.exists() or compat_kernel.read_text(encoding="utf-8") != compat_text:
        compat_kernel.write_text(compat_text, encoding="utf-8")

    # The generated CUDA file still includes native_rollout_bot.cuh by name.
    # Add the real source directory to NVCC's include search path rather than
    # copying headers into the build cache.
    sources = [
        str(root / "native_rollout.cpp"),
        str(compat_kernel),
    ]

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
            "-DUSE_CUDA",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
            "-std=c++17",
            "-gencode=arch=compute_120,code=sm_120",
            f"-I{root}",
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
