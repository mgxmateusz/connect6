from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


_EXTENSION = None


def _find_vcvars64_candidates() -> list[Path]:
    """Find all vcvars64.bat installations visible to Visual Studio Installer."""
    candidates: list[Path] = []
    pf86 = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"

    if vswhere.is_file():
        try:
            proc = subprocess.run(
                [
                    str(vswhere),
                    "-all",
                    "-products",
                    "*",
                    "-property",
                    "installationPath",
                ],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=10,
            )
            for raw in proc.stdout.splitlines():
                install = raw.strip()
                if install:
                    candidates.append(
                        Path(install) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                    )
        except OSError:
            pass

    roots = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio",
        Path(pf86) / "Microsoft Visual Studio",
    ]
    editions = ("BuildTools", "Community", "Professional", "Enterprise")
    for root in roots:
        for version in ("18", "2026", "2022", "2019"):
            for edition in editions:
                candidates.append(
                    root / version / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                )

    result: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            result.append(path)
    return result


def _installed_msvc_toolsets(vcvars: Path) -> list[tuple[tuple[int, int], str]]:
    """Return installed toolset directories, newest first."""
    vc_root = vcvars.parents[2]
    toolsets_root = vc_root / "Tools" / "MSVC"
    if not toolsets_root.is_dir():
        return []

    out: list[tuple[tuple[int, int], str]] = []
    for child in toolsets_root.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"^(\d+)\.(\d+)(?:\.|$)", child.name)
        if not m:
            continue
        out.append(((int(m.group(1)), int(m.group(2))), child.name))
    out.sort(reverse=True)
    return out


def _find_x64_tool_bin(toolset_root: Path) -> Path | None:
    """Find a host compiler capable of producing x64 code."""
    candidates = [
        toolset_root / "bin" / "Hostx64" / "x64",
        toolset_root / "bin" / "HostX64" / "x64",
        toolset_root / "bin" / "Hostx86" / "x64",
        toolset_root / "bin" / "HostX86" / "x64",
    ]
    seen: set[str] = set()
    for bin_dir in candidates:
        key = str(bin_dir).lower()
        if key in seen:
            continue
        seen.add(key)
        if (bin_dir / "cl.exe").is_file():
            return bin_dir
    return None


def _select_cuda_host_toolchain() -> tuple[Path, str, str, Path, Path]:
    """Select newest installed pre-v145 MSVC compiler usable by CUDA 12.9.

    Visual Studio 2026 can install legacy v143 compiler/header components without
    duplicating the old runtime libraries. That is fine: NVCC only needs the
    compatible host compiler and matching headers. Final C++ compilation and
    linking use the current complete MSVC toolset initialized by vcvars64.bat.

    Returns (vcvars64.bat, selector, full_version, toolset_root, bin_dir).
    """
    installations = _find_vcvars64_candidates()
    if not installations:
        raise RuntimeError(
            "Nie znaleziono Visual Studio Build Tools. Zainstaluj workload "
            "'Programowanie aplikacji klasycznych w języku C++'."
        )

    compatible: list[tuple[tuple[int, int], str, Path, Path, Path]] = []
    detected: list[str] = []
    incomplete: list[str] = []

    for vcvars in installations:
        vc_root = vcvars.parents[2]
        toolsets = _installed_msvc_toolsets(vcvars)
        if toolsets:
            detected.append(f"{vc_root}: " + ", ".join(v for _, v in toolsets))

        for key, full_version in toolsets:
            # CUDA 12.9 rejects VS2026/v145 (_MSC_VER >= 1950). Any installed
            # VS2022-era v143 compiler below 14.50 is a candidate.
            if not ((14, 10) <= key < (14, 50)):
                continue

            toolset_root = vc_root / "Tools" / "MSVC" / full_version
            bin_dir = _find_x64_tool_bin(toolset_root)
            include_dir = toolset_root / "include"
            missing: list[str] = []
            if bin_dir is None:
                missing.append("cl.exe target x64")
            if not (include_dir / "vector").is_file():
                missing.append("include\\vector")
            if not (include_dir / "yvals_core.h").is_file():
                missing.append("include\\yvals_core.h")
            if missing:
                incomplete.append(f"{full_version}: brak: " + ", ".join(missing))
                continue

            assert bin_dir is not None
            compatible.append((key, full_version, vcvars, toolset_root, bin_dir))

    if not compatible:
        details = "\n".join(detected) if detected else "(nie wykryto katalogów MSVC)"
        incomplete_details = (
            "\n".join(incomplete) if incomplete else "(brak kompatybilnych katalogów v143)"
        )
        raise RuntimeError(
            "Nie znaleziono kompilatora MSVC zgodnego z CUDA 12.9.\n"
            "Potrzebny jest v143 / VS 2022 C++ x64/x86 build tools z cl.exe i nagłówkami.\n"
            f"Wykryte toolsety:\n{details}\n"
            f"Niekompletne kompatybilne katalogi:\n{incomplete_details}"
        )

    compatible.sort(reverse=True, key=lambda x: (x[0], x[1]))
    key, full_version, vcvars, toolset_root, bin_dir = compatible[0]
    selector = f"{key[0]}.{key[1]}"
    return vcvars, selector, full_version, toolset_root, bin_dir


def _run_vcvars_and_capture_environment(
    vcvars: Path, vcvars_ver: str | None = None
) -> dict[str, str]:
    marker = "__CONNECT6_ENV_BEGIN__"
    args = f" -vcvars_ver={vcvars_ver}" if vcvars_ver else ""
    script = (
        "@echo off\r\n"
        f'call "{vcvars}"{args}\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        f"echo {marker}\r\n"
        "set\r\n"
    )

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".cmd",
            delete=False,
            encoding="utf-8",
            newline="",
        ) as f:
            f.write(script)
            temp_path = Path(f.name)

        proc = subprocess.run(
            ["cmd.exe", "/d", "/c", str(temp_path)],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )
    except OSError as exc:
        raise RuntimeError(f"Nie udało się uruchomić cmd.exe dla MSVC: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        selector_text = f" (-vcvars_ver={vcvars_ver})" if vcvars_ver else ""
        raise RuntimeError(
            f"vcvars64.bat{selector_text} zwrócił błąd:\n"
            + (combined if combined else "(brak komunikatu)")
        )

    lines = (proc.stdout or "").splitlines()
    try:
        start = lines.index(marker) + 1
    except ValueError as exc:
        raise RuntimeError(f"Nie udało się odczytać środowiska MSVC. Output:\n{combined}") from exc

    env: dict[str, str] = {}
    for line in lines[start:]:
        if line and not line.startswith("=") and "=" in line:
            key, value = line.split("=", 1)
            env[key] = value
    return env


def _prepend_env_path(variable: str, path: Path) -> None:
    current = os.environ.get(variable, "")
    item = str(path)
    parts = current.split(os.pathsep) if current else []
    if item.lower() not in {p.lower() for p in parts}:
        os.environ[variable] = item + (os.pathsep + current if current else "")


def _msvc_toolset_root_from_cl(cl_path: Path) -> Path:
    """Return .../VC/Tools/MSVC/<version> from .../bin/Host*/x64/cl.exe."""
    resolved = cl_path.resolve()
    try:
        root = resolved.parents[3]
    except IndexError as exc:
        raise RuntimeError(f"Nieprawidłowa ścieżka cl.exe: {resolved}") from exc
    if root.parent.name.lower() != "msvc":
        raise RuntimeError(f"Nie rozpoznano katalogu toolsetu MSVC dla: {resolved}")
    return root


def _bootstrap_msvc_environment() -> None:
    if platform.system() != "Windows":
        return

    vcvars, selector, cuda_host_version, cuda_host_root, cuda_host_bin = (
        _select_cuda_host_toolchain()
    )
    cuda_host_include = cuda_host_root / "include"

    # Important split for VS2026 + CUDA 12.9:
    # - initialize the CURRENT/default MSVC toolset (v145) for normal C++, link,
    #   Windows SDK and runtime libraries;
    # - pass the legacy v143 compiler explicitly to NVCC via -ccbin, together
    #   with its matching STL headers. CUDA validates only that host compiler.
    # This also makes PyTorch's Windows ninja linker select the current link.exe,
    # because it derives link.exe from the first cl.exe found in PATH.
    env = _run_vcvars_and_capture_environment(vcvars, None)
    os.environ.update(env)

    runtime_cl = shutil.which("cl.exe")
    runtime_link = shutil.which("link.exe")
    if not runtime_cl or not runtime_link:
        raise RuntimeError(
            "Domyślne vcvars64.bat nie wystawiło cl.exe/link.exe do PATH.\n"
            f"vcvars={vcvars}\nPATH={os.environ.get('PATH', '')}"
        )

    runtime_root = _msvc_toolset_root_from_cl(Path(runtime_cl))
    runtime_lib = runtime_root / "lib" / "x64"
    required_runtime = [
        runtime_lib / "msvcprt.lib",
        runtime_lib / "msvcrt.lib",
        runtime_lib / "vcruntime.lib",
    ]
    missing_runtime = [p.name for p in required_runtime if not p.is_file()]
    if missing_runtime:
        raise RuntimeError(
            "Domyślny toolset MSVC nie ma bibliotek runtime x64.\n"
            f"Toolset: {runtime_root}\n"
            f"lib x64: {runtime_lib} (exists={runtime_lib.is_dir()})\n"
            f"Brak: {', '.join(missing_runtime)}"
        )

    # Ensure the current runtime libraries are explicit even if vcvars ordering
    # changes in a future VS update.
    _prepend_env_path("LIB", runtime_lib)
    _prepend_env_path("LIBPATH", runtime_lib)

    os.environ["CONNECT6_NVCC_CCBIN"] = str(cuda_host_bin)
    os.environ["CONNECT6_NVCC_MSVC_INCLUDE"] = str(cuda_host_include)
    os.environ["CONNECT6_MSVC_LIB_X64"] = str(runtime_lib)
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"

    cuda_cl = cuda_host_bin / "cl.exe"
    if not cuda_cl.is_file():
        raise RuntimeError(f"Zniknął wybrany host compiler CUDA: {cuda_cl}")

    print(f"[NATIVE BUILD] C++/link MSVC: {runtime_root.name}")
    print(f"[NATIVE BUILD] C++ cl.exe: {runtime_cl}")
    print(f"[NATIVE BUILD] link.exe: {runtime_link}")
    print(
        f"[NATIVE BUILD] CUDA host MSVC: {selector} ({cuda_host_version}) "
        f"| {cuda_cl}"
    )
    print(f"[NATIVE BUILD] CUDA host include: {cuda_host_include}")
    print(f"[NATIVE BUILD] MSVC runtime lib x64: {runtime_lib}")


def _require_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Native championship wymaga CUDA.")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise RuntimeError(
            f"Native championship jest zoptymalizowany pod SM120/RTX 50; "
            f"wykryto compute capability {major}.{minor}."
        )
    if CUDA_HOME is None:
        raise RuntimeError(
            "Nie znaleziono CUDA Toolkit/NVCC. Zainstaluj CUDA Toolkit 12.8+; "
            "sam wheel PyTorch CUDA nie wystarcza do kompilacji rozszerzenia."
        )
    _bootstrap_msvc_environment()


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
        ]
        ldflags: list[str] = []

        if is_windows:
            ccbin = os.environ.get("CONNECT6_NVCC_CCBIN")
            cuda_msvc_include = os.environ.get("CONNECT6_NVCC_MSVC_INCLUDE")
            runtime_lib = os.environ.get("CONNECT6_MSVC_LIB_X64")
            if not ccbin or not cuda_msvc_include or not runtime_lib:
                raise RuntimeError("Brak wewnętrznej konfiguracji mieszanego toolchainu MSVC.")

            # torch.utils.cpp_extension quotes Windows arguments containing
            # spaces, so keep each NVCC option as one argument.
            cuda_flags.extend(
                [
                    f"-ccbin={ccbin}",
                    f"-I{cuda_msvc_include}",
                ]
            )
            ldflags.append(f"/LIBPATH:{runtime_lib}")

        _EXTENSION = load(
            name="connect6_cuda_championship_sm120_v12",
            sources=sources,
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_flags,
            extra_ldflags=ldflags,
            with_cuda=True,
            verbose=verbose,
        )
    finally:
        if old_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch

    return _EXTENSION
