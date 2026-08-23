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
        if (bin_dir / "cl.exe").is_file() and (bin_dir / "link.exe").is_file():
            return bin_dir
    return None


def _select_windows_toolchain() -> tuple[Path, str, str, Path, Path]:
    """Select newest COMPLETE MSVC toolset compatible with CUDA 12.9.

    A directory under VC/Tools/MSVC is not enough: optional v143 libraries can
    create a version directory without cl.exe. We therefore only rank toolsets
    that actually contain cl.exe + link.exe capable of targeting x64.

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
            if not ((14, 10) <= key < (14, 50)):
                continue

            toolset_root = vc_root / "Tools" / "MSVC" / full_version
            bin_dir = _find_x64_tool_bin(toolset_root)
            if bin_dir is None:
                incomplete.append(f"{full_version}: brak cl.exe/link.exe target x64")
                continue

            compatible.append((key, full_version, vcvars, toolset_root, bin_dir))

    if not compatible:
        details = "\n".join(detected) if detected else "(nie wykryto katalogów MSVC)"
        incomplete_details = (
            "\n".join(incomplete) if incomplete else "(brak kompatybilnych katalogów v143)"
        )
        raise RuntimeError(
            "Nie znaleziono kompletnego toolsetu MSVC zgodnego z CUDA 12.9.\n"
            "Potrzebny jest v143 / VS 2022 C++ x64/x86 build tools z cl.exe i link.exe.\n"
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
        raise RuntimeError(
            f"vcvars64.bat (-vcvars_ver={vcvars_ver}) zwrócił błąd:\n"
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


def _prepend_path(path: Path) -> None:
    current = os.environ.get("PATH", "")
    item = str(path)
    parts = current.split(os.pathsep) if current else []
    if item.lower() not in {p.lower() for p in parts}:
        os.environ["PATH"] = item + (os.pathsep + current if current else "")


def _bootstrap_msvc_environment() -> None:
    if platform.system() != "Windows":
        return

    vcvars, selector, full_version, toolset_root, bin_dir = _select_windows_toolchain()

    # vcvars supplies Windows SDK/CRT include+lib paths. Then force the exact
    # complete v143 compiler directory to the front of PATH so VS2026/v145
    # cannot win host-compiler resolution.
    env = _run_vcvars_and_capture_environment(vcvars, selector)
    os.environ.update(env)
    _prepend_path(bin_dir)

    cl_path = bin_dir / "cl.exe"
    link_path = bin_dir / "link.exe"

    os.environ["VCToolsInstallDir"] = str(toolset_root) + os.sep
    os.environ["VCToolsVersion"] = full_version
    os.environ["DISTUTILS_USE_SDK"] = "1"
    os.environ["MSSdk"] = "1"

    cl = shutil.which("cl.exe")
    link = shutil.which("link.exe")
    if not cl or not link:
        raise RuntimeError(
            "Wewnętrzny błąd konfiguracji PATH: wybrane binarki istnieją, ale system ich nie widzi.\n"
            f"cl={cl_path}\nlink={link_path}\nPATH={os.environ.get('PATH', '')}"
        )

    selected_root = str(toolset_root.resolve()).lower()
    if not str(Path(cl).resolve()).lower().startswith(selected_root):
        raise RuntimeError(
            "PATH wskazuje inny cl.exe niż wybrany kompletny v143.\n"
            f"Wybrany: {cl_path}\nZnaleziony: {cl}"
        )

    cl_lower = cl.lower().replace("/", "\\")
    m = re.search(r"\\msvc\\(\d+)\.(\d+)", cl_lower)
    if m and (int(m.group(1)), int(m.group(2))) >= (14, 50):
        raise RuntimeError(
            f"Wybrano nieobsługiwany kompilator mimo wymuszenia v143: {cl}"
        )

    print(f"[NATIVE BUILD] MSVC selector: {selector} ({full_version})")
    print(f"[NATIVE BUILD] host layout: {bin_dir.parent.name} -> x64")
    print(f"[NATIVE BUILD] cl.exe: {cl}")
    print(f"[NATIVE BUILD] link.exe: {link}")


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
        cflags = ["/O2", "/std:c++17"] if platform.system() == "Windows" else ["-O3", "-std=c++17"]
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
        _EXTENSION = load(
            name="connect6_cuda_championship_sm120_v10",
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