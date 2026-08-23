from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


_EXTENSION = None


def _find_vcvars64() -> Path | None:
    """Znajduje vcvars64.bat dla zainstalowanego Visual Studio/Build Tools."""
    candidates: list[Path] = []

    # Najpewniejsza ścieżka: vswhere instalowany razem z Visual Studio Installer.
    pf86 = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.exists():
        try:
            result = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            install = result.stdout.strip()
            if install:
                candidates.append(
                    Path(install) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                )
        except OSError:
            pass

    # Fallback dla systemów bez działającego vswhere oraz dla kilku generacji VS.
    roots = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio",
        Path(pf86) / "Microsoft Visual Studio",
    ]
    editions = ("BuildTools", "Community", "Professional", "Enterprise")
    for root in roots:
        for year in ("18", "2026", "2022", "2019"):
            for edition in editions:
                candidates.append(
                    root / year / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                )

    seen: set[str] = set()
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


def _run_vcvars_and_capture_environment(vcvars: Path) -> dict[str, str]:
    """Uruchamia vcvars64.bat przez tymczasowy .cmd, bez kruchego cmd quoting."""
    marker = "__CONNECT6_ENV_BEGIN__"
    script = (
        "@echo off\r\n"
        f'call "{vcvars}"\r\n'
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
            "vcvars64.bat zwrócił błąd. Pełny output:\n"
            + (combined if combined else "(brak komunikatu)")
        )

    lines = (proc.stdout or "").splitlines()
    try:
        start = lines.index(marker) + 1
    except ValueError as exc:
        raise RuntimeError(
            "vcvars64.bat zakończył się bez błędu, ale nie udało się odczytać "
            f"środowiska. Output:\n{combined}"
        ) from exc

    env: dict[str, str] = {}
    for line in lines[start:]:
        if not line or line.startswith("=") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            env[key] = value
    return env


def _bootstrap_msvc_environment() -> None:
    """Udostępnia cl/link/Windows SDK w zwykłym PowerShellu.

    PyTorch cpp_extension zakłada na Windows, że środowisko MSVC jest już
    skonfigurowane. Zamiast wymagać uruchamiania Developer PowerShell, sami
    wykonujemy vcvars64.bat i importujemy jego zmienne do bieżącego procesu.
    """
    if platform.system() != "Windows":
        return
    if shutil.which("cl.exe") and shutil.which("link.exe"):
        return

    vcvars = _find_vcvars64()
    if vcvars is None:
        raise RuntimeError(
            "Nie znaleziono kompilatora MSVC (cl.exe). Zainstaluj Visual Studio "
            "Build Tools z workloadem 'Desktop development with C++' "
            "(Microsoft.VisualStudio.Workload.VCTools), razem z x64 MSVC i Windows SDK."
        )

    env = _run_vcvars_and_capture_environment(vcvars)
    os.environ.update(env)

    cl = shutil.which("cl.exe")
    link = shutil.which("link.exe")
    if not cl or not link:
        raise RuntimeError(
            f"Znaleziono i uruchomiono {vcvars}, ale po vcvars64.bat nadal brakuje "
            "cl.exe/link.exe. W Visual Studio Installer doinstaluj workload "
            "'Desktop development with C++' wraz z MSVC x64/x86 i Windows SDK."
        )

    print(f"[NATIVE BUILD] MSVC environment: {vcvars}")
    print(f"[NATIVE BUILD] cl.exe: {cl}")


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
            name="connect6_cuda_championship_sm120_v5",
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
