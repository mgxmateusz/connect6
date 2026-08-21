from pathlib import Path

TRAIN = Path("connect6/train.py")
TESTS = Path("tests/test_training_features.py")
CONFIG = Path("configs/train.yaml")

text = TRAIN.read_text(encoding="utf-8")

old = '''def _history_dtype(device: torch.device, dtype_name: str) -> torch.dtype:\n    if device.type != "cuda":\n        return torch.float32\n    name = dtype_name.lower()\n    if name == "bfloat16":\n        return torch.bfloat16\n    if name == "float16":\n        return torch.float16\n    if name == "float32":\n        return torch.float32\n    raise ValueError(\n        "historical_inference_dtype musi być jednym z: bfloat16 | float16 | float32"\n    )\n\n\n'''
new = old + '''def _cuda_cache_threshold_reached(reserved_bytes: int, threshold_gb: float) -> bool:\n    """Return True when PyTorch reserved CUDA memory reached the configured cap."""\n    threshold_gb = float(threshold_gb)\n    if threshold_gb <= 0.0:\n        return False\n    return int(reserved_bytes) >= int(threshold_gb * 1_000_000_000)\n\n\n'''
if old not in text:
    raise SystemExit("history dtype anchor not found")
text = text.replace(old, new, 1)

old = '''    cuda_cache_clear_every_percent = float(\n        tr.get("cuda_cache_clear_every_percent", 1.0)\n    )\n    if cuda_cache_clear_every_percent < 0.0:\n        raise ValueError("cuda_cache_clear_every_percent nie może być ujemne")\n    cuda_cache_clear_interval = (\n'''
new = '''    cuda_cache_clear_every_percent = float(\n        tr.get("cuda_cache_clear_every_percent", 1.0)\n    )\n    if cuda_cache_clear_every_percent < 0.0:\n        raise ValueError("cuda_cache_clear_every_percent nie może być ujemne")\n    cuda_cache_clear_reserved_gb = float(\n        tr.get("cuda_cache_clear_reserved_gb", 10.0)\n    )\n    if cuda_cache_clear_reserved_gb < 0.0:\n        raise ValueError("cuda_cache_clear_reserved_gb nie może być ujemne")\n    cuda_cache_clear_interval = (\n'''
if old not in text:
    raise SystemExit("config anchor not found")
text = text.replace(old, new, 1)

old = '''    if device.type == "cuda" and cuda_cache_clear_interval:\n        print(\n            f"[cuda-cache] empty_cache co {cuda_cache_clear_every_percent:g}% treningu "\n            f"(~{cuda_cache_clear_interval:,} update'ów)"\n        )\n'''
new = '''    if device.type == "cuda" and cuda_cache_clear_reserved_gb > 0.0:\n        print(\n            f"[cuda-cache] adaptive empty_cache przy reserved >= "\n            f"{cuda_cache_clear_reserved_gb:g} GB"\n        )\n    if device.type == "cuda" and cuda_cache_clear_interval:\n        print(\n            f"[cuda-cache] dodatkowo empty_cache co {cuda_cache_clear_every_percent:g}% treningu "\n            f"(~{cuda_cache_clear_interval:,} update'ów)"\n        )\n'''
if old not in text:
    raise SystemExit("startup print anchor not found")
text = text.replace(old, new, 1)

old = '''            historical_ensemble = None\n            cuda_cache_cleared = 0.0\n            cuda_cache_clear_seconds = 0.0\n            if (\n                device.type == "cuda"\n                and cuda_cache_clear_interval > 0\n                and update % cuda_cache_clear_interval == 0\n            ):\n                clear_started = time.perf_counter()\n                torch.cuda.synchronize(device)\n                torch.cuda.empty_cache()\n                cuda_cache_clear_seconds = time.perf_counter() - clear_started\n                cuda_cache_cleared = 1.0\n\n            if device.type == "cuda":\n                torch.cuda.reset_peak_memory_stats(device)\n'''
new = '''            historical_ensemble = None\n            cuda_cache_cleared = 0.0\n            cuda_cache_clear_seconds = 0.0\n            cuda_cache_reserved_before_clear_gb = 0.0\n            cuda_cache_threshold_hit = False\n            cuda_cache_periodic_hit = False\n            if device.type == "cuda":\n                reserved_before_clear = torch.cuda.memory_reserved(device)\n                cuda_cache_reserved_before_clear_gb = reserved_before_clear / 1e9\n                cuda_cache_threshold_hit = _cuda_cache_threshold_reached(\n                    reserved_before_clear,\n                    cuda_cache_clear_reserved_gb,\n                )\n                cuda_cache_periodic_hit = (\n                    cuda_cache_clear_interval > 0\n                    and update % cuda_cache_clear_interval == 0\n                )\n\n                if cuda_cache_threshold_hit or cuda_cache_periodic_hit:\n                    clear_started = time.perf_counter()\n                    torch.cuda.synchronize(device)\n                    torch.cuda.empty_cache()\n                    cuda_cache_clear_seconds = time.perf_counter() - clear_started\n                    cuda_cache_cleared = 1.0\n                    if cuda_cache_threshold_hit:\n                        print(\n                            f"[cuda-cache] reserved={cuda_cache_reserved_before_clear_gb:.2f} GB "\n                            f">= {cuda_cache_clear_reserved_gb:g} GB -> empty_cache()"\n                        )\n\n                torch.cuda.reset_peak_memory_stats(device)\n'''
if old not in text:
    raise SystemExit("cache clear block anchor not found")
text = text.replace(old, new, 1)

old = '''                "cuda_cache_cleared": cuda_cache_cleared,\n                "cuda_cache_clear_seconds": cuda_cache_clear_seconds,\n'''
new = '''                "cuda_cache_cleared": cuda_cache_cleared,\n                "cuda_cache_clear_seconds": cuda_cache_clear_seconds,\n                "cuda_cache_clear_reserved_gb": cuda_cache_clear_reserved_gb,\n                "cuda_cache_reserved_before_clear_gb": cuda_cache_reserved_before_clear_gb,\n                "cuda_cache_threshold_hit": 1.0 if cuda_cache_threshold_hit else 0.0,\n'''
if old not in text:
    raise SystemExit("metrics anchor not found")
text = text.replace(old, new, 1)
TRAIN.write_text(text, encoding="utf-8")

text = TESTS.read_text(encoding="utf-8")
old = '''    CompleteGameBuffer,\n    _forced_random_opening_mask,\n'''
new = '''    CompleteGameBuffer,\n    _cuda_cache_threshold_reached,\n    _forced_random_opening_mask,\n'''
if old not in text:
    raise SystemExit("test import anchor not found")
text = text.replace(old, new, 1)
append = '''\n\ndef test_cuda_cache_threshold_clears_at_configured_reserved_limit():\n    threshold_gb = 10.0\n    assert not _cuda_cache_threshold_reached(9_999_999_999, threshold_gb)\n    assert _cuda_cache_threshold_reached(10_000_000_000, threshold_gb)\n    assert _cuda_cache_threshold_reached(13_250_000_000, threshold_gb)\n    assert not _cuda_cache_threshold_reached(99_000_000_000, 0.0)\n'''
if "def test_cuda_cache_threshold_clears_at_configured_reserved_limit" not in text:
    text += append
TESTS.write_text(text, encoding="utf-8")

text = CONFIG.read_text(encoding="utf-8")
old = '''  # PyTorch trzyma zwolnione bloki VRAM w caching allocatorze. To jest dobre dla\n  # szybkości, ale Menedżer zadań/NVIDIA może pokazywać dużo więcej VRAM niż żywe\n  # tensory. Raz na 1% całego treningu bezpiecznie oddajemy WYŁĄCZNIE wolny cache\n  # do sterownika. Wartość 0 wyłącza okresowe czyszczenie.\n  cuda_cache_clear_every_percent: 1.0\n'''
new = '''  # Jeżeli PyTorch reserved osiągnie 10 GB, na granicy update'u oddajemy wyłącznie\n  # nieużywany cache do sterownika przez torch.cuda.empty_cache(). Żywe tensory\n  # (allocated) nie są zwalniane. 0 wyłącza próg adaptacyjny.\n  cuda_cache_clear_reserved_gb: 10.0\n\n  # Dodatkowy rzadki fallback czasowy; niezależny od progu reserved.\n  # Wartość 0 wyłącza okresowe czyszczenie.\n  cuda_cache_clear_every_percent: 1.0\n'''
if old not in text:
    raise SystemExit("yaml cache anchor not found")
text = text.replace(old, new, 1)
CONFIG.write_text(text, encoding="utf-8")
