# Connect6 AI Lab

Projekt do trenowania i ewaluacji agenta CNN grającego w Connect6. Główny pipeline obejmuje GPU-native self-play/PPO, checkpointy historyczne, championship CUDA oraz testy przeciw botom taktycznym i Cloudictowi.

## Najważniejsze pliki

W katalogu głównym zostają tylko launchery używane na co dzień:

```text
run_train.py              trening / resume
run_gui.py                GUI
run_championship.py       championship checkpointów
run_cloudict_arena.py     arena CNN vs Cloudict D2/D3/D4
```

Kod właściwy znajduje się w `connect6/`, konfiguracje w `configs/`, testy w `tests/`.

Rzadziej używane narzędzia są w `tools/`:

```text
tools/check_gpu.py
tools/run_benchmark.py
tools/run_bot_arena.py
tools/run_bot_benchmark.py
tools/run_evaluate.py
tools/run_native_rollout_smoke.py
```

Pliki developerskie są w `dev/`, a luźne notatki w `docs/`.

## Instalacja

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r dev/requirements.txt
python tools/check_gpu.py
```

## Trening

```bat
python run_train.py --config configs/train.yaml
```

## Championship

```bat
python run_championship.py
```

Konfiguracja: `configs/championship.yaml`.

## Arena Cloudict

```bat
python run_cloudict_arena.py
```

Domyślny test biegnie etapami i zachowuje wyniki każdego etapu osobno:

```text
D2, VCF OFF  -> 361 pól startowych x 2 kolory = 722 partie
D3, VCF OFF  -> środkowe 3x3 x 2 kolory = 18 partii
D4, VCF OFF  -> środkowe 3x3 x 2 kolory = 18 partii
```

Istniejące CSV są wznawiane zamiast liczone od początku. `--reset` kasuje wyniki danego przebiegu zgodnie z logiką launchera.

## GUI

```bat
python run_gui.py
```

## Testy

```bat
pytest -c dev/pytest.ini -q
```

## Struktura

```text
configs/       konfiguracje treningu, aren i championship
connect6/      właściwy kod Pythona i CUDA
dev/           requirements + konfiguracja pytest
docs/          notatki
tests/         testy
tools/         benchmarki, smoke-testy i narzędzia pomocnicze
runs/          lokalne checkpointy i wyniki; ignorowane przez Git
```

`runs/` jest katalogiem roboczym i nie jest częścią repozytorium. Checkpointy oraz wyniki benchmarków/championship mogą być duże i pozostają lokalnie.
