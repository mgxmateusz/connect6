# Connect6 AI Lab — CNN

Projekt do trenowania jednej sieci **CNN** grającej w Connect6 przeciwko samej sobie metodą PPO.

Najważniejsze elementy:

- szybkie headless self-play na wielu planszach jednocześnie;
- trening PPO na GPU;
- do treningu trafiają wyłącznie pełne, zakończone partie;
- kanoniczne wejście przestrzenne `3 x 19 x 19`;
- ośmiowarstwowy backbone CNN z SiLU;
- POLICY jako `Conv 1x1 -> 361 logitów`;
- w pełni konwolucyjny VALUE head;
- online augmentacja D4 (obroty + odbicia);
- historyczny self-play z wieloma checkpointami liczonymi grouped CNN;
- losowe otwarcia czarnych jako eksploracja;
- autosave checkpointów i automatyczne wznawianie;
- dashboard HTML, GUI, benchmark i headless model-vs-model.

## Instalacja i uruchamianie

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python check_gpu.py
```

Trening:

```bat
python run_train.py --config configs/train.yaml
```

GUI:

```bat
python run_gui.py
```

Benchmark treningowy:

```bat
python run_benchmark.py --config configs/train.yaml --steps 500
```

Championship:

```bat
python run_championship.py
```

Testy:

```bat
pytest -q
```

## GPU-native championship — RTX 5070 / SM120

`run_championship.py` używa osobnego, natywnego silnika CUDA wyspecjalizowanego dla aktualnej architektury CNN i Connect6 19x19. Stary Pythonowy scheduler i championship autotuner nie znajdują się w hot-path.

Wymagania dodatkowe na Windows:

- RTX 50 / compute capability 12.0;
- CUDA Toolkit **12.8 lub nowszy** z `nvcc` dostępnym przez `CUDA_HOME`;
- Visual Studio Build Tools z workloadem **Desktop development with C++**;
- `ninja` instalowany przez `requirements.txt`.

Pierwsze uruchomienie kompiluje rozszerzenie CUDA pod `sm_120`. Wynik jest cache'owany przez PyTorch, więc kolejne uruchomienia używają gotowej binarki dopóki źródła CUDA lub środowisko kompilacji się nie zmienią.

Hot-path turnieju pozostaje na GPU:

- checkpointy są jednorazowo przepakowywane do FP16 w układzie używanym bezpośrednio przez WMMA;
- stan plansz jest przechowywany jako bitboardy;
- pierwsza warstwa odtwarza kanały `moje / przeciwnika / ostatni kamień tury` bez materializowania tensora wejściowego;
- warstwy CNN używają własnego implicit-GEMM WMMA FP16 z akumulacją FP32;
- nie ma `F.unfold`, `index_select` wag ani per-move transferów CPU/GPU;
- policy, maskowanie legalnych ruchów i `argmax` są scalone;
- GPU samo wykrywa koniec gry, zapisuje wynik i pobiera następny job;
- cała pętla ruchów działa jako conditional CUDA Graph `WHILE` i nie wraca do CPU między ruchami.

Domyślnie `configs/championship.yaml` używa 4096 stale dostępnych slotów GPU. Po zakończeniu device-side przebiegu CPU służy już tylko do zapisania `matches.csv`, `ranking.csv`, `championship.html` i `native_run.json`.

## Wejście modelu

Dla planszy 19x19 model dostaje tensor:

```text
[B, 3, 19, 19]
```

Kanały:

```text
0  moje kamienie          0/1
1  kamienie przeciwnika   0/1
2  ostatni kamień tury    0/1
```

Trzeci kanał jest cały wypełniony zerami albo jedynkami. `1` oznacza, że aktualna decyzja jest ostatnim kamieniem bieżącej tury (`stones_left == 1`).

Kolor gracza nie jest wejściem. Plansza jest zawsze przedstawiana jako `ja / przeciwnik`, więc strategicznie identyczna pozycja ma identyczną reprezentację niezależnie od fizycznego koloru kamieni.

## Architektura CNN

Domyślna architektura znajduje się w `configs/train.yaml`:

```text
INPUT  3 x 19 x 19

I      23x23   3  -> 32   + SiLU
II      3x3   32  -> 32   + SiLU
III     3x3   32  -> 64   + SiLU
IV      3x3   64  -> 64   + SiLU
V       3x3   64  -> 64   + SiLU
VI      3x3   64  -> 96   + SiLU
VII     3x3   96  -> 96   + SiLU
VIII    3x3   96  -> 96   + SiLU

POLICY  1x1   96  -> 1    -> 19x19 -> 361 logits
VALUE   1x1   96  -> 1    -> global mean -> tanh
```

Wszystkie warstwy używają `stride=1` i paddingu zachowującego `19x19`. Receptive field rośnie:

```text
23 -> 25 -> 27 -> 29 -> 31 -> 33 -> 35 -> 37
```

czyli na końcu także predykcja pola w rogu może zależeć od przeciwległego rogu planszy.

Konfiguracja:

```yaml
model:
  architecture_version: 4
  kernels: [23, 3, 3, 3, 3, 3, 3, 3]
  channels: [32, 32, 64, 64, 64, 96, 96, 96]
```

## POLICY i VALUE

To nadal jeden model z jednym wspólnym backbone.

`POLICY` daje mapę `1 x 19 x 19`, która jest spłaszczana do 361 logitów. Zajęte pola są maskowane przed utworzeniem rozkładu ruchów.

`VALUE` jest wymagane przez PPO. Zamiast MLP używa osobnej konwolucji `1x1`; jej mapa jest uśredniana po planszy i przechodzi przez `tanh`, dając jedną ocenę pozycji w `[-1, 1]`.

## Symetrie i losowe otwarcie

`D4` pozostaje w treningu. Zwykły CNN współdzieli filtry przestrzennie, ale nie jest automatycznie ekwiwariantny na obrót 90° ani odbicie lustrzane. Collector nadal cyklicznie pokazuje osiem orientacji i odpowiednio transformuje akcje.

Losowy pierwszy kamień Czarnych również pozostaje. Nie jest częścią PPO i służy wyłącznie do zwiększenia różnorodności otwarć oraz częstszego pokazywania modelowi pozycji poza centrum.

## Historyczny self-play

Stare modele tej samej architektury CNN nadal mogą być używane jako zamrożeni przeciwnicy. Zamiast wcześniejszego batched MLP, ich warstwy są składane w **grouped convolution**, dzięki czemu wiele checkpointów jest liczonych równolegle na GPU bez osobnego forwardu dla każdego modelu.

Checkpointy MLP `architecture_version: 3` nie są zgodne z CNN `architecture_version: 4` i są ignorowane jako historyczni przeciwnicy.

## Checkpointy

Domyślny run:

```text
runs/connect6_cnn_01/
```

Przy `resume: "auto"` trening wznawia najnowszy checkpoint z tego runu. Zmiana nazwy runu celowo odcina automatyczne wznawianie starego modelu MLP.

## GUI i ewaluacja

GUI, benchmark treningowy oraz model-vs-model korzystają z tego samego `network_input()`, więc pracują bezpośrednio z wejściem CNN.

```bat
python run_gui.py
python run_benchmark.py --config configs/train.yaml --steps 500
python run_evaluate.py SCIEZKA_A.pt SCIEZKA_B.pt --games 100
```
