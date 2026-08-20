# Connect6 AI Lab — wersja MLP

Projekt do trenowania jednej zwykłej sieci **MLP** grającej w Connect6 przeciwko samej sobie.

Najważniejsze elementy:

- szybkie headless self-play na wielu planszach jednocześnie;
- trening PPO na GPU;
- do treningu trafiają wyłącznie pełne, zakończone partie;
- niedokończone partie są wyrzucane po zapełnieniu bufora;
- jeden zwykły wektor wejściowy MLP;
- brak CNN, convolutions, residual blocks i innych architektur przestrzennych;
- dowolnie edytowalne warstwy `Linear`;
- osobna normalizacja, aktywacja i dropout dla każdej warstwy;
- SiLU dostępne jako podstawowa aktywacja;
- autosave checkpointów i automatyczne wznawianie;
- dashboard HTML z wykresami treningu;
- GUI: człowiek vs człowiek, człowiek vs AI, AI vs AI;
- benchmark wydajności;
- headless test model vs model.

## 1. Instalacja

Na RTX 5070 użyj PyTorch z obsługą odpowiedniej wersji CUDA dla kart Blackwell.
W projekcie znajduje się pomocniczy plik:

```bat
install_windows_rtx50.bat
```

Możesz też zainstalować ręcznie środowisko i wymagania:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python check_gpu.py
```

## 2. Uruchamianie

Trening:

```bat
python run_train.py --config configs/train.yaml
```

GUI:

```bat
python run_gui.py
```

Benchmark:

```bat
python run_benchmark.py --config configs/train.yaml --steps 500
```

Test checkpoint kontra checkpoint:

```bat
python run_evaluate.py SCIEZKA_A.pt SCIEZKA_B.pt --games 100
```

Testy projektu:

```bat
pytest -q
```

## 3. Wejście modelu

Standardowa plansza Connect6 ma `19 x 19 = 361` pól.

Model dostaje jeden zwykły wektor:

```text
361 wartości  moje kamienie
361 wartości  kamienie przeciwnika
1 wartość     ile kamieni zostało w tej turze / 2
1 wartość     czy aktualny gracz jest czarny
-----------------------------------------------
724 wartości razem
```

Kolejność wejścia:

```text
0..360     moje kamienie
361..721   kamienie przeciwnika
722        stones_left / 2
723        is_black
```

Kamienie są zawsze przedstawiane z perspektywy gracza wykonującego decyzję.
Dlatego jedna sieć może grać zarówno czarnymi, jak i białymi.

## 4. Architektura MLP

Całą architekturę edytujesz w:

```text
configs/train.yaml
```

Przykład:

```yaml
model:
  architecture_version: 3

  layers:
    - neurons: 1024
      norm: layer
      activation: silu
      dropout: 0.0

    - neurons: 512
      norm: none
      activation: silu
      dropout: 0.0

    - neurons: 256
      norm: layer
      activation: silu
      dropout: 0.0

    - neurons: 128
      norm: none
      activation: silu
      dropout: 0.0
```

Daje to zwykłą sieć:

```text
724 -> 1024 -> 512 -> 256 -> 128
```

Chcesz inną szerokość? Wpisujesz inną liczbę neuronów.

Na przykład:

```yaml
layers:
  - neurons: 2048
    norm: layer
    activation: silu
    dropout: 0.0

  - neurons: 512
    norm: none
    activation: silu
    dropout: 0.0

  - neurons: 128
    norm: none
    activation: silu
    dropout: 0.0

  - neurons: 256
    norm: layer
    activation: silu
    dropout: 0.0

  - neurons: 64
    norm: none
    activation: silu
    dropout: 0.0
```

Daje:

```text
724 -> 2048 -> 512 -> 128 -> 256 -> 64
```

Nie ma wymogu, aby kolejne warstwy były coraz mniejsze.

### Parametry pojedynczej warstwy

```yaml
- neurons: 512
  norm: layer
  activation: silu
  dropout: 0.0
```

`neurons` — liczba neuronów warstwy.

`norm`:

```text
none
layer
batch
```

`activation`:

```text
silu
gelu
relu
tanh
sigmoid
none
```

`dropout`:

```text
0.0 = brak dropout
0.1 = 10%
0.2 = 20%
```

Każda warstwa ma własne ustawienia.

## 5. Dlaczego są POLICY i VALUE

To nadal jest **jeden model**.

Po ostatniej wspólnej warstwie sieć ma dwie końcówki:

```text
                 wspólne MLP
                     |
              +------+------+
              |             |
           POLICY         VALUE
              |             |
        361 logitów      1 wartość
```

`POLICY` odpowiada za wybór ruchu.

Ma dokładnie `361` wyjść — po jednym na każde pole planszy.
Zajęte pola są maskowane, a z pozostałych logitów powstaje rozkład softmax.

`VALUE` ocenia aktualną pozycję jedną liczbą w zakresie `[-1, +1]`.

Dodatkowe warstwy tylko dla POLICY można dopisać w:

```yaml
policy_layers:
```

Dodatkowe warstwy tylko dla VALUE można dopisać w:

```yaml
value_layers:
```

Jeśli `policy_layers: []`, POLICY wychodzi bezpośrednio z końca wspólnego MLP.

## 6. Pełne partie w buforze

Wagi modelu są zamrożone podczas zbierania danych jednego update'u.

Przykład:

```text
1024 plansze zaczynają od zera
        ↓
grają równolegle
        ↓
zakończona partia -> cała trafia do bufora
        ↓
kolejna zakończona partia -> cała trafia do bufora
        ↓
>= completed_positions_per_update
        ↓
niedokończone partie -> kosz
        ↓
PPO aktualizuje model
        ↓
wszystkie plansze od zera na nowych wagach
```

Każda decyzja w kompletnej partii dostaje prawdziwy końcowy wynik:

```text
+1  gracz wykonujący decyzję ostatecznie wygrał
-1  gracz wykonujący decyzję ostatecznie przegrał
 0  remis
```

## 7. Najważniejsze parametry treningu

```yaml
training:
  num_envs: 1024
  completed_positions_per_update: 65536
  ppo_epochs: 4
  minibatch_size: 2048
  learning_rate: 0.0003
```

`num_envs` — liczba równoległych plansz.

`completed_positions_per_update` — minimalna liczba pozycji pochodzących z pełnych partii przed jednym update'em PPO.

`ppo_epochs` — ile razy PPO przejdzie po jednym zebranym buforze.

`minibatch_size` — ile pozycji trafia jednocześnie do jednego kroku optymalizatora.

## 8. Checkpointy

Domyślny run tej wersji:

```text
runs/connect6_mlp_01/
```

Checkpointy:

```text
runs/connect6_mlp_01/checkpoints/
  latest.pt
  model_update_00000024.pt
  model_update_00000049.pt
  ...
```

Przy:

```yaml
resume: "auto"
```

trening automatycznie wczyta `latest.pt`.

Stare checkpointy CNN nie są zgodne z tą wersją MLP.

## 9. Dashboard

Po rozpoczęciu treningu powstają:

```text
runs/connect6_mlp_01/metrics.csv
runs/connect6_mlp_01/dashboard.html
```

Dashboard pokazuje m.in.:

- loss;
- policy loss;
- value loss;
- entropy;
- KL;
- win-rate czarnych i białych;
- liczbę zakończonych partii;
- liczbę odrzuconych pozycji;
- szybkość self-play;
- całkowitą przepustowość treningu.

## 10. GUI

Uruchamiaj przez:

```bat
python run_gui.py
```

Nie przez:

```bat
python connect6/gui.py
```

ponieważ pliki w katalogu `connect6` korzystają z importów pakietowych.

GUI automatycznie wyszukuje checkpointy w `runs/**/checkpoints/*.pt`.
Możesz uruchomić:

- człowiek vs człowiek;
- człowiek vs AI;
- AI vs AI;
- stary checkpoint vs nowy checkpoint.

Można ustawić opóźnienie ruchu AI, aby oglądać grę w normalnym tempie.
