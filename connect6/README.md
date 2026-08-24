# Pakiet `connect6`

Kod jest podzielony według odpowiedzialności zamiast trzymania wszystkich modułów w jednym katalogu.

- `engine/` — zasady gry, model, checkpointy, historia modeli, PPO oraz GPU-native collector treningowy.
- `bots/` — ręcznie napisani przeciwnicy; GPU Tactical Bot V1 oraz bardziej zaawansowany, nadal stateless GPU Tactical Bot V2.
- `ui/` — GUI do Human / Model / Bot V1 / Bot V2.
- `evaluation/` — benchmarki i bezpośrednia ewaluacja modeli; benchmark bota porównuje CNN, V1 i V2.
- `championship/` — championship modeli oraz autosave-vs-bot arena.
- `cuda_native/` — natywne C++/CUDA kernele i loadery.

## CNN V5

Aktualny model treningowy używa czterech kanałów wejściowych:

1. moje kamienie,
2. kamienie przeciwnika,
3. maska prawdziwej planszy (`1` na 19x19, `0` w paddingu),
4. informacja, czy aktualna decyzja jest ostatnim kamieniem tej tury.

Osiem bloków backbone ma postać `Conv -> GroupNorm -> SiLU`. GroupNorm ma zawsze dokładnie 8 kanałów na grupę. Głowice policy/value pozostają bez GroupNorm.

## GPU-native rollout

`run_train.py` uruchamia trening V5 przez `engine/train_v5.py`. Collector zachowuje stan wszystkich plansz na GPU i wykonuje pętlę ruchów wewnątrz warunkowego CUDA Graph. Python przygotowuje update, pakuje current+history oraz przydziały stołów, wywołuje collector raz i odzyskuje gotowy bufor oraz małe liczniki po zakończeniu collectu.

Mix stołów:

- 50% current vs current,
- 25% current vs historical,
- 25% current vs bot; połowa V1 i połowa V2.

Historyczne checkpointy są losowane z powtórzeniami: 50% z całej historii, 25% z najnowszej połowy i 25% z najnowszej ćwiartki.

Granica PPO update nie resetuje niedokończonych plansz. Gdy target pełnych próbek zostanie osiągnięty, wszystkie partie kończące się w tym samym kroku są zaliczane, doświadczenia z nadal niedokończonych segmentów są odrzucane, a ich stan planszy przechodzi do następnego update'u.

Target collectu jest definiowany jako `num_envs * completed_positions_per_update`; nie zależy od `minibatch_size`.

Do szybkiej walidacji kompilacji i działania natywnego collectora służy:

```powershell
python run_native_rollout_smoke.py
```

Publiczne skrypty uruchomieniowe pozostają w katalogu głównym repo (`run_train.py`, `run_gui.py`, `run_championship.py`, itd.), żeby codzienne użycie nie wymagało pamiętania ścieżek modułów.
