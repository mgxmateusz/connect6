# Pakiet `connect6`

Kod jest podzielony według odpowiedzialności zamiast trzymania wszystkich modułów w jednym katalogu.

- `engine/` — zasady gry, środowisko wektorowe, model, checkpointy, historia modeli, trening PPO, logowanie i konfiguracja.
- `bots/` — ręcznie napisani przeciwnicy; GPU Tactical Bot V1 oraz bardziej zaawansowany, nadal stateless GPU Tactical Bot V2.
- `ui/` — GUI do Human / Model / Bot V1 / Bot V2.
- `evaluation/` — benchmarki i bezpośrednia ewaluacja modeli; benchmark bota porównuje CNN, V1 i V2.
- `championship/` — championship modeli oraz autosave-vs-bot arena. Arena może liczyć oba boty i generuje wspólny raport V1 vs V2.
- `cuda_native/` — natywne C++/CUDA kernele i loadery.

Publiczne skrypty uruchomieniowe pozostają w katalogu głównym repo (`run_train.py`, `run_gui.py`, `run_championship.py`, itd.), żeby codzienne użycie nie wymagało pamiętania ścieżek modułów.

Stare eksperymentalne implementacje championship (stream/resident/fast/tunery) zostały usunięte. Aktualną ścieżką turnieju jest `championship/native_championship.py`.
