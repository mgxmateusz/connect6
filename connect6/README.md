# Pakiet `connect6`

Kod jest podzielony według odpowiedzialności zamiast trzymania wszystkich modułów w jednym katalogu.

- `engine/` — zasady gry, środowisko wektorowe, model, checkpointy, historia modeli, trening PPO, logowanie i konfiguracja.
- `bots/` — ręcznie napisani przeciwnicy; obecnie GPU Tactical Bot.
- `ui/` — GUI do Human / Model / Bot.
- `evaluation/` — benchmarki i bezpośrednia ewaluacja modeli.
- `championship/` — championship modeli oraz autosave-vs-bot arena.
- `cuda_native/` — natywne C++/CUDA kernele i loadery.

Publiczne skrypty uruchomieniowe pozostają w katalogu głównym repo (`run_train.py`, `run_gui.py`, `run_championship.py`, itd.), żeby codzienne użycie nie wymagało pamiętania ścieżek modułów.

Stare eksperymentalne implementacje championship (stream/resident/fast/tunery) zostały usunięte. Aktualną ścieżką turnieju jest `championship/native_championship.py`.
