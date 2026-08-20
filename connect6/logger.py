from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


class TrainingLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.html_path = self.run_dir / "dashboard.html"
        self._fieldnames: list[str] | None = None

    def log(self, metrics: dict[str, Any], write_dashboard: bool = True) -> None:
        clean = {key: _scalar(value) for key, value in metrics.items()}

        if self._fieldnames is None:
            if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
                with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                    self._fieldnames = next(csv.reader(f))
            else:
                self._fieldnames = list(clean.keys())

        new_fields = [key for key in clean if key not in self._fieldnames]
        if new_fields:
            old_rows: list[dict[str, str]] = []
            if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
                with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                    old_rows = list(csv.DictReader(f))

            self._fieldnames.extend(new_fields)
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
                for old_row in old_rows:
                    writer.writerow(
                        {name: old_row.get(name, "") for name in self._fieldnames}
                    )

        new_file = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        row = {name: clean.get(name, "") for name in self._fieldnames}
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

        if write_dashboard:
            self.write_dashboard()

    def write_dashboard(self) -> None:
        if not self.csv_path.exists():
            return

        with self.csv_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return

        def series(name: str) -> list[float | None]:
            out: list[float | None] = []
            for row in rows:
                try:
                    raw = row.get(name, "")
                    out.append(float(raw) if raw not in (None, "") else None)
                except (TypeError, ValueError):
                    out.append(None)
            return out

        def latest_float(name: str, default: float = 0.0) -> float:
            try:
                raw = rows[-1].get(name, "")
                return float(raw) if raw not in (None, "") else default
            except (TypeError, ValueError):
                return default

        def fmt_number(value: float) -> str:
            if abs(value) >= 1_000_000:
                return f"{value / 1_000_000:.2f} mln"
            if abs(value) >= 1_000:
                return f"{value / 1_000:.1f} tys."
            return f"{value:.2f}"

        updates = series("update")

        charts = [
            {
                "id": "game_length_chart",
                "title": "Średnia długość partii",
                "metrics": [("mean_game_length", "Śr. długość")],
                "description": "Zmiana długości partii pomaga wychwycić przejścia między strategiami oraz nagłe załamania polityki.",
            },
            {
                "id": "games_chart",
                "title": "Zakończone partie na update",
                "metrics": [("games_this_update", "Partie")],
                "description": "Liczba pełnych partii zakończonych podczas jednego collectora PPO.",
            },
            {
                "id": "wins_chart",
                "title": "Czarne / białe / remisy",
                "metrics": [
                    ("black_win_rate", "Black win rate"),
                    ("white_win_rate", "White win rate"),
                    ("draw_rate", "Draw rate"),
                ],
                "description": "Służy głównie do wykrywania nierównowagi kolorów; 50/50 w mirror self-play nie jest miarą absolutnej siły.",
            },
            {
                "id": "historical_chart",
                "title": "Current kontra historyczne checkpointy",
                "metrics": [
                    ("historical_win_rate", "Win rate current"),
                    ("historical_score_rate", "Score rate current"),
                ],
                "description": "Najważniejszy wykres anti-forgetting: pokazuje wynik bieżącej polityki przeciw losowanej puli starszych modeli.",
            },
            {
                "id": "historical_games_chart",
                "title": "Wyniki current vs history na update",
                "metrics": [
                    ("historical_wins", "Wygrane"),
                    ("historical_losses", "Przegrane"),
                    ("historical_draws", "Remisy"),
                ],
                "description": "Surowe liczby wyników przeciw historycznym checkpointom w bieżącym update.",
            },
            {
                "id": "history_pool_chart",
                "title": "Historyczne stoły i modele w VRAM",
                "metrics": [
                    ("historical_tables", "Stoły historyczne"),
                    ("historical_models_loaded", "Modele historyczne")
                ],
                "description": "Kontrola, czy zadany udział stołów i liczba załadowanych checkpointów są faktycznie używane.",
            },
            {
                "id": "entropy_chart",
                "title": "Entropia polityki",
                "metrics": [("entropy", "Entropy")],
                "description": "Pozwala szybko zobaczyć ponowny collapse polityki lub zbyt agresywne wyostrzenie zachowania.",
            },
            {
                "id": "loss_chart",
                "title": "Funkcja straty",
                "metrics": [
                    ("loss", "Total"),
                    ("policy_loss", "Policy"),
                    ("value_loss", "Value"),
                ],
                "description": "Loss PPO nie jest bezpośrednią miarą siły, ale jego skoki są użyteczne diagnostycznie.",
            },
            {
                "id": "ppo_chart",
                "title": "Diagnostyka PPO",
                "metrics": [
                    ("approx_kl", "Approx KL"),
                    ("clip_fraction", "Clip fraction"),
                ],
                "description": "Approx KL i clip fraction pokazują, jak duży krok robi aktualizacja polityki.",
            },
            {
                "id": "gradient_chart",
                "title": "Gradient przed clippingiem",
                "metrics": [
                    ("grad_norm_mean", "Mean"),
                    ("grad_norm_p95", "P95"),
                    ("grad_norm_max", "Max"),
                    ("grad_limit", "Limit"),
                ],
                "description": "Normy są mierzone przed clippingiem; duże odchylenia od limitu pokazują, jak agresywnie działa zabezpieczenie.",
            },
            {
                "id": "gradient_clip_chart",
                "title": "Intensywność gradient clippingu",
                "metrics": [
                    ("grad_clip_fraction", "Część minibatchy obcięta"),
                    ("grad_scale_mean", "Śr. mnożnik gradientu"),
                ],
                "description": "Pokazuje jak często clipping działa oraz jaka średnia część pierwotnej skali gradientu pozostaje.",
            },
            {
                "id": "buffer_chart",
                "title": "Collector / bufor",
                "metrics": [
                    ("generated_positions_this_update", "Generated"),
                    ("completed_positions_this_update", "Used PPO"),
                    ("discarded_positions_this_update", "Discarded"),
                ],
                "description": "Do PPO wchodzą tylko bieżące decyzje current policy z terminalnych segmentów.",
            },
            {
                "id": "discard_chart",
                "title": "Udział odrzucanej historii",
                "metrics": [("discard_fraction", "Discard fraction")],
                "description": "Historia niedokończonych segmentów jest odrzucana, ale zwykłe self-play plansze są kontynuowane po PPO.",
            },
            {
                "id": "speed_chart",
                "title": "Szybkość",
                "metrics": [
                    ("selfplay_positions_per_second", "Self-play pos/s"),
                    ("positions_per_second", "Train pos/s"),
                ],
                "description": "Pozwala ocenić koszt historical self-play względem wcześniejszej wersji collectora.",
            },
            {
                "id": "schedule_chart",
                "title": "Schedule / credit assignment",
                "metrics": [
                    ("learning_rate", "Learning rate"),
                    ("temperature", "Temperature"),
                    ("gamma", "Gamma"),
                ],
                "description": "Parametry sterujące wielkością kroku PPO, eksploracją i dyskontowaniem terminalnego wyniku.",
            },
        ]

        chart_html: list[str] = []
        chart_js: list[str] = []
        for chart in charts:
            traces = [
                {
                    "x": updates,
                    "y": series(metric),
                    "mode": "lines",
                    "name": label,
                    "line": {"width": 2},
                }
                for metric, label in chart["metrics"]
            ]
            chart_id = str(chart["id"])
            chart_html.append(
                "<section class='panel'>"
                f"<div id='{html.escape(chart_id)}' class='chart'></div>"
                f"<p class='opis'>{html.escape(str(chart['description']))}</p>"
                "</section>"
            )
            layout = {
                "title": {"text": chart["title"], "font": {"size": 18}},
                "margin": {"t": 52, "l": 58, "r": 20, "b": 55},
                "hovermode": "x unified",
                "xaxis": {"title": "Update", "gridcolor": "#e5e7eb"},
                "yaxis": {"gridcolor": "#e5e7eb"},
                "legend": {"orientation": "h", "y": -0.2},
                "paper_bgcolor": "#ffffff",
                "plot_bgcolor": "#ffffff",
            }
            chart_js.append(
                f"Plotly.newPlot('{chart_id}', {json.dumps(traces)}, {json.dumps(layout)}, "
                "{responsive:true, displaylogo:false});"
            )

        latest = rows[-1]
        cards = [
            ("Update", latest.get("update", "-")),
            ("Global step", latest.get("global_step", "-")),
            ("Partie łącznie", f"{latest_float('games_completed'):,.0f}".replace(",", " ")),
            ("Partie / update", latest.get("games_this_update", "-")),
            ("Śr. długość", f"{latest_float('mean_game_length'):.1f}"),
            ("History gry / update", int(latest_float("historical_games_this_update"))),
            ("History wins", int(latest_float("historical_wins"))),
            ("History losses", int(latest_float("historical_losses"))),
            ("History draws", int(latest_float("historical_draws"))),
            ("History WR", f"{latest_float('historical_win_rate') * 100:.2f}%"),
            ("History score", f"{latest_float('historical_score_rate') * 100:.2f}%"),
            ("History stoły", int(latest_float("historical_tables"))),
            ("History modele w VRAM", int(latest_float("historical_models_loaded"))),
            ("Gamma", f"{latest_float('gamma', 1.0):.6f}"),
            ("Entropy", f"{latest_float('entropy'):.3f}"),
            ("Value loss", f"{latest_float('value_loss'):.4f}"),
            ("Policy loss", f"{latest_float('policy_loss'):.4f}"),
            ("Approx KL", f"{latest_float('approx_kl'):.4f}"),
            ("Grad mean", f"{latest_float('grad_norm_mean'):.3f}"),
            ("Grad p95", f"{latest_float('grad_norm_p95'):.3f}"),
            ("Grad clip", f"{latest_float('grad_clip_fraction') * 100:.1f}%"),
            ("Discard", f"{latest_float('discard_fraction') * 100:.1f}%"),
            ("Self-play", fmt_number(latest_float("selfplay_positions_per_second")) + " poz./s"),
            ("Cały trening", fmt_number(latest_float("positions_per_second")) + " poz./s"),
            ("GPU", f"{latest_float('gpu_memory_gb'):.2f} GB"),
        ]
        cards_html = "".join(
            "<div class='card'><div class='card-label'>"
            + html.escape(str(label))
            + "</div><div class='card-value'>"
            + html.escape(str(value))
            + "</div></div>"
            for label, value in cards
        )

        doc = f"""<!doctype html>
<html lang='pl'>
<head>
<meta charset='utf-8'>
<meta http-equiv='refresh' content='15'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Connect6 AI — dashboard treningu</title>
<script src='https://cdn.plot.ly/plotly-3.0.1.min.js'></script>
<style>
*{{box-sizing:border-box}}
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f3f5f8;color:#1f2937}}
main{{max-width:1700px;margin:0 auto;padding:24px}}
h1{{margin:0 0 6px;font-size:30px}}
.subtitle{{color:#64748b;margin-bottom:18px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin:14px 0 18px}}
.card{{background:white;padding:12px 14px;border-radius:10px;box-shadow:0 1px 4px #0002;min-width:0}}
.card-label{{font-size:.78rem;color:#64748b}}
.card-value{{font-size:1.20rem;font-weight:650;margin-top:4px;overflow-wrap:anywhere}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(380px,1fr));gap:14px}}
.panel{{background:white;border-radius:11px;padding:8px 10px 12px;box-shadow:0 1px 4px #0002;min-width:0}}
.chart{{min-height:350px}}
.opis{{margin:0 14px 8px;padding-top:9px;border-top:1px solid #e5e7eb;color:#64748b;font-size:13px;line-height:1.45}}
@media(max-width:950px){{.grid{{grid-template-columns:1fr}} main{{padding:14px}}}}
</style>
</head>
<body><main>
<h1>Connect6 AI — dashboard treningu</h1>
<div class='subtitle'>Auto-refresh co 15 s. History = wyniki aktualnej polityki przeciw zamrożonym checkpointom; ruchy starych modeli nie są materiałem PPO.</div>
<div class='cards'>{cards_html}</div>
<div class='grid'>{''.join(chart_html)}</div>
<script>{''.join(chart_js)}</script>
</main></body></html>"""

        self.html_path.write_text(doc, encoding="utf-8")


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
