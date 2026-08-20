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

        updates = series("update")

        charts = [
            {
                "id": "game_length_chart",
                "title": "Średnia długość partii",
                "metrics": [("mean_game_length", "Śr. długość")],
            },
            {
                "id": "wins_chart",
                "title": "Czarne / białe / remisy",
                "metrics": [
                    ("black_win_rate", "Black win rate"),
                    ("white_win_rate", "White win rate"),
                    ("draw_rate", "Draw rate"),
                ],
            },
            {
                "id": "historical_chart",
                "title": "Current kontra historyczne checkpointy",
                "metrics": [
                    ("historical_win_rate", "Win rate current"),
                    ("historical_score_rate", "Score rate current"),
                ],
            },
            {
                "id": "historical_games_chart",
                "title": "Wyniki current vs history na update",
                "metrics": [
                    ("historical_wins", "Wygrane"),
                    ("historical_losses", "Przegrane"),
                    ("historical_draws", "Remisy"),
                ],
            },
            {
                "id": "loss_chart",
                "title": "Loss",
                "metrics": [
                    ("loss", "Total"),
                    ("policy_loss", "Policy"),
                    ("value_loss", "Value"),
                ],
            },
            {
                "id": "policy_chart",
                "title": "Polityka / PPO",
                "metrics": [
                    ("entropy", "Entropy"),
                    ("approx_kl", "Approx KL"),
                    ("clip_fraction", "Clip fraction"),
                ],
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
            },
            {
                "id": "buffer_chart",
                "title": "Collector / bufor",
                "metrics": [
                    ("generated_positions_this_update", "Generated"),
                    ("completed_positions_this_update", "Used PPO"),
                    ("discarded_positions_this_update", "Discarded"),
                ],
            },
            {
                "id": "speed_chart",
                "title": "Szybkość",
                "metrics": [
                    ("selfplay_positions_per_second", "Self-play pos/s"),
                    ("positions_per_second", "Train pos/s"),
                ],
            },
            {
                "id": "schedule_chart",
                "title": "Schedule / credit",
                "metrics": [
                    ("learning_rate", "Learning rate"),
                    ("temperature", "Temperature"),
                    ("gamma", "Gamma"),
                ],
            },
        ]

        chart_html: list[str] = []
        chart_js: list[str] = []
        for chart in charts:
            traces = []
            for metric, label in chart["metrics"]:
                traces.append(
                    {
                        "x": updates,
                        "y": series(metric),
                        "mode": "lines",
                        "name": label,
                    }
                )
            chart_id = chart["id"]
            chart_html.append(
                f"<section class='panel'><div id='{html.escape(chart_id)}' class='chart'></div></section>"
            )
            layout = {
                "title": chart["title"],
                "margin": {"t": 48, "l": 58, "r": 20, "b": 45},
                "legend": {"orientation": "h", "y": -0.2},
            }
            chart_js.append(
                f"Plotly.newPlot('{chart_id}', {json.dumps(traces)}, {json.dumps(layout)}, "
                "{responsive:true, displaylogo:false});"
            )

        latest = rows[-1]
        hist_games = int(latest_float("historical_games_this_update"))
        cards = [
            ("Update", latest.get("update", "-")),
            ("Global step", latest.get("global_step", "-")),
            ("Partie / update", latest.get("games_this_update", "-")),
            ("History gry / update", hist_games),
            ("History wins", int(latest_float("historical_wins"))),
            ("History losses", int(latest_float("historical_losses"))),
            ("History WR", f"{latest_float('historical_win_rate') * 100:.2f}%"),
            ("History modele w VRAM", int(latest_float("historical_models_loaded"))),
            ("Gamma", f"{latest_float('gamma', 1.0):.6f}"),
            ("Entropy", f"{latest_float('entropy'):.3f}"),
            ("Approx KL", f"{latest_float('approx_kl'):.4f}"),
            ("GPU GB", f"{latest_float('gpu_memory_gb'):.2f}"),
        ]
        cards_html = "".join(
            "<div class='card'><div class='label'>"
            + html.escape(str(label))
            + "</div><div class='value'>"
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
main{{max-width:1500px;margin:0 auto;padding:20px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 18px}}
.card{{background:white;padding:10px 14px;border-radius:9px;box-shadow:0 1px 4px #0002;min-width:140px}}
.label{{font-size:.8rem;color:#6b7280}}
.value{{font-size:1.25rem;font-weight:600;margin-top:4px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(360px,1fr));gap:14px}}
.panel{{background:white;border-radius:10px;padding:8px;box-shadow:0 1px 4px #0002}}
.chart{{min-height:330px}}
small{{color:#6b7280}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<h1>Connect6 AI — trening</h1>
<small>Auto-refresh co 15 s. History = wyniki aktualnej polityki przeciw zamrożonym checkpointom.</small>
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
