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

        def as_float(row: dict[str, str], name: str, default: float | None = None):
            try:
                raw = row.get(name, "")
                if raw in (None, ""):
                    return default
                return float(raw)
            except (TypeError, ValueError):
                return default

        def series(name: str, fallback: str | None = None) -> list[float | None]:
            out: list[float | None] = []
            for row in rows:
                value = as_float(row, name, None)
                if value is None and fallback is not None:
                    value = as_float(row, fallback, None)
                out.append(value)
            return out

        def latest_float(name: str, default: float = 0.0, fallback: str | None = None) -> float:
            value = as_float(rows[-1], name, None)
            if value is None and fallback is not None:
                value = as_float(rows[-1], fallback, None)
            return default if value is None else value

        def fmt_number(value: float) -> str:
            if abs(value) >= 1_000_000:
                return f"{value / 1_000_000:.2f} mln"
            if abs(value) >= 1_000:
                return f"{value / 1_000:.1f} tys."
            return f"{value:.2f}"

        updates = series("update")

        charts = [
            {
                "id": "historical_chart",
                "title": "Wygrane current vs history",
                "metrics": [
                    {"key": "historical_win_rate", "label": "Win rate current", "percent": True},
                ],
                "y_range": [0, 100],
                "description": "Jedna prosta miara anti-forgetting: procent wygranych bieżącej polityki przeciw losowanej puli starszych checkpointów.",
            },
            {
                "id": "bots_chart",
                "title": "Wygrane current vs boty",
                "metrics": [
                    {"key": "bot_v1_win_rate", "label": "vs Bot V1", "percent": True},
                    {"key": "bot_v2_win_rate", "label": "vs Bot V2", "percent": True},
                ],
                "y_range": [0, 100],
                "description": "Bardzo czytelny zewnętrzny benchmark siły: odsetek wygranych bieżącej polityki osobno przeciw GPU Tactical Bot V1 i V2. Liczniki powstają w tym samym collectcie bez dodatkowej synchronizacji CPU-GPU.",
            },
            {
                "id": "kl_chart",
                "title": "Approx KL — przebieg i limity early-stop",
                "metrics": [
                    {"key": "approx_kl_mean", "fallback": "approx_kl", "label": "KL mean"},
                    {"key": "approx_kl_p95", "label": "KL p95"},
                    {"key": "approx_kl_max", "label": "KL max"},
                    {"key": "approx_kl_last", "label": "KL last"},
                    {"key": "target_kl", "label": "Soft target", "smooth": False},
                    {"key": "kl_hard_limit", "label": "Hard limit", "smooth": False},
                ],
                "description": "Soft-stop używa średniej kroczącej KL, a hard-stop pojedynczego skrajnego minibatcha. Minibatch wywołujący stop nie jest aplikowany.",
            },
            {
                "id": "ppo_execution_chart",
                "title": "Wykonanie PPO / early-stop",
                "metrics": [
                    {"key": "ppo_completion_fraction", "label": "Wykonane PPO", "percent": True},
                    {"key": "ppo_early_stop", "label": "Early-stop rate", "percent": True},
                ],
                "y_range": [0, 100],
                "description": "Wykonane PPO = procent zaplanowanych minibatchy wykonanych przed końcem update. Early-stop po wygładzeniu staje się odsetkiem update'ów uciętych przez KL.",
            },
            {
                "id": "ppo_batches_chart",
                "title": "Minibatche PPO",
                "metrics": [
                    {"key": "ppo_minibatches_completed", "label": "Wykonane"},
                    {"key": "ppo_minibatches_possible", "label": "Planowane", "smooth": False},
                ],
                "description": "Pozwala zobaczyć bezpośrednio ile kroków optymalizatora wykonano względem planu wynikającego z ppo_epochs.",
            },
            {
                "id": "game_length_chart",
                "title": "Średnia długość partii",
                "metrics": [{"key": "mean_game_length", "label": "Śr. długość"}],
                "description": "Zmiana długości partii pomaga wychwycić przejścia między strategiami oraz nagłe załamania polityki.",
            },
            {
                "id": "games_chart",
                "title": "Zakończone partie na update",
                "metrics": [{"key": "games_this_update", "label": "Partie"}],
                "description": "Liczba pełnych partii zakończonych podczas jednego collectora PPO.",
            },
            {
                "id": "wins_chart",
                "title": "Balans kolorów w self-play",
                "metrics": [
                    {"key": "black_win_rate", "label": "Black", "percent": True},
                    {"key": "white_win_rate", "label": "White", "percent": True},
                    {"key": "draw_rate", "label": "Remisy", "percent": True},
                ],
                "y_range": [0, 100],
                "description": "Służy do wykrywania nierównowagi kolorów; 50/50 w mirror self-play nie jest miarą absolutnej siły.",
            },
            {
                "id": "entropy_chart",
                "title": "Entropia polityki",
                "metrics": [{"key": "entropy", "label": "Entropy"}],
                "description": "Pozwala szybko zobaczyć collapse polityki lub zbyt agresywne wyostrzenie zachowania.",
            },
            {
                "id": "loss_chart",
                "title": "Funkcja straty",
                "metrics": [
                    {"key": "loss", "label": "Total"},
                    {"key": "policy_loss", "label": "Policy"},
                    {"key": "value_loss", "label": "Value"},
                ],
                "description": "Loss PPO nie jest bezpośrednią miarą siły, ale jego skoki są użyteczne diagnostycznie.",
            },
            {
                "id": "clip_chart",
                "title": "PPO clipping",
                "metrics": [{"key": "clip_fraction", "label": "Clip fraction", "percent": True}],
                "y_range": [0, 100],
                "description": "Odsetek próbek, dla których ratio wychodzi poza zakres PPO clip. Wysoki poziom razem z KL zwykle oznacza za duży krok polityki.",
            },
            {
                "id": "gradient_chart",
                "title": "Gradient przed clippingiem",
                "metrics": [
                    {"key": "grad_norm_mean", "label": "Mean"},
                    {"key": "grad_norm_p95", "label": "P95"},
                    {"key": "grad_norm_max", "label": "Max"},
                    {"key": "grad_limit", "label": "Limit", "smooth": False},
                ],
                "description": "Normy są mierzone przed clippingiem; duże odchylenia od limitu pokazują, jak agresywnie działa zabezpieczenie.",
            },
            {
                "id": "gradient_clip_chart",
                "title": "Intensywność gradient clippingu",
                "metrics": [
                    {"key": "grad_clip_fraction", "label": "Minibatche obcięte", "percent": True},
                    {"key": "grad_scale_mean", "label": "Śr. mnożnik gradientu"},
                ],
                "description": "Pokazuje jak często clipping działa oraz jaka średnia część pierwotnej skali gradientu pozostaje.",
            },
            {
                "id": "buffer_chart",
                "title": "Collector / bufor",
                "metrics": [
                    {"key": "generated_positions_this_update", "label": "Generated"},
                    {"key": "completed_positions_this_update", "label": "Used PPO"},
                    {"key": "discarded_positions_this_update", "label": "Discarded"},
                ],
                "description": "Do PPO wchodzą tylko bieżące decyzje current policy z terminalnych segmentów.",
            },
            {
                "id": "discard_chart",
                "title": "Udział odrzucanej historii",
                "metrics": [{"key": "discard_fraction", "label": "Discard", "percent": True}],
                "y_range": [0, 100],
                "description": "Historia niedokończonych segmentów jest odrzucana, ale zwykłe self-play plansze są kontynuowane po PPO.",
            },
            {
                "id": "speed_chart",
                "title": "Szybkość",
                "metrics": [
                    {"key": "selfplay_positions_per_second", "label": "Self-play pos/s"},
                    {"key": "positions_per_second", "label": "Train pos/s"},
                ],
                "description": "Pozwala ocenić koszt historycznego self-play i PPO względem wcześniejszych etapów treningu.",
            },
            {
                "id": "timing_chart",
                "title": "Czas poszczególnych części update'u",
                "metrics": [
                    {"key": "history_load_seconds", "label": "History load CPU"},
                    {"key": "history_build_seconds", "label": "History build GPU submit"},
                    {"key": "collector_seconds", "label": "Collector"},
                    {"key": "ppo_seconds", "label": "PPO"},
                    {"key": "update_seconds", "label": "Cały update"},
                ],
                "description": "Rozdziela koszt ładowania/stakowania przeciwników, self-play i PPO. Build GPU jest czasem hosta na przygotowanie/kopie; pierwsze użycie wag może dokończyć się asynchronicznie w collectorze.",
            },
            {
                "id": "vram_chart",
                "title": "VRAM — żywe tensory vs cache PyTorch",
                "metrics": [
                    {"key": "gpu_allocated_gb", "fallback": "gpu_memory_gb", "label": "Allocated"},
                    {"key": "gpu_reserved_gb", "label": "Reserved"},
                    {"key": "gpu_peak_allocated_gb", "fallback": "gpu_memory_gb", "label": "Peak allocated"},
                    {"key": "gpu_peak_reserved_gb", "label": "Peak reserved"},
                ],
                "description": "Allocated = pamięć faktycznie zajęta przez żywe tensory. Reserved = allocated + wolne bloki zatrzymane przez caching allocator; tę drugą liczbę zwykle widzi system/NVIDIA.",
            },
            {
                "id": "history_cache_chart",
                "title": "Cache historycznych checkpointów w RAM",
                "metrics": [
                    {"key": "historical_ram_cache_models", "label": "Modele w RAM"},
                    {"key": "historical_ram_cache_limit", "label": "Limit RAM", "smooth": False},
                    {"key": "historical_ram_cache_hit_rate", "label": "Hit rate", "percent": True},
                ],
                "description": "LRU cache ogranicza ponowne torch.load z dysku. Po osiągnięciu limitu najdawniej używany model jest usuwany z RAM.",
            },
            {
                "id": "history_pool_chart",
                "title": "Historyczne stoły i modele w VRAM",
                "metrics": [
                    {"key": "historical_tables", "label": "Stoły historyczne"},
                    {"key": "historical_models_loaded", "label": "Modele historyczne"},
                ],
                "description": "Kontrola, czy zadany udział stołów i liczba załadowanych checkpointów są faktycznie używane.",
            },
            {
                "id": "schedule_chart",
                "title": "Schedule / credit assignment",
                "metrics": [
                    {"key": "learning_rate", "label": "Learning rate"},
                    {"key": "temperature", "label": "Temperature"},
                    {"key": "gamma", "label": "Gamma"},
                ],
                "description": "Parametry sterujące wielkością kroku PPO, eksploracją i dyskontowaniem terminalnego wyniku.",
            },
        ]

        raw_series: dict[str, list[float | None]] = {"update": updates}
        for chart in charts:
            for metric in chart["metrics"]:
                key = str(metric["key"])
                fallback = metric.get("fallback")
                if key not in raw_series:
                    raw_series[key] = series(key, str(fallback) if fallback else None)

        chart_html: list[str] = []
        for chart in charts:
            chart_id = str(chart["id"])
            chart_html.append(
                "<section class='panel'>"
                f"<div id='{html.escape(chart_id)}' class='chart'></div>"
                f"<p class='opis'>{html.escape(str(chart['description']))}</p>"
                "</section>"
            )

        latest = rows[-1]
        ppo_stop = latest_float("ppo_early_stop") >= 0.5
        ppo_status = "EARLY STOP — KL" if ppo_stop else "pełny update"
        status_class = "warn" if ppo_stop else "ok"
        ppo_equiv = latest_float("ppo_epochs_equivalent")
        ppo_target = latest_float("ppo_epochs_target")
        cache_cleared = latest_float("cuda_cache_cleared") >= 0.5

        cards = [
            ("Update", latest.get("update", "-"), ""),
            ("Global step", latest.get("global_step", "-"), ""),
            ("PPO status", ppo_status, status_class),
            ("PPO wykonane", f"{latest_float('ppo_completion_fraction') * 100:.1f}%", status_class),
            ("Epoki PPO", f"{ppo_equiv:.2f} / {ppo_target:.0f}", ""),
            ("KL mean", f"{latest_float('approx_kl_mean', fallback='approx_kl'):.5f}", ""),
            ("KL p95", f"{latest_float('approx_kl_p95'):.5f}", ""),
            ("KL max", f"{latest_float('approx_kl_max'):.5f}", status_class if ppo_stop else ""),
            ("KL last", f"{latest_float('approx_kl_last'):.5f}", ""),
            ("Target KL", f"{latest_float('target_kl'):.5f}", ""),
            ("History win rate", f"{latest_float('historical_win_rate') * 100:.2f}%", ""),
            ("Bot V1 win rate", f"{latest_float('bot_v1_win_rate') * 100:.2f}%", ""),
            ("Bot V2 win rate", f"{latest_float('bot_v2_win_rate') * 100:.2f}%", ""),
            ("Bot V1 gry / update", int(latest_float("bot_v1_games_this_update")), ""),
            ("Bot V2 gry / update", int(latest_float("bot_v2_games_this_update")), ""),
            ("History gry / update", int(latest_float("historical_games_this_update")), ""),
            ("History RAM cache", f"{int(latest_float('historical_ram_cache_models'))} / {int(latest_float('historical_ram_cache_limit'))}", ""),
            ("RAM cache hit", f"{latest_float('historical_ram_cache_hit_rate') * 100:.1f}%", ""),
            ("Partie / update", latest.get("games_this_update", "-"), ""),
            ("Śr. długość", f"{latest_float('mean_game_length'):.1f}", ""),
            ("Entropy", f"{latest_float('entropy'):.3f}", ""),
            ("Value loss", f"{latest_float('value_loss'):.4f}", ""),
            ("Policy loss", f"{latest_float('policy_loss'):.4f}", ""),
            ("Grad mean", f"{latest_float('grad_norm_mean'):.3f}", ""),
            ("Grad p95", f"{latest_float('grad_norm_p95'):.3f}", ""),
            ("Grad clip", f"{latest_float('grad_clip_fraction') * 100:.1f}%", ""),
            ("Discard", f"{latest_float('discard_fraction') * 100:.1f}%", ""),
            ("Self-play", fmt_number(latest_float("selfplay_positions_per_second")) + " poz./s", ""),
            ("Cały trening", fmt_number(latest_float("positions_per_second")) + " poz./s", ""),
            ("Collector", f"{latest_float('collector_seconds'):.2f} s", ""),
            ("PPO", f"{latest_float('ppo_seconds'):.2f} s", ""),
            ("VRAM allocated", f"{latest_float('gpu_allocated_gb', fallback='gpu_memory_gb'):.2f} GB", ""),
            ("VRAM reserved", f"{latest_float('gpu_reserved_gb'):.2f} GB", ""),
            ("VRAM peak", f"{latest_float('gpu_peak_allocated_gb', fallback='gpu_memory_gb'):.2f} GB", ""),
            ("CUDA cache", "wyczyszczony" if cache_cleared else "normalny", "ok" if cache_cleared else ""),
        ]
        cards_html = "".join(
            "<div class='card " + html.escape(css_class) + "'><div class='card-label'>"
            + html.escape(str(label))
            + "</div><div class='card-value'>"
            + html.escape(str(value))
            + "</div></div>"
            for label, value, css_class in cards
        )

        charts_json = json.dumps(charts, ensure_ascii=False)
        data_json = json.dumps(raw_series, ensure_ascii=False)

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
main{{max-width:1800px;margin:0 auto;padding:24px}}
h1{{margin:0 0 6px;font-size:30px}}
.subtitle{{color:#64748b;margin-bottom:14px}}
.controls{{display:flex;gap:12px;flex-wrap:wrap;align-items:end;background:#fff;padding:12px 14px;border-radius:10px;box-shadow:0 1px 4px #0002;margin-bottom:14px}}
.control{{display:flex;flex-direction:column;gap:5px}}
.control label{{font-size:12px;color:#64748b;font-weight:600}}
.control select{{border:1px solid #cbd5e1;border-radius:7px;padding:7px 9px;background:#fff;color:#1f2937;min-width:145px}}
.control-note{{font-size:12px;color:#64748b;align-self:center}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:9px;margin:14px 0 18px}}
.card{{background:white;padding:11px 13px;border-radius:10px;box-shadow:0 1px 4px #0002;min-width:0;border-left:4px solid transparent}}
.card.ok{{border-left-color:#16a34a}}
.card.warn{{border-left-color:#dc2626;background:#fff7f7}}
.card-label{{font-size:.76rem;color:#64748b}}
.card-value{{font-size:1.16rem;font-weight:650;margin-top:4px;overflow-wrap:anywhere}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(390px,1fr));gap:14px}}
.panel{{background:white;border-radius:11px;padding:8px 10px 12px;box-shadow:0 1px 4px #0002;min-width:0}}
.chart{{min-height:355px}}
.opis{{margin:0 14px 8px;padding-top:9px;border-top:1px solid #e5e7eb;color:#64748b;font-size:13px;line-height:1.45}}
@media(max-width:950px){{.grid{{grid-template-columns:1fr}} main{{padding:14px}}}}
</style>
</head>
<body><main>
<h1>Connect6 AI — dashboard treningu</h1>
<div class='subtitle'>Auto-refresh co 15 s. Wygładzanie działa jako średnia krocząca po update'ach. Ustawienia widoku są zapamiętywane w przeglądarce.</div>
<div class='controls'>
  <div class='control'>
    <label for='smoothWindow'>Wygładzanie</label>
    <select id='smoothWindow'>
      <option value='1'>brak — 1 update</option>
      <option value='3'>3 update'y</option>
      <option value='5'>5 update'ów</option>
      <option value='10'>10 update'ów</option>
      <option value='25'>25 update'ów</option>
      <option value='50'>50 update'ów</option>
      <option value='100'>100 update'ów</option>
      <option value='250'>250 update'ów</option>
    </select>
  </div>
  <div class='control'>
    <label for='rangeWindow'>Zakres wykresów</label>
    <select id='rangeWindow'>
      <option value='0'>cały trening</option>
      <option value='100'>ostatnie 100</option>
      <option value='250'>ostatnie 250</option>
      <option value='500'>ostatnie 500</option>
      <option value='1000'>ostatnie 1000</option>
      <option value='2500'>ostatnie 2500</option>
      <option value='5000'>ostatnie 5000</option>
    </select>
  </div>
  <div class='control-note'>Wygładzanie nie zmienia stałych limitów, np. target KL, hard KL i grad limit.</div>
</div>
<div class='cards'>{cards_html}</div>
<div class='grid'>{''.join(chart_html)}</div>
<script>
const chartDefs = {charts_json};
const raw = {data_json};
const smoothSelect = document.getElementById('smoothWindow');
const rangeSelect = document.getElementById('rangeWindow');

smoothSelect.value = localStorage.getItem('connect6Smooth') || '10';
rangeSelect.value = localStorage.getItem('connect6Range') || '500';

function movingAverage(values, windowSize) {{
  if (windowSize <= 1) return values.slice();
  const out = new Array(values.length).fill(null);
  const queue = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {{
    const value = values[i];
    if (value === null || Number.isNaN(value)) {{
      queue.push(null);
    }} else {{
      queue.push(value);
      sum += value;
    }}
    if (queue.length > windowSize) {{
      const removed = queue.shift();
      if (removed !== null) sum -= removed;
    }}
    const valid = queue.filter(v => v !== null);
    out[i] = valid.length ? sum / valid.length : null;
  }}
  return out;
}}

function tail(values, count) {{
  if (!count || count <= 0 || values.length <= count) return values;
  return values.slice(values.length - count);
}}

function renderCharts() {{
  const smoothWindow = parseInt(smoothSelect.value, 10) || 1;
  const rangeWindow = parseInt(rangeSelect.value, 10) || 0;
  localStorage.setItem('connect6Smooth', String(smoothWindow));
  localStorage.setItem('connect6Range', String(rangeWindow));

  const x = tail(raw.update, rangeWindow);
  for (const chart of chartDefs) {{
    const traces = chart.metrics.map(metric => {{
      let y = raw[metric.key] || [];
      if (metric.smooth !== false) y = movingAverage(y, smoothWindow);
      if (metric.percent) y = y.map(v => v === null ? null : v * 100.0);
      y = tail(y, rangeWindow);
      return {{
        x: x,
        y: y,
        mode: 'lines',
        name: metric.label,
        line: {{width: metric.smooth === false ? 1.7 : 2.2, dash: metric.smooth === false ? 'dash' : 'solid'}},
        connectgaps: false,
      }};
    }});

    const layout = {{
      title: {{text: chart.title, font: {{size: 18}}}},
      margin: {{t: 52, l: 62, r: 20, b: 55}},
      hovermode: 'x unified',
      xaxis: {{title: 'Update', gridcolor: '#e5e7eb'}},
      yaxis: {{gridcolor: '#e5e7eb'}},
      legend: {{orientation: 'h', y: -0.2}},
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
    }};
    if (chart.y_range) layout.yaxis.range = chart.y_range;
    Plotly.react(chart.id, traces, layout, {{responsive:true, displaylogo:false}});
  }}
}}

smoothSelect.addEventListener('change', renderCharts);
rangeSelect.addEventListener('change', renderCharts);
renderCharts();
</script>
</main></body></html>"""

        self.html_path.write_text(doc, encoding="utf-8")


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
