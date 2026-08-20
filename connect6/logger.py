from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


class TrainingLogger:

    def __init__(
        self,
        run_dir: str | Path,
    ):

        self.run_dir = Path(
            run_dir
        )

        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        self.csv_path = (
            self.run_dir
            / "metrics.csv"
        )


        self.html_path = (
            self.run_dir
            / "dashboard.html"
        )


        self._fieldnames: list[
            str
        ] | None = None



    # =========================================================================
    # ZAPIS METRYK
    # =========================================================================

    def log(
        self,
        metrics: dict[str, Any],
        write_dashboard: bool = True,
    ) -> None:

        clean = {

            key:
                _scalar(value)

            for key, value
            in metrics.items()
        }


        # =====================================================================
        # WCZYTANIE ISTNIEJĄCEGO NAGŁÓWKA CSV
        # =====================================================================

        if self._fieldnames is None:

            if (
                self.csv_path.exists()
                and
                self.csv_path.stat().st_size > 0
            ):

                with self.csv_path.open(
                    "r",
                    newline="",
                    encoding="utf-8",
                ) as file:

                    reader = csv.reader(
                        file
                    )


                    self._fieldnames = next(
                        reader
                    )


            else:

                self._fieldnames = list(
                    clean.keys()
                )


        # =====================================================================
        # AUTOMATYCZNE DODANIE NOWYCH KOLUMN
        # =====================================================================
        #
        # Dzięki temu jeżeli metrics.csv już istnieje,
        # a dodamy np.:
        #
        # grad_norm_mean
        # grad_norm_p95
        # grad_clip_fraction
        #
        # to nie trzeba kasować historii treningu.
        #
        # Stare wiersze dostają puste wartości.
        # =====================================================================

        new_fields = [

            key

            for key
            in clean.keys()

            if (
                key
                not in self._fieldnames
            )
        ]


        if new_fields:

            old_rows: list[
                dict[str, str]
            ] = []


            if (
                self.csv_path.exists()
                and
                self.csv_path.stat().st_size > 0
            ):

                with self.csv_path.open(
                    "r",
                    newline="",
                    encoding="utf-8",
                ) as file:

                    old_rows = list(
                        csv.DictReader(
                            file
                        )
                    )


            self._fieldnames.extend(
                new_fields
            )


            with self.csv_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=
                        self._fieldnames,
                )


                writer.writeheader()


                for old_row in old_rows:

                    writer.writerow({

                        name:
                            old_row.get(
                                name,
                                "",
                            )

                        for name
                        in self._fieldnames
                    })


        # =====================================================================
        # DOPISANIE NOWEGO WIERSZA
        # =====================================================================

        row = {

            name:
                clean.get(
                    name,
                    "",
                )

            for name
            in self._fieldnames
        }


        new_file = (
            not self.csv_path.exists()
            or
            self.csv_path.stat().st_size == 0
        )


        with self.csv_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=
                    self._fieldnames,
            )


            if new_file:

                writer.writeheader()


            writer.writerow(
                row
            )


        if write_dashboard:

            self.write_dashboard()



    # =========================================================================
    # DASHBOARD
    # =========================================================================

    def write_dashboard(
        self,
    ) -> None:

        if not self.csv_path.exists():

            return


        with self.csv_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            rows = list(
                csv.DictReader(
                    file
                )
            )


        if not rows:

            return


        # =====================================================================
        # POMOCNICZE
        # =====================================================================

        def series(
            name: str,
        ) -> list[
            float | None
        ]:

            values: list[
                float | None
            ] = []


            for row in rows:

                try:

                    values.append(
                        float(
                            row.get(
                                name,
                                "",
                            )
                        )
                    )


                except (
                    TypeError,
                    ValueError,
                ):

                    values.append(
                        None
                    )


            return values



        def latest_float(
            name: str,
            default: float = 0.0,
        ) -> float:

            try:

                return float(
                    rows[-1].get(
                        name,
                        default,
                    )
                )


            except (
                TypeError,
                ValueError,
            ):

                return default



        def first_valid_float(
            name: str,
            default: float = 0.0,
        ) -> float:

            for row in rows:

                try:

                    return float(
                        row.get(
                            name,
                            "",
                        )
                    )


                except (
                    TypeError,
                    ValueError,
                ):

                    continue


            return default



        def format_number(
            value: float,
        ) -> str:

            if abs(value) >= 1_000_000:

                return (
                    f"{value / 1_000_000:.2f} mln"
                )


            if abs(value) >= 1_000:

                return (
                    f"{value / 1_000:.1f} tys."
                )


            return (
                f"{value:.2f}"
            )


        updates = series(
            "update"
        )


        # =====================================================================
        # DEFINICJE WYKRESÓW
        # =====================================================================

        charts = [

            # -----------------------------------------------------------------
            # DŁUGOŚĆ PARTII
            # -----------------------------------------------------------------

            {
                "id":
                    "game_length_chart",

                "title":
                    "Średnia długość partii",

                "metrics": [

                    (
                        "mean_game_length",
                        "Średnia długość partii",
                    ),
                ],

                "description":
                    (
                        "Jedna z najbardziej użytecznych metryk. "
                        "Jeżeli model uczy się skuteczniej tworzyć "
                        "6 kamieni w rzędzie, partie zwykle kończą "
                        "się szybciej."
                    ),
            },


            # -----------------------------------------------------------------
            # ZAKOŃCZONE GRY
            # -----------------------------------------------------------------

            {
                "id":
                    "games_chart",

                "title":
                    "Zakończone partie na update",

                "metrics": [

                    (
                        "games_this_update",
                        "Partie zakończone",
                    ),
                ],

                "description":
                    (
                        "Pokazuje ile prawdziwych partii zakończyło się "
                        "w każdym update. Przy krótszych grach ta liczba "
                        "zwykle rośnie."
                    ),
            },


            # -----------------------------------------------------------------
            # ENTROPIA
            # -----------------------------------------------------------------

            {
                "id":
                    "entropy_chart",

                "title":
                    "Entropia polityki",

                "metrics": [

                    (
                        "entropy",
                        "Entropia",
                    ),
                ],

                "description":
                    (
                        "Entropia mówi jak bardzo losowa jest polityka. "
                        "Stopniowy spadek zwykle oznacza, że model coraz "
                        "wyraźniej preferuje konkretne ruchy."
                    ),
            },


            # -----------------------------------------------------------------
            # TEMPERATURA
            # -----------------------------------------------------------------

            {
                "id":
                    "temperature_chart",

                "title":
                    "Temperatura eksploracji",

                "metrics": [

                    (
                        "temperature",
                        "Temperatura",
                    ),
                ],

                "description":
                    (
                        "Niższa temperatura wyostrza rozkład ruchów "
                        "i zwiększa skłonność do wybierania ruchów "
                        "o najwyższych logitach."
                    ),
            },


            # -----------------------------------------------------------------
            # LOSS
            # -----------------------------------------------------------------

            {
                "id":
                    "loss_chart",

                "title":
                    "Funkcja straty",

                "metrics": [

                    (
                        "loss",
                        "Loss całkowity",
                    ),

                    (
                        "policy_loss",
                        "Policy loss",
                    ),

                    (
                        "value_loss",
                        "Value loss",
                    ),
                ],

                "description":
                    (
                        "W PPO całkowity loss nie musi stale maleć. "
                        "Policy loss odpowiada za zmianę polityki, "
                        "a value loss za przewidywanie końcowego wyniku."
                    ),
            },


            # -----------------------------------------------------------------
            # PPO
            # -----------------------------------------------------------------

            {
                "id":
                    "ppo_chart",

                "title":
                    "Diagnostyka PPO",

                "metrics": [

                    (
                        "approx_kl",
                        "Approx KL",
                    ),

                    (
                        "clip_fraction",
                        "PPO clip fraction",
                    ),
                ],

                "description":
                    (
                        "Approx KL mierzy wielkość zmiany polityki. "
                        "PPO clip fraction pokazuje jaka część próbek "
                        "została ograniczona przez clipping ilorazu PPO."
                    ),
            },


            # -----------------------------------------------------------------
            # NORMA GRADIENTU
            # -----------------------------------------------------------------

            {
                "id":
                    "gradient_norm_chart",

                "title":
                    "Norma gradientu przed clippingiem",

                "metrics": [

                    (
                        "grad_norm_mean",
                        "Średnia",
                    ),

                    (
                        "grad_norm_p95",
                        "95 percentyl",
                    ),

                    (
                        "grad_norm_max",
                        "Maksimum",
                    ),

                    (
                        "grad_limit",
                        "Limit clippingu",
                    ),
                ],

                "description":
                    (
                        "Wszystkie normy są mierzone przed gradient clippingiem. "
                        "Jeżeli średnia i 95 percentyl stale leżą dużo powyżej "
                        "limitu, max_grad_norm prawdopodobnie jest zbyt agresywny. "
                        "Maksimum służy głównie do wykrywania pojedynczych skoków."
                    ),
            },


            # -----------------------------------------------------------------
            # CLIPPING GRADIENTU
            # -----------------------------------------------------------------

            {
                "id":
                    "gradient_clip_chart",

                "title":
                    "Intensywność gradient clippingu",

                "metrics": [

                    (
                        "grad_clip_fraction",
                        "Część minibatchy obcięta",
                    ),

                    (
                        "grad_scale_mean",
                        "Średni mnożnik gradientu",
                    ),
                ],

                "description":
                    (
                        "Grad clip fraction = 1.0 oznacza, że 100% minibatchy "
                        "było obcinanych. Grad scale = 0.25 oznacza, że po "
                        "clippingu zostawało średnio około 25% pierwotnej "
                        "skali gradientu."
                    ),
            },


            # -----------------------------------------------------------------
            # SZYBKOŚĆ
            # -----------------------------------------------------------------

            {
                "id":
                    "speed_chart",

                "title":
                    "Szybkość treningu",

                "metrics": [

                    (
                        "selfplay_positions_per_second",
                        "Self-play pozycji/s",
                    ),

                    (
                        "positions_per_second",
                        "Trening pozycji/s",
                    ),
                ],

                "description":
                    (
                        "Self-play pokazuje szybkość generowania pozycji. "
                        "Druga wartość obejmuje cały update razem z PPO "
                        "i backpropagation."
                    ),
            },


            # -----------------------------------------------------------------
            # BUFOR
            # -----------------------------------------------------------------

            {
                "id":
                    "buffer_chart",

                "title":
                    "Pozycje generowane i używane",

                "metrics": [

                    (
                        "generated_positions_this_update",
                        "Wygenerowane",
                    ),

                    (
                        "completed_positions_this_update",
                        "Użyte do PPO",
                    ),

                    (
                        "discarded_positions_this_update",
                        "Historia odrzucona",
                    ),
                ],

                "description":
                    (
                        "Do PPO trafiają tylko decyzje z segmentów, które "
                        "zakończyły się wynikiem terminalnym w bieżącym update. "
                        "Historia niedokończonych segmentów jest odrzucana, "
                        "ale ich plansze pozostają i są kontynuowane po PPO."
                    ),
            },


            # -----------------------------------------------------------------
            # DISCARD
            # -----------------------------------------------------------------

            {
                "id":
                    "discard_chart",

                "title":
                    "Udział odrzucanej historii",

                "metrics": [

                    (
                        "discard_fraction",
                        "Odrzucona część",
                    ),
                ],

                "description":
                    (
                        "To nie oznacza już kasowania plansz. Wartość mówi "
                        "tylko jaka część ruchów bieżącego modelu nie dostała "
                        "wyniku terminalnego przed PPO i dlatego nie została "
                        "użyta do uczenia. Sama pozycja na planszy pozostaje."
                    ),
            },


            # -----------------------------------------------------------------
            # CZARNE / BIAŁE
            # -----------------------------------------------------------------

            {
                "id":
                    "wins_chart",

                "title":
                    "Czarne kontra białe",

                "metrics": [

                    (
                        "black_win_rate",
                        "Wygrane czarnych",
                    ),

                    (
                        "white_win_rate",
                        "Wygrane białych",
                    ),

                    (
                        "draw_rate",
                        "Remisy",
                    ),
                ],

                "description":
                    (
                        "Ponieważ ta sama sieć gra przeciwko sobie, wynik około "
                        "50/50 nie oznacza braku postępu. Ten wykres służy głównie "
                        "do wykrywania przewagi jednego koloru."
                    ),
            },
        ]


        # =====================================================================
        # TWORZENIE WYKRESÓW PLOTLY
        # =====================================================================

        chart_scripts: list[
            str
        ] = []


        chart_blocks: list[
            str
        ] = []


        for chart in charts:

            traces = []


            for (
                metric_name,
                label,
            ) in chart[
                "metrics"
            ]:

                traces.append({

                    "x":
                        updates,

                    "y":
                        series(
                            metric_name
                        ),

                    "mode":
                        "lines",

                    "name":
                        label,

                    "line": {
                        "width":
                            2
                    },
                })


            layout = {

                "title": {

                    "text":
                        chart[
                            "title"
                        ],

                    "font": {
                        "size":
                            18
                    },
                },


                "margin": {

                    "t":
                        55,

                    "l":
                        60,

                    "r":
                        25,

                    "b":
                        55,
                },


                "paper_bgcolor":
                    "#ffffff",

                "plot_bgcolor":
                    "#ffffff",

                "hovermode":
                    "x unified",


                "xaxis": {

                    "title":
                        "Update",

                    "gridcolor":
                        "#e5e7eb",
                },


                "yaxis": {

                    "gridcolor":
                        "#e5e7eb",
                },


                "legend": {

                    "orientation":
                        "h",

                    "y":
                        -0.20,
                },
            }


            chart_scripts.append(

                "Plotly.newPlot("

                + json.dumps(
                    chart[
                        "id"
                    ]
                )

                + ", "

                + json.dumps(
                    traces
                )

                + ", "

                + json.dumps(
                    layout
                )

                + ", "

                + json.dumps({

                    "responsive":
                        True,

                    "displaylogo":
                        False,
                })

                + ");"
            )


            chart_blocks.append(

                """
                <section class="panel">

                    <div
                        id="__CHART_ID__"
                        class="chart">
                    </div>

                    <p class="opis">
                        __DESCRIPTION__
                    </p>

                </section>
                """

                .replace(

                    "__CHART_ID__",

                    html.escape(
                        chart[
                            "id"
                        ]
                    ),
                )

                .replace(

                    "__DESCRIPTION__",

                    html.escape(
                        chart[
                            "description"
                        ]
                    ),
                )
            )


        # =====================================================================
        # KARTY
        # =====================================================================

        cards_data = [

            (
                "Update",
                f"{latest_float('update'):.0f}",
            ),

            (
                "Partie łącznie",
                f"{latest_float('games_completed'):,.0f}"
                .replace(
                    ",",
                    " ",
                ),
            ),

            (
                "Śr. długość partii",
                f"{latest_float('mean_game_length'):.1f}",
            ),

            (
                "Entropia",
                f"{latest_float('entropy'):.3f}",
            ),

            (
                "Policy loss",
                f"{latest_float('policy_loss'):.4f}",
            ),

            (
                "Value loss",
                f"{latest_float('value_loss'):.4f}",
            ),

            (
                "Approx KL",
                f"{latest_float('approx_kl'):.4f}",
            ),

            (
                "PPO clip fraction",
                f"{latest_float('clip_fraction'):.3f}",
            ),

            (
                "Gradient średni",
                f"{latest_float('grad_norm_mean'):.3f}",
            ),

            (
                "Gradient p95",
                f"{latest_float('grad_norm_p95'):.3f}",
            ),

            (
                "Gradient max",
                f"{latest_float('grad_norm_max'):.3f}",
            ),

            (
                "Minibatche obcięte",
                f"{latest_float('grad_clip_fraction') * 100:.1f}%",
            ),

            (
                "Śr. skala gradientu",
                f"{latest_float('grad_scale_mean'):.3f}",
            ),

            (
                "Limit gradientu",
                f"{latest_float('grad_limit'):.3f}",
            ),

            (
                "Temperatura",
                f"{latest_float('temperature'):.3f}",
            ),

            (
                "Self-play",

                format_number(
                    latest_float(
                        "selfplay_positions_per_second"
                    )
                )
                +
                " poz./s",
            ),

            (
                "Cały trening",

                format_number(
                    latest_float(
                        "positions_per_second"
                    )
                )
                +
                " poz./s",
            ),

            (
                "Odrzucona historia",
                f"{latest_float('discard_fraction') * 100:.1f}%",
            ),

            (
                "Pamięć GPU",
                f"{latest_float('gpu_memory_gb'):.2f} GB",
            ),
        ]


        cards_html = ""


        for (
            label,
            value,
        ) in cards_data:

            cards_html += (

                "<div class='card'>"

                "<div class='card-label'>"

                + html.escape(
                    label
                )

                + "</div>"

                "<div class='card-value'>"

                + html.escape(
                    value
                )

                + "</div>"

                "</div>"
            )


        # =====================================================================
        # AUTOMATYCZNE PODSUMOWANIE
        # =====================================================================

        first_game_length = (
            first_valid_float(
                "mean_game_length"
            )
        )


        mean_game_length = (
            latest_float(
                "mean_game_length"
            )
        )


        summary_items: list[
            str
        ] = []


        if (
            first_game_length > 0
            and
            mean_game_length > 0
        ):

            difference_percent = (

                (
                    mean_game_length
                    /
                    first_game_length
                )

                - 1.0

            ) * 100.0


            summary_items.append(

                "Średnia długość partii "
                "zmieniła się z "

                f"{first_game_length:.1f} "

                "do "

                f"{mean_game_length:.1f} "

                f"({difference_percent:+.1f}%)."
            )


        summary_items.append(

            "Aktualna entropia wynosi "

            f"{latest_float('entropy'):.3f}, "

            "Approx KL "

            f"{latest_float('approx_kl'):.4f}, "

            "a PPO clip fraction "

            f"{latest_float('clip_fraction'):.3f}."
        )


        if (
            latest_float(
                "grad_limit"
            )
            > 0
        ):

            summary_items.append(

                "Gradient: średnia "

                f"{latest_float('grad_norm_mean'):.3f}, "

                "p95 "

                f"{latest_float('grad_norm_p95'):.3f}, "

                "limit "

                f"{latest_float('grad_limit'):.3f}; "

                "obcięto "

                f"{latest_float('grad_clip_fraction') * 100:.1f}% "

                "minibatchy, średni mnożnik "
                "po clippingu "

                f"{latest_float('grad_scale_mean'):.3f}."
            )


        summary_items.append(

            "W ostatnim update odrzucono "

            f"{latest_float('discard_fraction') * 100:.1f}% "

            "historii ruchów bez wyniku terminalnego. "
            "Plansze tych gier nie są resetowane przez PPO."
        )


        summary_html = ""


        for item in summary_items:

            summary_html += (

                "<li>"

                + html.escape(
                    item
                )

                + "</li>"
            )


        # =====================================================================
        # HTML
        # =====================================================================
        #
        # UWAGA:
        #
        # To NIE jest f-string.
        #
        # Dzięki temu klamry CSS:
        #
        # body { ... }
        #
        # nie są interpretowane przez Pythona.
        # =====================================================================

        document = """
<!doctype html>

<html lang="pl">

<head>

    <meta charset="utf-8">

    <meta
        http-equiv="refresh"
        content="15">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1">


    <title>
        Connect6 AI — dashboard treningu
    </title>


    <script
        src="https://cdn.plot.ly/plotly-3.0.1.min.js">
    </script>


    <style>

        * {
            box-sizing: border-box;
        }


        body {

            font-family:
                "Segoe UI",
                Arial,
                sans-serif;

            margin: 0;

            background:
                #f3f5f8;

            color:
                #1f2937;
        }


        .container {

            max-width:
                1700px;

            margin:
                0 auto;

            padding:
                24px;
        }


        h1 {

            margin:
                0 0 6px;

            font-size:
                30px;
        }


        .subtitle {

            color:
                #64748b;

            margin-bottom:
                20px;
        }


        .cards {

            display:
                grid;

            grid-template-columns:
                repeat(
                    auto-fit,
                    minmax(
                        170px,
                        1fr
                    )
                );

            gap:
                12px;

            margin-bottom:
                18px;
        }


        .card {

            background:
                white;

            border-radius:
                12px;

            padding:
                14px 16px;

            box-shadow:
                0 1px 4px
                rgba(
                    0,
                    0,
                    0,
                    0.08
                );
        }


        .card-label {

            color:
                #64748b;

            font-size:
                13px;

            margin-bottom:
                5px;
        }


        .card-value {

            font-size:
                22px;

            font-weight:
                700;
        }


        .summary {

            background:
                white;

            border-radius:
                12px;

            padding:
                14px 20px;

            margin-bottom:
                18px;

            box-shadow:
                0 1px 4px
                rgba(
                    0,
                    0,
                    0,
                    0.08
                );
        }


        .summary h2 {

            margin:
                0 0 8px;

            font-size:
                18px;
        }


        .summary ul {

            margin:
                0;

            padding-left:
                20px;

            line-height:
                1.6;
        }


        .grid {

            display:
                grid;

            grid-template-columns:
                repeat(
                    2,
                    minmax(
                        420px,
                        1fr
                    )
                );

            gap:
                16px;
        }


        .panel {

            background:
                white;

            border-radius:
                12px;

            padding:
                8px 10px 12px;

            box-shadow:
                0 1px 4px
                rgba(
                    0,
                    0,
                    0,
                    0.08
                );

            min-width:
                0;
        }


        .chart {

            width:
                100%;

            min-height:
                360px;
        }


        .opis {

            margin:
                0 14px 8px;

            padding-top:
                10px;

            border-top:
                1px solid
                #e5e7eb;

            color:
                #64748b;

            line-height:
                1.5;

            font-size:
                13px;
        }


        .footer {

            margin-top:
                18px;

            color:
                #64748b;

            font-size:
                12px;
        }


        @media (
            max-width:
                1000px
        ) {

            .grid {

                grid-template-columns:
                    1fr;
            }
        }

    </style>

</head>


<body>


<div class="container">


    <h1>
        Connect6 AI — dashboard treningu
    </h1>


    <div class="subtitle">

        Automatyczne odświeżanie co 15 sekund.

        Przy PPO nie oceniaj jakości modelu
        wyłącznie po całkowitym lossie.

    </div>


    <div class="cards">

        __CARDS__

    </div>


    <section class="summary">

        <h2>
            Co dzieje się teraz?
        </h2>

        <ul>

            __SUMMARY__

        </ul>

    </section>


    <div class="grid">

        __CHARTS__

    </div>


    <div class="footer">

        Dane są pobierane z metrics.csv.

        Plansze niedokończonych gier pozostają
        między update'ami PPO.

        Kasowana jest tylko historia ruchów
        bez wyniku terminalnego.

    </div>


</div>


<script>

__CHART_SCRIPTS__

</script>


</body>

</html>
"""


        document = (

            document

            .replace(
                "__CARDS__",
                cards_html,
            )

            .replace(
                "__SUMMARY__",
                summary_html,
            )

            .replace(
                "__CHARTS__",
                "".join(
                    chart_blocks
                ),
            )

            .replace(
                "__CHART_SCRIPTS__",
                "\n".join(
                    chart_scripts
                ),
            )
        )


        self.html_path.write_text(
            document,
            encoding="utf-8",
        )



# =============================================================================
# KONWERSJA DO ZWYKŁEJ LICZBY
# =============================================================================

def _scalar(
    value: Any,
) -> Any:

    if hasattr(
        value,
        "item",
    ):

        try:

            return value.item()


        except Exception:

            pass


    return value