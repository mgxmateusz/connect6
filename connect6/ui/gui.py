from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import torch

from .checkpoint import load_checkpoint, load_model_for_inference
from .game import BLACK, WHITE, Connect6Game
from .model import mask_logits


HUMAN = "Human"


class Connect6GUI:
    """Okno do rozgrywek człowiek/AI w Connect6.

    Logika gry jest celowo oddzielona od GUI. Ten moduł odpowiada tylko za
    wyświetlanie planszy, interakcję użytkownika i wybór modeli.
    """

    # ------------------------------------------------------------------
    # Paleta interfejsu - tutaj najłatwiej zmienić wygląd całego GUI.
    # ------------------------------------------------------------------
    BG = "#11151c"
    PANEL = "#191f29"
    PANEL_2 = "#202734"
    TEXT = "#eef2f7"
    MUTED = "#9aa6b2"
    ACCENT = "#5c8dff"
    ACCENT_ACTIVE = "#769fff"
    BORDER = "#2d3745"

    LIGHT_SQUARE = "#d8dde5"
    DARK_SQUARE = "#8793a3"
    BOARD_BORDER = "#303846"
    LAST_MOVE = "#ffb020"
    HOVER = "#66a3ff"

    BLACK_STONE = "#15191f"
    BLACK_STONE_EDGE = "#050607"
    WHITE_STONE = "#f7f8fb"
    WHITE_STONE_EDGE = "#aeb7c3"

    def __init__(self, root: tk.Tk, runs_dir: str | Path = "runs"):
        self.root = root
        self.root.title("Connect6 AI Arena")
        self.root.geometry("1120x940")
        self.root.minsize(980, 860)
        self.root.configure(bg=self.BG)

        self.runs_dir = Path(runs_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cache: dict[str, tuple[torch.nn.Module, dict]] = {}
        self.model_paths: dict[str, Path] = {}
        self.game = Connect6Game()
        self.paused = False
        self.ai_job: str | None = None
        self.hover_rc: tuple[int, int] | None = None

        self.black_source = tk.StringVar(value=HUMAN)
        self.white_source = tk.StringVar(value=HUMAN)
        self.delay_ms = tk.IntVar(value=400)
        self.temperature = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar(value="Ready")

        self._configure_styles()
        self._build_header()
        self._build_controls()
        self._build_board()
        self._build_status_bar()

        self.refresh_models()
        self.new_game()

    # ------------------------------------------------------------------
    # Układ i styl interfejsu
    # ------------------------------------------------------------------
    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("Panel2.TFrame", background=self.PANEL_2)

        style.configure(
            "Title.TLabel",
            background=self.BG,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 21),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BG,
            foreground=self.MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Field.TLabel",
            background=self.PANEL,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Status.TLabel",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 10),
            padding=(12, 8),
        )

        style.configure(
            "TCombobox",
            fieldbackground=self.PANEL_2,
            background=self.PANEL_2,
            foreground=self.TEXT,
            arrowcolor=self.TEXT,
            bordercolor=self.BORDER,
            lightcolor=self.BORDER,
            darkcolor=self.BORDER,
            padding=6,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.PANEL_2)],
            foreground=[("readonly", self.TEXT)],
            selectbackground=[("readonly", self.PANEL_2)],
            selectforeground=[("readonly", self.TEXT)],
        )

        style.configure(
            "TSpinbox",
            fieldbackground=self.PANEL_2,
            foreground=self.TEXT,
            background=self.PANEL_2,
            bordercolor=self.BORDER,
            arrowcolor=self.TEXT,
            padding=5,
        )

        style.configure(
            "Primary.TButton",
            background=self.ACCENT,
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
            padding=(14, 8),
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.ACCENT_ACTIVE), ("pressed", self.ACCENT_ACTIVE)],
        )

        style.configure(
            "Secondary.TButton",
            background=self.PANEL_2,
            foreground=self.TEXT,
            bordercolor=self.BORDER,
            padding=(12, 8),
            font=("Segoe UI", 9),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#293241"), ("pressed", "#303b4c")],
            foreground=[("active", "#ffffff")],
        )

    def _build_header(self) -> None:
        header = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18, 24, 8))
        header.pack(fill="x")

        left = ttk.Frame(header, style="App.TFrame")
        left.pack(side="left", fill="x", expand=True)
        ttk.Label(left, text="Connect6 AI Arena", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="Human vs AI / AI vs AI • checkpoint browser • live move playback",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        device_text = f"INFERENCE: {str(self.device).upper()}"
        badge = tk.Label(
            header,
            text=device_text,
            bg="#18345c" if self.device.type == "cuda" else "#343b45",
            fg="#dce9ff",
            font=("Segoe UI Semibold", 9),
            padx=12,
            pady=7,
        )
        badge.pack(side="right", anchor="n")

    def _build_controls(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=(24, 6, 24, 10))
        outer.pack(fill="x")

        panel = ttk.Frame(outer, style="Panel.TFrame", padding=14)
        panel.pack(fill="x")
        for col in range(4):
            panel.columnconfigure(col, weight=1)

        ttk.Label(panel, text="BLACK PLAYER", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(panel, text="WHITE PLAYER", style="Section.TLabel").grid(row=0, column=2, sticky="w", padx=(18, 0))

        self.black_combo = ttk.Combobox(
            panel,
            textvariable=self.black_source,
            state="readonly",
            width=42,
        )
        self.black_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 12), padx=(0, 9))

        self.white_combo = ttk.Combobox(
            panel,
            textvariable=self.white_source,
            state="readonly",
            width=42,
        )
        self.white_combo.grid(row=1, column=2, columnspan=2, sticky="ew", pady=(5, 12), padx=(9, 0))

        settings = ttk.Frame(panel, style="Panel.TFrame")
        settings.grid(row=2, column=0, columnspan=4, sticky="ew")

        ttk.Label(settings, text="AI delay [ms]", style="Field.TLabel").pack(side="left")
        ttk.Spinbox(
            settings,
            from_=0,
            to=5000,
            increment=50,
            textvariable=self.delay_ms,
            width=8,
        ).pack(side="left", padx=(7, 20))

        ttk.Label(settings, text="Temperature", style="Field.TLabel").pack(side="left")
        ttk.Spinbox(
            settings,
            from_=0.0,
            to=2.0,
            increment=0.1,
            textvariable=self.temperature,
            width=7,
        ).pack(side="left", padx=(7, 20))

        ttk.Label(settings, text="0 = deterministic / argmax", style="Field.TLabel").pack(side="left")

        buttons = ttk.Frame(panel, style="Panel.TFrame")
        buttons.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(13, 0))

        ttk.Button(buttons, text="New game", command=self.new_game, style="Primary.TButton").pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Pause / Resume", command=self.toggle_pause, style="Secondary.TButton").pack(side="left", padx=3)
        ttk.Button(buttons, text="Step AI", command=self.step_ai_once, style="Secondary.TButton").pack(side="left", padx=3)
        ttk.Button(buttons, text="Refresh models", command=self.refresh_models, style="Secondary.TButton").pack(side="left", padx=3)
        ttk.Button(buttons, text="Runs folder", command=self.choose_runs_dir, style="Secondary.TButton").pack(side="left", padx=3)

    def _build_board(self) -> None:
        board_outer = ttk.Frame(self.root, style="App.TFrame", padding=(24, 0, 24, 0))
        board_outer.pack(fill="both", expand=True)

        board_panel = ttk.Frame(board_outer, style="Panel.TFrame", padding=14)
        board_panel.pack(anchor="center")

        # 760 px daje pola około 40 px na standardowej planszy 19x19.
        self.canvas_size = 760
        self.margin = 10
        self.canvas = tk.Canvas(
            board_panel,
            width=self.canvas_size,
            height=self.canvas_size,
            bg=self.BOARD_BORDER,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>", self._on_mouse_leave)

    def _build_status_bar(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=(24, 10, 24, 18))
        outer.pack(fill="x")
        status_panel = ttk.Frame(outer, style="Panel.TFrame")
        status_panel.pack(fill="x")
        ttk.Label(status_panel, textvariable=self.status, style="Status.TLabel").pack(fill="x")

    # ------------------------------------------------------------------
    # Wybór checkpointów
    # ------------------------------------------------------------------
    def choose_runs_dir(self) -> None:
        folder = filedialog.askdirectory(
            initialdir=str(self.runs_dir if self.runs_dir.exists() else Path.cwd())
        )
        if folder:
            self.runs_dir = Path(folder)
            self.model_cache.clear()
            self.refresh_models()

    def refresh_models(self) -> None:
        self.model_paths.clear()
        if self.runs_dir.exists():
            checkpoints = list(self.runs_dir.rglob("*.pt"))
            checkpoints = [p for p in checkpoints if p.parent.name == "checkpoints"]
            # latest.pt jest zawsze pierwszy, potem checkpointy od najnowszego.
            checkpoints.sort(
                key=lambda p: (p.name == "latest.pt", p.stat().st_mtime),
                reverse=True,
            )
            for p in checkpoints:
                try:
                    rel = p.relative_to(self.runs_dir)
                    label = str(rel).replace("\\", "/")
                except ValueError:
                    label = str(p)
                self.model_paths[label] = p

        values = [HUMAN] + list(self.model_paths.keys())
        self.black_combo["values"] = values
        self.white_combo["values"] = values
        if self.black_source.get() not in values:
            self.black_source.set(HUMAN)
        if self.white_source.get() not in values:
            self.white_source.set(HUMAN)
        self.status.set(f"Models found: {len(values) - 1}  •  device: {self.device}")

    # ------------------------------------------------------------------
    # Rysowanie gry i planszy
    # ------------------------------------------------------------------
    def new_game(self) -> None:
        self._cancel_ai_job()
        size = self._selected_board_size()
        self.game = Connect6Game(board_size=size, win_length=6)
        self.paused = False
        self.hover_rc = None
        self.draw_board()
        self._update_status()
        self._schedule_ai_if_needed()

    def _selected_board_size(self) -> int:
        sizes = []
        for src in (self.black_source.get(), self.white_source.get()):
            if src != HUMAN and src in self.model_paths:
                try:
                    payload = load_checkpoint(self.model_paths[src], map_location="cpu")
                    sizes.append(int(payload["game_config"]["board_size"]))
                except Exception as exc:
                    messagebox.showerror("Checkpoint error", f"Cannot inspect {src}:\n{exc}")
        if sizes and len(set(sizes)) > 1:
            raise ValueError("Selected checkpoints use different board sizes")
        return sizes[0] if sizes else 19

    def draw_board(self) -> None:
        """Rysuje planszę jak w szachach/warcabach - kamienie leżą na środku pól."""
        self.canvas.delete("all")
        n = self.game.board_size
        step = self._cell_size()

        # Cień i ramka planszy.
        x0 = self.margin
        y0 = self.margin
        x1 = self.margin + n * step
        y1 = self.margin + n * step
        self.canvas.create_rectangle(
            x0 - 2,
            y0 - 2,
            x1 + 2,
            y1 + 2,
            fill=self.BOARD_BORDER,
            outline=self.BOARD_BORDER,
            width=0,
        )

        # Naprzemienne pola jak na szachownicy.
        for r in range(n):
            for c in range(n):
                left = self.margin + c * step
                top = self.margin + r * step
                fill = self.LIGHT_SQUARE if (r + c) % 2 == 0 else self.DARK_SQUARE
                self.canvas.create_rectangle(
                    left,
                    top,
                    left + step + 0.5,
                    top + step + 0.5,
                    fill=fill,
                    outline=fill,
                    width=0,
                )

        # Podświetlenie pola pod kursorem gracza.
        if (
            self.hover_rc is not None
            and not self.game.done
            and not self.paused
            and self.current_source() == HUMAN
        ):
            hr, hc = self.hover_rc
            if self.game.board[hr, hc] == 0:
                left = self.margin + hc * step
                top = self.margin + hr * step
                self.canvas.create_rectangle(
                    left + 2,
                    top + 2,
                    left + step - 2,
                    top + step - 2,
                    outline=self.HOVER,
                    width=2,
                )

        # Ostatni ruch jest zaznaczony prostą ramką pod kamieniem.
        if self.game.last_action is not None:
            r, c = self.game.action_to_rc(self.game.last_action)
            left = self.margin + c * step
            top = self.margin + r * step
            self.canvas.create_rectangle(
                left + 2,
                top + 2,
                left + step - 2,
                top + step - 2,
                outline=self.LAST_MOVE,
                width=3,
            )

        # Kamienie są wyśrodkowane wewnątrz pól, jak pionki w warcabach.
        radius = max(7.0, step * 0.37)
        shadow_offset = max(1.5, step * 0.045)
        for r in range(n):
            for c in range(n):
                val = int(self.game.board[r, c])
                if val == 0:
                    continue

                x, y = self._xy(r, c)

                # Mały cień dodaje głębi bez wyglądu drewnianej planszy Go.
                self.canvas.create_oval(
                    x - radius + shadow_offset,
                    y - radius + shadow_offset,
                    x + radius + shadow_offset,
                    y + radius + shadow_offset,
                    fill="#242a33",
                    outline="",
                )

                if val == BLACK:
                    fill = self.BLACK_STONE
                    edge = self.BLACK_STONE_EDGE
                    highlight = "#3a414b"
                else:
                    fill = self.WHITE_STONE
                    edge = self.WHITE_STONE_EDGE
                    highlight = "#ffffff"

                self.canvas.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=fill,
                    outline=edge,
                    width=max(1, int(step * 0.04)),
                )

                # Delikatny refleks poprawia czytelność kamieni na obu kolorach pól.
                h = radius * 0.28
                self.canvas.create_oval(
                    x - radius * 0.48,
                    y - radius * 0.48,
                    x - radius * 0.48 + h,
                    y - radius * 0.48 + h,
                    fill=highlight,
                    outline="",
                )

    def _cell_size(self) -> float:
        n = self.game.board_size
        return (self.canvas_size - 2 * self.margin) / n

    def _xy(self, row: int, col: int) -> tuple[float, float]:
        step = self._cell_size()
        return (
            self.margin + (col + 0.5) * step,
            self.margin + (row + 0.5) * step,
        )

    def _nearest_rc(self, x: float, y: float) -> tuple[int, int] | None:
        n = self.game.board_size
        step = self._cell_size()
        local_x = x - self.margin
        local_y = y - self.margin
        if local_x < 0 or local_y < 0:
            return None
        col = int(local_x // step)
        row = int(local_y // step)
        if not (0 <= row < n and 0 <= col < n):
            return None
        return row, col

    def _on_mouse_move(self, event: tk.Event) -> None:
        rc = self._nearest_rc(event.x, event.y)
        if rc != self.hover_rc:
            self.hover_rc = rc
            self.draw_board()

    def _on_mouse_leave(self, _event: tk.Event) -> None:
        if self.hover_rc is not None:
            self.hover_rc = None
            self.draw_board()

    # ------------------------------------------------------------------
    # Sterowanie człowiekiem i AI
    # ------------------------------------------------------------------
    def current_source(self) -> str:
        return self.black_source.get() if self.game.current_player == BLACK else self.white_source.get()

    def on_click(self, event: tk.Event) -> None:
        if self.game.done or self.paused or self.current_source() != HUMAN:
            return
        rc = self._nearest_rc(event.x, event.y)
        if rc is None:
            return
        r, c = rc
        if self.game.board[r, c] != 0:
            return
        self.game.step(self.game.rc_to_action(r, c))
        self.draw_board()
        self._update_status()
        self._schedule_ai_if_needed()

    def _get_model(self, source: str) -> tuple[torch.nn.Module, dict]:
        if source in self.model_cache:
            return self.model_cache[source]
        path = self.model_paths[source]
        model, payload = load_model_for_inference(path, self.device)
        self.model_cache[source] = (model, payload)
        return model, payload

    @torch.inference_mode()
    def _ai_action(self, source: str) -> int:
        model, _ = self._get_model(source)

        # ---------------------------------------------------------
        # Budujemy wejście DOKŁADNIE tak jak podczas treningu:
        #
        # 0..360   = MOJE kamienie
        # 361..721 = kamienie PRZECIWNIKA
        # 722      = stones_left / 2
        # 723      = is_black
        #
        # "MOJE" zawsze oznacza gracza, który TERAZ wykonuje ruch.
        # ---------------------------------------------------------

        board = torch.from_numpy(self.game.board).to(self.device)
        player = int(self.game.current_player)

        me = board.eq(player).reshape(-1).to(torch.float32)
        opp = board.eq(-player).reshape(-1).to(torch.float32)

        game_info = torch.tensor(
            [
                float(self.game.stones_left_in_turn) / 2.0,
                1.0 if player == BLACK else 0.0,
            ],
            dtype=torch.float32,
            device=self.device,
        )

        network_input = torch.cat(
            [me, opp, game_info],
            dim=0,
        ).unsqueeze(0)

        # Kontrola bezpieczeństwa.
        expected = int(model.input_size)
        if network_input.shape != (1, expected):
            raise RuntimeError(
                f"Zły input GUI: {tuple(network_input.shape)}, "
                f"model oczekuje (1, {expected})"
            )

        legal = (
            torch.from_numpy(self.game.legal_mask())
            .unsqueeze(0)
            .to(self.device)
        )

        logits, _ = model(network_input)
        logits = mask_logits(logits.float(), legal)

        temp = float(self.temperature.get())

        if temp <= 0:
            return int(logits.argmax(dim=1).item())

        probs = torch.softmax(logits / max(temp, 1e-4), dim=1)
        return int(torch.multinomial(probs, 1).item())

    def _schedule_ai_if_needed(self) -> None:
        self._cancel_ai_job()
        if self.paused or self.game.done or self.current_source() == HUMAN:
            return
        self.ai_job = self.root.after(max(0, int(self.delay_ms.get())), self._do_ai_step)

    def _do_ai_step(self) -> None:
        self.ai_job = None
        if self.paused or self.game.done:
            return
        source = self.current_source()
        if source == HUMAN:
            return
        try:
            action = self._ai_action(source)
            self.game.step(action)
        except Exception as exc:
            messagebox.showerror("AI error", str(exc))
            self.paused = True
            return
        self.draw_board()
        self._update_status()
        self._schedule_ai_if_needed()

    def step_ai_once(self) -> None:
        self._cancel_ai_job()
        if self.game.done or self.current_source() == HUMAN:
            return
        was_paused = self.paused
        self.paused = False
        self._do_ai_step()
        self.paused = was_paused
        self._update_status()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self._cancel_ai_job()
        else:
            self._schedule_ai_if_needed()
        self._update_status()

    def _cancel_ai_job(self) -> None:
        if self.ai_job is not None:
            try:
                self.root.after_cancel(self.ai_job)
            except tk.TclError:
                pass
            self.ai_job = None

    def _update_status(self) -> None:
        if self.game.done:
            if self.game.winner == BLACK:
                msg = "Game over  •  BLACK wins"
            elif self.game.winner == WHITE:
                msg = "Game over  •  WHITE wins"
            else:
                msg = "Game over  •  DRAW"
        else:
            color = "BLACK" if self.game.current_player == BLACK else "WHITE"
            msg = (
                f"{color} to move"
                f"  •  stones left this turn: {self.game.stones_left_in_turn}"
                f"  •  moves: {self.game.move_count}"
            )
            if self.paused:
                msg += "  •  PAUSED"
        self.status.set(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect6 GUI: human/model/model-vs-model")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    root = tk.Tk()
    Connect6GUI(root, args.runs_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
