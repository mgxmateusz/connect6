"""Benchmarks and model evaluation utilities."""

# Evaluation modules keep their original relative imports after the package
# split; bind those names to the canonical engine/bot modules.
import sys

from connect6.bots import gpu_bot as _gpu_bot
from connect6.engine import checkpoint as _checkpoint
from connect6.engine import config as _config
from connect6.engine import game as _game
from connect6.engine import model as _model
from connect6.engine import vector_env as _vector_env

for _name, _module in {
    "checkpoint": _checkpoint,
    "config": _config,
    "game": _game,
    "model": _model,
    "vector_env": _vector_env,
    "gpu_bot": _gpu_bot,
}.items():
    sys.modules.setdefault(f"{__name__}.{_name}", _module)
