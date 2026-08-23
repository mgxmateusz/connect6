"""Tournament engines and the fixed-bot autosave gauntlet."""

# The championship modules are kept byte-for-byte while being grouped here.
# Their historical relative imports are mapped to the canonical engine modules.
import sys

import connect6.cuda_native as _cuda_native
from connect6.bots import gpu_bot as _gpu_bot
from connect6.engine import checkpoint as _checkpoint
from connect6.engine import history as _history
from connect6.engine import model as _model
from connect6.engine import vector_env as _vector_env

for _name, _module in {
    "checkpoint": _checkpoint,
    "history": _history,
    "model": _model,
    "vector_env": _vector_env,
    "cuda_native": _cuda_native,
    "gpu_bot": _gpu_bot,
}.items():
    sys.modules.setdefault(f"{__name__}.{_name}", _module)
