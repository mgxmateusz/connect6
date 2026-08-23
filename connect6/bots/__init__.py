"""Hand-written Connect6 opponents."""

# The tactical bot predates the package split and imports `.cuda_native`.
# Keep that dependency canonical without duplicating CUDA modules.
import sys

import connect6.cuda_native as _cuda_native
from connect6.cuda_native import bot_loader as _bot_loader

sys.modules.setdefault(__name__ + ".cuda_native", _cuda_native)
sys.modules.setdefault(__name__ + ".cuda_native.bot_loader", _bot_loader)
