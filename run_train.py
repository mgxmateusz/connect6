import torch

from connect6.engine.train_v5 import main


_CUDA_RESERVED_CACHE_LIMIT_BYTES = 10_000_000_000
_original_reset_peak_memory_stats = torch.cuda.reset_peak_memory_stats


def _reset_peak_memory_stats_with_cache_guard(device=None):
    """Release unused CUDA cache at update boundaries once reserved hits 10 GB."""
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved(device)
        if reserved >= _CUDA_RESERVED_CACHE_LIMIT_BYTES:
            allocated = torch.cuda.memory_allocated(device)
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            print(
                f"[cuda-cache] reserved={reserved / 1e9:.2f} GB, "
                f"allocated={allocated / 1e9:.2f} GB -> empty_cache()"
            )

    return _original_reset_peak_memory_stats(device)


torch.cuda.reset_peak_memory_stats = _reset_peak_memory_stats_with_cache_guard


if __name__ == "__main__":
    main()
