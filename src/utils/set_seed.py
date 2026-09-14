"""Deterministic seeding helper for Paper 2.

Sets all RNGs (Python, numpy, PyTorch CPU/CUDA/MPS) and configures backend
flags so that runs reproduce within the tolerance band documented in the
paper (Appendix A, Reproducibility, and README).

Usage at the top of every entry point::

    from utils.set_seed import set_seed
    g_torch = set_seed(42)
    # ...
    loader = DataLoader(ds, ..., generator=g_torch,
                        worker_init_fn=get_worker_init_fn(42))

Notes
-----
* On CUDA, ``CUBLAS_WORKSPACE_CONFIG`` is set to ``':4096:8'`` because
  ``torch.use_deterministic_algorithms(True)`` requires it.
* On Apple MPS, full determinism is not yet supported for every op
  (notably reduction kernels). Seeding still yields close-to-deterministic
  results within the tolerance band stated in requirements.txt.
* This helper is idempotent — calling it twice with the same seed is safe.
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> torch.Generator:
    """Seed every RNG and configure determinism flags.

    Parameters
    ----------
    seed : int
        The random seed used for Python random, numpy, and torch.
    deterministic : bool, default True
        When True, set ``torch.use_deterministic_algorithms(True)`` and
        the corresponding cuDNN flags. When False, only seeds are set
        (useful for speed comparisons; not used in reported experiments).

    Returns
    -------
    torch.Generator
        A torch.Generator seeded with ``seed``, suitable for passing
        to a DataLoader via the ``generator=`` keyword.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    # 1) Python and numpy
    random.seed(seed)
    np.random.seed(seed)

    # 2) PyTorch base seeds (CPU)
    torch.manual_seed(seed)

    # 3) CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Required by torch.use_deterministic_algorithms on CUDA:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # 4) Apple MPS
    if torch.backends.mps.is_available():
        # torch.mps.manual_seed was added in PyTorch 2.0+
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            warnings.warn(
                "torch.mps.manual_seed not available; "
                "MPS RNG state may not be fully reproducible.",
                stacklevel=2,
            )

    # 5) Determinism flags
    if deterministic:
        # CUDA backends:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Global flag (PyTorch >= 1.8):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, AttributeError) as e:
            # warn_only=True downgrades non-deterministic-op errors to
            # warnings; older PyTorch may not accept the keyword.
            try:
                torch.use_deterministic_algorithms(True)
            except (RuntimeError, AttributeError):
                warnings.warn(
                    f"torch.use_deterministic_algorithms not fully applied: {e}",
                    stacklevel=2,
                )

    # 6) DataLoader generator
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_worker_init_fn(seed: int = 42):
    """Return a worker_init_fn that seeds each DataLoader worker deterministically.

    Pass to DataLoader via ``worker_init_fn=get_worker_init_fn(seed)``.
    """
    def _worker_init(worker_id: int) -> None:
        # Each worker gets a deterministic but distinct sub-seed.
        worker_seed = seed * 1000 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _worker_init
