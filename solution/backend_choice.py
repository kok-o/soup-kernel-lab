"""Optional installed PyTorch BLAS preference; no precision/workspace overrides.

API: https://docs.pytorch.org/docs/2.11/backends.html#torch.backends.cuda.preferred_blas_library
"""


def configure_blas():
    from ._speed_config import BLAS_LIBRARY

    if BLAS_LIBRARY is not None:
        import torch

        torch.backends.cuda.preferred_blas_library(BLAS_LIBRARY)
