"""Exercise a source-selected BLAS preference in standalone solution tests too."""
import pytest


@pytest.fixture(scope="session", autouse=True)
def selected_blas_backend():
    import torch

    from solution.backend_choice import configure_blas

    if not torch.cuda.is_available():
        yield
        return
    original = torch.backends.cuda.preferred_blas_library()
    configure_blas()
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(original)
