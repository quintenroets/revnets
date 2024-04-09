import pytest
import torch
from revnets.main.main import main

gpu_available = torch.cuda.is_available()


@pytest.mark.skipif(
    not gpu_available, reason="Only test model training if GPU is available"
)
def test_main() -> None:
    main()
