import pytest
import torch
from revnets.main.main import Experiment

gpu_available = torch.cuda.is_available()


@pytest.mark.skipif(
    not gpu_available,
    reason="Only test model training if GPU is available",
)
@pytest.mark.usefixtures("test_context")
def test_main() -> None:
    Experiment().run()
