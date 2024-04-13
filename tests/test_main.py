import pytest
import torch
from revnets.main.main import Experiment

gpu_available = torch.cuda.is_available()


@pytest.mark.skipif(
    not gpu_available, reason="Only test model training if GPU is available"
)
def test_main(test_context: None) -> None:
    Experiment().run()
