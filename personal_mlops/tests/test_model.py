import torch
import torch.nn as nn
from personal_mlops.model import DummyNeuralNetwork


def test_my_dummy_model():
    """Test the DummyNeuralNetwork."""

    model = DummyNeuralNetwork()
    assert isinstance(model, nn.Module), "Model is not an instance of nn.Module."
    assert model(torch.randn(32, 100, 3)).shape == (32, 100, 1), "Incorrect output shape."