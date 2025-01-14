import torch
import torch.nn as nn

class DummyNeuralNetwork(nn.Module):
    def __init__(self):
        super(DummyNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input features are 3, output features are 64
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)  # Output a single value per point

    def forward(self, x):
        # x is of shape (B, N, 3)
        x = self.fc1(x)  # Shape: (B, N, 64)
        x = torch.relu(x)
        x = self.fc2(x)  # Shape: (B, N, 128)
        x = torch.relu(x)
        x = self.fc3(x)  # Shape: (B, N, 64)
        x = torch.relu(x)
        x = self.fc4(x)  # Shape: (B, N, 1)
        return x

# Example usage
if __name__ == "__main__":
    model = DummyNeuralNetwork()
    input_data = torch.randn(32, 100, 3)  # Batch size of 32, N=100 points, 3 features per point
    output = model(input_data)
    print(output.shape)  # Should print torch.Size([32, 100, 1])