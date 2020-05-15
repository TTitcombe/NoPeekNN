"""
Neural networks
"""
import torch


class SplitNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encode = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(9216, 500),
        )

        self.decode = torch.nn.Sequential(
            torch.nn.Linear(500, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        intermediate = self.encode(x)
        return self.decode(intermediate), intermediate
