"""
Neural networks
"""
import torch


model_part1 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, 3, 1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(32, 64, 3, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    torch.nn.Linear(9216, 500),
)


model_part2 = torch.nn.Sequential(
    torch.nn.Linear(500, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
    torch.nn.Softmax(dim=1),
)


class SplitNN(torch.nn.Module):
    def __init__(self, models, optimisers) -> None:
        super().__init__()

        self.models = models
        self.optimisers = optimisers

        self.outputs = [None] * len(self.models)
        self.inputs = [None] * len(self.models)

    def forward(self, x):
        self.inputs[0] = x
        self.outputs[0] = self.models[0](self.inputs[0])

        for i in range(1, len(self.models)):
            self.inputs[i] = self.outputs[i - 1].detach().requires_grad_()
            if self.outputs[i - 1].location != self.models[i].location:
                self.inputs[i] = (
                    self.inputs[i].move(self.models[i].location).requires_grad_()
                )
            self.outputs[i] = self.models[i](self.inputs[i])

        return self.outputs[-1], self.outputs[0]

    def backward(self):
        for i in range(len(self.models) - 2, -1, -1):
            grad_in = self.inputs[i + 1].grad.copy()
            if self.outputs[i].location != self.inputs[i + 1].location:
                grad_in = grad_in.move(self.outputs[i].location)
            self.outputs[i].backward(grad_in)

    def zero_grads(self):
        for opt in self.optimisers:
            opt.zero_grad()

    def step(self):
        for opt in self.optimisers:
            opt.step()

    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()

    @property
    def location(self):
        return self.models[0].location if self.models and len(self.models) else None
