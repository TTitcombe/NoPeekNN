"""Script for training a SplitNN"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from tqdm import tqdm

import syft as sy

from src import SplitNN, NoPeekLoss, model_part1, model_part2

# Set torch-hook
hook = sy.TorchHook(torch)


def train_epoch(model, criterion, train_loader, device) -> Tuple[float, float]:
    train_loss = 0.0

    correct = 0
    total = 0

    first_model_location = model.location
    last_model_location = model.models[-1].location

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device).send(first_model_location)
        targets = targets.to(device).send(last_model_location)

        model.zero_grads()

        outputs, intermediates = model(inputs)

        losses = criterion(inputs, intermediates, outputs, targets)

        _step_loss = 0.0
        for loss in losses:
            loss.backward()
            _step_loss += loss.get().item()

        model.backward()
        model.step()

        train_loss += _step_loss

        outputs = outputs.get()
        targets = targets.get()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100 * correct / total, train_loss


def test(model, test_loader, device) -> float:
    # Evaluate on test data
    correct_test = 0
    total_test = 0

    first_model_location = model.location
    last_model_location = model.models[-1].location

    model.eval()
    for test_inputs, test_targets in test_loader:
        test_inputs = test_inputs.to(device).send(first_model_location)
        test_targets = test_targets.to(device).send(last_model_location)

        with torch.no_grad():
            outputs, _ = model(test_inputs)

        outputs = outputs.get()
        test_targets = test_targets.get()

        _, predicted = outputs.max(1)
        total_test += test_targets.size(0)
        correct_test += predicted.eq(test_targets).sum().item()

    return 100 * correct_test / total_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SplitNN with NoPeek loss")
    parser.add_argument(
        "--nopeek_weight",
        type=float,
        required=True,
        help="Weighting of NoPeek loss term. If 0.0, NoPeek is not used. Required.",
    )
    parser.add_argument(
        "--epochs", default=5, type=int, help="Number of epochs to run for (default 5)",
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size (default 64)"
    )
    parser.add_argument(
        "--learning_rate",
        default=0.6,
        type=float,
        help="Starting learning rate (default 0.6)",
    )
    parser.add_argument(
        "--saveas",
        default="nopeek",
        type=str,
        help="Name of model to save as (default is 'nopeek')."
        "Note that '_{nopeek_weight}weight' will be appended to the end of the name",
    )
    parser.add_argument(
        "--n_train_data",
        default=10_000,
        type=int,
        help="Number of training points to use (default 10'000)",
    )

    args = parser.parse_args()
    weighting = args.nopeek_weight

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File paths
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    root_model_path = project_root / "models"

    # Model name
    model_name = args.saveas + f"_{weighting}weight".replace(".", "")
    MODEL_SAVE_PATH = (root_model_path / model_name).with_suffix(".pth")

    summary_writer_path = project_root / "models" / ("tb_" + model_name)

    # ----- Model Parts -----
    models = [model_part1, model_part2]
    optims = [torch.optim.SGD(model.parameters(), lr=args.lr,) for model in models]

    # ----- Users -----
    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")

    for model, location in zip(models, [alice, bob]):
        model.send(location)

    # Create model
    split_model = SplitNN([model_part1, model_part2], optims)
    split_model.train()

    # ----- Data -----
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # PyTorch examples; https://github.com/pytorch/examples/blob/master/mnist/main.py
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_data = MNIST(data_dir, download=True, train=True, transform=data_transform)

    # We only want to use a subset of the data to force overfitting
    train_data.data = train_data.data[: args.n_train_data]
    train_data.targets = train_data.targets[: args.n_train_data]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)

    # Test data
    test_data = MNIST(data_dir, download=True, train=False, transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024)

    # ----- Train -----
    n_epochs = args.epochs

    best_accuracy = 0.0

    # writer = SummaryWriter(summary_writer_path)

    criterion = NoPeekLoss(weighting)

    epoch_pbar = tqdm(total=n_epochs)

    print("Starting training...")
    for epoch in range(n_epochs):
        train_acc, train_loss = train_epoch(
            split_model, criterion, train_loader, DEVICE
        )
        test_acc = test(split_model, test_loader, DEVICE)

        # Update tensorboard
        # writer.add_scalars("Accuracy", {"train": train_acc, "test": test_acc}, epoch)
        # writer.add_scalar("Loss/train", train_loss, epoch)

        # Save model if it's an improvement
        if test_acc > best_accuracy:
            best_accuracy = test_acc

            state_dict = {
                "model_state_dict": split_model.state_dict(),
                "epoch": epoch,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
            torch.save(state_dict, MODEL_SAVE_PATH)

        # Update prog bar text
        epoch_pbar.set_description(
            f"Train {train_acc: .2f}%; "
            f"Test {test_acc : .2f}%; "
            f"Best test {best_accuracy : .2f}%"
        )
        epoch_pbar.update(1)

    epoch_pbar.close()
    # writer.close()
