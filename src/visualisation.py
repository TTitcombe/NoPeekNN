from os import PathLike
from typing import Optional

import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def plot_images(tensors, rows: Optional[int] = None, cols: Optional[int] = None, save_path: Optional[PathLike] = None):
    """
    Plot normalised MNIST tensors as images
    """
    fig = plt.figure(figsize=(20, 10))

    n_tensors = len(tensors)
    n_cols = cols if cols else min(n_tensors, 4)
    n_rows = rows if rows else int((n_tensors - 1) / 4) + 1

    # De-normalise an MNIST tensor
    mu = torch.tensor([0.1307], dtype=torch.float32)
    sigma = torch.tensor([0.3081], dtype=torch.float32)
    Unnormalise = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())

    for row in range(n_rows):
        for col in range(n_cols):
            idx = n_cols * row + col

            if idx > n_tensors - 1:
                break

            ax = fig.add_subplot(n_rows, n_cols, idx + 1)
            tensor = Unnormalise(tensors[idx])

            # Clip image values so we can plot
            tensor[tensor < 0] = 0
            tensor[tensor > 1] = 1

            tensor = tensor.squeeze(0)  # remove batch dim

            ax.imshow(transforms.ToPILImage()(tensor), interpolation="bicubic")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
