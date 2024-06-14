import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loss import CategoricalKLQP
from storchastic import StochasticMeta
from tqdm.auto import tqdm, trange
from typing import Optional
from pathlib import Path
from pprint import pprint


def get_mnist_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return trainset, testset


def get_mnist_loaders(batch_size=64):
    trainset, testset = get_mnist_data()
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StochasticMLP(MLP, metaclass=StochasticMeta):
    def average_forward(self, x, repeats: int = 1):
        return torch.log(sum(torch.softmax(self(x), dim=1) for _ in range(repeats)) / repeats)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    for x, y in tqdm(loader, leave=False, position=1, desc="Train"):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if torch.isnan(loss).any():
            raise ValueError("Loss is NaN")


def evaluate(model, loader, criterion, model_call_fn=None):
    model.eval()
    if model_call_fn is None:
        model_call_fn = model
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, position=1, desc="Eval"):
            logits = model_call_fn(x)
            total_loss += criterion(logits, y).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def train(model, crit, epochs, logs: Optional[Path] = None):
    if logs is not None and not logs.exists():
        logs.mkdir(parents=True)
    writer = SummaryWriter(log_dir=logs) if logs is not None else None

    train_loader, test_loader = get_mnist_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    def _checkpoint(ep):
        nonlocal writer, logs, model, optimizer, test_loader, crit
        test_loss, test_acc = evaluate(model, test_loader, crit)
        if logs is not None:
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
            torch.save(state, logs / f"checkpoint_ep{ep:03d}.pth")
        if writer is not None:
            writer.add_scalar("Test Loss", test_loss, ep)
            writer.add_scalar("Test Accuracy", test_acc, ep)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, ep)

    starting_epoch = 0
    if logs is not None:
        latest_ckpt = max(
            logs.glob("checkpoint_ep*.pth"), key=lambda p: int(p.stem.split("_ep")[-1])
        )
        if latest_ckpt.exists():
            state = torch.load(latest_ckpt)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            starting_epoch = int(latest_ckpt.stem.split("_ep")[-1])

    for ep in trange(starting_epoch, epochs, desc="Epochs", leave=False, position=0):
        _checkpoint(ep)
        train_one_epoch(model, train_loader, optimizer, crit)
    _checkpoint(epochs)

    return model


if __name__ == "__main__":
    logs_root = Path("logs")
    print("Training Stochastic MLP (lambda=1.0, encourage mixture)")
    smlp_lamda_1 = train(
        model=StochasticMLP(),
        crit=CategoricalKLQP(num_classes=10, lam=1.0),
        epochs=20,
        logs=logs_root / "smlp-1.0",
    )
    print("Training Stochastic MLP (lambda=0.0, encourage single-component VI solution)")
    smlp_lamda_0 = train(
        model=StochasticMLP(),
        crit=CategoricalKLQP(num_classes=10, lam=0.0),
        epochs=20,
        logs=logs_root / "smlp-0.0",
    )
    print("Training MLP")
    mlp = train(
        model=MLP(),
        crit=CategoricalKLQP(num_classes=10, lam=0.0),
        epochs=20,
        logs=logs_root / "mlp",
    )

    stats = {"test_ce": {}, "test_acc": {}}

    # Evaluate MLP the old-fashioned way
    stats["test_ce"]["mlp"], stats["test_acc"]["mlp"] = evaluate(
        mlp, get_mnist_loaders()[1], nn.CrossEntropyLoss()
    )

    # Evaluate StochasticMLPs using mixture posterior for different #s of samples
    # TODO - track these stats in tensorboard
    k_values = [1, 10, 100]
    for k in k_values:
        stats["test_ce"][f"smlp_1_{k}"], stats["test_acc"][f"smlp_1_{k}"] = evaluate(
            smlp_lamda_1,
            get_mnist_loaders()[1],
            nn.CrossEntropyLoss(),
            model_call_fn=lambda x: smlp_lamda_1.average_forward(x, k),
        )
        stats["test_ce"][f"smlp_0_{k}"], stats["test_acc"][f"smlp_0_{k}"] = evaluate(
            smlp_lamda_0,
            get_mnist_loaders()[1],
            nn.CrossEntropyLoss(),
            model_call_fn=lambda x: smlp_lamda_0.average_forward(x, k),
        )

    pprint(stats)
