import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from distributions import Categorical
from storchastic import StochasticMeta
from tqdm.auto import tqdm, trange
from typing import Optional
from pathlib import Path
from utils import cross_entropy_stats, accuracy_stats, StatsFn
from collections import defaultdict
from functools import partial
import matplotlib.pyplot as plt


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
    init_mean = 0.0
    init_sigma = 0.1

    def average_forward(self, x, repeats: int = 1):
        # TODO - generalize with torch.distributions.MixtureSameFamily
        return torch.log(torch.mean(torch.softmax(self(x, repeats=repeats), dim=-1), dim=-2))


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


def evaluate(
    model, loader, stat_fns: list[StatsFn], model_call_fn=None
) -> dict[str, float | torch.Tensor]:
    model.eval()
    if model_call_fn is None:
        model_call_fn = model
    stats = defaultdict(float)
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False, position=1, desc="Eval"):
            logits = model_call_fn(x)
            for fn in stat_fns:
                for name, value in fn(logits, y, reduction="sum").items():
                    stats[name] += value
    return {k: v / len(loader.dataset) for k, v in stats.items()}


def train(
    model,
    crit,
    epochs,
    model_eval_fn=None,
    stats_fns: Optional[list[StatsFn]] = None,
    logs: Optional[Path] = None,
):
    if logs is not None and not logs.exists():
        logs.mkdir(parents=True)
    writer = SummaryWriter(log_dir=logs) if logs is not None else None

    stats_fns = stats_fns or []

    train_loader, test_loader = get_mnist_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    history = []

    def _checkpoint(ep):
        nonlocal writer, logs, model, optimizer, test_loader, crit
        test_stats = evaluate(model, test_loader, stats_fns, model_call_fn=model_eval_fn)
        history.append(test_stats)
        if logs is not None:
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                },
                logs / f"checkpoint_ep{ep:03d}.pth",
            )
        if writer is not None:
            for k, v in test_stats.items():
                writer.add_scalar(f"test_{k}", v, ep)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, ep)

    starting_epoch = 0
    if logs is not None:
        latest_ckpt = None
        for ckpt in logs.glob("checkpoint_ep*.pth"):
            state = torch.load(ckpt)
            if state["epoch"] > starting_epoch:
                latest_ckpt = ckpt
                starting_epoch = state["epoch"]
        if latest_ckpt is not None:
            state = torch.load(latest_ckpt)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            starting_epoch = state["epoch"]

    for ep in trange(starting_epoch, epochs, desc="Epochs", leave=False, position=0):
        _checkpoint(ep)
        train_one_epoch(model, train_loader, optimizer, crit)
        scheduler.step()
    _checkpoint(epochs)

    return model


if __name__ == "__main__":
    logs_root = Path("logs")

    last_layer_distribution_alpha_1 = Categorical(num_classes=10, label_smoothing=0.1, alpha=1.0)
    last_layer_distribution_alpha_0 = Categorical(num_classes=10, label_smoothing=0.1, alpha=0.0)

    stats_fns = [
        cross_entropy_stats,
        accuracy_stats,
        partial(last_layer_distribution_alpha_1.mixture_kl_loss, return_components=True),
    ]

    print("Training Stochastic MLP (alpha=1.0, encourage mixture)")
    model = StochasticMLP(repeats=10)
    smlp_lamda_1 = train(
        model=model,
        crit=partial(last_layer_distribution_alpha_1.mixture_kl_loss, dim_samples=-2),
        epochs=50,
        model_eval_fn=partial(model.average_forward, repeats=10),
        stats_fns=stats_fns,
        logs=logs_root / "smlp-1.0",
    )
    print("Training Stochastic MLP (alpha=0.0, encourage single-component VI solution)")
    model = StochasticMLP(repeats=10)
    smlp_lamda_0 = train(
        model=model,
        crit=partial(last_layer_distribution_alpha_0.mixture_kl_loss, dim_samples=-2),
        epochs=50,
        model_eval_fn=partial(model.average_forward, repeats=10),
        stats_fns=stats_fns,
        logs=logs_root / "smlp-0.0",
    )
    print("Training Deterministic MLP (alpha=0.0, encourage single-component VI solution)")
    mlp_0 = train(
        model=MLP(),
        crit=last_layer_distribution_alpha_0.mixture_kl_loss,
        epochs=50,
        stats_fns=stats_fns,
        logs=logs_root / "mlp-0.0",
    )

    # Final stats - compare MLPs and SMLPs with different # of mixture components
    test_data = get_mnist_loaders()[1]
    final_stats = {
        "MLP (alpha=0.0)": evaluate(mlp_0, test_data, stats_fns),
        "SMLP (alpha=1.0, k=1)": evaluate(
            smlp_lamda_1,
            test_data,
            stats_fns,
            model_call_fn=partial(smlp_lamda_1.average_forward, repeats=1),
        ),
        "SMLP (alpha=1.0, k=10)": evaluate(
            smlp_lamda_1,
            test_data,
            stats_fns,
            model_call_fn=partial(smlp_lamda_1.average_forward, repeats=10),
        ),
        "SMLP (alpha=1.0, k=100)": evaluate(
            smlp_lamda_1,
            test_data,
            stats_fns,
            model_call_fn=partial(smlp_lamda_1.average_forward, repeats=100),
        ),
        "SMLP (alpha=0.0, k=1)": evaluate(
            smlp_lamda_0,
            test_data,
            stats_fns,
            model_call_fn=partial(smlp_lamda_1.average_forward, repeats=1),
        ),
        "SMLP (alpha=0.0, k=10)": evaluate(
            smlp_lamda_0,
            test_data,
            stats_fns,
            model_call_fn=partial(smlp_lamda_1.average_forward, repeats=10),
        ),
        "SMLP (alpha=0.0, k=100)": evaluate(
            smlp_lamda_0,
            test_data,
            stats_fns,
            model_call_fn=partial(smlp_lamda_1.average_forward, repeats=100),
        ),
    }

    num_stats = len(final_stats["MLP (alpha=0.0)"])
    figures = [plt.figure(figsize=(4, 3)) for _ in range(num_stats)]
    axes = [fig.add_subplot(1, 1, 1) for fig in figures]
    for x, (name, stats) in enumerate(final_stats.items()):
        for i, (stat_name, stat_value) in enumerate(stats.items()):
            axes[i].bar(x=x, height=stat_value)
    for ax, name in zip(axes, final_stats["MLP (alpha=0.0)"].keys()):
        ax.set_title(name)
        ax.set_xticks(range(len(final_stats)), labels=final_stats.keys(), rotation=45)
    for fig in figures:
        fig.tight_layout()
    plt.show()
