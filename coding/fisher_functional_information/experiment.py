"""
Main experiment: Does Training Obey a Fisher Fundamental Theorem?

Measures functional information I_f(t) and Fisher trace T_F(t) at each epoch
across three settings (MLP/MNIST, CNN/CIFAR-10, Transformer/SyntheticText).

Tests two hypotheses:
  Part A: I_f(t) increases monotonically during training.
  Part E: dI_f/dt ∝ T_F(t)  (Fisher fundamental theorem analog).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))

from models import MLP, SmallCNN, SmallTransformer, count_parameters, init_weights_kaiming
from data import get_mnist, get_cifar10, get_synthetic_text, randomize_labels
from measurements import (
    compute_loss_on_dataset,
    cache_random_network_losses,
    compute_functional_information,
    compute_fisher_trace_efficient,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


SETTINGS = {
    "mlp_mnist": {
        "model_cls": MLP,
        "model_kwargs": {"input_dim": 784, "hidden_dim": 256, "num_classes": 10},
        "data_fn": get_mnist,
        "num_classes": 10,
        "epochs": 20,
        "lr": 0.01,
        "M": 1000,
        "fisher_samples": 128,
        "batch_limit_random": 10,
    },
    "cnn_cifar10": {
        "model_cls": SmallCNN,
        "model_kwargs": {"num_classes": 10},
        "data_fn": get_cifar10,
        "num_classes": 10,
        "epochs": 20,
        "lr": 0.01,
        "M": 1000,
        "fisher_samples": 128,
        "batch_limit_random": 10,
    },
    "transformer_text": {
        "model_cls": SmallTransformer,
        "model_kwargs": {
            "vocab_size": 10000, "d_model": 128, "nhead": 4,
            "num_layers": 2, "num_classes": 4, "max_len": 64,
        },
        "data_fn": get_synthetic_text,
        "num_classes": 4,
        "epochs": 20,
        "lr": 0.001,
        "M": 1000,
        "fisher_samples": 64,
        "batch_limit_random": 10,
    },
}


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


def run_single_experiment(setting_name, config, device, random_labels=False,
                          optimizer_name="sgd", lr_override=None, weight_decay=0.0):
    """Run one full training + measurement loop."""
    print(f"\n{'='*70}")
    print(f"Setting: {setting_name} | optimizer={optimizer_name} | "
          f"random_labels={random_labels} | wd={weight_decay}")
    print(f"{'='*70}")

    lr = lr_override if lr_override is not None else config["lr"]
    epochs = config["epochs"]

    # Load data
    print("Loading data...")
    train_loader, test_loader = config["data_fn"](batch_size=128)

    if random_labels:
        train_loader = randomize_labels(
            train_loader, num_classes=config["num_classes"]
        )

    # Initialize model
    model = config["model_cls"](**config["model_kwargs"]).to(device)
    init_weights_kaiming(model)
    print(f"Model parameters: {count_parameters(model):,}")

    # Setup optimizer
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd_momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Cache random network losses (computed ONCE)
    print(f"\nCaching M={config['M']} random network losses...")
    t0 = time.time()
    sorted_random_losses, tail_model = cache_random_network_losses(
        model_cls=config["model_cls"],
        model_kwargs=config["model_kwargs"],
        data_loader=train_loader,
        device=device,
        M=config["M"],
        batch_limit=config["batch_limit_random"],
    )
    print(f"  Cached in {time.time()-t0:.1f}s")

    # Measure at initialization (epoch 0)
    init_loss = compute_loss_on_dataset(model, train_loader, device, max_batches=50)
    init_If, init_F, init_method = compute_functional_information(
        init_loss, sorted_random_losses, tail_model
    )
    init_fisher = compute_fisher_trace_efficient(
        model, train_loader, device, num_samples=config["fisher_samples"]
    )

    results = {
        "setting": setting_name,
        "optimizer": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "random_labels": random_labels,
        "num_params": count_parameters(model),
        "M": config["M"],
        "tail_model": tail_model,
        "epochs": [],
    }

    epoch_data = {
        "epoch": 0,
        "train_loss": init_loss,
        "functional_info": init_If,
        "F_hat": init_F,
        "fi_method": init_method,
        "fisher_trace": init_fisher,
        "train_acc": None,
    }
    results["epochs"].append(epoch_data)
    print(f"  Epoch 0: loss={init_loss:.4f}  I_f={init_If:.4f}  "
          f"F_hat={init_F:.2e}  T_F={init_fisher:.4f}  [{init_method}]")

    # Training loop with measurements
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)

        # Measure functional information
        eval_loss = compute_loss_on_dataset(model, train_loader, device, max_batches=50)
        I_f, F_hat, fi_method = compute_functional_information(
            eval_loss, sorted_random_losses, tail_model
        )

        # Measure Fisher trace
        fisher_trace = compute_fisher_trace_efficient(
            model, train_loader, device, num_samples=config["fisher_samples"]
        )

        epoch_data = {
            "epoch": epoch,
            "train_loss": eval_loss,
            "train_acc": train_acc,
            "functional_info": I_f,
            "F_hat": F_hat,
            "fi_method": fi_method,
            "fisher_trace": fisher_trace,
        }
        results["epochs"].append(epoch_data)

        dt = time.time() - t_epoch
        print(f"  Epoch {epoch:3d}: loss={eval_loss:.4f}  acc={train_acc:.4f}  "
              f"I_f={I_f:.2f}  T_F={fisher_trace:.2e}  [{fi_method}] ({dt:.1f}s)")

    return results


def save_results(results, tag):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Fisher Fundamental Theorem Experiment"
    )
    parser.add_argument(
        "--setting", type=str, default="mlp_mnist",
        choices=list(SETTINGS.keys()) + ["all"],
    )
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "sgd_momentum", "adam"])
    parser.add_argument("--random-labels", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    settings_to_run = list(SETTINGS.keys()) if args.setting == "all" else [args.setting]

    all_results = []
    for name in settings_to_run:
        config = SETTINGS[name]
        results = run_single_experiment(
            setting_name=name,
            config=config,
            device=device,
            random_labels=args.random_labels,
            optimizer_name=args.optimizer,
            lr_override=args.lr,
            weight_decay=args.weight_decay,
        )
        tag = f"{name}_{args.optimizer}"
        if args.random_labels:
            tag += "_randomlabels"
        if args.weight_decay > 0:
            tag += f"_wd{args.weight_decay}"
        if args.lr is not None:
            tag += f"_lr{args.lr}"

        save_results(results, tag)
        all_results.append(results)

    return all_results


if __name__ == "__main__":
    main()
