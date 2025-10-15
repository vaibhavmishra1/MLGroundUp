import os
import json
import numpy as np
import torch

from utils.functions import load_model_class
from models.losses import IGNORE_LABEL_ID


def load_config_and_build_model(checkpoint_dir: str, dataset_root: str, global_batch_size: int = 1):
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    import yaml
    with open(config_path, "rt") as f:
        cfg = yaml.safe_load(f)

    # Use dataset_root (absolute) for metadata
    train_meta = _read_dataset_metadata(os.path.join(dataset_root, "test"))

    model_cfg = dict(
        **cfg["arch"],
        batch_size=global_batch_size,
        vocab_size=train_meta["vocab_size"],
        seq_len=train_meta["seq_len"],
        num_puzzle_identifiers=train_meta["num_puzzle_identifiers"],
    )

    model_cls = load_model_class(cfg["arch"]["name"])  # models.recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1
    loss_head_cls = load_model_class(cfg["arch"]["loss"]["name"])  # models.losses@ACTLossHead

    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **{k: v for k, v in cfg["arch"]["loss"].items() if k != "name"})
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

    return cfg, model


def _read_dataset_metadata(split_dir: str):
    with open(os.path.join(split_dir, "dataset.json"), "r") as f:
        return json.load(f)


def load_single_example(dataset_root: str, index: int = 0):
    test_dir = os.path.join(dataset_root, "test")
    inputs = np.load(os.path.join(test_dir, "all__inputs.npy"))
    labels = np.load(os.path.join(test_dir, "all__labels.npy"))
    puzzle_identifiers = np.load(os.path.join(test_dir, "all__puzzle_identifiers.npy"))
    puzzle_indices = np.load(os.path.join(test_dir, "all__puzzle_indices.npy"))

    # dataset.json defines ignore_label_id used during training conversion; for evaluation we pass raw labels
    meta = _read_dataset_metadata(test_dir)

    # Select one item, then batch dimension of 1
    x = inputs[index:index + 1].astype(np.int32)
    y = labels[index:index + 1].astype(np.int32)
    # Convert ignore label id to the training IGNORE_LABEL_ID used by the loss
    if meta.get("ignore_label_id", None) is not None:
        y = np.where(y == int(meta["ignore_label_id"]), IGNORE_LABEL_ID, y).astype(np.int32)
    # Map example index to its puzzle id (replicates loader behavior)
    pidx = int(np.searchsorted(puzzle_indices, index, side="right") - 1)
    pid = puzzle_identifiers[pidx:pidx + 1].astype(np.int32)

    batch = {
        "inputs": torch.from_numpy(x).cuda(),
        "labels": torch.from_numpy(y).cuda(),
        "puzzle_identifiers": torch.from_numpy(pid).cuda(),
    }
    return batch, meta


def decode_grid(tokens_row: np.ndarray):
    # Sudoku vocab_size is 11 in this dataset.json (0 pad, 1..9 digits, maybe 10 as blank). Map 1..9 -> digits.
    return tokens_row


def main():
    checkpoint_file = "/root/TinyRecursiveModels/checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_4882"
    checkpoint_dir = os.path.dirname(checkpoint_file)
    dataset_root = "/root/TinyRecursiveModels/data/sudoku-extreme-1k-aug-1000"

    torch.random.manual_seed(0)

    cfg, model = load_config_and_build_model(checkpoint_dir, dataset_root, global_batch_size=1)
    
    # Load weights
    print(f"Loading checkpoint from {checkpoint_file}")
    state_dict = torch.load(checkpoint_file, map_location="cuda")
    
    # Handle potential puzzle embedding shape mismatch (mirror pretrain.load_checkpoint behavior)
    puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
    if hasattr(model, "model") and hasattr(model.model, "puzzle_emb"):
        expected_shape = model.model.puzzle_emb.weights.shape  # type: ignore[attr-defined]
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding: found {tuple(puzzle_emb.shape)} expected {tuple(expected_shape)}")
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )

    # Newer torch allows assign=True to load into compiled wrapper
    model.load_state_dict(state_dict, assign=True)
    model.eval()
    model.cuda()
    # Prepare batch
    batch, meta = load_single_example(dataset_root, index=0)

    # Build initial carry and run until halted (like evaluate())
    with torch.inference_mode():
        return_keys = {"preds"}
        # Ensure carry tensors are allocated on CUDA like in pretrain.py
        with torch.device("cuda"):
            carry = model.initial_carry(batch)
        inference_steps = 0
        while True:
            carry, loss, metrics, preds, all_finish = model(carry=carry, batch=batch, return_keys=return_keys)
            inference_steps += 1
            if all_finish:
                break

    preds_tokens = preds["preds"].squeeze(0).cpu().numpy() - 1
    inputs_tokens = batch["inputs"].squeeze(0).cpu().numpy() - 1
    labels_tokens = batch["labels"].squeeze(0).cpu().numpy() - 1

    # Print concise outputs
    print(f"Inference steps: {inference_steps}")
    # metrics fields are tensors; convert to Python
    if isinstance(metrics, dict):
        m = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}
        print("Metrics:", m)
    def print_grid(name, arr):
        print(f"{name}:")
        for row in arr.reshape(9, 9):
            print(" ".join(str(int(x)) for x in row))
        print()

    print_grid("Input tokens (9x9)", inputs_tokens)
    print_grid("Label tokens (9x9)", labels_tokens)
    print_grid("Pred tokens  (9x9)", preds_tokens)
    


if __name__ == "__main__":
    main()


