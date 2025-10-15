import torch
from model_trm import TinyReasoningModel
from model_transformer import TransformerModel
from dataloader import SudokuExtremeDataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.init as init 
import os
import datetime
import logging
device = "mps"
vocab_size = 11
embedding_dim = 512
seq_length = 81
number_of_transformer_units = 2
n = 6
T = 3
epochs = 500
batch_size = 64
subsample = 0
deep_supervision_steps = 16
model_trm = TinyReasoningModel(vocab_size=vocab_size, embedding_dim=embedding_dim, seq_length=seq_length, num_of_transformer_units=number_of_transformer_units, n=n, T=T)

dataset = SudokuExtremeDataset(data_path="/Users/vaibhav/Desktop/MLGroundUp/coding/reasoning/less_is_more_recursive_reasoning_with_tiny_networks/myimplementation/data/sudoku-extreme-1k-aug-1000", split="train", subsample=subsample)
print("Length of dataset:", len(dataset))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True )
model_trm.train()
criterion = nn.CrossEntropyLoss()

optimizer_trm = torch.optim.Adam(model_trm.parameters(), lr=0.0001)
def initialize_carry(batch_size):
    z_init = torch.empty(embedding_dim, dtype=torch.float32)
    init.trunc_normal_(z_init, mean=0.0, std=1.0, a=-2.0, b=2.0)  # Truncated normal to avoid outliers
    
    y_init = torch.empty(embedding_dim, dtype=torch.float32)
    init.trunc_normal_(y_init, mean=0.0, std=1.0, a=-2.0, b=2.0)
    
    # Broadcast to full shape (repeat the same init vector across sequence positions)
    full_shape = (batch_size, seq_length , embedding_dim)
    z = z_init.expand(full_shape).clone().detach()
    y = y_init.expand(full_shape).clone().detach()
    return z, y

# Run directories and logger setup
run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_dir = os.path.join(os.path.dirname(__file__), "outputs", run_id)
ckpt_trm_dir = os.path.join(base_dir, "checkpoints_trm")
ckpt_tx_dir = os.path.join(base_dir, "checkpoints_transformer")
os.makedirs(ckpt_trm_dir, exist_ok=True)
os.makedirs(ckpt_tx_dir, exist_ok=True)
log_path = os.path.join(base_dir, "train.log")
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)

print("training model trm ----------")
logger.info("training model trm ----------")

for epoch in range(epochs):
    running_loss = 0.0
    num_batches = 0
    running_correct = 0
    running_total = 0
    for batch in dataloader:
        inputs, labels = batch
        z, y = model_trm.initialize_carry(inputs.shape[0])
        for i in range(deep_supervision_steps):
            outputs, z, y = model_trm(inputs, z, y)
            z = z.detach()
            y = y.detach()
            mask = (inputs == 1).view(-1)
            masked_outputs = outputs.view(-1, outputs.shape[-1])[mask]
            masked_labels = labels.view(-1)[mask]
            loss_calculation = criterion(masked_outputs, masked_labels)
            optimizer_trm.zero_grad(set_to_none=True)

            loss_calculation.backward()
            torch.nn.utils.clip_grad_norm_(model_trm.parameters(), 1.0)
            optimizer_trm.step()
        running_loss += loss_calculation.item()
        num_batches += 1

        # Accuracy calculation
        preds = torch.argmax(masked_outputs, dim=-1)
        running_correct += (preds == masked_labels).sum().item()
        running_total += masked_labels.numel()

    epoch_loss = running_loss / max(1, num_batches)
    epoch_acc = running_correct / running_total if running_total > 0 else 0
    msg = f"TRM Epoch {epoch+1}, Loss: {epoch_loss:.6f}, Accuracy: {epoch_acc:.6f}"
    print(msg)
    logger.info(msg)

    # Save TRM checkpoint
    ckpt_path = os.path.join(ckpt_trm_dir, f"epoch_{epoch+1:03d}.pt")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model_trm.state_dict(),
        "optimizer_state_dict": optimizer_trm.state_dict(),
        "loss": epoch_loss,
        "accuracy": epoch_acc,
    }, ckpt_path)
    logger.info(f"Saved TRM checkpoint: {ckpt_path}")


print("training model transformer ----------")


number_of_transformer_units = 4

model_transformer = TransformerModel(vocab_size=vocab_size, embedding_dim=embedding_dim, seq_length=seq_length, num_heads=4, num_of_transformer_units=number_of_transformer_units)
model_transformer.train()
optimizer_transformer = torch.optim.Adam(model_transformer.parameters(), lr=0.0001)

for epoch in range(epochs):
    running_loss = 0.0
    num_batches = 0
    running_correct = 0
    running_total = 0
    for batch in dataloader:
        inputs, labels = batch
        outputs = model_transformer(inputs)
        
        mask = (inputs == 1).view(-1)
        masked_outputs = outputs.view(-1, outputs.shape[-1])[mask]
        masked_labels = labels.view(-1)[mask]
        loss_calculation = criterion(masked_outputs, masked_labels)
        optimizer_transformer.zero_grad(set_to_none=True)

        loss_calculation.backward()
        torch.nn.utils.clip_grad_norm_(model_transformer.parameters(), 1.0)
        optimizer_transformer.step()
        running_loss += loss_calculation.item()
        num_batches += 1

        # Accuracy calculation
        preds = torch.argmax(masked_outputs, dim=-1)
        running_correct += (preds == masked_labels).sum().item()
        running_total += masked_labels.numel()

    epoch_loss = running_loss / max(1, num_batches)
    epoch_acc = running_correct / running_total if running_total > 0 else 0
    msg = f"Transformer Epoch {epoch+1}, Loss: {epoch_loss:.6f}, Accuracy: {epoch_acc:.6f}"
    print(msg)
    logger.info(msg)

    # Save Transformer checkpoint
    ckpt_path_tx = os.path.join(ckpt_tx_dir, f"epoch_{epoch+1:03d}.pt")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model_transformer.state_dict(),
        "optimizer_state_dict": optimizer_transformer.state_dict(),
        "loss": epoch_loss,
        "accuracy": epoch_acc,
    }, ckpt_path_tx)
    logger.info(f"Saved Transformer checkpoint: {ckpt_path_tx}")
