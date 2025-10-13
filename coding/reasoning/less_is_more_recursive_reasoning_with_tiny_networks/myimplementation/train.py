import torch
from model import TinyReasoningModel
from dataloader import SudokuExtremeDataset
from torch.utils.data import DataLoader
from torch import nn

device = "mps"
model = TinyReasoningModel(vocab_size=11, embedding_dim=16, seq_length=81, num_of_transformer_units=4, n=6, T=1)
dataset = SudokuExtremeDataset(data_path="data/sudoku-extreme-1k-aug-1000", split="train")
print("Length of dataset:", len(dataset))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True )
model.train()
criterion = nn.CrossEntropyLoss()
epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    running_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        inputs, labels = batch
        outputs = model(inputs)
        
        outputs = outputs.view(-1, outputs.shape[-1])
        labels = labels.view(-1)
        loss_calculation = criterion(outputs, labels)
        optimizer.zero_grad(set_to_none=True)

        loss_calculation.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss_calculation.item()
        num_batches += 1
    print(f"Epoch {epoch+1}, Loss: {running_loss / max(1, num_batches):.6f}")
    
        