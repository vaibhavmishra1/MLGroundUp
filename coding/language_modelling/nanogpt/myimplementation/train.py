from model import BigramGPT, Transformer
import torch
import torch.optim as optim
from torch.nn import functional as F
from tqdm import trange
def sample_batch( encode, block_size, batch_size):
    rand_int = torch.randint(0, len(encode) - block_size, (batch_size,))
    x = torch.stack([encode[i:i+block_size] for i in rand_int])
    y = torch.stack([encode[i+1:i+block_size+1] for i in rand_int])
    return x, y
        


def prepare_data():
    with open('input.txt', 'r') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    stoi = { c:i for i,c in enumerate(chars)}
    itos = { i:c for i,c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    train_data = text[:int(0.9*len(text))]
    val_data = text[int(0.9*len(text)):]
    encode_train = torch.tensor(encode(train_data), dtype=torch.long)
    encode_val = torch.tensor(encode(val_data), dtype=torch.long)
    return encode_train, encode_val, chars

def train(model, optimizer, epochs, batch_size, vocab_size, n_embed, encode_train, val_data, block_size):
    
    for i in trange(epochs):
        x,y = sample_batch(encode_train, block_size, batch_size)
        out = model(x)
        out = out.view(-1, vocab_size)
        y = y.view(-1)
        loss = F.cross_entropy(out, y)
        print("train loss: ", loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 1000 == 0:
            evaluate(model, val_data, block_size, vocab_size)


def evaluate(model, val_data, block_size, vocab_size):
    with torch.no_grad():
        x,y = sample_batch(val_data, block_size, 1000)
        out = model(x)
        out = out.view(-1, vocab_size)
        y = y.view(-1)
        loss = F.cross_entropy(out, y)
        print("val loss: ", loss.item())

def main():
    epochs = 5000
    batch_size = 64
    block_size = 256
    lr=1e-3
    n_embed = 384
    n_heads = 6
    n_layers = 6
    train_data, val_data , chars= prepare_data()

    stoi = { c:i for i,c in enumerate(chars)}
    itos = { i:c for i,c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    vocab_size = len(chars)
    



    model = Transformer(vocab_size, n_embed, block_size, n_heads, n_layers)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train(model, optimizer, epochs, batch_size, vocab_size, n_embed, train_data, val_data, block_size)
    
    # output = decode(model.generate(torch.zeros((1,1), dtype=torch.long), 3000).tolist()[0])
    # print(output)

if __name__ == "__main__":
    main()