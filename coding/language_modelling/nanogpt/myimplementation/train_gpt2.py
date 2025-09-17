from dataclasses import dataclass
from typing import Optional
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import os
import tiktoken
import torch


from torch.distributed import init_process_group, destroy_process_group
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available.")

    elif torch.backends.mps.is_available():
        device = "mps"
        print("MPS is available.")
    else:
        device = "cpu"
        print("MPS is not available.")


torch.set_float32_matmul_precision('high')

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, n_head, T, hs

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, n_head, T, hs
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, n_head, T, hs

        # implement attention using flash attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # B, n_head, T, T
        # # print(att[0,0,:,:])
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y =  att @ v # (B, n_head, T, T) x (B, n_head, T, hs) -> (B, n_head, T, hs)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True
    dropout: float = 0.1

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),

        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weught sharing 
        self.transformer.wte.weight = self.lm_head.weight
        #self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        loss = None
        
        logits = self.lm_head(x) # B,T,vocab_size
        # y  = B,T
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
num_sequence = 5
max_length = 30

model = GPT(GPTConfig())
model.eval()
model.to(device=device)
if device == "cuda":            
    model = torch.compile(model)
if ddp:
    model = torch.nn.DistributedDataParallel(model, device_ids=[ddp_local_rank])
import tiktoken
enc = tiktoken.get_encoding("gpt2")
text = "Hi i am a language model"
tokens = enc.encode(text)
tokens = torch.tensor(tokens, dtype=torch.long, device=device)
# tokens = tokens.unsqueeze(0)
tokens = torch.stack([tokens for _ in range(num_sequence)])
x =  tokens.to(device=device)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)
        



class DataLoader:
    def __init__(self, text, B, T, process_rank, process_world_size):
        with open('input.txt', 'r') as f:
            text = f.read()
        self.text = text
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.process_world_size = process_world_size
        enc = tiktoken.get_encoding("gpt2")
        self.text_encoding = torch.tensor(enc.encode(text), dtype=torch.long)
        print(f"loaded {len(self.text_encoding)} tokens")
        print(f"1 epoch will take {len(self.text_encoding)//(self.B*self.T)} batches")
        self.current_idx = B * self.T * process_rank
    def get_batch(self):
        buf = torch.tensor(self.text_encoding[self.current_idx:self.current_idx+self.B*self.T+1], dtype=torch.long, device=device)
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_idx += self.B*self.T * self.process_world_size
        if self.current_idx + self.B*self.T*self.process_world_size >= len(self.text_encoding):
            self.current_idx = B * self.T * self.process_rank
        return x, y

total_batch_size = 4 * 128 * 32

B,T = 4, 128
assert total_batch_size % (B*T) == 0, "make sure total_batch_size is divisible by B*T"
grad_accumulation_steps = total_batch_size // (B*T*ddp_world_size)

# Training hyperparameters
max_learning_rate = 6e-4  # Reduced learning rate
min_learning_rate = 6e-5
max_iters = 1000
warmup_iters = max_iters // 10  # Warmup period
max_grad_norm = 1.0  # Gradient clipping threshold
weight_decay = 0.1   # L2 regularization

dataloader = DataLoader(text, B, T, process_rank=ddp_rank, process_world_size=ddp_world_size)
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=max_learning_rate, betas=(0.9, 0.95), eps=1e-8)

# Learning rate scheduler with linear warmup
def get_lr(it):
    # Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    # Linear decay after warmup
    if it > max_iters:
        return min_learning_rate
    coeff = 0.5 * (1.0 + math.cos(math.pi * (it - warmup_iters) / (max_iters - warmup_iters)))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


import time
import torch.distributed as dist
for i in range(max_iters):
    # Update learning rate
    start_time = time.time()
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    loss_accumulator = 0
    optimizer.zero_grad()
    for micro_step in range(grad_accumulation_steps):
        x, y = dataloader.get_batch()
        if device == "cuda":
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss = loss / grad_accumulation_steps
        loss_accumulator += loss.item()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accumulation_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accumulator, op=dist.ReduceOp.AVG)
    # Gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    if master_process:
        if i % 10 == 0:  # Print less frequently
            exec_time_ms = (time.time() - start_time) * 1000
            token_throughput = dataloader.B*dataloader.T/(exec_time_ms/1000) * grad_accumulation_steps * ddp_world_size
            print(f"Step: {i}, Loss: {loss_accumulator:.6f}, LR: {lr:.2e}, execution time: {exec_time_ms:.2f} ms, Token throughput: {token_throughput:.2f} tokens/second , clip norm: {norm:.2f}")
            print("--------------------------------")

if ddp:
    destroy_process_group()
"""
torchrun --standalone --nproc_per_node=4 train_gpt2.py
"""