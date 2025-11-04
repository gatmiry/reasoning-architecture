import torch.nn as nn
import torch.nn.functional as F
import torch
import math
device = 'cuda'
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.fc_2 = nn.Linear(config.n_embd * 3, config.n_embd)
        self.NANO_SCALE_GPT = True
    def forward(self, x):
        return self.fc_2(self.gelu(self.fc_1(x)))

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads
        self.block_size = config.block_size # Store block_size
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Removed bias registration
        self.c_proj.NANOGPT_SCALE_INIT = True

    def forward(self, x, last_k_no_attend=0, window_size=0):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1,2)
        attn = q @ k.transpose(-1,-2) * (k.size(-1)) ** -0.5
        # Create bias mask directly in forward and ensure it's on the correct device
        bias = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(bias == 0, float('-inf'))
        if last_k_no_attend > 0:
          last_k_no_attend = min(last_k_no_attend, T)
          attn[:,-last_k_no_attend:] = float('-inf')
          if window_size > 0:
            window_size = min(window_size, T - last_k_no_attend)
            i = torch.arange(T).view(T, 1)
            j = torch.arange(T).view(1, T)
            recent_band = (i - j) >= 0 & (i - j) <= window_size & i >= T - last_k_no_attend
            attn[recent_band] = float('-inf')

        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = CasualSelfAttention(config)
        self.c_fc = MLP(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x, last_k_no_attend=0, window_size=0):
        x = x + self.c_attn(self.ln_1(x), last_k_no_attend=0, window_size=0)
        return x + self.c_fc(self.ln_2(x))

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
        self.config = config # Store config

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=std)

    def forward(self, idx, targets=None, last_k_no_attend=0, window_size=0):
        B, T = idx.size()
        pe = torch.arange(0, T, dtype=torch.long, device=idx.device) # Ensure pe is on the same device as idx
        pe_vecs = self.transformer.wpe(pe)
        if last_k_no_attend > 0:
          last_k_no_attend = min(last_k_no_attend, T)
          pe_vecs[-last_k_no_attend:,:] = 0.0
        x = pe_vecs + self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, last_k_no_attend=last_k_no_attend, window_size=window_size)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # The loss calculation in the training loop is outside this method
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            pass # Keep this pass or remove the if targets is not None block if not used internally
        return logits, loss

class GPTConfig():
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers = 2
    n_heads = 1
    n_embd = 64

    def __init__(self, block_size, vocab_size):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        #self.n_heads = n_heads
        #self.n_layers = n_layers