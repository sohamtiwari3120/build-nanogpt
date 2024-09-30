from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time

# ---------------------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    # vocab_size: int = 50257 # number of tokens: 50,000 BPE + 256 bytes tokens + 1 <|endoftext}> token
    vocab_size: int = 50304 # this is a multiple of 128, extra tokens would be ignored, being a muktiple of two makes it use more optimal number of kernels (and no need for kernels for stragglers). Despite the more memory, the computation was faster for Andrej Karpathy by ~4%.
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # q, k, v
        # output_projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask but following the the OpenAI/HF naming
        self.register_buffer("bias", torch.tril(
            torch.ones(config.block_size, config.block_size)
        ).view(1, 1, config.block_size, config.block_size)) 
        # [1, 0, 0, 0]
        # [1, 1, 0, 0]
        # [1, 1, 1, 0]
        # [1, 1, 1, 1]
        # tril creates a lower triangular matrix from an input tensor
        # creates a causal self attn mask of shape (seqLen, seqLen)
        
    def forward(self, x):
        B, T, C = x.size()
        qkv: torch.Tensor = self.c_attn(x)
        q, k, v = qkv.split(split_size = self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T, T) matrix for queries and keys)
        # att: torch.Tensor = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # making the attention causal
        # att = F.softmax(att, dim=-1) # upper triangle will be zero, since exp(-inf) ~= 0, and we are replacing with -inf where mask == 0
        # # hence causal self attention
        # y = att @ v # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # re-assemble all heads output side by side
        y = self.c_proj(y) # output projection
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_fc  = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config: GPTConfig = config
    
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([
                Block(config) for _ in range(config.n_layer)
            ]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # GPT-2 uses no bias

        # weight sharing scheme, they both have the same job, just in reverse
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # std weight determined from source code of gpt-2 relased by openai
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # 1 / √(2* num of times noise added by the residual layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T},\
            block size is {self.config.block_size}"
        # forward the token and pos embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # pe of shape (T, n_embd), this will be broadcasted later
        tok_emb = self.transformer.wte(idx) # te of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and classifier
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            ) # flattening logits (B, T, vocab_size) -> (B * T, vocab_size)
            # # flattening targets from (B, T) -> (B * T )
            # this will automatically compute the softmax of the input logits before computing NLL
        return logits, loss



    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model weights from huggingface

        Args:
            model_type (str): Model type should be one of 
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / bias buffer

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 
                      'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a  vanilla
        # this means we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
def get_device():
    """
    Returns the best available device: CUDA, MPS (Apple Silicon), or CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

# ---------------------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T) -> None:
        self.B = B
        self.T = T
        
        # at init load tokens from disc and store into memory
        with open('input.txt', 'r') as f:
            text = f.read()
            
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Number of characters in text {len(text)} characters")
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        
        # state
        self.current_position = 0
        
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + (B * T) + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        
        # advance the position in the tensor
        self.current_position += B * T
        
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
        
        
def test_code():
    seed = 1337
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    # model = GPT.from_pretrained('gpt2')

    device = get_device()

    # get a data batch
    B, T = 8, 32
    train_loader = DataLoaderLite(B, T)
    torch.set_float32_matmul_precision('high')
    # get logits
    model = GPT(GPTConfig())
    print(f"didn't crash yay!")
    model.to(device)
    
    # model = torch.compile(model)
    # logits, loss = model(x, y)
    if device == torch.device('cuda'):
        sync = lambda: torch.cuda.synchronize()
    elif device == torch.device('mps'):
        sync = lambda: torch.mps.synchronize()
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # with torch.autocast(device_type=device, dtype=torch.bfloat16): # not supported for mps, only for ampere nvidia gpus and above
        logits, loss = model(x, y) # refer to video about which specific page to refer https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast
        loss.backward() #.backward() always does += on existing gradients, hence important to zero grad (unless grad accumulation)
        optimizer.step()
        sync()
        t1 = time.time()
        dt = (t1 - t0) * 1000 # time diff in milliseconds
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
        print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
    
    # NOTE: IMPORTANT INSIGHT
    # 1. We expect the loss to still decrease on the above small dataset.
    # 2. Two things to note: 1) Our dataset is very biased, and covers only a very small portion of the 50,257 tokens
    #  2) Hence when training, the model would just try to eliminate/"forget" the importance of the other tokens that never occur in the dataset
    # by for example driving the bias for these terms to -inf. This is the cause behind the easy gains that will be made .
    # 3. Compression ratio is 3:1 (3 characters ~= 1 token)
    
if __name__ == "__main__":
    test_code()