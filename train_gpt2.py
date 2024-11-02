from dataclasses import dataclass
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import inspect
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from hellaswag import get_most_likely_row, iterate_examples, render_example
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
    
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups, any parameters that is 2D will be weight decayed, otherwise no
        # i.e., all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.device('cuda') == device
        print(f'Using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer 
    
    
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
# LR Scheduler
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # range (-1, 1) -> (0, 1)
    return min_lr + coeff * (max_lr - min_lr) # should this not be max_lr - coeff * (max_lr - min_lr) ??

# ---------------------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename: str):
    # load a numpy array of uint16 tokens from a binary file
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank: int, num_processes: int, split: str) -> None:
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        self.split = split
        data_root = os.path.join(os.path.dirname(__file__), 'edu_fineweb10')
        shards = os.listdir(data_root)
        shards = [shard for shard in shards if split in shard]
        assert len(shards) > 0, f"No shards found for split: {split}"
        self.shards = shards
        self.starting_position = self.B * self.T * self.process_rank
        self.reset()
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # state
        self.current_position = self.starting_position
        
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + (B * T) + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T) # targets
        
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes 
        # if loading the next batch will be out of bounds, then go to the next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens): # checking if all the parallel processes can get another valid batch of data or not
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.starting_position
        return x, y
    

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py


        
def test_code():
    # simple launch
    # python train_gpt2.py
    # DDP launch for e.g. 8 GPUs
    #torchrun --standalone --nproc_per_node=8 train_gpt2.py
    
    # using torch distributed data parallel to train in multi-gpu setup. Do not use torch.nn.DataParallel
    # as it is deprecated, it is legacy, and possibly slower than torch.distributed.
    from torch.distributed import init_process_group, destroy_process_group
    enc = tiktoken.get_encoding("gpt2")
    
    # setup DDP distributed data parallel
    # torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    # torch.distributed.init_process_group() will use these environment variables to initialize the distributed process group
    
    
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        # use of DDP at the moment demands CUDA, we setthe the device appropriately according to rank
        assert torch.cuda.is_available(), "for now I think DDP requires CUDA"
        init_process_group(backend='nccl') # what is NCCL? NCCL (NVIDIA Collective Communications Library) is a high-performance GPU-to-GPU communication library optimized for NVIDIA GPUs. It's used as the backend for distributed training in PyTorch, enabling efficient multi-GPU and multi-node communication, which is crucial for distributed deep learning workloads.
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = get_device()
        master_process = True
        
        
    seed = 1337
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
    # model = GPT.from_pretrained('gpt2')

    device = get_device()
    
    total_batch_size = 2**19 # 524_288, ~0.5M
    # B, T = 8, 32 # for my mac
    B = 64 # micro batch size
    T = 1024 # sequence length
    eval_steps = 250
    save_steps = 5000
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total batch size is divisible by B * T * ddp_world_size"    
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    
    if master_process:
        print(f'Total desired batch size: {total_batch_size}')
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # get a data batch
    train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
    torch.set_float32_matmul_precision('high')
    # get logits
    model = GPT(GPTConfig())
    print(f"didn't crash yay!")
    model.to(device)
    use_compile = torch.cuda.is_available() and False
    
    # TODO: uncomment when want to compile model on CUDA
    if use_compile:
        model = torch.compile(model)
        
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank]) # ddp does all_reduce and computes the average of the gradients across all the GPUs, and then deposits the averaged gradients to each of the GPUs
    raw_model = model.module if ddp else model # if ddp, then we need to access the module of the model, else just the model
    
    # logits, loss = model(x, y)
    if device == torch.device('cuda'):
        sync = lambda: torch.cuda.synchronize()
    elif device == torch.device('mps'):
        sync = lambda: torch.mps.synchronize()
        
    # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
    
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass


    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == (max_steps - 1))
        loss_accum = 0.0
        if step % eval_steps == 0 or last_step:
            model.eval() 
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"step {step:4d} | val loss: {val_loss_accum.item():.4f}", file=f)
                with open(log_file, "a") as f:
                    f.write(f"step {step:4d} | val loss: {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % save_steps == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "config": raw_model.config,
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                    }
                    torch.save(checkpoint, checkpoint_path)
                    
        if (step % eval_steps == 0 or last_step) and (not use_compile):
            num_correct_norm = 0
            num_total = 0
            
            for i, example in enumerate(iterate_examples("val")):
                # hellaswag evaluation
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(tokens, mask)
                    _, pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all the processes
            if ddp:
                num_correct_norm = torch.tensor(num_correct_norm, device=device, dtype=torch.long)
                num_total = torch.tensor(num_total, device=device, dtype=torch.long)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                num_correct_norm = num_correct_norm.item()
                num_total = num_total.item()
            
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"step {step:4d} | Hellaswag acc: {num_correct_norm}/{num_total} = {acc_norm:.4f}", file=f)
                with open(log_file, "a") as f:
                    f.write(f"step {step:4d} | Hellaswag acc: {num_correct_norm}/{num_total} = {acc_norm:.4f}\n")
        
        if ((step > 0 and step % eval_steps == 0) or last_step) and (not use_compile) and master_process:
            model.eval()
            num_return_sequences = 4
            max_length = 32
            tokens = torch.tensor(enc.encode("Hello, I am a language model"), dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(seed + ddp_rank)
            while xgen.size(1) < max_length:
                with torch.no_grad():
                    logits, _ = model(xgen)
                logits = logits[:, -1, :] # last time step
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top k probabilities
                # note: multinomdial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, dim=-1, index=ix)
                xgen = torch.cat((xgen, xcol), dim=1)
            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_length].tolist()
                decoded = enc.decode(tokens)
                print(f'rank {ddp_rank} sample {i}: {decoded}')
                
        # train
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
                
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # NOTE with torch.autocast(device_type=device, dtype=torch.bfloat16): # not supported for mps, only for ampere nvidia gpus and above
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y) # refer to video about which specific page to refer https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast
            loss = loss / grad_accum_steps 
            # NOTE we need to do this, because lets say a batch size of 8, the loss is mean reduction of the batch, so uses the normalizer 1/8 for the 
            # sum of each elements loss. hence to in case of grad accum, if each batch size of 1 is accumulated over 8 steps, then the 1/8 normalizer is lost.
            loss_accum += loss.detach() 
            if ddp:
                # if we don't do this, then the gradients will be synchronized across the GPUs, at every backward step. We want to do at the end of all grad accum steps.
                # the below hack is from Andrej, and not use the no_sync context manager, as it was leading to code duplication.
                # we only want to sync the gradients at the end of the grad accum steps, not after every micro_step 
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward() #.backward() always does += on existing gradients, hence important to zero grad (unless grad accumulation)
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # will take average across all the ranks/gpus and redistribute the averaged loss to all the GPUs
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # if we get a bad data batch, a really high loss can give a really high gradient, which can provide a big shock to the model
        # gradient norm clipping is then used to ensure that the updates to the model are not very big/shocking.
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        sync()
        t1 = time.time()
        dt = (t1 - t0)  # time diff in milliseconds
        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / dt
        if master_process:
            print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e}  | norm: {norm:.4f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

    
    if ddp:
        destroy_process_group()
    
    # NOTE: IMPORTANT INSIGHT
    # 1. We expect the loss to still decrease on the above small dataset.
    # 2. Two things to note: 1) Our dataset is very biased, and covers only a very small portion of the 50,257 tokens
    #  2) Hence when training, the model would just try to eliminate/"forget" the importance of the other tokens that never occur in the dataset
    # by for example driving the bias for these terms to -inf. This is the cause behind the easy gains that will be made .
    # 3. Compression ratio is 3:1 (3 characters ~= 1 token)
    
if __name__ == "__main__":
    test_code()