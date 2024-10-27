"""$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
FineWeb is a 10B token dataset of English web pages, filters were applied using LLama3 - 70B model.
"""

import os
import multiprocessing as mp
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm

# -------------------------------------
local_dir = "edu_fineweb10"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

#download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text tokens

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc['text'])) # encode ordinary will ignore special tokens
    tokens_np = np.array(tokens)
    assert (0<=tokens_np).all() and (tokens_np<2**16).all(), "token dictionary"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename: str, tokens_np: np.ndarray):
    # writes a numpy array of uint16 tokens to a binary file
    with open(filename, "wb") as f:
        f.write(tokens_np.tobytes())
        
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # pre allocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simple append tokens to end of the current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, desc=f"Sharding {shard_index}", unit="tokens")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            # write the current shard to disk
            shard_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb10_{remote_name}_{split}_{shard_index:06d}.npy")
            # split the document into whatever fits in the current shard, the remainder goes to the next shard
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(shard_filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0: len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        shard_filename = os.path.join(DATA_CACHE_DIR, f"edufineweb10_{remote_name}_{split}_{shard_index:06d}.npy")
        write_datafile(shard_filename, all_tokens_np[:token_count])
