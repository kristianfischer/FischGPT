import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Directory to save tokenized shards
local_dir = "./src/data/datasets/oasst1_sft"
os.makedirs(local_dir, exist_ok=True)

# Download the OpenAssistant dataset
print("Loading OpenAssistant/oasst1 dataset...")
dataset = load_dataset("OpenAssistant/oasst1", split="train")

# Build a mapping from message_id to message
id2msg = {msg['message_id']: msg for msg in dataset}

# Build a parent -> children map
children_map = {}
for msg in dataset:
    parent_id = msg['parent_id']
    if parent_id:
        children_map.setdefault(parent_id, []).append(msg)

# Extract all (prompt, response) pairs from full paths
def extract_paths():
    def dfs(path):
        last = path[-1]
        for child in children_map.get(last['message_id'], []):
            new_path = path + [child]
            if child['role'] == 'assistant':
                prompt = "\n".join(f"<|{m['role']}|>{m['text']}" for m in new_path[:-1])
                response = child['text']
                yield (prompt, response)
            yield from dfs(new_path)

    roots = [m for m in dataset if m['parent_id'] is None]
    for root in roots:
        yield from dfs([root])

# Collect all prompt/response pairs
print("Extracting prompt/response pairs...")
pairs = list(extract_paths())
print(f"Extracted {len(pairs)} prompt/response pairs.")

# Create train/val split BEFORE tokenizing - take first 2% for validation
val_split_ratio = 0.02  # 2% for validation
val_size = int(len(pairs) * val_split_ratio)
val_pairs = pairs[:val_size]
train_pairs = pairs[val_size:]

print(f"Validation pairs: {len(val_pairs)} ({len(val_pairs)/len(pairs)*100:.1f}%)")
print(f"Training pairs: {len(train_pairs)} ({len(train_pairs)/len(pairs)*100:.1f}%)")

# Tokenizer setup
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def format_and_tokenize(pair):
    prompt, response = pair
    text = f"{prompt}<|assistant|>{response}<|endoftext|>"
    tokens = enc.encode(text, allowed_special={'<|endoftext|>'})
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np

def save_split(pairs, split_name):
    """Save a split (train or val) to numbered shards"""
    shard_size = int(1e6)  # 1M tokens per shard
    shard = []
    token_count = 0
    shard_idx = 0
    
    for pair in tqdm(pairs, desc=f"Tokenizing {split_name} pairs"):
        tokens_np = format_and_tokenize(pair)
        if token_count + len(tokens_np) > shard_size:
            filename = os.path.join(local_dir, f"oasst1_sft_{split_name}_{shard_idx:06d}.npy")
            np.save(filename, np.concatenate(shard))
            print(f"Saved {filename} ({token_count} tokens)")
            shard = []
            token_count = 0
            shard_idx += 1
        shard.append(tokens_np)
        token_count += len(tokens_np)
    
    # Save any remaining tokens
    if shard:
        filename = os.path.join(local_dir, f"oasst1_sft_{split_name}_{shard_idx:06d}.npy")
        np.save(filename, np.concatenate(shard))
        print(f"Saved {filename} ({token_count} tokens)")

# Save validation split first (smaller)
print("\nSaving validation split...")
save_split(val_pairs, "val")

# Save training split
print("\nSaving training split...")
save_split(train_pairs, "train")

print("Done! Shards are ready for your DataLoader.")