import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_loader import DataLoaderLite
from src.config.gpt_config import GPTConfig
from src.model.gpt import GPT
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time

pretrained_chkp = "./pretrain_log/log/model_80000.pt"

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
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
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
torch.set_float32_matmul_precision('high')

B = 16  # Smaller batch for more gradient updates
T = 1024
max_steps = 20000  # ~5-10 epochs through dataset
max_lr = 8e-6  # Much lower LR for fine-tuning pretrained model
min_lr = max_lr * 0.1
warmup_steps = 600  # Longer warmup for stability

total_batch_size = 16384  # Total tokens per optimization step
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
    split="train", dataset_name="oasst1_sft"
)

# Proper validation split - first shard is held out for validation
val_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
    split="val", dataset_name="oasst1_sft"
)

# Load pretrained model
checkpoint = torch.load(pretrained_chkp, map_location=device, weights_only=False)
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model'])
model.to(device)
if ddp: 
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = .5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=0.01, learning_rate=max_lr, device_type=device)

# Set up tokenizer for generation
enc = tiktoken.get_encoding("gpt2")

def generate_sample(model, prompt, max_length=200, temperature=0.8, top_k=50):
    """Generate a sample response during training to monitor quality"""
    model.eval()
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_length):
            if tokens.size(1) >= 1024:  # Respect block size
                break
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Stop at end of text
            if next_token.item() == enc._special_tokens['<|endoftext|>']:
                break
    
    return enc.decode(tokens[0].tolist())

# Test prompts for generation during training
test_prompts = [
    "<|user|>Hello! How are you doing today?<|assistant|>",
    "<|user|>Can you explain quantum computing in simple terms?<|assistant|>",
    "<|user|>Write a short poem about coding<|assistant|>",
]

log_dir = "./logs/sft_log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # Evaluate validation loss periodically (like pretrain every 250 steps)
    if step % 1000 == 0 or last_step:
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
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            # Save checkpoint after validation evaluation (like pretrain)
            if step > 0 and (step % 1500 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

    # Generate samples every 1500 steps
    if master_process and step > 0 and step % 1500 == 0:
        print(f"\n=== Generation Samples at Step {step} ===")
        for i, prompt in enumerate(test_prompts):
            print(f"\nPrompt {i+1}: {prompt}")
            response = generate_sample(raw_model, prompt)
            # Extract just the assistant's response
            if "<|assistant|>" in response:
                assistant_response = response.split("<|assistant|>")[-1].split("<|endoftext|>")[0]
                print(f"Response: {assistant_response.strip()}")
            else:
                print(f"Response: {response}")
        print("=" * 50 + "\n")
        
        # Log generations to file
        with open(os.path.join(log_dir, f"generations_{step:05d}.txt"), "w") as f:
            f.write(f"=== Generation Samples at Step {step} ===\n\n")
            for i, prompt in enumerate(test_prompts):
                response = generate_sample(raw_model, prompt)
                f.write(f"Prompt {i+1}: {prompt}\n")
                if "<|assistant|>" in response:
                    assistant_response = response.split("<|assistant|>")[-1].split("<|endoftext|>")[0]
                    f.write(f"Response: {assistant_response.strip()}\n\n")
                else:
                    f.write(f"Response: {response}\n\n")

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
