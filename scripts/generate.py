import torch
import tiktoken
import argparse
import os
from src.config.gpt_config import GPTConfig
from src.model.gpt import GPT


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config'] if 'config' in checkpoint else GPTConfig()
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model


def generate(model, enc, prompt, device, max_length=30, num_return_sequences=5, top_k=50, temperature=1.0):
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)
    results = []
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        results.append(decoded)
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT model.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--prompt', type=str, default="Hello, I'm a language model,", help='Prompt to start generation')
    parser.add_argument('--max_length', type=int, default=30, help='Maximum length of generated text')
    parser.add_argument('--num_return_sequences', type=int, default=5, help='Number of samples to generate')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding('gpt2')
    model = load_model(args.checkpoint, device)
    samples = generate(
        model, enc, args.prompt, device,
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
        top_k=args.top_k,
        temperature=args.temperature
    )
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}: {sample}")

if __name__ == "__main__":
    main() 