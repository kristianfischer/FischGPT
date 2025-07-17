# FischGPT: Understanding GPT-2 from First Principles

This repository implements GPT-2 from scratch, organized to mirror how top AI engineers at OpenAI, Meta, and Google structure their transformer codebases.

## Repository Structure

```
FischGPT/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py      # Self-attention mechanism
│   │   ├── mlp.py           # Feed-forward network
│   │   ├── block.py         # Transformer block
│   │   └── gpt.py           # Main GPT model
│   ├── config/
│   │   ├── __init__.py
│   │   └── gpt_config.py    # Model configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py   # Data loading utilities
│   │   └── get_dataset.py  # get dataset from hf
├── scripts/
│   ├── train.py             # Training script
│   └── generate.py          # Generation script
├── requirements.txt
└── README.md
```

## Why This Structure?

### 1. **Separation of Concerns**
Each component is isolated, making it easier to:
- Understand individual components
- Debug specific issues
- Test components independently
- Modify one part without affecting others

### 2. **Professional Standards**
This mirrors how major AI labs organize their code:
- **OpenAI**: Separates attention, MLP, and embeddings
- **Meta**: Modular transformer components
- **Google**: Clear separation between model, data, and training

### 3. **Intuitive Understanding**
- **attention.py**: The heart of transformers - understand this first
- **mlp.py**: The "thinking" component
- **block.py**: How attention and MLP work together
- **gpt.py**: The complete system

## Getting Started


## Training

```bash
python scripts/train.py
```

## Generation

```bash
python scripts/generate.py --checkpoint path/to/model.pt --prompt "Your prompt here"
```


This structure helps you think like a top AI engineer - understanding each component deeply before seeing how they work together.