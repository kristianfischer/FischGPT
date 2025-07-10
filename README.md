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
│   │   ├── embeddings.py    # Token and position embeddings
│   │   └── gpt.py          # Main GPT model
│   ├── config/
│   │   ├── __init__.py
│   │   └── model_config.py  # Model configuration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py     # Text tokenization
│   │   └── dataloader.py    # Data loading utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── optimizer.py     # Training optimizers
│   │   └── trainer.py       # Training loop
│   └── generation/
│       ├── __init__.py
│       └── sampler.py       # Text generation
├── scripts/
│   ├── train.py            # Training script
│   ├── generate.py         # Generation script
│   └── pretrained.py       # Load pretrained weights
├── tests/
│   ├── test_attention.py
│   ├── test_mlp.py
│   └── test_gpt.py
└── requirements.txt
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
- **embeddings.py**: How words become numbers
- **gpt.py**: The complete system

## Key Insights for Understanding

### Attention Mechanism (attention.py)
- **Query, Key, Value**: Each word asks "What am I looking for?", offers "What am I offering?", says "What am I actually saying?"
- **Multi-head**: Multiple perspectives on the same text
- **Masked attention**: Can't see the future (causal)

### MLP (mlp.py)
- **Expansion**: 4x wider for deep thinking
- **Activation**: GELU for non-linearity
- **Compression**: Back to original size

### Block (block.py)
- **Residual connections**: Keep original info while adding insights
- **Layer normalization**: Keep numbers stable
- **Attention + MLP**: One complete thinking cycle

## Getting Started

1. **Start with attention.py** - This is the revolutionary part
2. **Then mlp.py** - The thinking component
3. **Then block.py** - How they work together
4. **Finally gpt.py** - The complete system

## Training

```bash
python scripts/train.py
```

## Generation

```bash
python scripts/generate.py
```

## Testing

```bash
python -m pytest tests/
```

This structure helps you think like a top AI engineer - understanding each component deeply before seeing how they work together.