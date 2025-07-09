from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CasualSelfAttention(nn.Module):
    """
    This is the heart of GPT-2 - the attention mechanism that makes it understand context.
    Think of it like having multiple spotlights that can focus on different parts of a sentence at once.
    Each spotlight (attention head) can look at different words and understand how they relate to each other.
    This is what makes GPT-2 so powerful - it can see patterns across the entire sentence, not just word by word.
    """

    def __init__(self, config):
        super().__init__()
        # We need to make sure we can split the model into equal attention heads
        # This is like making sure you can divide a pizza into equal slices - if you can't, some slices will be bigger than others
        assert config.n_embd % config.n_head == 0
        
        # This creates three different "lenses" (Query, Key, Value) for each attention head
        # Think of it like having 3 different colored glasses for each person in a room
        # Each person can look through red, blue, and green glasses to see different aspects of the same thing
        # The 3* means we're creating 3 times as many connections as normal
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # This combines all the attention results back into one coherent output
        # Like taking all the different colored pictures and merging them into one final image
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Store the configuration so we know how many heads and how big each piece should be
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # This creates a "mask" that prevents the model from looking into the future
        # It's like putting blinders on a horse - the model can only see what came before, not what comes after
        # This is crucial because in real life, you can't see the future when reading a sentence
        # The tril() function creates a triangle shape where only the lower left part is filled with 1s
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # x is our input text, but in number form. B=batch size (how many sentences we're processing at once)
        # T=sequence length (how many words), C=embedding size (how much information each word carries)
        B, T, C = x.size()
        
        # This is where the magic happens - we create Query, Key, and Value for each word
        # Think of it like this: for each word, we ask "What am I looking for?" (Query)
        # "What am I offering?" (Key), and "What am I actually saying?" (Value)
        qkv = self.c_attn(x)
        
        # Split the output into three equal parts - Query, Key, and Value
        # This is like taking a long string and cutting it into three equal pieces
        q, k, v = qkv.chunk(3, dim=2)
        
        # Reshape the data so each attention head can work on its own piece
        # Think of it like taking a big group of people and organizing them into smaller teams
        # Each team (head) can focus on different aspects of the problem
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # This is the attention calculation - how much should each word pay attention to every other word?
        # It's like calculating how important each person in a room is to every other person
        # The sqrt() part is crucial - it prevents the numbers from getting too big and causing problems
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply the mask to prevent looking into the future
        # This is like putting on blinders - you can only see what's in front of you, not behind
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # Convert the attention scores into probabilities (numbers that add up to 1)
        # This tells us exactly how much attention each word should pay to every other word
        att = F.softmax(att, dim=-1)
        
        # Apply the attention to the values - this is where the actual "understanding" happens
        # It's like taking all the important information from other words and combining it
        y = att @ v # (B, nh, T, T) x (B. nh, T, hs) => (B, nh, T, hs)
        
        # Reshape back to the original format and combine all the attention heads
        # This is like taking all the different perspectives and merging them into one coherent view
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Apply the final transformation to get the output
        # This is like the final polish that makes everything work together smoothly
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """
    This is the "thinking" part of the model - it takes the attention results and processes them further
    Think of attention as gathering information from other words, and MLP as thinking about that information
    It's like having a conversation where you first listen to everyone (attention), then think about what they said (MLP)
    """

    def __init__(self, config):
        super().__init__()
        # This expands the information to 4 times its original size
        # Think of it like taking a simple idea and exploring it from many different angles
        # The 4* is crucial - it gives the model room to think deeply about each piece of information
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        
        # This is the activation function that adds "non-linearity" - the ability to make complex decisions
        # Without this, the model would only be able to make simple linear decisions
        # GELU is like a smart switch that can make complex, nuanced decisions
        self.gelu = nn.GELU(approximate = 'tanh')
        
        # This compresses the information back to its original size
        # Think of it like taking all your thoughts and summarizing them into a clear conclusion
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        # Expand the information to think more deeply about it
        x = self.c_fc(x)
        
        # Apply the activation function to make complex decisions
        x = self.gelu(x)
        
        # Compress back to the original size with the new understanding
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """
    This is one complete "thinking cycle" of the model
    It combines attention (gathering information) with MLP (thinking about that information)
    Think of it as one complete step in understanding a sentence
    """

    def __init__(self, config):
        super().__init__()
        # Layer normalization helps keep the numbers in a reasonable range
        # Without this, the numbers can get too big or too small, causing problems
        # It's like having a thermostat that keeps the temperature comfortable
        self.ln_1 = nn.LayerNorm(config.n_embd)
        
        # The attention mechanism that gathers information from other words
        self.attn = CasualSelfAttention(config)
        
        # Another layer normalization after attention
        self.ln_2 = nn.LayerNorm(config.n_embd)
        
        # The MLP that thinks about the gathered information
        self.mlp = MLP(config)

    def forward(self, x):
        # First, normalize the input, then apply attention, then add it back (residual connection)
        # The residual connection is crucial - it's like keeping the original information while adding new insights
        # Without this, the model would lose information as it processes it
        x = x + self.attn(self.ln_1(x))
        
        # Same process for the MLP - normalize, think, add back
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass 
class GPTConfig:
    """
    This defines the "architecture" of the model - how big and complex it is
    Think of it like the blueprint for a building - it defines how many rooms, how big they are, etc.
    """
    block_size: int = 256  # How many words the model can look at at once (like how many words you can remember)
    vocab_size: int = 65   # How many different words/tokens the model knows about
    n_layer: int = 6       # How many thinking cycles the model goes through (like how many times you read a sentence)
    n_head: int = 6        # How many different perspectives the model can take at once
    n_embd: int = 384      # How much information each word carries (like how detailed your memory of each word is)

class GPT(nn.Module):
    """
    This is the complete GPT-2 model - it combines everything into one working system
    Think of it as a complete brain that can understand and generate text
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # This creates the complete model structure
        # wte = word token embeddings (how words are represented as numbers)
        # wpe = word position embeddings (where each word is in the sentence)
        # h = the thinking blocks (the actual processing)
        # ln_f = final layer normalization (the final polish)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Converts words to numbers that carry meaning
            wpe = nn.Embedding(config.block_size, config.n_embd), # Adds position information (1st word, 2nd word, etc.)
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # The thinking layers
            ln_f = nn.LayerNorm(config.n_embd), # Final normalization
        ))
        
        # This converts the final understanding back into word predictions
        # It's like taking all your thoughts and choosing the next word to say
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx is our input text as numbers, targets would be the correct next words (for training)
        B, T = idx.size()
        
        # Make sure we're not trying to process more words than the model can handle
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Create position numbers (0, 1, 2, 3...) for each word
        # This tells the model where each word is in the sentence
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # Convert positions to embeddings (numbers that carry position meaning)
        pos_emb = self.transformer.wpe(pos)
        
        # Convert word tokens to embeddings (numbers that carry word meaning)
        tok_emb = self.transformer.wte(idx)
        
        # Combine word meaning and position meaning
        # This is crucial - the model needs to know both WHAT the word is and WHERE it is
        x = tok_emb + pos_emb
        
        # Pass through all the thinking blocks
        # Each block adds more understanding to the text
        for block in self.transformer.h:
            x = block(x)
        
        # Apply final normalization
        x = self.transformer.ln_f(x)
        
        # Convert the final understanding into predictions for the next word
        logits = self.lm_head(x)
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """
        This loads a pre-trained model that has already learned to understand language
        Think of it like downloading a brain that already knows how to read and write
        Instead of starting from scratch, we get a model that already understands language
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # Different model sizes have different capacities
        # Bigger models can understand more complex patterns but need more computational power
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M parameters - like a smart teenager
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M parameters - like a college graduate
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M parameters - like a PhD
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M parameters - like a genius
        }[model_type]
        
        # GPT-2 always uses the same vocabulary size and context length
        # This ensures compatibility with the pre-trained weights
        config_args['vocab_size'] = 50257 # The number of different tokens GPT-2 knows about
        config_args['block_size'] = 1024 # How many words GPT-2 can look at at once
        
        # Create our model with the right architecture
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # Skip the attention mask - it's not a learned parameter

        # Load the pre-trained model from Hugging Face
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy the learned weights from the pre-trained model to our model
        # This is like transferring knowledge from one brain to another
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # Skip buffers
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # Skip attention masks
        
        # Some weights need to be transposed because of different implementations
        # This is like translating between two different languages that mean the same thing
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # Copy each weight, handling the transposed ones specially
        for k in sd_keys:
            if k in sd_keys_hf:
                if any(k.endswith(w) for w in transposed):
                    # For transposed weights, we need to flip the dimensions
                    if sd_hf[k].shape[::-1] != sd[k].shape:
                        print(f"Shape mismatch for {k}: HF shape {sd_hf[k].shape} vs our shape {sd[k].shape}")
                        continue
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k].t())
                else:
                    # For normal weights, just copy directly
                    if sd_hf[k].shape != sd[k].shape:
                        print(f"Shape mismatch for {k}: HF shape {sd_hf[k].shape} vs our shape {sd[k].shape}")
                        continue
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k])
            else:
                print(f"Skipping key {k} - not found in HF model")

        return model
    
# Set up the device (CPU, GPU, or Apple Silicon)
# This determines where the model runs - GPU is much faster for big models
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"  # Use NVIDIA GPU if available
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"   # Use Apple Silicon GPU if available

# Generation parameters
num_return_sequences = 5  # How many different versions to generate
max_length = 30          # How long each generated text should be

# Load the pre-trained model and put it in evaluation mode
model = GPT.from_pretrained('gpt2')
model.eval()  # Turn off training mode - we only want to generate text, not learn
model.to(device)  # Move the model to the right device (CPU/GPU)

# Set up the tokenizer - this converts text to numbers and back
import tiktoken
enc = tiktoken.get_encoding('gpt2')  # Get the same tokenizer that GPT-2 was trained with

# Convert our starting text to tokens (numbers)
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # Create multiple copies for multiple generations
x = tokens.to(device)

# Set random seeds for reproducible results
# This ensures we get the same results each time we run the code
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# The main generation loop - this is where the magic happens
while x.size(1) < max_length:
    # Get predictions from the model (what should come next?)
    with torch.no_grad():  # Don't calculate gradients - we're not training
        logits = model(x)  # Get raw scores for each possible next word
        logits = logits[:, -1, :]  # Only look at the last word's predictions
        
        # Convert scores to probabilities and sample the next word
        probs = F.softmax(logits, dim=-1)  # Convert scores to probabilities (0-1, sum to 1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # Get the 50 most likely words
        ix = torch.multinomial(topk_probs, 1)  # Randomly choose from the top 50
        xcol = torch.gather(topk_indices, -1, ix)  # Get the actual word index
        x = torch.cat((x, xcol), dim=1)  # Add the new word to our sequence

# Convert the generated numbers back to text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()  # Get the tokens for this generation
    decoded = enc.decode(tokens)  # Convert numbers back to text
    print(">", decoded)

"""
THE COMPLETE STORY OF HOW GPT-2 WORKS

Imagine you're trying to understand a sentence like "The cat sat on the mat." Here's what GPT-2 does:

1. TOKENIZATION (Converting to Numbers):
   - First, it converts each word to a number: "The"=1, "cat"=2, "sat"=3, etc.
   - But it's smarter than just numbers - each word gets a rich representation (like a 768-dimensional vector)
   - Think of it like each word having 768 different characteristics (color, size, emotion, etc.)

2. POSITION EMBEDDINGS (Knowing Where Words Are):
   - The model also learns where each word is in the sentence
   - "The" at position 0, "cat" at position 1, "sat" at position 2, etc.
   - This is crucial because "cat" means different things in "The cat sat" vs "The cat ran"

3. ATTENTION (The Magic Part):
   - For each word, the model asks: "How much should I pay attention to every other word?"
   - "cat" might pay 80% attention to "The" (because it's the subject), 10% to "sat" (the action), etc.
   - This happens through Query, Key, Value:
     * Query: "What am I looking for?" (each word asks this)
     * Key: "What am I offering?" (each word answers this)
     * Value: "What am I actually saying?" (the real information)

4. MULTIPLE PERSPECTIVES (Multi-Head Attention):
   - The model doesn't just do this once - it does it 12 times (12 attention heads)
   - Each head can focus on different aspects: one might focus on grammar, another on meaning, another on context
   - It's like having 12 different experts looking at the same text

5. THINKING (MLP):
   - After gathering information through attention, the model "thinks" about it
   - It expands the information to 4 times its size, processes it, then compresses it back
   - This is where complex patterns and relationships are learned

6. LAYERS (Deep Processing):
   - This attention + thinking process happens 12 times (12 layers)
   - Each layer builds on the understanding of the previous layer
   - Layer 1 might understand basic grammar, Layer 6 might understand context, Layer 12 might understand complex reasoning

7. GENERATION (Predicting the Next Word):
   - When generating text, the model looks at all the words so far
   - It calculates how likely each possible next word is
   - Instead of always picking the most likely word (which would be boring), it samples from the top candidates
   - This creates variety while maintaining coherence

THE KEY INSIGHTS THAT MAKE THIS WORK:

1. SELF-ATTENTION: The model can look at any word in the sentence, not just nearby words. This is revolutionary because it can understand long-range dependencies.

2. RESIDUAL CONNECTIONS: The model keeps the original information while adding new insights. Without this, information would get lost as it's processed.

3. LAYER NORMALIZATION: This keeps the numbers in a reasonable range, preventing them from getting too big or too small.

4. MASKED ATTENTION: The model can only look at previous words, not future words. This mimics how we actually read and write.

5. SCALED ATTENTION: The attention scores are divided by the square root of the embedding size. This prevents the numbers from getting too large and causing numerical instability.

6. MULTI-HEAD ATTENTION: Different heads can learn different types of relationships, making the model much more powerful than single-head attention.

This architecture allows GPT-2 to understand context, grammar, meaning, and even generate coherent text that follows the patterns it learned from its training data. The key is that it can see relationships between any words in the sentence, no matter how far apart they are.
"""