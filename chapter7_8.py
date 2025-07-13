import streamlit as st

def render_7_8():
    """Renders the detailed code explanation section."""
    st.subheader("7.8: Full Code Explanation")
    st.markdown("""
    This section provides a detailed, block-by-block explanation of the complete Python script from the previous section. The goal is to demystify the code and connect each part back to the concepts we've learned.
    """)

    st.subheader("1. Hyperparameters")
    st.markdown("These are the settings that control the model's architecture and the training process. We define them at the top for easy modification.")
    st.code("""
# --- Hyperparameters ---
batch_size = 64 # How many independent sequences will we process in parallel?
block_size = 256 # What is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # The dimension of each word embedding
n_head = 6   # The number of attention heads
n_layer = 6  # The number of Transformer blocks to stack
dropout = 0.2 # The dropout rate to prevent overfitting
    """, language="python")
    st.markdown("- **`block_size`**: This is the model's context window. It can never see more than this many tokens into the past when making a prediction.")
    st.markdown("- **`n_embd`**: The size of our vectors. A larger dimension allows the model to encode more nuanced information.")
    st.markdown("- **`n_head` & `n_layer`**: These define the model's depth and complexity. More heads and layers mean a more powerful model, but also one that is slower to train.")

    st.subheader("2. Tokenizer and Data Loading")
    st.markdown("Here, we set up our character-level tokenizer and a function to feed random batches of data to the model during training.")
    st.code("""
# --- Tokenizer ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [char_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_char[i] for i in l])

# --- Data Loading ---
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Generate random starting points for our batches
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create the input chunks (x)
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Create the target chunks (y), which are shifted by one character
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
    """, language="python")
    st.markdown("- **`get_batch`**: This function is crucial. It grabs a `batch_size` number of random chunks from our text. For each chunk `x`, it creates the corresponding target `y` by shifting the chunk one position to the right. This is how we create the `(input, target)` pairs for our self-supervised learning task.")

    st.subheader("3. The `LanguageModel` Class")
    st.markdown("This is the main class that defines our entire nano-GPT. Let's look at its three main methods.")
    
    st.markdown("#### `__init__(self, ...)` - The Constructor")
    st.markdown("This method defines the layers of our model.")
    st.code("""
def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
    super().__init__()
    # A lookup table for token embeddings
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    # A lookup table for position embeddings
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    # A sequence of Transformer Blocks
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
    # A final layer normalization
    self.ln_f = nn.LayerNorm(n_embd)
    # The final linear layer to map to vocabulary scores
    self.lm_head = nn.Linear(n_embd, vocab_size)
    """, language="python")
    st.markdown("- **`token_embedding_table`**: The lookup table for our character vectors (from Chapter 2).")
    st.markdown("- **`position_embedding_table`**: The lookup table that provides a vector for each position, giving the model a sense of order.")
    st.markdown("- **`blocks`**: This creates a stack of `n_layer` number of the `Block`s we defined in section 7.3.")
    st.markdown("- **`lm_head`**: The final layer that takes the processed token representation and converts it into a vector of `vocab_size` scores (logits).")

    st.markdown("#### `forward(self, ...)` - The Forward Pass")
    st.markdown("This method defines what happens when we pass data through the model.")
    st.code("""
def forward(self, idx, targets=None):
    B, T = idx.shape

    # 1. Get initial token and position embeddings
    tok_emb = self.token_embedding_table(idx) # (Batch, Time, Embedding Dim)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (Time, Embedding Dim)
    
    # 2. Add them together
    x = tok_emb + pos_emb # (B, T, C)
    
    # 3. Pass through the Transformer blocks
    x = self.blocks(x) # (B, T, C)
    
    # 4. Final normalization
    x = self.ln_f(x) # (B, T, C)
    
    # 5. Get the final scores (logits)
    logits = self.lm_head(x) # (B, T, vocab_size)

    # (Loss calculation code for training)
    # ...
    
    return logits, loss
    """, language="python")
    st.markdown("This method executes the exact architecture we've discussed: it gets the embeddings, adds positional information, passes them through the stack of attention blocks, and finally produces the output logits.")

    st.markdown("#### `generate(self, ...)` - The Generation Method")
    st.markdown("This method uses the trained model to generate new text.")
    st.code("""
def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # 1. Crop context to the last 'block_size' tokens
        idx_cond = idx[:, -self.block_size:]
        
        # 2. Get the predictions (logits) by running a forward pass
        logits, loss = self(idx_cond)
        
        # 3. Focus only on the last time step
        logits = logits[:, -1, :] # Becomes (B, C)
        
        # 4. Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        
        # 5. Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        
        # 6. Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
    return idx
    """, language="python")
    st.markdown("This is the autoregressive loop we saw in the previous demo. It repeatedly gets a prediction, appends it to the context, and feeds the new context back into the model to get the next prediction.")
