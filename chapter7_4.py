import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Re-define all components from previous sections for this self-contained example ---
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- Main Render Function ---
def render_7_4():
    """Renders the Full Model Assembly section."""
    st.subheader("7.4: Step 4 - Creating the Full Language Model")

    st.subheader("Motivation: Putting It All Together")
    st.markdown("""
    We have all our LEGO bricks: the tokenizer, the embedding tables, and the Transformer `Block`. Now it's time to assemble the final castle.

    Our full language model needs to perform the following steps:
    1.  Take a sequence of input tokens (integers).
    2.  Get the embedding for each token.
    3.  Get the positional encoding for each token's position.
    4.  Combine these two embeddings.
    5.  Pass the combined embeddings through a series of Transformer `Block`s.
    6.  Apply one final normalization step.
    7.  Use a final linear layer to project the output into a vector of scores (logits) for each word in our vocabulary.
    """)

    st.subheader("🐍 The Python Code: `LanguageModel` Class")
    with st.expander("Show the PyTorch code for the complete Language Model"):
        st.code("""
class LanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Create a sequence of Transformer Blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Final linear layer to map to vocabulary size
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        # ... (loss calculation code omitted for clarity) ...
        
        return logits
        """, language='python')

    st.subheader("🛠️ Interactive Demo: A Forward Pass Through the Model")
    st.markdown("Let's instantiate our full model and send a sample sequence through it to see the input and output shapes at each stage.")

    # --- Hyperparameters ---
    st.markdown("##### Model Hyperparameters")
    c1, c2, c3, c4 = st.columns(4)
    vocab_size = c1.number_input("Vocabulary Size", value=65)
    n_embd = c2.number_input("Embedding Dimension", value=32)
    block_size = c3.number_input("Block Size (Context Length)", value=64)
    n_head = c4.number_input("Number of Attention Heads", value=4)
    n_layer = c1.number_input("Number of Transformer Blocks", value=4)
    dropout = c2.number_input("Dropout Rate", value=0.1)

    if st.button("Instantiate Model and Run Forward Pass"):
        # --- Define the full model class here for the demo ---
        class LanguageModel(nn.Module):
            def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
                super().__init__()
                self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
                self.position_embedding_table = nn.Embedding(block_size, n_embd)
                self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
                self.ln_f = nn.LayerNorm(n_embd)
                self.lm_head = nn.Linear(n_embd, vocab_size)
            def forward(self, idx):
                B, T = idx.shape
                tok_emb = self.token_embedding_table(idx)
                pos_emb = self.position_embedding_table(torch.arange(T))
                x = tok_emb + pos_emb
                x = self.blocks(x)
                x = self.ln_f(x)
                logits = self.lm_head(x)
                return logits

        # --- Run the demo ---
        st.markdown("---")
        st.write("1. **Instantiating the Model** with the specified hyperparameters.")
        model = LanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
        st.success("Model created successfully!")

        st.write("2. **Creating a Sample Input:** A batch of 1 sequence, with 8 tokens (characters).")
        # (B, T) -> Batch=1, Time=8 (8 characters in our sequence)
        sample_input = torch.randint(0, vocab_size, (1, 8))
        st.info(f"**Input Tensor Shape:** `{list(sample_input.shape)}` (Batch, Time)")
        st.write("Input Tensor (token integers):", sample_input)

        st.write("3. **Running the Forward Pass:** Passing the input through the model.")
        with st.spinner("Processing..."):
            logits = model(sample_input)
        
        st.write("4. **Receiving the Output (Logits):**")
        st.info(f"**Output Tensor Shape:** `{list(logits.shape)}` (Batch, Time, Vocab Size)")
        st.success("Forward pass complete!")

        st.markdown("""
        **Why this shape?** The output shape `(1, 8, 65)` tells us that for each of the 8 input tokens in our sequence, the model has produced a list of 65 scores. Each score corresponds to the model's prediction for what the *very next character* should be. For example, the output at `logits[0, -1, :]` is the model's prediction for the character that should follow the 8th and final token of our input. This is the vector we will use to generate new text.
        """)
