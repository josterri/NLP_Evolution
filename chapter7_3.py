import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# --- Re-define components from 7.2 for this self-contained example ---
class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# --- Main Render Function ---
def render_7_3():
    """Renders the Transformer Block section."""
    st.subheader("7.3: Step 3 - Assembling the Transformer Block")

    st.subheader("Motivation: Combining Communication and Computation")
    st.markdown("""
    We have our two main LEGO bricks: the **Self-Attention Head** (which allows tokens to communicate) and the **Feed-Forward Network** (which allows tokens to "think" about what they've learned).

    Now, we need to assemble them into a standard, reusable component: the **Transformer Block**. To do this, we need two more crucial concepts: **Multi-Head Attention** and **Residual Connections**.
    """)

    # --- Component 1: Multi-Head Attention ---
    st.markdown("---")
    st.subheader("Component 1: Multi-Head Attention")
    st.markdown("""
    A single self-attention head can learn to focus on one type of relationship. For example, it might learn to connect verbs to their subjects. But what if we want to learn multiple types of relationships at once?

    **Analogy:** Instead of one librarian asking one question, we have a committee of 8 librarians (8 "heads"). Each librarian has their own specialized Q, K, V lenses and asks a different kind of question.
    -   Head 1 might ask: "Who is the subject of this verb?"
    -   Head 2 might ask: "Which adjective is describing this noun?"
    -   Head 3 might ask: "Which pronoun does this noun refer to later?"

    By running multiple heads in parallel and then combining their results, the model can capture a much richer set of syntactic and semantic relationships from the text.
    """)
    st.subheader("🐍 The Python Code: `MultiHeadAttention` Class")
    with st.expander("Show the PyTorch code for Multi-Head Attention"):
        st.code("""
class MultiHeadAttention(nn.Module):
    \"\"\" multiple heads of self-attention in parallel \"\"\"

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        # Create a list of attention heads
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        # Add a projection layer to combine the head outputs
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run each head on the input in parallel and concatenate the results
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project the concatenated outputs back to the original embedding dimension
        out = self.dropout(self.proj(out))
        return out
        """, language='python')
        
        st.subheader("A Simple Example")
        st.markdown("**Why we need this for next-word prediction:** This allows the model to gather different kinds of contextual clues simultaneously before making a prediction. For the phrase `the brave crew must`, one head might focus on the `brave crew` relationship, while another focuses on what kind of words typically follow `must`. Combining these signals leads to a better prediction.")
        st.code("""
# --- Example ---
# Let's assume our embedding dimension is 32 (n_embd=32)
# We want 4 parallel attention heads (num_heads=4)
# Each head will therefore have a dimension of 32 / 4 = 8 (head_size=8)

# Create a Multi-Head Attention module
# (block_size and dropout are for the full model, we can use placeholders)
mha = MultiHeadAttention(num_heads=4, head_size=8, n_embd=32, block_size=64, dropout=0.1)

# Create a dummy input tensor:
# Batch size = 1, Sequence length = 5 tokens, Embedding dimension = 32
x = torch.randn(1, 5, 32)

# Pass the input through the multi-head attention layer
output = mha(x)

# The output has the exact same shape as the input!
# It has gathered information from all heads and projected it back.
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
        """, language='python')


    # --- Component 2: Residual Connections & Layer Norm ---
    st.markdown("---")
    st.subheader("Component 2: Residual Connections & Layer Normalization")
    st.markdown("""
    When we stack many layers on top of each other in a deep neural network, we can run into a problem: the signal from the original input can get lost or garbled as it passes through so many transformations. This is the infamous "vanishing gradient" problem.

    Transformers solve this with a simple yet powerful technique called a **Residual Connection** (or "skip connection").

    **Analogy:** Imagine a highway and a local road. The main information (the original input `x`) flows unimpeded down the highway. A copy of the information is sent down a local road to be processed by our Multi-Head Attention block. The result of that processing is then added back to the original highway traffic.

    This ensures that the model never loses the original information. It only learns to add the useful new context on top of it. After the addition, a **Layer Normalization** step is applied to re-center the data and keep the training process stable.
    """)

    # --- Final Assembly ---
    st.markdown("---")
    st.subheader("Final Assembly: The Transformer Block")
    st.markdown("We can now assemble our final `Block`. It consists of two main sub-layers:")
    st.markdown("""
    1.  A Multi-Head Self-Attention layer, followed by a residual connection and layer normalization.
    2.  A Feed-Forward Network, followed by another residual connection and layer normalization.
    """)
    st.image("https://jalammar.github.io/images/t/transformer_decoder_block_2.png", caption="The architecture of a single Transformer Decoder Block. Image by Jay Alammar.")
    
    st.subheader("🐍 The Python Code: `Block` Class")
    with st.expander("Show the PyTorch code for a full Transformer Block"):
        st.code("""
class Block(nn.Module):
    \"\"\" Transformer block: communication followed by computation \"\"\"

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        # The communication part: Multi-Head Self-Attention
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        # The computation part: Feed-Forward Network
        self.ffwd = FeedForward(n_embd, dropout)
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # First, the self-attention sub-layer with residual connection
        # x + ... is the residual connection
        x = x + self.sa(self.ln1(x))
        # Then, the feed-forward sub-layer with residual connection
        x = x + self.ffwd(self.ln2(x))
        return x
        """, language='python')

        st.subheader("A Simple Example")
        st.markdown("**Why we need this for next-word prediction:** A single block isn't enough. We need to stack these blocks to allow the model to learn progressively more complex patterns. The residual connections are vital for this stacking, ensuring that information from the initial embeddings can flow all the way to the final layer without being lost. This deep processing is what allows the model to make a highly informed final prediction.")
        st.code("""
# --- Example ---
# Let's create a single Transformer Block
# Embedding dimension = 32, Number of heads = 4
block = Block(n_embd=32, n_head=4, block_size=64, dropout=0.1)

# Create a dummy input tensor
x = torch.randn(1, 5, 32)

# Pass the input through the block
output = block(x)

# The output shape is identical to the input shape.
# This is crucial because it allows us to stack these blocks
# on top of each other, creating a deep network.
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
        """, language='python')

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Residual Connection:** In the `Block`'s `forward` method, what does the `x = x + ...` line represent? What would happen to the information flow if we changed it to just `x = ...`?
    2.  **Layer Normalization:** The `ln1` and `ln2` layers are applied *before* the main transformation in our code. This is a common variation called "pre-normalization". Why might it be beneficial to normalize the input *before* it goes into the attention or feed-forward layers?
    """)
