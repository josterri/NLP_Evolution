import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def render_7_2():
    """Renders the Model Components section."""
    st.subheader("7.2: Step 2 - Building the Model Components")

    st.subheader("Motivation: From LEGO Bricks to a Castle")
    st.markdown("""
    A Transformer model seems incredibly complex, but like a LEGO castle, it's built from a few simple, reusable types of bricks. Before we can assemble a full Transformer block, we need to understand and build each of these individual components.

    In this section, we will build the three core 'bricks' of our nano-GPT:
    1.  **The Self-Attention Head:** The mechanism that allows words to communicate with each other.
    2.  **The Feed-Forward Network:** A small neural network that "thinks" about the information gathered by attention.
    3.  **The Transformer Block:** The component that combines the Attention Head and the Feed-Forward Network.
    
    We will implement each of these as a Python class using the PyTorch library.
    """)
    st.warning("This section introduces PyTorch, a popular deep learning framework. The code is designed to be conceptual, but it reflects how real models are built.")

    # --- Component 1: The Self-Attention Head ---
    st.markdown("---")
    st.subheader("Component 1: The Self-Attention Head")
    st.markdown("""
    As we learned in Chapter 4, the core of the Transformer is the **Self-Attention Head**. Its job is to produce a new, context-aware vector for each token by creating a weighted blend of all the other token's `Value` vectors.

    **The Process Recap:**
    1.  For each input token, we generate a **Query**, a **Key**, and a **Value** vector using learned linear layers.
    2.  We calculate the attention scores by taking the dot product of the current token's **Query** with every other token's **Key**.
    3.  We apply a **mask** to hide future tokens (since this is a generative model).
    4.  We apply **softmax** to the scores to get the final attention weights.
    5.  We produce the final output for this token by taking a weighted sum of all the **Value** vectors.
    """)
    st.subheader("🐍 The Python Code: `Head` Class")
    with st.expander("Show the PyTorch code for a single Self-Attention Head"):
        st.code("""
import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    \"\"\" one head of self-attention \"\"\"

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        # Linear layers to create Q, K, V vectors
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # A buffer for the attention mask (tril), not a model parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 
        
        # Apply the mask to block future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        
        # Apply softmax to get attention weights
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        
        return out
        """, language='python')

    # --- Component 2: The Feed-Forward Network ---
    st.markdown("---")
    st.subheader("Component 2: The Feed-Forward Network")
    st.markdown("""
    After the attention mechanism allows the tokens to communicate and blend their information, we need a component that can *process* this new information. This is the job of the **Feed-Forward Network (FFN)**.

    You can think of this as the "thinking" part of the Transformer block. While attention is about communication, the FFN is about computation. It takes the context-rich vector for each token and processes it independently.

    Its structure is very simple: two linear layers with a non-linear activation function (like ReLU) in between. The first layer expands the dimensionality of the vector, and the second layer compresses it back down.
    """)
    st.subheader("🐍 The Python Code: `FeedForward` Class")
    with st.expander("Show the PyTorch code for the Feed-Forward Network"):
        st.code("""
import torch.nn as nn

class FeedForward(nn.Module):
    \"\"\" a simple linear layer followed by a non-linearity \"\"\"

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # First layer expands the embedding dimension by 4x
            nn.Linear(n_embd, 4 * n_embd),
            # ReLU activation function
            nn.ReLU(),
            # Second layer projects it back to the original dimension
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
        """, language='python')

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Role of the FFN:** Why is the Feed-Forward Network necessary? What would happen if we just stacked attention layers on top of each other without an FFN in between?
    2.  **`masked_fill`:** In the `Head` class, what is the purpose of the line `wei = wei.masked_fill(...)`? What would happen if we removed it?
    3.  **Dropout:** Both components use a `Dropout` layer. What is the purpose of dropout in training a neural network?
    """)
