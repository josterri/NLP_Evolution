import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Full Language Model Class Definition (copied from previous sections) ---
# We need the full model definition here to instantiate and train it.

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
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

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Main Render Function ---
def render_7_7():
    """Renders the real training and generation section."""
    st.subheader("7.7: Real Training & Generation Workbench")
    st.markdown("""
    This is the final step where everything comes together. We will use the text you provide to perform a **real, live training run** on our nano-GPT model. This will take some time, but the results will be far more coherent than the random output of the untrained model.
    """)
    st.warning("Training is computationally intensive. This demo runs on your browser's machine and may be slow. A smaller text corpus and fewer training iterations will be faster.")

    # --- Step 1: The Corpus ---
    st.markdown("---")
    st.markdown("#### Step 1: The 'Knowledge Base' (Corpus)")
    text = st.text_area("Enter the text for the model to learn from:", 
                        "The quick brown fox jumps over the lazy dog. A lazy dog is a happy dog. The fox is not a dog.", 
                        height=150)
    
    # --- Step 2: Training ---
    st.markdown("---")
    st.markdown("#### Step 2: Train the Model")
    st.markdown("Here, we will instantiate and train our `LanguageModel` on the text above.")
    
    max_iters = st.slider("Select number of training iterations:", 100, 5000, 1000)

    if st.button("Train nano-GPT"):
        # --- Setup ---
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        char_to_int = {ch: i for i, ch in enumerate(chars)}
        int_to_char = {i: ch for i, ch in enumerate(chars)}
        encode = lambda s: [char_to_int.get(c, 0) for c in s] # Default to 0 for safety
        decode = lambda l: ''.join([int_to_char.get(i, '?') for i in l])
        data = torch.tensor(encode(text), dtype=torch.long)

        # --- Model Definition ---
        n_embd = 64
        block_size = 32
        n_head = 4
        n_layer = 4
        dropout = 0.1
        model = LanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
        
        # --- Training Loop ---
        st.markdown("##### Training Progress")
        st.markdown("The code below is now running live. The loss should decrease over time, showing that the model is learning.")
        
        with st.expander("Show the Python Training Loop Code"):
            st.code("""
# --- Training Loop ---
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
losses = []

for i in range(max_iters):
    # Simple batching for the demo
    # In a real scenario, you'd have separate train/validation data
    ix = torch.randint(len(data) - block_size, (16,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Forward pass, calculate loss
    logits, loss = model(x, y)
    
    # Backward pass, update weights
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
            """, language="python")

        progress_bar = st.progress(0)
        status_text = st.empty()
        losses = []
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for i in range(max_iters):
            # Simple batching for the demo
            ix = torch.randint(len(data) - block_size, (16,))
            x = torch.stack([data[i:i+block_size] for i in ix])
            y = torch.stack([data[i+1:i+block_size+1] for i in ix])
            
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            # Update progress bar and status text
            progress = (i + 1) / max_iters
            progress_bar.progress(progress)
            status_text.text(f"Iteration {i+1}/{max_iters} | Loss: {loss.item():.4f}")

        status_text.text("Training complete!")
        st.success("Training complete!")
        
        # --- Store results in session state ---
        st.session_state.trained_model = model
        st.session_state.tokenizer = {'encode': encode, 'decode': decode}
        st.session_state.losses = losses
        
        # Display loss curve
        st.subheader("Training Loss Over Time")
        loss_df = pd.DataFrame(losses, columns=['Loss'])
        st.line_chart(loss_df)
        st.caption("A decreasing loss curve shows that the model was successfully learning the patterns in the text.")

    # --- Step 3: Generation ---
    if 'trained_model' in st.session_state:
        st.markdown("---")
        st.subheader("Step 3: Generate Text from Your Trained Model")
        st.markdown("Now, use the model you just trained to generate new text. The output should resemble the style and content of your corpus.")
        
        model = st.session_state.trained_model
        tokenizer = st.session_state.tokenizer
        
        start_char = st.text_input("Enter a starting character:", "t")
        num_to_generate = st.slider("Number of characters to generate:", 20, 500, 100)

        if st.button("Generate Text"):
            with st.spinner("Generating..."):
                context = torch.tensor([tokenizer['encode'](start_char)], dtype=torch.long)
                generated_tokens = model.generate(context, max_new_tokens=num_to_generate)[0].tolist()
                generated_text = tokenizer['decode'](generated_tokens)
                
                st.subheader("Final Generated Text")
                st.info(generated_text)
                st.warning("**Analysis:** Compare this to the gibberish from the untrained model. This text, while not perfect, has clearly learned the vocabulary, word structures, and common patterns from your input corpus. This demonstrates the power of the training process.")
