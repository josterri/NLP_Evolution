import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# --- Main Render Function ---
def render_7_6():
    """Renders the final code and generation section."""
    st.subheader("7.6: The Grand Finale - Full Code & Generation")

    st.subheader("Motivation: The Finished Product")
    st.markdown("""
    This is it! We have built all the individual components of a Transformer-based language model. Now, we will put them all together into a single, complete Python script.

    This script contains everything needed to:
    1.  Load and tokenize a text file.
    2.  Define the full `LanguageModel` architecture.
    3.  Train the model on the text data.
    4.  Generate new text from the trained model.

    After the full code, we'll walk through an interactive demonstration of the `generate` function to see exactly how the model writes new text, character by character.
    """)

    st.subheader("🐍 The Complete Python Code for nano-GPT")
    with st.expander("Show the full, runnable Python script"):
        st.code("""
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Hyperparameters ---
batch_size = 64 # How many independent sequences will we process in parallel?
block_size = 256 # What is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# --------------------

# We recommend creating a file named 'input.txt' in the same folder
# and pasting some text into it (e.g., a short story or poem).
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("'input.txt' not found. Using default text.")
    text = "hello world"

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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- Model Components ---
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

# --- Full Language Model ---
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Training ---
# model = LanguageModel(...)
# m = model.to(device)
# optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
# for iter in range(max_iters):
#     xb, yb = get_batch('train')
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
# print("Training complete.")

# --- Generation ---
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# generated_tokens = m.generate(context, max_new_tokens=500)[0].tolist()
# print(decode(generated_tokens))
        """, language='python')

    st.subheader("🛠️ Interactive End-to-End Workbench")
    st.markdown("Let's walk through the entire process using the text you provide as our model's complete 'knowledge base'.")

    # --- Step 1: The Corpus ---
    st.markdown("---")
    st.markdown("#### Step 1: The 'Knowledge Base' (Corpus)")
    text = st.text_area("Enter a short paragraph for the model to learn from:", 
                        "hello world this is a test of the emergency broadcast system. this is only a test.", 
                        height=150)
    
    # --- Step 2: Tokenization ---
    st.markdown("---")
    st.markdown("#### Step 2: Tokenization")
    st.markdown("We build a character-level tokenizer from the text. This defines our vocabulary and how to convert text to numbers.")
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [char_to_int.get(c, -1) for c in s if c in char_to_int]
    decode = lambda l: ''.join([int_to_char.get(i, '?') for i in l])
    
    with st.expander("Show Tokenizer Details"):
        st.write(f"**Vocabulary Size:** {vocab_size}")
        st.json(char_to_int)

    # --- Step 3: Model Initialization ---
    st.markdown("---")
    st.markdown("#### Step 3: Building the Model")
    st.markdown("We will now instantiate a real (but very small) `LanguageModel` with random weights.")
    
    # Hyperparameters for the demo model
    n_embd = 32
    block_size = 64
    n_head = 4
    n_layer = 4
    dropout = 0.0

    # Instantiate the model
    model = LanguageModel(vocab_size, n_embd, block_size, n_head, n_layer, dropout)
    st.success("A new, untrained nano-GPT model has been created!")

    # --- Step 4: Generating from the UNTRAINED Model ---
    st.markdown("---")
    st.subheader("Step 4: Generating from the UNTRAINED Model")
    st.markdown("Let's see what this 'blank slate' model predicts. Its weights are random, so its predictions will be gibberish.")
    
    start_text_untrained = st.text_input("Enter a starting character:", "t", key="gen_start_untrained")
    
    if st.button("Generate with UNTRAINED Model"):
        st.markdown("---")
        generated_text = start_text_untrained
        gen_placeholder = st.empty()

        for i in range(10): # Generate 10 characters
            with gen_placeholder.container():
                st.markdown(f"#### Generating character #{i+1}...")
                st.info(f"Current Text: `{generated_text}`")
                input_tensor = torch.tensor([encode(generated_text)], dtype=torch.long)
                with torch.no_grad():
                    logits, _ = model(input_tensor)
                last_token_logits = logits[:, -1, :]
                probs = F.softmax(last_token_logits, dim=-1)
                next_token_int = torch.multinomial(probs, num_samples=1).item()
                next_char = int_to_char[next_token_int]
                st.success(f"**Predicted Next Character:** '{next_char}'")
                generated_text += next_char
                st.markdown("---")
            time.sleep(0.5)
        
        st.subheader("Final Untrained Result")
        st.info(generated_text)

    # --- Step 5: Generating from a TRAINED Model (Simulated) ---
    st.markdown("---")
    st.subheader("Step 5: Generating from a TRAINED Model (Simulated)")
    st.markdown("Actually training the model is too slow for a web app. Instead, we'll **simulate** a trained model. Our simulation will 'know' the common character patterns from your text and will be more likely to predict them.")

    start_text_trained = st.text_input("Enter a starting character:", "t", key="gen_start_trained")

    if st.button("Generate with TRAINED Model"):
        st.markdown("---")
        # --- Build a simple bigram frequency map for our simulation ---
        bigram_model = defaultdict(Counter)
        for i in range(len(text) - 1):
            bigram_model[text[i]][text[i+1]] += 1

        generated_text = start_text_trained
        gen_placeholder = st.empty()

        for i in range(20): # Generate more characters
            with gen_placeholder.container():
                st.markdown(f"#### Generating character #{i+1}...")
                st.info(f"Current Text: `{generated_text}`")
                
                last_char = generated_text[-1]
                
                st.write(f"**1. Forward Pass (Simulated):** The model produces `logits` (scores). Because it's 'trained', it gives higher scores to characters that commonly follow `'{last_char}'`.")
                
                # Simulate biased logits
                logits = torch.zeros(vocab_size)
                if last_char in bigram_model:
                    for char, count in bigram_model[last_char].items():
                        logits[char_to_int[char]] = count # Score is based on frequency
                
                st.write(f"**2. Softmax:** The scores are converted into probabilities.")
                probs = F.softmax(logits, dim=-1).unsqueeze(0)
                
                with st.expander("Show Probability Distribution"):
                    prob_df = pd.DataFrame(probs.tolist()[0], index=chars, columns=["Probability"])
                    st.bar_chart(prob_df)

                st.write(f"**3. Sampling:** The model chooses the next character. Notice how the probabilities are no longer random!")
                next_token_int = torch.multinomial(probs, num_samples=1).item()
                next_char = int_to_char[next_token_int]
                st.success(f"**Predicted Next Character:** '{next_char}'")
                
                generated_text += next_char
                st.markdown("---")
            time.sleep(0.5)
        
        st.subheader("Final Trained Result")
        st.info(generated_text)
        st.warning("**Analysis:** Compare this to the output from the untrained model. This text, while simple, starts to look like real language because the model has learned the basic patterns from your corpus. This is the magic of the training process!")
