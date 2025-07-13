import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import pandas as pd

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
    def __init__(self, head_size):
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
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- Full Language Model ---
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
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
# model = LanguageModel()
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

    st.subheader("🛠️ Interactive Demo: The Generation Loop")
    st.markdown("Let's trace the generation process to see exactly how the model writes new text.")

    # --- Setup for Demo ---
    st.markdown("#### 1. The 'Knowledge Base' (Corpus)")
    st.markdown("Paste a short paragraph of text here. This will be the only text our model learns from. The vocabulary for our tokenizer will be built from this text.")
    text = st.text_area("Enter a short paragraph for the model to learn from:", 
                        "hello world this is a test of the emergency broadcast system. this is only a test.", 
                        height=150)
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [char_to_int.get(c, -1) for c in s if c in char_to_int] # Handle chars not in vocab
    decode = lambda l: ''.join([int_to_char.get(i, '?') for i in l])
    
    st.markdown("#### 2. The Generation Prompt")
    start_text = st.text_input("Enter a starting character or phrase:", "h")
    num_to_generate = st.slider("Number of characters to generate:", 1, 50, 10)

    if st.button("Generate Text"):
        st.markdown("---")
        generated_text = start_text
        placeholder = st.empty()

        for i in range(num_to_generate):
            with placeholder.container():
                st.markdown(f"#### Generating character #{i+1}...")
                
                # Step 1: Input
                st.write(f"**1. Input:** The model receives the current raw text and tokenizes it into a sequence of integers.")
                st.info(f"Current Text: `{generated_text}`")
                encoded_input = encode(generated_text)
                if -1 in encoded_input:
                    st.error("Input contains characters not found in the corpus vocabulary. Generation stopped.")
                    break
                input_tensor = torch.tensor([encoded_input], dtype=torch.long)
                st.code(f"Input Tensor: {input_tensor}")

                # Step 2: Forward Pass
                st.write(f"**2. Forward Pass:** The model processes the input tensor and produces `logits` - a raw, unnormalized score for every character in our vocabulary. We only care about the logits for the very last token in the sequence, as that's what we use to predict the *next* token.")
                # We simulate this process with random numbers for the demo
                simulated_logits = torch.randn(1, len(generated_text), vocab_size)
                last_token_logits = simulated_logits[:, -1, :]
                with st.expander("Show Logits Tensor (Scores for each character)"):
                    st.write(last_token_logits)
                
                # Step 3: Softmax
                st.write(f"**3. Softmax:** The model converts these raw scores into a probability distribution. The probabilities for all {vocab_size} characters now sum to 1. A higher probability means the model is more confident in that prediction.")
                probs = F.softmax(last_token_logits, dim=-1)
                with st.expander("Show Probability Distribution"):
                    prob_df = pd.DataFrame(probs.tolist()[0], index=chars, columns=["Probability"])
                    st.bar_chart(prob_df)

                # Step 4: Sampling
                st.write(f"**4. Sampling:** The model chooses the next character from this probability distribution. For this demo, we'll use **greedy sampling** and just pick the character with the highest probability. A real model might sample randomly to be more creative.")
                next_token_int = torch.argmax(probs).item()
                next_char = int_to_char[next_token_int]
                st.success(f"**Predicted Next Character:** '{next_char}'")

                # Step 5: Append
                generated_text += next_char
                st.write(f"**5. Append:** The new character is appended to the sequence. The next loop will start with the updated text: `{generated_text}`")
                st.markdown("---")
            time.sleep(1)
        
        st.subheader("Final Result")
        st.info(generated_text)
