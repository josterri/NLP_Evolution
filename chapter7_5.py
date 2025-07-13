import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

# --- Main Render Function ---
def render_7_5():
    """Renders the Training Loop section."""
    st.subheader("7.5: Step 5 - The Training Loop")

    st.subheader("Motivation: Teaching the Model")
    st.markdown("""
    We have assembled a complete language model, but its brain is empty. All of its weights and parameters are just random numbers. It has no knowledge of language.

    The **training loop** is the process of teaching the model. We will repeatedly show it examples from our text data and adjust its internal weights so that its predictions get progressively less wrong. This is the heart of the machine learning process.
    """)

    st.subheader("🧠 The Method: A Cycle of Prediction and Correction")
    st.markdown("""
    The training loop is a cycle that repeats thousands of times. Each cycle has four main steps:
    1.  **Get a Batch of Data:** We don't show the model the entire text at once. We take a small, random chunk of text (a "batch") to learn from. For each sequence in the batch, we create an `input` and a `target`. The target is simply the input shifted one character to the right.
    2.  **Forward Pass:** We feed the input batch into the model to get its predictions (the `logits`).
    3.  **Calculate Loss:** We compare the model's predictions (`logits`) to the correct answers (`targets`). The "loss" is a single number that measures how wrong the model was. A high loss is bad, a low loss is good. We use a standard function called **Cross-Entropy Loss**.
    4.  **Backward Pass & Optimization:** This is the learning step. We use the loss value to calculate the gradient for every weight in the model. The gradient tells us which direction to "nudge" each weight to make the loss smaller. We then use an **optimizer** (like AdamW) to update all the weights.
    """)

    st.subheader("🛠️ Interactive Example: One Step of Training")
    st.markdown("Let's trace a single training step with a simple example.")

    # --- Example Setup ---
    text = "hello world"
    vocab = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(vocab)}
    int_to_char = {i: ch for i, ch in enumerate(vocab)}
    
    st.markdown("#### 1. The Raw Text & The Batch")
    st.markdown(f"Imagine our entire text is just `'{text}'`. We'll take a random chunk of 8 characters to form our input batch.")
    
    input_text = "hello wo"
    target_text = "ello wor"
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Input `x`:**")
        st.info(f"`{input_text}`")
    with c2:
        st.write("**Target `y`:**")
        st.success(f"`{target_text}`")
    st.caption("The model's goal: when it sees the input `h`, it should predict `e`. When it sees `he`, it should predict `l`, and so on.")

    st.markdown("#### 2. Forward Pass & Loss Calculation")
    st.markdown("We feed the input into our (untrained) model. The model makes a prediction, and we compare it to the target to calculate the loss.")

    # Simulate a forward pass
    # In a real scenario, this would involve the full model from 7.4
    simulated_logits = torch.randn(1, 8, len(vocab)) # (Batch, Time, Vocab Size)
    input_tensor = torch.tensor([[char_to_int[c] for c in input_text]], dtype=torch.long)
    target_tensor = torch.tensor([[char_to_int[c] for c in target_text]], dtype=torch.long)
    
    # PyTorch's cross_entropy function expects logits as (B, C, T)
    loss = F.cross_entropy(simulated_logits.view(-1, len(vocab)), target_tensor.view(-1))
    
    st.write(f"The model makes its (random) predictions. We compare them to the correct targets and get a **Loss Value**.")
    st.metric(label="Initial Loss", value=f"{loss.item():.4f}")
    st.caption("A high loss value means the predictions are very wrong.")

    st.markdown("#### 3. Backward Pass & Optimization")
    st.markdown("This is where the learning happens. The model uses the loss to calculate how to adjust all of its internal weights. After one optimization step, if we run the same input through the model again, the loss will be slightly lower.")
    
    st.metric(label="Loss After One Optimization Step", value=f"{loss.item() * 0.9:.4f} (Simulated)")
    st.success("The model is now slightly better at predicting the next character for this specific chunk of text. We repeat this process thousands of times with different chunks to make the model generalize.")

    st.subheader("🐍 The Python Code: The Training Loop")
    with st.expander("Show the PyTorch code for the Training Loop"):
        st.code("""
import torch

# Assume 'model' is an instance of our LanguageModel class
# Assume 'train_data' is a tensor of our tokenized text data

# Create a PyTorch optimizer
# The optimizer holds the model's parameters and updates them
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Define a function to get a random batch of data
def get_batch(data, batch_size, block_size):
    # Generate random starting points for our batches
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create the input chunks (x)
    x = torch.stack([data[i:i+block_size] for i in ix])
    # Create the target chunks (y), shifted by one character
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# --- The Training Loop ---
for steps in range(10000): # Example: 10,000 steps
    
    # 1. Get a batch of data
    xb, yb = get_batch(train_data, batch_size=32, block_size=64)

    # 2. Forward pass: get predictions and calculate loss
    logits = model(xb)
    B, T, C = logits.shape
    logits_flat = logits.view(B*T, C)
    targets_flat = yb.view(B*T)
    loss = F.cross_entropy(logits_flat, targets_flat)
    
    # 3. Backward pass: calculate gradients
    optimizer.zero_grad(set_to_none=True) # Reset old gradients
    loss.backward() # Calculate new gradients
    
    # 4. Update weights: take a step with the optimizer
    optimizer.step()

print(f"Training complete. Final loss: {loss.item()}")
        """, language='python')

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Batch Size:** What is the purpose of training on a "batch" of data instead of just one sequence at a time? How might a larger batch size affect the training process?
    2.  **Learning Rate (lr):** The optimizer has a `lr` (learning rate) parameter. What would happen if this number was too high? What if it was too low?
    """)
