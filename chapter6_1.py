import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

def render_6_1():
    """Renders the Decoder-Only Architecture section."""
    st.subheader("6.1: The Decoder-Only Architecture (GPT-style)")
    
    st.subheader("Motivation: Why Throw Half the Model Away?")
    st.markdown("""
    In the original Transformer architecture (which we'll explore more in later chapters on translation and summarization), there were two distinct parts:
    1.  **An Encoder:** Its job was to read and understand a source sentence (e.g., in English).
    2.  **A Decoder:** Its job was to take that understanding and generate a new sentence in a target language (e.g., in German).

    However, for pure text generation, we don't have a separate source sentence to understand. Our only goal is to **continue a given sequence of text**. This led to a brilliant simplification pioneered by models like GPT (Generative Pre-trained Transformer): **why not just use the Decoder part?**

    By discarding the Encoder, we create a more focused and efficient architecture whose sole purpose is next-word prediction.
    """)

    st.subheader("🧠 The Core Component: Masked Self-Attention")
    st.markdown("""
    The most crucial component that makes this possible is **Masked Self-Attention**. This is the key difference between an Encoder block and a Decoder block.
    
    In a normal self-attention block (like in an Encoder), every word can "look at" every other word, both before and after it. This is great for building a deep understanding of an entire sentence. But for next-word prediction, this would be **cheating!**

    To predict the word "sat" in the sequence `the cat sat`, the model should only be allowed to see "the" and "cat". It cannot see the word "sat" itself, or any words that might come after. The model must predict the future based only on the past.

    The **mask** is a simple but brilliant trick that enforces this rule. Before the softmax step in the attention calculation, the model adds a large negative number (`-infinity`) to the attention scores for all "future" positions. When the softmax is applied, these large negative scores become zero, effectively hiding the future words from the attention mechanism.
    """)

    st.subheader("🛠️ Interactive Demo: Visualizing the Attention Mask")
    st.markdown("Enter a short sentence to see the attention mask that a generative model would use. `1` means the word is allowed to attend, `0` means it is blocked.")

    sentence = st.text_input("Enter a sentence for the mask demo:", "the cat sat on the mat")
    tokens = sentence.lower().split()

    if tokens:
        size = len(tokens)
        mask = np.tril(np.ones((size, size)))
        df = pd.DataFrame(mask, index=[f"Query: '{w}'" for w in tokens], columns=[f"Key: '{w}'" for w in tokens])
        
        fig, ax = plt.subplots()
        sns.heatmap(df, annot=True, cmap="viridis_r", fmt=".0f", ax=ax, cbar=False, linewidths=.5)
        ax.set_title("Attention Mask")
        st.pyplot(fig)
        st.caption("Notice how each word (row) can only 'see' itself and the words that came before it (columns). For example, the query from 'sat' can attend to the keys of 'the', 'cat', and 'sat', but is masked from seeing 'on', 'the', and 'mat'.")

    st.subheader("The Generation Process: Autoregressive Generation")
    st.markdown("""
    This masked architecture allows for a process called **autoregressive generation**, which simply means generating the sequence one step at a time, where each new step depends on all the previous ones.
    """)

    st.markdown("Click the button below to see a step-by-step animation of this process.")
    if st.button("Animate Generation"):
        placeholder = st.empty()
        prompt = ["the", "cat"]
        for i in range(3):
            with placeholder.container():
                st.markdown(f"#### Step {i+1}")
                st.write(f"**1. Input:** The model receives the current sequence:")
                st.info(f"`{' '.join(prompt)}`")
                st.write(f"**2. Process:** It passes this sequence through its stack of masked self-attention layers.")
                
                # Simulate prediction
                if prompt == ["the", "cat"]:
                    prediction = "sat"
                elif prompt == ["the", "cat", "sat"]:
                    prediction = "on"
                elif prompt == ["the", "cat", "sat", "on"]:
                    prediction = "the"
                
                st.write(f"**3. Predict:** Based on the context, it predicts the most likely next word:")
                st.success(f"`{prediction}`")

                prompt.append(prediction)
                st.write(f"**4. Append:** The model appends the new word. The input for the next step will be:")
                st.info(f"`{' '.join(prompt)}`")
                st.markdown("---")
            time.sleep(2)


    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **The Mask's Shape:** Why is the attention mask always a lower-triangular matrix (a triangle of 1s on and below the main diagonal)?
    2.  **What if there's no mask?** If we used a normal, unmasked self-attention (like in an Encoder) for next-word prediction, what would the model learn to do? Why would this be a problem? (Hint: What's the easiest way to predict the next word if you're allowed to see it?)
    3.  **Generation Speed:** Is the autoregressive generation process parallel or sequential? Why does this make generating text much slower than analyzing it with an Encoder?
    """)

    st.subheader("🐍 The Python Behind the Mask")
    with st.expander("Show the Python Code for Creating an Attention Mask"):
        st.code("""
import numpy as np

def create_attention_mask(sequence_length):
    \"\"\"
    Creates a lower-triangular matrix for masked self-attention.
    The mask has 1s on and below the diagonal, and 0s above.
    \"\"\"
    # np.tril creates a lower-triangular matrix of 1s
    mask = np.tril(np.ones((sequence_length, sequence_length)))
    return mask

def apply_mask_to_scores(attention_scores, mask):
    \"\"\"
    Applies the mask to attention scores before the softmax step.
    Where the mask is 0, we set the score to a very large negative number.
    \"\"\"
    # Add a very large negative number where the mask is 0
    masked_scores = attention_scores + (1.0 - mask) * -1e9
    return masked_scores

# --- Example ---
# For a sentence with 4 words
seq_len = 4
attention_scores = np.random.rand(seq_len, seq_len)
mask = create_attention_mask(seq_len)
masked_scores = apply_mask_to_scores(attention_scores, mask)

print("Original Scores:\\n", np.round(attention_scores, 2))
print("\\nMask:\\n", mask)
print("\\nScores after Masking (before softmax):\\n", np.round(masked_scores, 2))
# Notice the top-right triangle is now filled with large negative numbers.
        """, language='python')
