import streamlit as st
import matplotlib.pyplot as plt

def render_7_9():
    """Renders the 'Missing Pieces' section."""
    st.subheader("7.9: What's Missing? From nano-GPT to State-of-the-Art")
    st.markdown("""
    Congratulations on building a nano-GPT! You have now implemented the core components of a modern language model. However, to go from our simple model to a state-of-the-art LLM like GPT-3, Gemini, or Llama, there are several massive leaps in scale, architecture, and training methodology.

    Let's break down the biggest missing pieces.
    """)

    st.subheader("1. Scale, Scale, and More Scale")
    st.markdown("""
    This is the most significant difference.
    -   **Data:** Our model trained on a few sentences. A modern LLM trains on a significant portion of the public internet—trillions of words from web pages, books, and articles.
    -   **Parameters:** Our model has a few thousand parameters. A large model has billions or even trillions. This is like the difference between a brain with a few neurons and a human brain. More parameters allow the model to store more knowledge and learn more complex patterns.
    -   **Compute:** Training our model takes a few minutes on a laptop. Training a state-of-the-art model requires thousands of specialized GPUs running for months, costing millions of dollars.
    """)

    st.subheader("2. A Smarter Tokenizer")
    st.markdown("""
    We used a simple character-level tokenizer. Modern models use more advanced **subword tokenizers** like **Byte-Pair Encoding (BPE)** or **SentencePiece**.
    -   **The Idea:** These tokenizers break text down into common subword units. For example, the word "unhappily" might be tokenized into `['un', 'happi', 'ly']`.
    -   **The Advantage:** This is the best of both worlds. The vocabulary size remains manageable (e.g., 32,000 to 100,000 tokens), but the model can still represent any word by combining subword units. It's much more efficient than character-level tokenization.
    """)

    st.subheader("3. Architectural Improvements")
    st.markdown("""
    While the core Transformer block remains similar, modern LLMs incorporate many small but important architectural tweaks for better performance and stability, such as:
    -   **RMSNorm:** A different type of layer normalization that is simpler and more efficient.
    -   **SwiGLU:** A more advanced activation function that often performs better than the standard ReLU.
    -   **Rotary Positional Embeddings (RoPE):** A clever way to incorporate positional information that has better properties for handling long sequences.
    """)

    st.subheader("4. The Alignment Process (The Secret Sauce)")
    st.markdown("""
    This is perhaps the most crucial missing piece. A model trained only on next-word prediction is good at completing text, but it's not necessarily helpful or safe. It might generate harmful, biased, or nonsensical text.

    To make the model a useful assistant, it goes through a post-training **alignment** process. The most common method is **Reinforcement Learning from Human Feedback (RLHF)**.
    """)
    
    # --- Visualization of RLHF ---
    st.markdown("##### The RLHF Process:")
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis('off')
    steps = ["1. Supervised\nFine-Tuning", "2. Reward Model\nTraining", "3. RL\nOptimization"]
    descriptions = [
        "Fine-tune the base model on a small, high-quality dataset of human-written conversations.",
        "Humans rank different model responses to the same prompt. A 'reward model' is trained to predict these human preferences.",
        "The language model is further fine-tuned using reinforcement learning, where it gets 'points' from the reward model for generating helpful and harmless responses."
    ]
    
    for i, step in enumerate(steps):
        ax.text(i * 3.3 + 1.5, 0.8, step, ha='center', va='center', size=10, bbox=dict(boxstyle="round,pad=0.5", fc="lightblue"))
        ax.text(i * 3.3 + 1.5, 0.1, descriptions[i], ha='center', va='center', size=8, wrap=True)
        if i < len(steps) - 1:
            ax.arrow(i * 3.3 + 2.8, 0.8, 0.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.2)
    st.pyplot(fig)

    st.success("This alignment process is what teaches the model to follow instructions, refuse harmful requests, and behave like a helpful assistant, rather than just a text completion machine.")

