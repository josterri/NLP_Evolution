import streamlit as st
import numpy as np

def render_2_7():
    """Renders the technical details of Word2Vec training."""
    st.subheader("2.7: Under the Hood - Training Word2Vec")
    st.markdown("""
    We've seen *what* Word2Vec does, but *how* does it actually learn the vectors? The answer lies in a simple, shallow neural network. Word2Vec isn't a deep learning model in the modern sense; its power comes from a clever training objective and a crucial optimization.

    Let's focus on the **Skip-Gram** architecture for this explanation.
    """)

    st.subheader("🧠 The Neural Network Architecture")
    st.markdown("""
    The network is surprisingly simple and consists of three layers:
    1.  **Input Layer:** A one-hot encoded vector representing the center word from a training pair (e.g., "fox"). If our vocabulary has 50,000 words, this is a 50,000-dimensional vector.
    2.  **Hidden Layer (The "Embedding Layer"):** This layer has no activation function (it's a linear layer). Its size is the dimensionality of our desired embeddings (e.g., 100). The weight matrix between the input and hidden layer, with dimensions `[vocab_size x embedding_size]`, is the **embedding matrix**. This is what we are actually trying to learn.
    3.  **Output Layer:** A layer with the same size as the vocabulary (`vocab_size`), followed by a **softmax** function. This layer produces a probability distribution over the entire vocabulary, predicting the likelihood of each word being the correct context word.
    """)
#    st.image("https://i.imgur.com/13y4x1c.png", caption="Simplified architecture for one Skip-Gram prediction.")

    st.subheader("The Objective: Maximizing Probability")
    st.markdown("""
    The goal is to adjust the weights of the embedding matrix so that, given an input center word, the probability of the *actual* context words appearing at the output layer is maximized.

    Mathematically, we want to maximize the log probability of the correct predictions, which is equivalent to minimizing the **Negative Log-Likelihood loss**.
    """)

    st.subheader("The Problem with Softmax")
    st.markdown("""
    This architecture has a massive computational bottleneck. For every single training pair, calculating the softmax function requires performing a calculation involving the score of *every single word* in the vocabulary. For a 50,000-word vocabulary, this is incredibly slow and makes training on large corpora impractical.
    """)

    st.subheader("The Solution: Negative Sampling")
    st.markdown("""
    To solve this, the Word2Vec authors introduced a highly efficient optimization called **Negative Sampling**. Instead of updating weights for the entire vocabulary, the model does something much smarter:
    1.  It takes the true context word (the **positive sample**) and a small number (`k`, usually 5-20) of randomly chosen words from the vocabulary (the **negative samples**).
    2.  It then trains a set of simple logistic regression models. The goal is to train the network to output a high probability for the positive sample and low probabilities for all the negative samples.

    This brilliant trick converts a complex, computationally expensive multinomial classification problem into a series of much faster binary classification problems. The model is no longer answering "Which word is it?", but rather a series of "Is this specific word the context word? (Yes/No)" questions. This is what made Word2Vec feasible to train on huge datasets.
    """)

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **The Embedding Matrix:** In the described architecture, where do the final word embeddings actually come from after the model is trained?
    2.  **Negative Sampling's Role:** Why is it more efficient to predict "Is this the right word?" `k+1` times than it is to predict "Which of the 50,000 words is it?" once?
    """)
