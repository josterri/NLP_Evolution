import streamlit as st
import numpy as np
import pandas as pd

def render_4_4():
    """Renders the Attention Calculation section."""
    st.subheader("4.4: The Attention Calculation Step-by-Step")
    st.markdown("""
    Now that we have our specialized Query (Q), Key (K), and Value (V) vectors for each word, how do we use them to create the final contextual representation? The process follows a specific formula, which we can break down into four simple steps.

    The famous formula is:
    $$ \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

    Let's demystify this by focusing on how we calculate the new, context-aware embedding for just **one word**: the word **'cat'** in the sentence "the cat sat".
    """)

    # --- Setup for the Demo ---
    st.subheader("🛠️ Interactive Demo: Calculating the New 'cat' Vector")
    st.markdown("Our goal is to create a new vector for 'cat' that is a blend of the other words in the sentence, based on how relevant they are.")

    # Define simple, 2-dimensional Q, K, V vectors for our words
    q_vectors = {
        'the': np.array([1.0, 0.1]),
        'cat': np.array([0.2, 0.8]),
        'sat': np.array([0.9, 0.5])
    }
    k_vectors = {
        'the': np.array([0.9, 0.2]),
        'cat': np.array([0.3, 0.7]),
        'sat': np.array([0.8, 0.6])
    }
    v_vectors = {
        'the': np.array([0.1, 0.2]),
        'cat': np.array([0.3, 0.4]),
        'sat': np.array([0.5, 0.6])
    }
    words = ['the', 'cat', 'sat']
    d_k = 2 # The dimension of our key vectors

    st.info("We will use the **Query** from 'cat' and compare it to the **Key** of every other word.")

    # --- Step 1: Score ---
    st.markdown("---")
    st.markdown("#### Step 1: Score (Dot Product)")
    st.markdown("First, we calculate a 'compatibility' score between the Query of 'cat' and the Key of every word in the sentence (including itself).")

    q_cat = q_vectors['cat']
    k_the, k_cat, k_sat = k_vectors['the'], k_vectors['cat'], k_vectors['sat']
    
    score_the = np.dot(q_cat, k_the)
    score_cat = np.dot(q_cat, k_cat)
    score_sat = np.dot(q_cat, k_sat)

    st.write("`Query('cat')` • `Key('the')` = ", f"`{score_the:.2f}`")
    st.write("`Query('cat')` • `Key('cat')` = ", f"`{score_cat:.2f}`")
    st.write("`Query('cat')` • `Key('sat')` = ", f"`{score_sat:.2f}`")
    st.caption("A high score means the word is highly relevant to 'cat'.")

    # --- Step 2: Scale ---
    st.markdown("---")
    st.markdown("#### Step 2: Scale")
    st.markdown(f"To stabilize training, we divide the scores by the square root of the key dimension (`sqrt({d_k})` ≈ 1.41).")
    
    scaled_score_the = score_the / np.sqrt(d_k)
    scaled_score_cat = score_cat / np.sqrt(d_k)
    scaled_score_sat = score_sat / np.sqrt(d_k)

    st.write("Scaled score for 'the':", f"`{scaled_score_the:.2f}`")
    st.write("Scaled score for 'cat':", f"`{scaled_score_cat:.2f}`")
    st.write("Scaled score for 'sat':", f"`{scaled_score_sat:.2f}`")

    # --- Step 3: Softmax ---
    st.markdown("---")
    st.markdown("#### Step 3: Softmax")
    st.markdown("We apply a softmax function to the scaled scores. This turns them into probabilities that all add up to 1. These are our final **attention weights**.")

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    scores = np.array([scaled_score_the, scaled_score_cat, scaled_score_sat])
    attention_weights = softmax(scores)
    
    st.write("Attention weight for 'the':", f"`{attention_weights[0]:.2f}`")
    st.write("Attention weight for 'cat':", f"`{attention_weights[1]:.2f}`")
    st.write("Attention weight for 'sat':", f"`{attention_weights[2]:.2f}`")
    st.caption("This tells the model how much 'Value' to take from each word to build the new 'cat' vector.")

    # --- Step 4: Weighted Sum ---
    st.markdown("---")
    st.markdown("#### Step 4: Weighted Sum of Values")
    st.markdown("Finally, we create the new 'cat' vector by multiplying each word's **Value** vector by its attention weight and summing them up.")

    v_the, v_cat, v_sat = v_vectors['the'], v_vectors['cat'], v_vectors['sat']
    
    output_vector = (attention_weights[0] * v_the) + \
                    (attention_weights[1] * v_cat) + \
                    (attention_weights[2] * v_sat)

    st.write(f"`{attention_weights[0]:.2f}` * `Value('the')` + `{attention_weights[1]:.2f}` * `Value('cat')` + `{attention_weights[2]:.2f}` * `Value('sat')`")
    st.latex("=")
    st.write(f"**New context-aware vector for 'cat':** `{np.round(output_vector, 2)}`")

    st.success("This new vector for 'cat' is now context-aware! It contains a blend of information from the other words in the sentence, weighted by their relevance.")

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **The Output:** The final output for the word 'cat' is a new vector. What information does this vector contain?
    2.  **The Softmax Role:** What would happen if we skipped the softmax step and multiplied the scaled scores directly by the Value vectors?
    """)
