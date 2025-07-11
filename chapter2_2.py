import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_2_2():
    """Renders the Vector Space & Analogies section."""
    st.subheader("2.2: Exploring the Vector Space & Analogies")
    st.markdown("""
    The true power of word embeddings isn't just that similar words are close together, but that the *relationships* and *directions* between words are also captured mathematically. This allows us to solve analogies using simple vector arithmetic.
    """)

    st.subheader("🧠 The Theory: Vector Arithmetic for Meaning")
    st.markdown("""
    The most famous example of this is the "king" and "queen" analogy. The relationship between "man" and "woman" (a gender relationship) is encoded in the vector that connects them. This vector is surprisingly similar to the one that connects "king" and "queen".

    This leads to the classic equation:
    $$ \text{vector}(\text{'king'}) - \text{vector}(\text{'man'}) + \text{vector}(\text{'woman'}) \approx \text{vector}(\text{'queen'}) $$

    In essence, we are saying: "Start at 'king', take away the concept of 'man', and add the concept of 'woman'. Where do you end up?" The answer, in the vector space, is very close to the location for 'queen'.
    """)

    st.subheader("🛠️ Interactive Demo: Solving Analogies")
    st.markdown("Perform your own vector arithmetic to solve an analogy. See if the resulting vector lands close to the word you expect.")
    
    # --- Visualization Demo ---
    vocab = {
        'king': np.array([9.5, 9]), 'queen': np.array([9.5, 1]),
        'man': np.array([8.5, 9]), 'woman': np.array([8.5, 1]),
        'prince': np.array([7.5, 8.5]), 'princess': np.array([7.5, 1.5]),
        'strong': np.array([9, 6]), 'stronger': np.array([9.5, 6]),
        'fast': np.array([8, 6]), 'faster': np.array([8.5, 6]),
    }
    words = list(vocab.keys())

    def find_closest_word(target_vec, vocab_dict):
        closest_word = None
        min_dist = float('inf')
        for word, vec in vocab_dict.items():
            dist = np.linalg.norm(vec - target_vec)
            if dist < min_dist:
                min_dist = dist
                closest_word = word
        return closest_word

    c1, c2, c3 = st.columns(3)
    word1 = c1.selectbox("Start with:", options=words, index=words.index('king'))
    word2 = c2.selectbox("Subtract:", options=words, index=words.index('man'))
    word3 = c3.selectbox("Add:", options=words, index=words.index('woman'))
    
    # Perform calculation
    vec1, vec2, vec3 = vocab[word1], vocab[word2], vocab[word3]
    result_vec = vec1 - vec2 + vec3
    result_word = find_closest_word(result_vec, vocab)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    for word, vec in vocab.items():
        ax.scatter(vec[0], vec[1], s=50, color='gray')
        ax.text(vec[0] + 0.1, vec[1] + 0.1, word, fontsize=10, color='gray')
    
    # Highlight the analogy words
    ax.scatter(vec1[0], vec1[1], s=150, color='blue', label=word1)
    ax.scatter(vec2[0], vec2[1], s=150, color='red', label=word2)
    ax.scatter(vec3[0], vec3[1], s=150, color='green', label=word3)
    
    # Plot the result
    ax.scatter(result_vec[0], result_vec[1], s=200, marker='*', color='gold', label=f"Result (Closest: {result_word})")
    
    ax.set_title("Analogy in Vector Space")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.success(f"Result of `{word1} - {word2} + {word3}` is closest to **{result_word}**.")

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Comparative Analogy:** Try solving for `stronger` using the analogy `strong - fast + faster`. Does it work?
    2.  **Create Your Own:** What words would you need to solve the analogy "Paris is to France as Tokyo is to ___"?
    """)

    st.subheader("🐍 The Python Behind the Analogy")
    with st.expander("Show the Python Code for Solving Analogies"):
        st.code("""
import numpy as np

def find_closest_word(target_vec, vocab_dict):
    closest_word = None
    min_dist = float('inf')
    for word, vec in vocab_dict.items():
        # Calculate Euclidean distance
        dist = np.linalg.norm(vec - target_vec)
        if dist < min_dist:
            min_dist = dist
            closest_word = word
    return closest_word

# --- Example ---
# vocab = {'king': np.array(...), ...}
vec1 = vocab['king']
vec2 = vocab['man']
vec3 = vocab['woman']

# Perform the vector arithmetic
result_vec = vec1 - vec2 + vec3

# Find the word in our vocabulary closest to the result vector
predicted_word = find_closest_word(result_vec, vocab)
# predicted_word will be 'queen'
        """, language='python')
