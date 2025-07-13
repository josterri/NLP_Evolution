import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def render_3_3():
    """Renders the ELMo / Words in Disguise section."""
    st.subheader("3.3: The Breakthrough - Seeing the Whole Picture")
    st.markdown("""
    The "rolling context" of a simple RNN was a step forward, but it had a critical flaw: it was like reading a sentence with one eye closed. It only knew about the words that came *before* the target word.

    To truly understand a word like "bank" in the sentence "I went to the river bank to fish", the model needs to see the word "fish" which comes *after*. This need for a complete view led to the first major breakthrough in contextual embeddings: a model called **ELMo (Embeddings from Language Models)**.
    """)

    st.subheader("🧠 The Core Idea: Look Both Ways")
    st.markdown("""
    ELMo's innovation was simple in concept but profound in impact. It used a powerful sequential model (a deep LSTM) and had it read the sentence **twice**:
    1.  **A Forward Pass:** It reads the sentence from left-to-right, creating a "forward memory" at each step.
    2.  **A Backward Pass:** It reads the sentence from right-to-left, creating a "backward memory".

    The final, contextual embedding for any given word is a combination of three things:
    - Its original, static embedding (like from Word2Vec).
    - The "forward memory" at that word's position.
    - The "backward memory" at that word's position.

    By combining information from both directions, the model can finally understand that "bank" is related to geography because it has "seen" the word "river" before it and "fish" after it.
    """)

    st.subheader("Visualizing the Bidirectional Context")
    st.markdown("Let's visualize how the two passes combine to create a final, context-aware understanding.")
    
    # --- Visualization Demo ---
    sentence = ["he", "went", "to", "the", "bank", "to", "deposit", "money"]
    st.info(f"**Example Sentence:** `{' '.join(sentence)}`")

    # Simulate memory states
    forward_memory = "Memory of: 'he went to the'"
    backward_memory = "Memory of: 'money deposit to'"
    
    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### Forward Pass →")
        st.text_area("Left-to-Right Memory", forward_memory, height=100)
    
    with cols[1]:
        st.markdown("#### Target Word")
        st.success("# bank")

    with cols[2]:
        st.markdown("#### ← Backward Pass")
        st.text_area("Right-to-Left Memory", backward_memory, height=100)
    
    st.markdown("<h4 style='text-align: center;'>↓</h4>", unsafe_allow_html=True)
    st.success("""
    **Combined Contextual Embedding for "bank"**
    The final vector for "bank" is a rich combination of its static meaning, the forward context ("he went to the"), and the backward context ("money deposit to"). This makes it unambiguously financial.
    """)

    st.subheader("🛠️ Interactive Demo: The Influence of Context")
    st.markdown("Let's explore how specific context words 'pull' the meaning of an ambiguous word in one direction or another. Enter a sentence and select the ambiguous word to analyze.")

    sentence_input = st.text_input("Enter a sentence with an ambiguous word:", "the bat flew out of the cave at night")
    tokens = sentence_input.lower().split()

    if tokens:
        ambiguous_word = st.selectbox("Select the ambiguous word to analyze:", options=set(tokens))

        if st.button("Analyze Context"):
            # Define keywords for our two meanings
            sports_keywords = {'swing', 'hit', 'game', 'ball', 'player', 'team'}
            animal_keywords = {'fly', 'flew', 'animal', 'cave', 'night', 'mammal'}

            # Find context words
            context_words = [word for word in tokens if word != ambiguous_word]
            
            # Calculate influence scores
            sports_score = sum(1 for word in context_words if word in sports_keywords)
            animal_score = sum(1 for word in context_words if word in animal_keywords)

            st.markdown("---")
            st.write(f"Analyzing the context for **'{ambiguous_word}'**...")

            cols = st.columns(2)
            with cols[0]:
                st.write("Context Words Found:")
                st.json(context_words)
            with cols[1]:
                st.write("Influence Scores:")
                st.metric(label="Sports Context Score", value=sports_score)
                st.metric(label="Animal Context Score", value=animal_score)

            # Visualize the result
            fig, ax = plt.subplots()
            categories = ['Sports Meaning', 'Animal Meaning']
            scores = [sports_score, animal_score]
            ax.bar(categories, scores, color=['red', 'blue'])
            ax.set_ylabel("Contextual Influence Score")
            ax.set_title(f"Contextual 'Pull' on the word '{ambiguous_word}'")
            st.pyplot(fig)

            if sports_score > animal_score:
                st.success("The context words strongly suggest a **sports** meaning.")
            elif animal_score > sports_score:
                st.success("The context words strongly suggest an **animal** meaning.")
            else:
                st.warning("The context is ambiguous or contains no strong keywords.")


    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Bidirectional Need:** Why would a purely forward-looking RNN fail to correctly interpret the word "apple" in the sentence "I just bought the new apple laptop"?
    2.  **Create a Pair:** Write two sentences where the word "right" has completely different meanings (e.g., direction vs. correctness). What are the key context words (both before and after) that help define the meaning in each case?
    """)

    st.subheader("🐍 The Python Behind the Idea")
    with st.expander("Show the Python Pseudo-code for a Bidirectional Model"):
        st.code("""
# This is pseudo-code to illustrate the concept.
# Real libraries like AllenNLP (for ELMo) or Hugging Face Transformers are used in practice.

def get_contextual_embedding(sentence_tokens, word_index):
    # 1. Process the sentence from left-to-right
    forward_memory_states = forward_rnn.process(sentence_tokens)
    forward_vec = forward_memory_states[word_index]

    # 2. Process the reversed sentence from left-to-right (which is right-to-left)
    reversed_tokens = reversed(sentence_tokens)
    backward_memory_states = backward_rnn.process(reversed_tokens)
    # Get the corresponding backward state (index needs to be adjusted)
    backward_vec = backward_memory_states[len(sentence_tokens) - 1 - word_index]

    # 3. Get the word's static embedding
    static_vec = get_static_embedding(sentence_tokens[word_index])

    # 4. Combine them to create the final, rich embedding
    final_contextual_vector = combine(static_vec, forward_vec, backward_vec)
    
    return final_contextual_vector

# --- Example ---
sentence = ["a", "bat", "flew", "from", "the", "cave"]
# The vector for "bat" will be influenced by the forward memory of "a"
# AND the backward memory of "cave the from flew".
bat_vector = get_contextual_embedding(sentence, 1)
        """, language='python')
