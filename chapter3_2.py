import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt

def render_3_2():
    """Renders the Rolling Context section."""
    st.subheader("3.2: The Idea of a 'Rolling Context'")
    st.markdown("""
    To solve the polysemy problem, a model needs to understand the context surrounding a word. The first major attempt at this used an idea that mimics how humans read: **one word at a time**.

    This is the core concept behind **Recurrent Neural Networks (RNNs)**. Instead of looking at a fixed window of words, an RNN processes a sentence sequentially, maintaining a 'memory' of what it has seen so far.
    """)

    st.subheader("🧠 The Analogy: Reading with Memory")
    st.markdown("""
    Imagine reading a sentence. Your understanding of each new word is colored by the words you've just read.
    - If you read "The river...", your brain is already primed for words related to geography or nature.
    - If you read "The financial...", your brain is primed for words about money or business.

    An RNN tries to simulate this. It has a **hidden state** (its memory) that gets updated after reading each word. The information from the current word is combined with the memory from the previous word to form a new memory. This updated memory is then used to help interpret the next word. This creates a **rolling context** that builds up as the model reads through the sentence.
    """)

    st.subheader("Visualizing the Rolling Context")
    st.markdown("Enter a sentence below to see a simplified animation of how a model's 'memory' vector might change as it reads each word.")

    sentence_input = st.text_input("Enter a sentence to visualize:", "the river bank was very steep")
    
    if st.button("Animate Rolling Context"):
        tokens = sentence_input.lower().split()
        if not tokens:
            st.warning("Please enter a sentence.")
        else:
            placeholder = st.empty()
            # Start with a neutral memory vector
            memory_vector = np.array([0.5, 0.5]) 
            
            for i, word in enumerate(tokens):
                with placeholder.container():
                    # Simulate a change in the memory vector based on the word
                    if word in ['river', 'water', 'steep', 'flowed']:
                        memory_vector = memory_vector * 0.7 + np.array([0, 0.3]) # Nudge towards "geography"
                    elif word in ['money', 'loan', 'financial', 'cash']:
                        memory_vector = memory_vector * 0.7 + np.array([0.3, 0]) # Nudge towards "finance"
                    else:
                        memory_vector = memory_vector * 0.9 # Slightly decay memory for neutral words
                    
                    memory_vector = np.clip(memory_vector, 0, 1) # Keep values between 0 and 1

                    st.markdown(f"**Step {i+1}:** Reading word `'{word}'`")
                    
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.barh(['Memory State'], [memory_vector[0]], color='red', label='Finance')
                    ax.barh(['Memory State'], [memory_vector[1]], left=[memory_vector[0]], color='blue', label='Geography')
                    ax.set_xlim(0, 1)
                    ax.set_yticks([])
                    ax.set_title("Model's 'Memory' State")
                    ax.legend(loc='lower right')
                    st.pyplot(fig)
                    plt.close(fig) # Prevent figure from being displayed twice

                    if word == "bank":
                        if memory_vector[1] > memory_vector[0] + 0.2: # If geography is much higher
                            st.success("The model's memory is now strongly influenced by 'river'. It interprets 'bank' as a geographical term.")
                        elif memory_vector[0] > memory_vector[1] + 0.2: # If finance is much higher
                             st.success("The model's memory is now strongly influenced by 'financial'. It interprets 'bank' as a financial term.")
                        else:
                            st.warning("The model has seen 'bank', but the context is still ambiguous.")

                time.sleep(1)

    st.subheader("The Problem of Long-Term Memory")
    st.markdown("""
    Simple RNNs have a major weakness: they have trouble remembering things for a long time. As new information comes in at each step, the memory of earlier words tends to fade. This is known as the **vanishing gradient problem**.

    For a sentence like, "The man who grew up in a small town in France and loves to bake croissants... speaks fluent French," the model might forget that the subject was "man" by the time it needs to predict the final word. To solve this, more advanced versions like **LSTMs** and **GRUs** were created with special "gates" to learn what to remember and what to forget, but the core idea of a rolling, sequential context remains the same.
    """)

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Ambiguity Point:** In the sentence "I saw a bat...", at what point is the meaning ambiguous? What single word could you add to the end to resolve the ambiguity one way or the other?
    2.  **Order Matters:** How would an RNN's final 'memory' state differ between the sentences "The food was great but the service was terrible" and "The service was terrible but the food was great"?
    """)

    st.subheader("🐍 The Python Behind the Idea")
    with st.expander("Show the Python Pseudo-code for a Rolling Context"):
        st.code("""
# This is pseudo-code to illustrate the concept of an RNN's loop.

def process_sentence_with_rnn(sentence_tokens):
    # Start with an initial 'memory' of all zeros.
    memory_state = np.zeros(100) # Example memory size of 100

    for token in sentence_tokens:
        # For each token, get its static embedding (from Chapter 2)
        token_embedding = get_static_embedding(token)
        
        # The core of the RNN:
        # The new memory is a function of the OLD memory and the CURRENT input.
        # The 'update_memory' function is what the network learns.
        memory_state = update_memory(memory_state, token_embedding)
        
    # After the loop, the final memory_state represents the "meaning"
    # of the entire sentence.
    return memory_state

# --- Example ---
sentence1 = ["the", "river", "bank"]
sentence2 = ["the", "financial", "bank"]

# The final memory state will be very different for these two sentences,
# because the memory was updated with different words along the way.
final_memory1 = process_sentence_with_rnn(sentence1)
final_memory2 = process_sentence_with_rnn(sentence2)
        """, language='python')
