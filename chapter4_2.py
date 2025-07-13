import streamlit as st
import numpy as np

def render_4_2():
    """Renders the Query, Key, and Value section."""
    st.subheader("4.2: Query, Key, and Value - A Smarter Dictionary")

    st.subheader("The Problem: An Information Bottleneck")
    st.markdown("""
    In the last chapter, we saw how sequential models like RNNs and LSTMs create a 'rolling context' by passing a memory state from one word to the next. This was a huge improvement, but it created a new problem: an **information bottleneck**.

    Imagine trying to summarize an entire novel into a single, short paragraph. You would inevitably lose crucial details and nuance. A sequential model faces the same issue. It must compress the meaning of an entire sentence, no matter how long, into a single, fixed-size vector (the final hidden state).

    For a long sentence like:
    > "The cats, which were sitting by the window in the warm afternoon sun, were tired."

    When the model gets to the word "were", the most important piece of context is the word "cats". But "cats" is many steps away in the sequence. The model's memory of "cats" might have been diluted by all the words in between ("window", "sun", etc.). This makes it difficult to learn long-range dependencies.
    
    **The core motivation for Attention was to break this bottleneck.** Instead of forcing all information through a single, sequential memory state, what if we could create direct shortcuts between words, no matter how far apart they are?
    """)

    st.markdown("---")
    st.markdown("""
    Attention solves this by treating the process like a sophisticated information retrieval system, similar to a database or a library. For every word, the model generates three specialized vectors: a **Query**, a **Key**, and a **Value**.
    """)

    st.subheader("🧠 The Library Analogy")
    st.markdown("""
    Imagine you're in a library and you want to find information about "cats".
    -   **Your Query:** You don't just shout "cats!". You formulate a specific question, like "I want to know about the behavior of domestic felines". This is your **Query vector**. It represents your specific information need.
    -   **The Book's Keywords (Keys):** Every book on the shelf has a set of keywords on its spine, like "feline", "animal", "pet", "lion", etc. These are the **Key vectors**. They advertise the content of the book.
    -   **The Book's Content (Value):** The actual information inside the book is the **Value vector**. It's the rich, detailed content you actually want to read.

    The process is simple: you compare your **Query** ("behavior of domestic felines") to every book's **Keywords (Keys)**. The books with the most relevant keywords get the highest scores. You then "read" a blend of the **content (Values)** from the highest-scoring books.
    """)

    st.subheader("🛠️ Interactive Demo: The Q-K Matchup")
    st.markdown("Let's simulate this. We have a small database of 'keys' representing different concepts. Enter a 'query' to see which keys it matches best.")

    # --- Interactive Demo ---
    keys_database = {
        "royalty": np.array([0.9, 0.1]),
        "animal": np.array([0.1, 0.9]),
        "fruit": np.array([0.1, 0.1]),
        "technology": np.array([0.5, 0.5]),
    }

    query_input = st.text_input("Enter a query word (e.g., 'king', 'dog', 'apple', 'computer'):", "king")

    if st.button("Find Best Match"):
        query_word = query_input.lower()
        
        # Simulate a query vector based on input
        query_vec = np.array([0.0, 0.0])
        if query_word in ['king', 'queen', 'prince']:
            query_vec = np.array([0.95, 0.05])
        elif query_word in ['dog', 'cat', 'bird']:
            query_vec = np.array([0.05, 0.95])
        elif query_word in ['apple', 'orange', 'banana']:
            query_vec = np.array([0.15, 0.15])
        elif query_word in ['computer', 'phone', 'software']:
            query_vec = np.array([0.55, 0.55])
        else:
            st.warning("Query not recognized, using a neutral vector.")

        # Calculate dot product scores
        scores = {name: np.dot(query_vec, key_vec) for name, key_vec in keys_database.items()}
        
        best_match = max(scores, key=scores.get)

        st.write(f"Your Query Vector (simulated): `{np.round(query_vec, 2)}`")
        st.write("Scores (Query • Key):")
        st.json(scores)
        st.success(f"The best match for your query is the **'{best_match}'** key!")

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Why Three Vectors?** Why doesn't the model just compare a word's embedding with every other word's embedding? Why does it need to create separate Q, K, and V vectors?
    2.  **The 'Self' in Self-Attention:** In our demo, the query was external. In a real self-attention model, *every* word generates its own Q, K, and V. What does it mean for the word "cat" to have its own query that it compares with its own key?
    """)

    st.subheader("💡 Answers to Exercises")
    with st.expander("Click to see the answers"):
        st.markdown("""
        **1. Why Three Vectors?**
        
        The reason for creating three separate vectors (Query, Key, and Value) from the original word embedding is **specialization**. It's like having three different specialized tools instead of one general-purpose tool.
        -   The **Query** vector is trained to become good at *asking* for relevant information.
        -   The **Key** vector is trained to become good at *advertising* its own information.
        -   The **Value** vector is trained to provide the rich, useful information once a match is made.
        
        This separation allows for much more complex and flexible relationships. A word can learn to ask for one type of information (e.g., "I am a verb, I need a subject") while simultaneously advertising itself in a different way (e.g., "I am an action word related to movement"). If it only had one vector, it would have to use that same vector for both asking and advertising, which is far less powerful.

        ---
        
        **2. The 'Self' in Self-Attention:**
        
        When a word's Query is compared with its own Key, it's essentially asking: **"How important is my own original meaning to my final contextual meaning?"**
        
        This "self-attention" score is crucial.
        -   If a word has a **high self-attention score**, it means the model has decided that the word's original, inherent meaning is very important and should be preserved. For a key noun like "cat" or "dragon", this is often the case.
        -   If a word has a **low self-attention score**, it means the model has decided that the word's meaning should be heavily influenced and redefined by the other words in the sentence. A vague pronoun like "it" or a stopword like "the" might have low self-attention, as their meaning is almost entirely derived from context.
        
        This allows the model to decide, on a word-by-word basis, whether to stick to its original meaning or to let itself be redefined by its neighbors.
        """)
