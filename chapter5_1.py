# chapter5_0.py
import streamlit as st
import matplotlib.pyplot as plt

def render_5_0():
    """Renders the introduction to text classification."""
    st.subheader("5.0: What is Text Classification?")
    st.markdown("""
    Now that we have a solid foundation in how models represent and understand language, let's apply these concepts to one of the most common and useful real-world NLP tasks: **Text Classification**.

    The goal is simple: **assign a predefined category or label to a piece of text.** This is a *supervised learning* problem, meaning we first need to train our model on a dataset of texts that have already been labeled by humans.
    """)

    st.subheader("Real-World Examples")
    st.markdown("""
    You interact with text classification systems every day:
    -   **Spam Detection:** Your email service reads an incoming email and labels it as **`Spam`** or **`Not Spam`**.
    -   **Sentiment Analysis:** An e-commerce site analyzes a product review and labels it as **`Positive`**, **`Negative`**, or **`Neutral`**.
    -   **Topic Labeling:** A news website reads an article and assigns it a topic like **`Sports`**, **`Politics`**, or **`Technology`**.
    -   **Language Detection:** A web browser detects the language of a page to offer translation.
    """)

    # --- Visualization ---
    st.subheader("Visualizing the Task")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')

    ax.text(0.5, 0.6, '"The movie was fantastic!"', ha='center', va='center', size=12, bbox=dict(boxstyle="round,pad=0.5", fc="lightblue"))
    ax.arrow(0.5, 0.5, 0, -0.2, head_width=0.05, head_length=0.05, fc='k', ec='k')
    ax.text(0.5, 0.2, "Classification Model", ha='center', va='center', size=12, bbox=dict(boxstyle="sawtooth,pad=0.5", fc="lightgray"))
    ax.arrow(0.5, 0.1, 0, -0.2, head_width=0.05, head_length=0.05, fc='k', ec='k')
    ax.text(0.5, -0.2, "Positive", ha='center', va='center', size=14, color="green", bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen"))

    st.pyplot(fig)
    st.markdown("In this chapter, we'll explore how the 'black box' of the classification model has evolved.")

# -------------------------------------------------------------------

# chapter5_1.py
import streamlit as st
from collections import Counter
import re
import pandas as pd
import numpy as np

def render_5_1():
    """Renders the Bag-of-Words and Naive Bayes section."""
    st.subheader("5.1: Method 1 - Bag-of-Words & Naive Bayes")
    st.markdown("""
    The earliest approach to text classification was purely statistical. It completely ignores grammar and word order, treating a sentence as just a **"bag" of its words**. The core idea is that the *frequency* of certain words can reliably predict the category.
    """)

    st.subheader("🧠 The Method: Naive Bayes")
    st.markdown("""
    The **Naive Bayes** classifier is a simple probabilistic model based on Bayes' Theorem. In simple terms, it calculates the probability of a text belonging to a certain category based on the words it contains.

    For spam detection, it learns to answer the question:
    > "What is the probability this email is **Spam**, given that it contains the words 'free', 'win', and 'prize'?"

    It's called "naive" because it makes a strong, simplifying assumption: it assumes that every word in the sentence is independent of the others. This isn't true (the word 'York' is much more likely to appear after 'New'), but the assumption makes the math much simpler and works surprisingly well in practice.
    """)

    st.subheader("🛠️ Interactive Demo: Naive Bayes for Spam Detection")
    st.markdown("Enter an email text below. Our simple model has learned from a tiny dataset that 'free', 'win', 'prize', and 'money' are common in spam, while 'report', 'meeting', and 'project' are common in normal emails (ham).")
    
    email_text = st.text_input("Enter email text:", "win free money in our prize giveaway")
    tokens = re.findall(r'\b\w+\b', email_text.lower())
    
    if st.button("Classify Email"):
        # Simple Naive Bayes simulation
        spam_keywords = {'free', 'win', 'prize', 'money', 'giveaway'}
        ham_keywords = {'report', 'meeting', 'project', 'presentation'}
        
        spam_score = 1.0 # Start with a base probability
        ham_score = 1.0
        
        for word in tokens:
            if word in spam_keywords:
                spam_score *= 2.0 # Double the score for each spam word
            if word in ham_keywords:
                ham_score *= 2.0 # Double the score for each ham word

        st.write(f"Spam Score (likelihood): `{spam_score:.2f}`")
        st.write(f"Ham Score (likelihood): `{ham_score:.2f}`")

        if spam_score > ham_score:
            st.error("Classification: **Spam**")
        else:
            st.success("Classification: **Not Spam (Ham)**")

    st.error("**Limitation:** This method has no concept of meaning. It doesn't know that 'cash' is similar to 'money', or that 'prize' is similar to 'award'.")

# -------------------------------------------------------------------

# chapter5_2.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_5_2():
    """Renders the Averaging Vectors section."""
    st.subheader("5.2: Method 2 - Averaging Word Embeddings")
    st.markdown("""
    The Bag-of-Words approach fails to capture the meaning of words. The next logical step was to use the **word embeddings** we learned about in Chapter 2. Instead of a sparse vector of word counts, we can create a single, dense vector that represents the overall meaning of the sentence.
    """)

    st.subheader("🧠 The Method: Averaging Word Vectors")
    st.markdown("""
    The simplest way to do this is to:
    1.  Get the pre-trained word embedding (e.g., from Word2Vec or GloVe) for every word in the sentence.
    2.  Average these vectors together element-wise.

    The resulting averaged vector is a single, dense representation of the sentence's meaning. This 'sentence embedding' can then be fed into a simple classifier (like Logistic Regression). This is a huge improvement because it understands that sentences like "The movie was great" and "The film was fantastic" should have very similar vectors.
    """)

    st.subheader("Visualizing the Sentence Vectors")
    st.markdown("Imagine a 2D space where we plot the final sentence vectors. Texts with similar meanings will cluster together, allowing a simple line to separate the categories.")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Positive examples
    ax.scatter([0.8, 0.9, 0.85], [0.8, 0.7, 0.9], s=100, color='green', label='Positive Reviews')
    # Negative examples
    ax.scatter([0.2, 0.1, 0.15], [0.2, 0.3, 0.1], s=100, color='red', label='Negative Reviews')
    
    # Decision boundary
    ax.plot([0, 1], [1, 0], 'k--', label='Decision Boundary')
    
    ax.set_title("Sentence Embeddings in a 2D Space")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend()
    st.pyplot(fig)
    
    st.error("**Limitation:** Averaging all the word vectors together loses all information about word order and grammar. The sentences 'The cat chased the dog' and 'The dog chased the cat' would have the exact same sentence embedding.")

# -------------------------------------------------------------------

# chapter5_3.py
import streamlit as st
import time

def render_5_3():
    """Renders the RNN/LSTM for Classification section."""
    st.subheader("5.3: Method 3 - The Power of Context (RNN/LSTM)")
    st.markdown("""
    Averaging embeddings ignores word order, which is crucial for understanding negation and complex grammar. The next step is to use a **sequential model** (from Chapter 3) to process the text.
    """)

    st.subheader("🧠 The Method: Using the Final 'Memory'")
    st.markdown("""
    An RNN or LSTM reads the sentence one word at a time, updating its 'memory' (the hidden state) at each step. By the time it reaches the end of the sentence, the final hidden state is a rich vector that represents the meaning of the *entire ordered sequence*.

    This single, context-aware vector is then fed into a classification layer. This approach is much more powerful because it can understand patterns like:
    - "The movie was **not good**." (The presence of "not" before "good" completely changes the meaning).
    - "The food was great, but the service was terrible." (The model can learn that words appearing after "but" might be more important for the overall sentiment).
    """)

    st.subheader("Visualizing the LSTM Classifier")
    st.markdown("Click the button to see an animation of an LSTM processing a sentence for classification.")
    
    if st.button("Animate LSTM Classification"):
        sentence = ["the", "movie", "was", "not", "good"]
        placeholder = st.empty()
        
        for i in range(1, len(sentence) + 1):
            with placeholder.container():
                context_so_far = " ".join(sentence[:i])
                current_word = sentence[i-1]
                
                st.markdown(f"**Step {i}:** Reading word `'{current_word}'`")
                st.info(f"**Context Seen So Far:** `{context_so_far}`")

                if "not" in context_so_far:
                    st.warning("The model's memory now contains the concept of negation...")
                
                if i == len(sentence):
                    st.success("End of sentence reached. The final 'memory' vector, which understands the full context including 'not', is now passed to the classifier.")
            time.sleep(1)

# -------------------------------------------------------------------

# chapter5_4.py
import streamlit as st

def render_5_4():
    """Renders the Fine-tuning Transformers section."""
    st.subheader("5.4: Method 4 - The Modern Approach (Fine-tuning)")
    st.markdown("""
    The modern, state-of-the-art approach uses the full power of the **Transformer architecture** (from Chapter 4) to solve both the meaning and word order problems simultaneously. Instead of training a classifier from scratch, we use a technique called **fine-tuning**.
    """)

    st.subheader("🧠 The Method: The [CLS] Token")
    st.markdown("""
    1.  **Start with a Pre-trained Model:** We take a massive Transformer model like **BERT** that has already been trained on a huge amount of general text.
    2.  **Add Special Tokens:** BERT adds a special `[CLS]` (for "classification") token to the beginning of every input sentence.
    3.  **Process the Text:** The entire sentence, including the `[CLS]` token, is passed through all the layers of the Transformer.
    4.  **Use the [CLS] Embedding:** The key insight is that the final output embedding corresponding to this `[CLS]` token is designed to be a rich, aggregated representation of the entire sentence's meaning.
    5.  **Classify:** We take this single `[CLS]` vector and feed it into a simple, untrained classification layer. We then fine-tune the whole system on our specific task.

    This is the most effective method because the `[CLS]` token's final embedding is a result of the deep, multi-headed attention mechanism considering all words and their relationships in parallel.
    """)
    st.image("https://miro.medium.com/max/1400/1*i2ss3_U4p2a2n5q2d4cW2g.png",
             caption="BERT uses the final hidden state of the [CLS] token as the aggregate sequence representation for classification tasks.")

# -------------------------------------------------------------------

# chapter5_5.py
import streamlit as st
import re

def render_5_5():
    """Renders the interactive classification workbench."""
    st.subheader("5.5: Interactive Classification Workbench")
    st.markdown("Let's compare how our different methods might classify a movie review. Enter a review below to see a simulated result from each model.")

    review_text = st.text_area("Enter a movie review:", "The acting was great and the story was fantastic, but the ending was terrible.")
    
    if st.button("Classify Review"):
        tokens = re.findall(r'\b\w+\b', review_text.lower())
        
        st.markdown("---")
        # --- Method 1: Bag-of-Words ---
        st.subheader("1. Statistical Model (Bag-of-Words)")
        positive_words = {'great', 'fantastic', 'good', 'amazing', 'love'}
        negative_words = {'terrible', 'bad', 'awful', 'boring', 'hate'}
        pos_score = sum(1 for word in tokens if word in positive_words)
        neg_score = sum(1 for word in tokens if word in negative_words)
        
        st.write(f"Positive word count: `{pos_score}`")
        st.write(f"Negative word count: `{neg_score}`")
        if pos_score > neg_score:
            st.success("Classification: **Positive**")
        else:
            st.error("Classification: **Negative**")
        st.caption("This model simply counts keywords. It is easily confused by the word 'but'.")

        st.markdown("---")
        # --- Method 2: Averaging Embeddings ---
        st.subheader("2. Embedding Model (Averaging)")
        st.write(f"Positive word score: `{pos_score}`")
        st.write(f"Negative word score: `{neg_score}`")
        if pos_score > neg_score:
            st.success("Classification: **Positive**")
        else:
            st.error("Classification: **Negative**")
        st.caption("This model understands word meanings are similar, but by averaging, it also gets confused by the word 'but' and just sees more positive words overall.")

        st.markdown("---")
        # --- Method 3: Transformer Model ---
        st.subheader("3. Transformer Model (Fine-tuned)")
        # A more nuanced simulation
        final_score = pos_score - (neg_score * 1.5) # Give more weight to negative words after 'but'
        st.write(f"Simulated Transformer Output Score: `{final_score:.2f}`")
        if final_score > 0:
             st.success("Classification: **Positive**")
        elif final_score < 0:
             st.error("Classification: **Negative**")
        else:
             st.warning("Classification: **Neutral**")
        st.caption("A real Transformer understands the whole sentence structure. It recognizes that 'but' often negates what came before, giving more weight to 'terrible' and likely classifying the review as **Negative**.")
