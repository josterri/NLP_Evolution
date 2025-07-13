# chapter5_0.py
import streamlit as st
import matplotlib.pyplot as plt

def render_5_0():
    """Renders the introduction to text classification."""
    st.subheader("5.0: What is Text Classification?")
    st.markdown("""
    Now that we have a solid foundation in how models represent and understand language, let's apply these concepts to one of the most common and useful real-world NLP tasks: **Text Classification**.

    The goal is simple: **assign a predefined category or label to a piece of text.**
    """)

    st.subheader("Real-World Examples")
    st.markdown("""
    You interact with text classification systems every day:
    -   **Spam Detection:** Your email service reads an incoming email and labels it as **`Spam`** or **`Not Spam`**.
    -   **Sentiment Analysis:** An e-commerce site analyzes a product review and labels it as **`Positive`**, **`Negative`**, or **`Neutral`**.
    -   **Topic Labeling:** A news website reads an article and assigns it a topic like **`Sports`**, **`Politics`**, or **`Technology`**.
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

def render_5_1():
    """Renders the Bag-of-Words section."""
    st.subheader("5.1: The Statistical Approach (Bag-of-Words)")
    st.markdown("""
    The earliest approach to text classification was purely statistical. It completely ignores grammar and word order, treating a sentence as just a **"bag" of its words**.

    The core idea is that the *frequency* of certain words can reliably predict the category. For example, an email with many occurrences of "free", "win", and "prize" is more likely to be spam.
    """)

    st.subheader("🧠 The Method: Counting Words")
    st.markdown("""
    1.  **Create a Vocabulary:** First, we build a vocabulary of all unique words from our entire training dataset.
    2.  **Create a Vector:** For each sentence, we create a vector that is the same length as the vocabulary. Each element in the vector corresponds to a word in the vocabulary.
    3.  **Count Frequencies:** We fill the vector by counting how many times each word from the vocabulary appears in our sentence.

    This vector, which represents the word counts, is then fed into a simple statistical model (like Naive Bayes) that learns to associate certain word counts with certain labels.
    """)

    st.subheader("🛠️ Interactive Demo: Create a Bag-of-Words Vector")
    st.markdown("Enter a sentence below to see it converted into a word count vector based on a fixed vocabulary.")
    
    vocab = sorted(['the', 'cat', 'dog', 'sat', 'on', 'mat', 'was', 'happy'])
    sentence = st.text_input("Enter a sentence:", "The cat sat on the mat, the cat was happy.")
    tokens = re.findall(r'\b\w+\b', sentence.lower())
    
    if tokens:
        word_counts = Counter(tokens)
        vector = [word_counts.get(word, 0) for word in vocab]
        
        df = pd.DataFrame([vector], columns=vocab, index=["Word Count Vector"])
        st.write("Vocabulary:", vocab)
        st.dataframe(df)
    
    st.error("**Limitation:** This method has no concept of meaning. It doesn't know that 'good' and 'great' are similar, or that 'not good' is the opposite of 'good'.")

    st.subheader("🐍 The Python Behind the Idea")
    with st.expander("Show the Python Code for Bag-of-Words"):
        st.code("""
from collections import Counter
import re

def create_bow_vector(sentence, vocabulary):
    # Tokenize and count words in the input sentence
    tokens = re.findall(r'\\b\\w+\\b', sentence.lower())
    word_counts = Counter(tokens)
    
    # Create the vector by looking up counts for each word in the vocabulary
    vector = [word_counts.get(word, 0) for word in vocabulary]
    
    return vector

# --- Example ---
vocab = ['the', 'quick', 'brown', 'fox', 'jumps']
sentence = "the fox is quick, the fox is brown"
bow_vector = create_bow_vector(sentence, vocab)
# Result: [2, 1, 1, 2, 0]
print(bow_vector)
        """, language='python')

# -------------------------------------------------------------------

# chapter5_2.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_5_2():
    """Renders the Averaging Vectors section."""
    st.subheader("5.2: The Embedding Approach (Averaging Vectors)")
    st.markdown("""
    The Bag-of-Words approach fails to capture the meaning of words. The next logical step was to use the **word embeddings** we learned about in Chapter 2.

    Instead of a sparse vector of word counts, we can create a single, dense vector that represents the overall meaning of the sentence.
    """)

    st.subheader("🧠 The Method: Averaging Word Vectors")
    st.markdown("""
    The simplest way to do this is to:
    1.  Get the pre-trained word embedding (e.g., from Word2Vec or GloVe) for every word in the sentence.
    2.  Average these vectors together element-wise.

    The resulting averaged vector is a single, dense representation of the sentence's meaning. This 'sentence embedding' can then be fed into a classifier. This is a huge improvement because it understands that sentences like "The movie was great" and "The film was fantastic" should have very similar vectors.
    """)

    st.subheader("Visualizing the Averaging Process")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    # Word vectors
    ax.text(0.5, 0.8, "The", ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))
    ax.text(2.0, 0.8, "movie", ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))
    ax.text(3.5, 0.8, "was", ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))
    ax.text(5.0, 0.8, "great", ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="lightblue"))

    # Arrows pointing down
    ax.arrow(0.5, 0.7, 0, -0.2, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(2.0, 0.7, 0, -0.2, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(3.5, 0.7, 0, -0.2, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(5.0, 0.7, 0, -0.2, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Averaging box
    ax.text(2.75, 0.2, "Average all vectors", ha='center', va='center', size=12, bbox=dict(boxstyle="sawtooth,pad=0.5", fc="lightgray"))
    ax.arrow(2.75, 0.1, 0, -0.2, head_width=0.1, head_length=0.1, fc='k', ec='k')

    # Final sentence vector
    ax.text(2.75, -0.2, "Sentence Embedding", ha='center', va='center', size=14, color="green", bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen"))
    st.pyplot(fig)
    
    st.error("**Limitation:** Averaging all the word vectors together loses all information about word order and grammar. The sentences 'The cat chased the dog' and 'The dog chased the cat' would have the exact same sentence embedding.")

    st.subheader("🐍 The Python Behind the Idea")
    with st.expander("Show the Python Code for Averaging Embeddings"):
        st.code("""
import numpy as np

def create_sentence_embedding(sentence, embedding_model):
    tokens = sentence.lower().split()
    
    # Get the vector for each word in the sentence, if it exists in the model
    word_vectors = [embedding_model[word] for word in tokens if word in embedding_model]
    
    if not word_vectors:
        # If no words are in the model, return a zero vector
        # You need to know the embedding dimension beforehand
        embedding_dim = 100 # Example dimension
        return np.zeros(embedding_dim)
        
    # Calculate the mean of all word vectors
    sentence_vector = np.mean(word_vectors, axis=0)
    
    return sentence_vector

# --- Example ---
# Assume `model` is a pre-trained Word2Vec or GloVe model
# sentence = "the movie was great"
# sentence_embedding = create_sentence_embedding(sentence, model)
# print(sentence_embedding.shape) # e.g., (100,)
        """, language='python')

# -------------------------------------------------------------------

# chapter5_3.py
import streamlit as st

def render_5_3():
    """Renders the Fine-tuning Transformers section."""
    st.subheader("5.3: The Modern Approach (Fine-tuning Transformers)")
    st.markdown("""
    The previous methods either ignore meaning (Bag-of-Words) or ignore word order (Averaging Embeddings). The modern, state-of-the-art approach uses the full power of the **Transformer architecture** (from Chapter 4) to solve both problems.

    Instead of training a classifier from scratch, we use a technique called **fine-tuning**.
    """)

    st.subheader("🧠 The Method: Standing on the Shoulders of Giants")
    st.markdown("""
    1.  **Start with a Pre-trained Model:** We take a massive Transformer model (like BERT or RoBERTa) that has already been trained on a huge amount of general text (like all of Wikipedia). This model already has a deep, contextual understanding of language.
    2.  **Add a Small Classification 'Head':** We add a very simple, untrained neural network layer on top of the Transformer. This is the only new part of the model.
    3.  **Fine-tune on a Specific Task:** We then train this combined model on our specific, smaller dataset (e.g., 10,000 movie reviews). The training process slightly adjusts the weights of the entire model, but mostly trains the new classification head to map the Transformer's sophisticated output to our desired labels (e.g., 'Positive' or 'Negative').

    This approach is incredibly powerful and efficient because we are leveraging the billions of parameters and the vast knowledge already encoded in the pre-trained model.
    """)
    st.image("https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/tasks/sequence_classification.svg",
             caption="A pre-trained Transformer model with a new classification head added for fine-tuning.")

    st.subheader("🐍 The Python Behind the Idea")
    with st.expander("Show the Python Code for Fine-tuning (Conceptual)"):
        st.code("""
# In practice, this is done using libraries like Hugging Face Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 1. Load a pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) # e.g., Positive/Negative

# 2. Prepare your dataset
# Your dataset would be a list of texts and their corresponding labels (0 or 1)
# train_texts = ["I love this movie!", "This was awful."]
# train_labels = [1, 0]
# tokenized_dataset = tokenizer(train_texts, padding=True, truncation=True)

# 3. Define training arguments and create a Trainer
# training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
#     # eval_dataset=...
# )

# 4. Fine-tune the model
# trainer.train()

# After training, the model can be used for classification on new sentences.
        """, language='python')

# -------------------------------------------------------------------

# chapter5_4.py
import streamlit as st
import re


def render_5_5():
    """Renders the interactive classification workbench."""
    st.subheader("5.5: Interactive Classification Workbench")
    st.markdown("Let's compare how our three methods might classify a movie review. Enter a review below to see a simulated result from each model.")

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
        elif neg_score > pos_score:
            st.error("Classification: **Negative**")
        else:
            st.warning("Classification: **Neutral**")
        st.caption("This model simply counts keywords. It can be easily confused by complex sentences.")

        st.markdown("---")
        # --- Method 2: Averaging Embeddings ---
        st.subheader("2. Embedding Model (Averaging)")
        # Simulate a score based on keywords
        sim_pos_score = pos_score * 0.9 - neg_score * 0.5 # Negative words slightly reduce the score
        sim_neg_score = neg_score * 0.9 - pos_score * 0.5 # Positive words slightly reduce the score
        st.write(f"Simulated Positive Similarity Score: `{sim_pos_score:.2f}`")
        st.write(f"Simulated Negative Similarity Score: `{sim_neg_score:.2f}`")
        if sim_pos_score > sim_neg_score:
            st.success("Classification: **Positive**")
        else:
            st.error("Classification: **Negative**")
        st.caption("This model understands that 'great' and 'fantastic' are similar, but it averages everything together, losing the nuance of the word 'but'.")

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
        st.caption("A real Transformer understands the whole sentence structure. It recognizes that 'but' often negates what came before, giving more weight to 'terrible' and likely classifying the review as Negative.")
