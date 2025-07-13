import streamlit as st

def render_5_6():
    """Renders the consolidated Python code section for Chapter 5."""
    st.subheader("5.6: Consolidated Python Code")
    st.markdown("""
    This section contains the complete, commented Python functions for the classification methods demonstrated in this chapter.
    """)

    st.subheader("1. Bag-of-Words & Naive Bayes Code")
    with st.expander("Show Bag-of-Words Vectorization Code"):
        st.code("""
from collections import Counter
import re

def create_bow_vector(sentence, vocabulary):
    \"\"\"Creates a vector of word counts for a sentence based on a fixed vocabulary.\"\"\"
    # Tokenize and count words in the input sentence
    tokens = re.findall(r'\\b\\w+\\b', sentence.lower())
    word_counts = Counter(tokens)
    
    # Create the vector by looking up counts for each word in the vocabulary
    vector = [word_counts.get(word, 0) for word in vocabulary]
    
    return vector

# --- Example ---
# vocab = ['the', 'quick', 'brown', 'fox', 'jumps']
# sentence = "the fox is quick, the fox is brown"
# bow_vector = create_bow_vector(sentence, vocab)
# Result: [2, 1, 1, 2, 0]
# print(bow_vector)
        """, language='python')

    st.subheader("2. Averaging Word Embeddings Code")
    with st.expander("Show Sentence Embedding Code"):
        st.code("""
import numpy as np

def create_sentence_embedding(sentence, embedding_model):
    \"\"\"Creates a single sentence vector by averaging the vectors of its words.\"\"\"
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

    st.subheader("3. Fine-tuning Transformers (Conceptual Code)")
    with st.expander("Show Conceptual Fine-tuning Code"):
        st.code("""
# In practice, this is done using libraries like Hugging Face Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def fine_tune_transformer_for_classification():
    # 1. Load a pre-trained model and tokenizer
    # 'distilbert-base-uncased' is a small, fast version of BERT
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load the model with a classification head for 2 labels (e.g., Positive/Negative)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 2. Prepare your dataset
    # Your dataset would be a list of texts and their corresponding labels (0 or 1)
    train_texts = ["I love this movie!", "This was awful."]
    train_labels = [1, 0]
    
    # Tokenize the dataset so the model can understand it
    tokenized_dataset = tokenizer(train_texts, padding=True, truncation=True)
    # This needs to be converted to a format the Trainer can use (e.g., a PyTorch Dataset)

    # 3. Define training arguments and create a Trainer
    # These arguments control the fine-tuning process
    training_args = TrainingArguments(
        output_dir="test_trainer",
        num_train_epochs=3,
        per_device_train_batch_size=8,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=your_prepared_dataset,
        # eval_dataset=your_prepared_eval_dataset
    )

    # 4. Fine-tune the model
    # This command starts the training process
    # trainer.train()

    # After training, the model can be used for classification on new sentences.
    print("Conceptual code for fine-tuning a Transformer.")

# fine_tune_transformer_for_classification()
        """, language='python')
