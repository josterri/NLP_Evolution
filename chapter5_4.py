# chapter5_4.py
import streamlit as st

def render_5_4():
    """Renders the Fine-tuning Transformers section."""
    st.subheader("5.4: The Modern Approach (Fine-tuning Transformers)")
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
