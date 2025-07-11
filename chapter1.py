import streamlit as st
import random
from collections import defaultdict, Counter
import re
import pandas as pd

def render_chapter_1():
    """Renders all content for Chapter 1."""
    
    # --- Helper Functions (Chapter 1) ---
    def build_ngram_model(text, n):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        if len(tokens) < n:
            return None, "The text is too short to create the model."
        model = defaultdict(list)
        for i in range(len(tokens) - n + 1):
            ngram = tokens[i:i+n]
            context, target_word = tuple(ngram[:-1]), ngram[-1]
            model[context].append(target_word)
        return {context: Counter(words) for context, words in model.items()}, tokens

    def get_word_counts_for_context(model, context_tuple):
        return model.get(context_tuple, None)

    # --- UI Rendering ---
    st.header("Chapter 1: The Foundation - Predicting the Next Word")
    st.markdown("We started with **N-grams**, a statistical method to predict the next word based on a fixed window of preceding words.")
    st.subheader("🧠 The Theory: What are N-grams?")
    st.markdown("An **N-gram** is a sequence of *N* words. An N-gram model calculates the probability of a word appearing given the `N-1` words that came before it.")

    st.subheader("🛠️ Interactive Demo: Build Your Own N-gram Model")
    n_value = st.select_slider("Select N-gram size (N):", options=[2, 3, 4, 5], value=3, key="ngram_slider_ch1")
    st.markdown(f"The model will use the last **{n_value - 1}** words as context.")
    default_text = "Once upon a time, in a land filled with green meadows and tall mountains, there lived a friendly dragon. This friendly dragon loved to fly over the green meadows every morning. The villagers would watch the friendly dragon and wave. The dragon was not like other dragons; this dragon was kind and gentle. One day, a lost knight stumbled upon the land. The knight was afraid of the dragon, but the friendly dragon offered the knight a warm smile. The knight, seeing the kind dragon, was no longer afraid. The knight and the dragon became the best of friends, and they would often fly over the green meadows together."
    user_text = st.text_area("Enter text to train the model:", default_text, height=150, key="ngram_text_ch1")
    
    model, tokens = build_ngram_model(user_text, n_value)

    if model:
        st.success(f"✅ Model built using {len(model)} unique contexts.")
        if tokens and len(tokens) >= n_value - 1:
            possible_contexts = list(model.keys())
            initial_context = " ".join(random.choice(possible_contexts)) if possible_contexts else ""
            test_input = st.text_input(f"Enter {n_value - 1} words of context:", value=initial_context, key="ngram_input_ch1")

            if st.button("Predict Next Word", key="ngram_button_ch1"):
                test_tokens = test_input.lower().split()
                if len(test_tokens) != n_value - 1:
                    st.warning(f"Please enter exactly {n_value - 1} words.")
                else:
                    context_tuple = tuple(test_tokens)
                    word_counts = get_word_counts_for_context(model, context_tuple)
                    st.markdown("---")
                    st.info(f"Input Context: `{test_input}`")
                    if word_counts:
                        prediction = word_counts.most_common(1)[0][0]
                        st.success(f"**Top Prediction: {prediction}**")
                    else:
                        st.error("Context not found. Try another phrase from the text!")

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Change the Prediction:** In the default text, the context `the friendly` is followed by `dragon` three times. Try editing the text so that `the friendly knight` appears more often. What does the model predict for `the friendly` now?
    2.  **Create Ambiguity:** The context `the knight` is followed by `was`, `seeing`, and `and`. What happens if you add the sentence "The knight and the king left."? How do the probabilities change?
    """)

    st.subheader("🐍 The Python Behind the Model")
    with st.expander("Show the Python Code for N-grams"):
        st.code("""
def build_ngram_model(text, n):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    if len(tokens) < n: return None
    model = defaultdict(list)
    for i in range(len(tokens) - n + 1):
        ngram = tokens[i:i+n]
        context, target_word = tuple(ngram[:-1]), ngram[-1]
        model[context].append(target_word)
    return {context: Counter(words) for context, words in model.items()}
        """, language='python')
