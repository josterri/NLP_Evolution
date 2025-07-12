import streamlit as st
import random
from collections import defaultdict, Counter
import re
import pandas as pd

def render_1_1():
    """Renders the N-grams demo section."""
    
    # --- Helper Functions ---
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



    import PyPDF2
    import streamlit as st

    st.success(f"PyPDF2 version: {PyPDF2.__version__}")
    # --- UI Rendering ---
    st.subheader("1.1: N-grams & The Interactive Demo")
    st.markdown("The journey into NLP begins not with complex neural networks, but with a surprisingly simple statistical idea: **we can predict the next word in a sequence based on the words that came before it.** This approach, known as N-gram modeling, formed the backbone of language technologies for decades.")

    st.subheader("🧠 The Theory: What are N-grams?")
    st.markdown("""
    An **N-gram** is a contiguous sequence of N items (words) from a given sample of text. The 'N' simply stands for a number:
    - **1-gram (unigram):** A single word. (`"the"`, `"cat"`, `"sat"`)
    - **2-gram (bigram):** A sequence of two words. (`"the cat"`, `"cat sat"`, `"sat on"`)
    - **3-gram (trigram):** A sequence of three words. (`"the cat sat"`, `"cat sat on"`, `"sat on the"`)

    #### The Markov Assumption
    N-gram models operate on a key simplifying principle called the **Markov Assumption**. This assumption states that the probability of the next word depends *only* on the previous `n-1` words. It conveniently ignores all words that came before that limited context window. For a bigram model (`n=2`), the model assumes the next word only depends on the single previous word. For a trigram model (`n=3`), it depends on the two previous words.

    While this is a drastic simplification of how human language works, it makes the problem computationally tractable.

    #### Calculating Probabilities
    The model works by counting occurrences in a large body of text (a corpus). For a bigram model, the probability of a word `w_i` following a word `w_{i-1}` is calculated as:

    $$
    P(w_i | w_{i-1}) = \\frac{Count(w_{i-1}, w_i)}{Count(w_{i-1})}
    $$

    In simple terms: "To find the probability of 'sat' following 'cat', count every time you've seen the phrase 'cat sat' and divide it by the total number of times you've seen the word 'cat'."
    """)

    st.subheader("🛠️ Interactive Demo: Build Your Own N-gram Model")
    n_value = st.select_slider("Select N-gram size (N):", options=[2, 3, 4, 5], value=3, key="ngram_slider_ch1_1")
    st.markdown(f"The model will use the last **{n_value - 1}** words as context.")
    default_text = "Once upon a time, in a land filled with green meadows and tall mountains, there lived a friendly dragon. This friendly dragon loved to fly over the green meadows every morning. The villagers would watch the friendly dragon and wave. The dragon was not like other dragons; this dragon was kind and gentle. One day, a lost knight stumbled upon the land. The knight was afraid of the dragon, but the friendly dragon offered the knight a warm smile. The knight, seeing the kind dragon, was no longer afraid. The knight and the dragon became the best of friends, and they would often fly over the green meadows together."
    user_text = st.text_area("Enter text to train the model:", default_text, height=150, key="ngram_text_ch1_1")
    
    model, tokens = build_ngram_model(user_text, n_value)

    if model:
        st.success(f"✅ Model built using {len(model)} unique contexts.")
        if tokens and len(tokens) >= n_value - 1:
            possible_contexts = list(model.keys())
            initial_context = " ".join(random.choice(possible_contexts)) if possible_contexts else ""
            test_input = st.text_input(f"Enter {n_value - 1} words of context:", value=initial_context, key="ngram_input_ch1_1")

            if st.button("Predict Next Word", key="ngram_button_ch1_1"):
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
