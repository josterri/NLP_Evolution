import streamlit as st
import random
from collections import defaultdict, Counter
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(
    page_title="The Evolution of NLP",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions (Chapter 1) ---

def generate_ngrams(tokens, n):
    """Generates n-grams from a list of tokens."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams

def build_ngram_model(text, n):
    """Builds a simple n-gram model from a given text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    if len(tokens) < n:
        return None, "The text is too short to create the model."
    model = defaultdict(list)
    ngrams = generate_ngrams(tokens, n)
    for ngram in ngrams:
        context, target_word = ngram[:-1], ngram[-1]
        model[context].append(target_word)
    prob_model = {context: Counter(words) for context, words in model.items()}
    return prob_model, tokens

def get_word_counts_for_context(model, context_tuple):
    """Gets the Counter object for a given context."""
    return model.get(context_tuple, None)

# --- Helper Functions (Chapter 2) ---

def get_embedding_vocab():
    """Creates a predefined vocabulary with 2D vector embeddings for the demo."""
    vocab = {
        'king': np.array([9.5, 9]), 'queen': np.array([9.5, 1]),
        'man': np.array([8.5, 9]), 'woman': np.array([8.5, 1]),
        'prince': np.array([7.5, 8.5]), 'princess': np.array([7.5, 1.5]),
        'apple': np.array([1, 5]), 'orange': np.array([1, 4]),
        'banana': np.array([1.5, 3.5]), 'grape': np.array([0.5, 3]),
        'dog': np.array([6, 8]), 'cat': np.array([6, 7]),
        'puppy': np.array([5, 8.5]), 'kitten': np.array([5, 7.5]),
        'strong': np.array([9, 6]), 'fast': np.array([8, 6]),
        'sweet': np.array([2, 2]), 'sour': np.array([2, 1])
    }
    return vocab

def find_closest_words(selected_word, vocab, n=3):
    """Finds the n closest words to a selected word based on Euclidean distance."""
    if selected_word not in vocab:
        return []
    
    selected_vector = vocab[selected_word]
    distances = []
    for word, vector in vocab.items():
        if word == selected_word:
            continue
        distance = np.linalg.norm(selected_vector - vector)
        distances.append((word, distance))
    
    distances.sort(key=lambda x: x[1])
    return distances[:n]

# --- Main App ---

st.title("📜 The Evolution of NLP")
st.markdown("From Simple Predictions to Deep Understanding")
st.markdown("---")

# --- Chapter 1 ---
st.header("Chapter 1: The Foundation - Predicting the Next Word")
st.markdown("""
Welcome to the first chapter in our journey through the history of Natural Language Processing! Before we could build sophisticated chatbots or translation tools, we had to solve a fundamental problem: **How can a machine understand and generate human language?**

The earliest and most intuitive approach was to teach the machine to guess the next word in a sentence. This simple idea laid the groundwork for everything that followed. It's like learning to speak by memorizing common phrases.

In this section, we'll explore the core technique behind this: **N-grams**.
""")

st.subheader("🧠 The Theory: What are N-grams?")
st.markdown("""
An **N-gram** is simply a sequence of *N* items (in our case, words) from a sample of text. The "N" can be any positive integer.

- **Unigram (1-gram):** A single word. (`"the"`, `"cat"`, `"sat"`)
- **Bigram (2-gram):** A sequence of two words. (`"the cat"`, `"cat sat"`, `"sat on"`)
- **Trigram (3-gram):** A sequence of three words. (`"the cat sat"`, `"cat sat on"`, `"sat on the"`)

The core idea of an N-gram model is to look at the previous `N-1` words (the *context*) and calculate the probability of what the *Nth* word will be. By analyzing a large amount of text, the model learns which words are likely to follow others.
""")

st.subheader("🛠️ Interactive Demo: Build Your Own N-gram Model")
n_value = st.select_slider("Select N-gram size (N):", options=[2, 3, 4, 5], value=3, key="ngram_slider_main")
st.markdown(f"You've selected **N = {n_value}**. This means the model will use the last **{n_value - 1}** words as context.")

st.markdown("#### Step 1: Provide Text to Train the Model")
default_text = "Once upon a time, in a land filled with green meadows and tall mountains, there lived a friendly dragon. This friendly dragon loved to fly over the green meadows every morning. The villagers would watch the friendly dragon and wave. The dragon was not like other dragons; this dragon was kind and gentle. One day, a lost knight stumbled upon the land. The knight was afraid of the dragon, but the friendly dragon offered the knight a warm smile. The knight, seeing the kind dragon, was no longer afraid. The knight and the dragon became the best of friends, and they would often fly over the green meadows together."
user_text = st.text_area("Enter a paragraph of text here:", default_text, height=200, key="ngram_text_main")

model, tokens = build_ngram_model(user_text, n_value)

if model is None:
    st.error(f"Error: The provided text is too short to build a model with N={n_value}.")
else:
    st.success(f"✅ Model successfully built from your text using {len(model)} unique contexts.")
    with st.expander("Click to see all contexts the model learned"):
        formatted_contexts = [f"`{' '.join(context)}`" for context in sorted(model.keys())]
        st.info(", ".join(formatted_contexts))

    st.markdown("#### Step 2: Test the Model")
    if tokens and len(tokens) >= n_value - 1:
        possible_contexts = list(model.keys())
        initial_context = " ".join(random.choice(possible_contexts)) if possible_contexts else ""
        test_input = st.text_input(f"Enter {n_value - 1} words of context:", value=initial_context)

        if st.button("Predict Next Word", key="ngram_button_main"):
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
                    st.subheader("🔍 Behind the Prediction: Probabilities")
                    total = sum(word_counts.values())
                    prob_data = [{"Word": word, "Probability": count/total, "Count": count} for word, count in word_counts.items()]
                    df = pd.DataFrame(prob_data).sort_values(by="Probability", ascending=False).reset_index(drop=True)
                    st.bar_chart(df.rename(columns={'Word':'index'}).set_index('index')['Probability'])
                    st.dataframe(df)
                else:
                    st.error("Context not found. Try another phrase from the text!")

st.subheader("✏️ Exercises")
st.markdown("""
1.  **Change the Prediction:** In the default text, the context `the friendly` is followed by `dragon` three times. Try editing the text so that `the friendly knight` appears more often. What does the model predict for `the friendly` now?
2.  **Create Ambiguity:** The context `the knight` is followed by `was`, `seeing`, and `and`. What happens if you add the sentence "The knight and the king left."? How do the probabilities change?
3.  **The Sparsity Problem:** Try predicting from a context that doesn't exist in the text, like `tall green`. What happens? This demonstrates the model's inability to generalize.
""")

st.subheader("🐍 The Python Behind the Model")
st.markdown("Curious how the N-gram model is built? It's surprisingly simple. Here are the core functions that power the demo above.")
with st.expander("Show the Python Code"):
    st.code("""
def build_ngram_model(text, n):
    # Convert text to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()

    # Ensure the text is long enough for the N-gram size
    if len(tokens) < n:
        return None, "Text is too short."

    # This dictionary will hold our model
    # The key is the context (an N-1 gram tuple)
    # The value is a list of all words that have followed that context
    model = defaultdict(list)
    
    # Slide a window of size N across the text
    for i in range(len(tokens) - n + 1):
        ngram = tokens[i:i+n]
        context, target_word = tuple(ngram[:-1]), ngram[-1]
        model[context].append(target_word)
    
    # To make predictions, we convert the lists of words into frequency counters
    # This tells us *how many times* each word followed the context
    prob_model = {context: Counter(words) for context, words in model.items()}
    return prob_model, tokens
    """, language='python')

st.markdown("---")

# --- Chapter 2 ---
st.header("Chapter 2: Giving Words Meaning - Word Embeddings")
st.markdown("""
The N-gram model's biggest flaws were its inability to understand word meaning and its failure to handle new word combinations (the sparsity problem). If the model never saw "the friendly knight", it couldn't predict what came next.

The next great leap in NLP was to stop treating words as unique symbols and start representing them based on their **meaning and context**. This is the world of **Word Embeddings**.
""")

st.subheader("🧠 The Theory: Words as Vectors")
st.markdown("""
The core idea is to represent each word as a list of numbers, called a **vector**. This isn't just a random list; it's carefully calculated so that words with similar meanings are represented by similar vectors.

- **Analogy:** Imagine a map. "London", "Paris", and "Tokyo" would be plotted far apart. But "cat", "kitten", and "puppy" would be clustered together in a "pets" neighborhood, while "apple", "orange", and "banana" would be in a "fruits" neighborhood.

This even allows for amazing **vector arithmetic**:
$$
\text{vector}(\text{'king'}) - \text{vector}(\text{'man'}) + \text{vector}(\text{'woman'}) \approx \text{vector}(\text{'queen'})
$$
""")

st.subheader("🛠️ Interactive Demo: Exploring a Word Vector Space")
st.markdown("Here is a simplified 2D 'map' of words. Each word is a point in space. Let's see what relationships we can find.")

vocab = get_embedding_vocab()
words = list(vocab.keys())
vectors = np.array(list(vocab.values()))

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(vectors[:, 0], vectors[:, 1], s=50)
for word, (x, y) in vocab.items():
    ax.text(x + 0.1, y + 0.1, word, fontsize=12)
ax.set_title("A 2D Map of Word Meanings", fontsize=16)
ax.set_xlabel("Vector Dimension 1 (e.g., 'Royalty/Power')")
ax.set_ylabel("Vector Dimension 2 (e.g., 'Gender/Age')")
ax.grid(True)
st.pyplot(fig)

st.markdown("#### Find Similar Words")
selected_word = st.selectbox("Choose a word:", options=words, index=0)
if st.button("Find Closest Words"):
    closest_words = find_closest_words(selected_word, vocab, n=3)
    if closest_words:
        st.write(f"The words most similar to **'{selected_word}'** are:")
        for word, dist in closest_words:
            st.info(f"- **{word}** (Distance: {dist:.2f})")

st.markdown("#### Test Vector Math")
c1, c2, c3 = st.columns(3)
word1 = c1.selectbox("Word 1", options=words, index=words.index('king'))
word2 = c2.selectbox("Word 2", options=words, index=words.index('man'))
word3 = c3.selectbox("Word 3", options=words, index=words.index('woman'))
if st.button("Calculate: Word 1 - Word 2 + Word 3"):
    result_vec = vocab[word1] - vocab[word2] + vocab[word3]
    closest_to_result = find_closest_words('placeholder', {**vocab, 'placeholder': result_vec}, n=1)
    st.info(f"Calculating: `{word1}` - `{word2}` + `{word3}`")
    if closest_to_result:
        result_word = closest_to_result[0][0]
        st.success(f"The result is closest to: **{result_word}**")

st.markdown("---")
st.header("Limitations of Static Embeddings")
st.markdown("""
Word embeddings were a massive breakthrough, but they still have a crucial flaw: **they are static**. Each word has only one vector, regardless of its context. Consider the word "**bank**":
- "I need to go to the **bank** to deposit money." (a financial institution)
- "We sat on the river **bank** and watched the boats." (the side of a river)
A traditional word embedding model gives "bank" a single vector, an average of these two meanings. It can't distinguish between them based on the sentence. This is the **problem of polysemy**.

**Coming up next...** We'll dive into the world of **Attention and Transformers**, the architecture that finally solved the context problem and powers modern NLP models like ChatGPT.
""")
