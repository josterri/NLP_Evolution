import streamlit as st
import pandas as pd
from collections import Counter

def render_1_3():
    """Renders the Smoothing Techniques section."""
    st.subheader("1.3: Smoothing - Donating Probability Mass")
    st.markdown("""
    How do we solve the Sparsity Problem? We can't just ignore unseen N-grams, as that would make our model blind to the creativity of language. The solution is **Smoothing**.

    The core idea is simple but powerful: **take a tiny fraction of the probability from the N-grams we *have* seen and redistribute it among all the N-grams we *haven't* seen.** This ensures that no word sequence has a probability of exactly zero, making the model more robust and realistic.
    """)

    st.subheader("Laplace (Add-One) Smoothing")
    st.markdown("""
    The simplest and most intuitive smoothing technique is **Laplace Smoothing**, often called **Add-One Smoothing**. It works by pretending we have seen every possible N-gram one more time than we actually have.

    Let's revisit our probability formula. The standard calculation for a bigram is:

    $$
    P(w_i | w_{i-1}) = \\frac{Count(w_{i-1}, w_i)}{Count(w_{i-1})}
    $$

    With Add-One smoothing, we add 1 to every count in the numerator. To keep it a valid probability distribution (i.e., to make sure all probabilities still sum to 1), we must also adjust the denominator. We add the total number of unique words in our vocabulary (V) to the denominator's count.

    The new formula becomes:
    $$ P_{add-1}(w_i | w_{i-1}) = \\frac{Count(w_{i-1}, w_i) + 1}{Count(w_{i-1}) + V} $$

    Where **V** is the size of the vocabulary.
    """)

    st.subheader("Visualizing the Effect of Smoothing")
    st.markdown("Let's revisit the example from the previous section and see how Add-One Smoothing fixes the 'zero probability' cliff.")

    # --- Visualization Demo ---
    training_text = "the cat sat on the mat the cat ate the food"
    tokens = training_text.split()
    vocab = sorted(list(set(tokens)))
    vocab_size = len(vocab)

    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    words_after_the = [bigram[1] for bigram in bigrams if bigram[0] == 'the']
    counts_after_the = Counter(words_after_the)
    total_count_after_the = len(words_after_the)

    st.write("Our training text is:")
    st.info(f"`{training_text}`")
    st.write(f"Our vocabulary size (V) is: **{vocab_size}**")

    # Create data for comparison
    prob_data = []
    for word in vocab:
        # Standard Probability (with sparsity)
        standard_count = counts_after_the.get(word, 0)
        standard_prob = standard_count / total_count_after_the if total_count_after_the > 0 else 0

        # Add-One Smoothed Probability
        smoothed_count = standard_count + 1
        smoothed_denominator = total_count_after_the + vocab_size
        smoothed_prob = smoothed_count / smoothed_denominator

        prob_data.append({
            "Possible Next Word": word,
            "Original Probability": standard_prob,
            "Smoothed Probability": smoothed_prob,
            "Original Count": standard_count,
            "Smoothed Count": smoothed_count
        })

    df = pd.DataFrame(prob_data)
    
    st.write("Comparing probabilities for words following the context **'the'**:")
    st.dataframe(df[['Possible Next Word', 'Original Probability', 'Smoothed Probability']].style.format({
        'Original Probability': '{:.2f}',
        'Smoothed Probability': '{:.2f}'
    }).apply(lambda row: ['background-color: #ccffcc'] * len(row) if row['Original Probability'] == 0.0 else [''] * len(row), axis=1))
    
    st.success("""
    Notice how the words that previously had a probability of 0 now have a small, non-zero probability. This probability mass was 'donated' from the words we had already seen ('cat', 'mat', 'food'), whose probabilities are now slightly lower. The model is no longer brittle and can handle unseen events!
    """)
    
    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Impact of Vocabulary Size:** In our example, `V = 7`. What would happen to the smoothed probabilities if our vocabulary was much larger, say `V = 10000`? How would that affect the probabilities of the words we *have* seen?
    2.  **Over-smoothing:** Add-One is often criticized for being too aggressive. It gives a lot of probability mass to unseen events. Can you think of a situation where this might be a problem? (Hint: What if you have a very large vocabulary and a very small training text?)
    3.  **Add-k Smoothing:** A variation is "Add-k" smoothing, where you add a fraction `k` (e.g., 0.1) instead of 1. How would using `k=0.01` change the results compared to `k=1`?
    """)

    st.subheader("🐍 The Python Behind the Smoothing")
    with st.expander("Show the Python Code for Add-One Smoothing"):
        st.code("""
from collections import Counter

def calculate_add_one_smoothed_prob(word, context, corpus_tokens):
    # Get the entire vocabulary and its size
    vocab = set(corpus_tokens)
    vocab_size = len(vocab)

    # Create all bigrams from the corpus
    bigrams = [(corpus_tokens[i], corpus_tokens[i+1]) for i in range(len(corpus_tokens)-1)]
    
    # Count occurrences
    context_count = corpus_tokens.count(context)
    bigram_count = bigrams.count((context, word))

    # The core calculation with Add-One smoothing
    numerator = bigram_count + 1
    denominator = context_count + vocab_size

    probability = numerator / denominator
    
    return probability

# --- Example ---
text = "the cat sat on the mat"
tokens = text.split()

# This will now be non-zero!
prob_of_food = calculate_add_one_smoothed_prob("food", "the", tokens) 

# This probability will be slightly lower than the original 0.5
prob_of_cat = calculate_add_one_smoothed_prob("cat", "the", tokens)
        """, language='python')
