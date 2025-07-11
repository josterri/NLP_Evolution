import streamlit as st
import pandas as pd
from collections import Counter

def render_1_2():
    """Renders the Sparsity Problem section."""
    st.subheader("1.2: The Sparsity Problem - When the Data Runs Dry")
    st.markdown("""
    The N-gram model has a critical, inherent weakness that stems directly from its reliance on counting. This is the **Sparsity Problem**, and it's one of the primary reasons researchers sought more advanced techniques.

    **In short: If an N-gram has never appeared in your training text, the model assigns it a probability of zero.**
    """)

    st.subheader("An Example of Failure")
    st.markdown("""
    Imagine your training text contains these sentences:
    - "The student read the book."
    - "The student wrote the essay."

    From this text, your bigram model learns that the word `the` can follow `student`.
    
    Now, you want to evaluate a new, perfectly valid sentence:
    > "The student passed the exam."

    The model calculates the probability of the sentence by chaining the probabilities of its bigrams:
    $$ P(\text{"The student passed the exam"}) = P(\text{student}|\text{The}) \\times P(\text{passed}|\text{student}) \\times P(\text{the}|\text{passed}) \\times ... $$

    But when it gets to `P(passed|student)`, it runs into a problem. The bigram `("student", "passed")` never appeared in the training data.
    - `Count("student", "passed")` = 0
    - Therefore, `P(passed|student)` = 0

    Because one part of the chain is zero, the probability of the *entire sentence* becomes zero. The model incorrectly concludes that this sentence is impossible, simply because it hasn't seen that specific word combination before.
    """)

    st.subheader("Visualizing the 'Zero Probability' Cliff")
    st.markdown("""
    Let's see this in action. We'll "train" a model on a tiny piece of text and then look at the probabilities for words following the context **'the'**.
    """)

    # --- Visualization Demo ---
    training_text = "the cat sat on the mat the cat ate the food"
    tokens = training_text.split()
    
    # Manually create bigrams for clarity
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    
    # Get all words following 'the'
    words_after_the = [bigram[1] for bigram in bigrams if bigram[0] == 'the']
    counts_after_the = Counter(words_after_the)

    st.write("Our training text is:")
    st.info(f"`{training_text}`")
    
    st.write("Let's find the probability of words that can follow the word **'the'** based *only* on this text.")

    # Create a dataframe for display
    vocab = sorted(list(set(tokens)))
    prob_data = []
    total_count = len(words_after_the)

    for word in vocab:
        count = counts_after_the.get(word, 0)
        prob = count / total_count if total_count > 0 else 0
        prob_data.append({
            "Possible Next Word": word,
            "Count('the', word)": count,
            "Probability": f"{prob:.2f}"
        })
    
    df = pd.DataFrame(prob_data)
    st.dataframe(df.style.apply(lambda row: ['background-color: #ffcccc'] * len(row) if row['Probability'] == '0.00' else [''] * len(row), axis=1))

    st.error("""
    Notice that for any word that never followed 'the' in our tiny text (like 'on', 'sat', etc.), the count is 0, and therefore the probability is 0. 
    The model believes the phrase "the on" is impossible. This is the sparsity problem visualized.
    """)

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Find the Failure Point:** Given the training text "I love my dog and I love my cat", what is the probability of the sentence "I love my bird"? At which word does the model fail?
    2.  **Real-World Example:** Think about a restaurant review website. If the training data contains many reviews like "great food" and "good food", but no one has ever written "amazing food", what probability would an N-gram model assign to that new phrase?
    3.  **Scaling Up:** Does this problem go away if we use a much larger N, like a 5-gram model? Why or why not? (Hint: Think about how many 5-word combinations exist vs. how many are likely to be in your training text).
    """)

    st.subheader("🐍 The Python Behind the Problem")
    with st.expander("Show the Python Code for Probability Calculation"):
        st.code("""
from collections import Counter

def calculate_bigram_probability(word, context, corpus_tokens):
    # Create all bigrams from the corpus
    bigrams = [(corpus_tokens[i], corpus_tokens[i+1]) for i in range(len(corpus_tokens)-1)]
    
    # Count occurrences
    context_count = corpus_tokens.count(context)
    bigram_count = bigrams.count((context, word))

    # Avoid division by zero if context never appears
    if context_count == 0:
        return 0

    # The core calculation
    # If bigram_count is 0, the probability is 0. This is the sparsity problem.
    probability = bigram_count / context_count
    
    return probability

# --- Example ---
text = "the cat sat on the mat"
tokens = text.split()

# This will be 0 because ("the", "food") is not in the text.
prob_of_food = calculate_bigram_probability("food", "the", tokens) 

# This will be 0.5 because "the" appears 2 times, and "the cat" appears once.
prob_of_cat = calculate_bigram_probability("cat", "the", tokens) 
        """, language='python')
