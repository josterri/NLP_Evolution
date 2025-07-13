import streamlit as st
import pandas as pd
import time

def render_6_2():
    """Renders the Causal Language Modeling section."""
    st.subheader("6.2: The Training Objective: Causal Language Modeling")
    
    st.subheader("Motivation: The Data Problem")
    st.markdown("""
    Most traditional machine learning tasks are **supervised**, meaning they require carefully labeled data. For example, to train a sentiment classifier, you need thousands of movie reviews that have been hand-labeled by humans as "Positive" or "Negative". This process is expensive, time-consuming, and creates a data bottleneck. It's simply not feasible to label the entire internet.

    The creators of generative models asked a brilliant question: **Can we get a model to learn about language without any human labels?**

    The answer is yes, if we frame the task correctly. The solution is **self-supervised learning**, where the data itself provides the labels. Instead of needing humans to create `(input, label)` pairs, we can generate them automatically from raw text.
    """)

    st.subheader("🧠 The Method: Learning from Raw Text")
    st.markdown("""
    The training process is surprisingly simple: it's just **next-word prediction on a massive scale**. This specific task is often called **Causal Language Modeling** (CLM), because the goal is to predict the next token based only on the sequence of tokens that came before it (the "cause").

    The model is given a huge amount of text from the internet (e.g., a Wikipedia article). It then automatically turns this text into millions of training examples. Its only goal is to predict the next word at every single position in the text.

    This simple task, when performed on a vast and diverse dataset, forces the model to learn grammar, facts, reasoning, and world knowledge in order to get better at its one job. To accurately predict the word "French" in the sentence "The man from Paris speaks fluent...", the model must learn that Paris is in France and that people from France speak French. It learns these complex relationships not because it was taught them, but because they improve its ability to predict the next word.
    """)

    st.subheader("🛠️ Interactive Demo: Creating Training Examples")
    st.markdown("Enter a sentence to see how the model automatically creates its own labeled training data. The model slides a window across the text, treating everything inside the window as the input (context) and the very next word as the output (target).")
    
    sentence = st.text_input("Enter a sentence:", "The cat sat on the mat")
    tokens = sentence.lower().split()
    
    if st.button("Generate Training Data"):
        if len(tokens) > 1:
            placeholder = st.empty()
            for i in range(1, len(tokens)):
                with placeholder.container():
                    context = tokens[:i]
                    target = tokens[i]
                    
                    st.markdown(f"#### Step {i}")
                    st.write("The model takes the context:")
                    st.info(f"`{' '.join(context)}`")
                    st.write("And learns to predict the target:")
                    st.success(f"`{target}`")
                    st.markdown("---")
                time.sleep(1.5)
            
            st.write("The final training set created from this one sentence:")
            training_data = []
            for i in range(1, len(tokens)):
                context_str = " ".join(tokens[:i])
                target_str = tokens[i]
                training_data.append({"Input (Context)": f"`{context_str}`", "Output (Target)": f"`{target_str}`"})
            df = pd.DataFrame(training_data)
            st.dataframe(df)

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Why "Self-Supervised"?** Why is this process called "self-supervised" instead of "unsupervised"? (Hint: Are there labels involved in the training process, and where do they come from?)
    2.  **Data Quality:** What would happen to a model that was only trained on a dataset of Shakespeare's plays? How would its generated text differ from a model trained on Twitter data? What does this imply about the importance of the pre-training corpus?
    3.  **Task vs. Pre-training:** How does this self-supervised pre-training objective differ from the supervised objective of a classification task (Chapter 5)?
    """)

    st.subheader("🐍 The Python Behind the Data Creation")
    with st.expander("Show the Python Code for Generating Training Pairs"):
        st.code("""
def create_causal_lm_training_pairs(tokens):
    \"\"\"
    Takes a list of tokens and creates a list of (context, target) pairs.
    \"\"\"
    training_pairs = []
    # We iterate from the second token to the end of the list
    for i in range(1, len(tokens)):
        # The context is all tokens up to the current position (exclusive)
        context = tokens[:i]
        # The target is the token at the current position
        target = tokens[i]
        training_pairs.append((context, target))
    return training_pairs

# --- Example ---
sentence = "the cat sat on the mat"
tokens = sentence.split()
pairs = create_causal_lm_training_pairs(tokens)

# Print the first training pair
# Context: ['the'], Target: 'cat'
print(f"Context: {pairs[0][0]}, Target: '{pairs[0][1]}'")

# Print the last training pair
# Context: ['the', 'cat', 'sat', 'on', 'the'], Target: 'mat'
print(f"Context: {pairs[-1][0]}, Target: '{pairs[-1][1]}'")
        """, language='python')
