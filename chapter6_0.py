# chapter6_0.py
import streamlit as st
import matplotlib.pyplot as plt

def render_6_0():
    """Renders the motivational recap section."""
    st.subheader("6.0: The Story So Far - A Recap on Next-Word Prediction")
    st.markdown("""
    Throughout this entire course, one simple task has been the driving force behind most of the major breakthroughs: **predicting the next word**. The methods for performing this task have become increasingly sophisticated, with each new technique building on the last to overcome a critical limitation.

    Let's trace this evolution step-by-step to understand where we are now and why generative models were the inevitable next step.
    """)

    st.subheader("The Evolution of Context for Next-Word Prediction")
    st.markdown("At each stage, the key difference was how the model defined and used **context**.")

    st.markdown("---")
    
    # --- Stage 1: N-grams ---
    st.markdown("#### Stage 1: Counting with a Fixed Window (N-grams)")
    st.markdown("""
    -   **What it considers (Context):** A fixed window of the last `N-1` words. For a trigram model (N=3), it only ever sees the last two words.
    -   **Example:** For the sentence `the cat sat on the ___`, the model only sees the context `('on', 'the')`.
    -   **Thought Process:** "In my training data, what was the single most frequent word that appeared after the exact phrase 'on the'?"
    -   **Key Limitation:** No understanding of meaning and fails completely if the context has never been seen before.
    """)

    # --- Stage 2: Static Embeddings ---
    st.markdown("---")
    st.markdown("#### Stage 2: Meaning of the Last Word (Static Embeddings)")
    st.markdown("""
    -   **What it considers (Context):** Only the single last word in the sequence.
    -   **Example:** For the sentence `the cat sat on the ___`, the model only sees the context `the`.
    -   **Thought Process:** "What is the meaning (vector) of the word 'the'? Now, I will search my entire vocabulary for the word with the most similar meaning."
    -   **Key Limitation:** While it understands word similarity, it has no memory of the preceding words. This leads to text that is semantically related but often drifts off-topic and lacks grammatical structure.
    """)

    # --- Stage 3: Sequential Memory ---
    st.markdown("---")
    st.markdown("#### Stage 3: A 'Memory' of the Past (RNNs/LSTMs)")
    st.markdown("""
    -   **What it considers (Context):** A 'memory' or 'hidden state' that is a compressed summary of *all* the words that came before.
    -   **Example:** For the sentence `the cat sat on the ___`, the model's memory contains a blended representation of having seen `the`, then `cat`, then `sat`, then `on`.
    -   **Thought Process:** "Based on my current memory, which has been influenced by the entire sequence so far, what word is most likely to come next?"
    -   **Key Limitation:** The 'memory' is a bottleneck. Information from early words (like 'cat') can get diluted or lost by the time the model has processed many other words, making it hard to handle long-range dependencies.
    """)

    # --- Stage 4: Focused Blending ---
    st.markdown("---")
    st.markdown("#### Stage 4: A Weighted Focus on the Past (Attention)")
    st.markdown("""
    -   **What it considers (Context):** All previous words, but not equally. It calculates attention scores to decide which past words are most important for predicting the next one.
    -   **Example:** For the sentence `the cat sat on the ___`, the model's attention mechanism might determine that 'sat' and 'on' are the most important clues for what comes next.
    -   **Thought Process:** "I need to predict the next word. I will look at all the words I've seen so far. I will create a new 'context vector' by blending the meaning of all previous words, but I will give more weight to the ones my attention scores tell me are most relevant. Now, what word is most similar to this new, blended, context-aware vector?"
    -   **Key Limitation:** This is incredibly powerful, but what if we build an entire architecture around just this one idea?
    """)

    st.subheader("The Next Big Question")
    st.markdown("""
    This leads to a revolutionary idea: What if we build an architecture that is *only* designed to do next-word prediction, but uses the most powerful tool we have—**the Attention mechanism**—to do it on a massive scale?

    What if, instead of using this for analysis (like classification), we just let the model keep predicting the next word, over and over again, to *create* new text?

    This is the core idea behind **Generative Models** like GPT.
    """)

# -------------------------------------------------------------------

# chapter6_1.py
import streamlit as st

def render_6_1():
    """Renders the Decoder-Only Architecture section."""
    st.subheader("6.1: The Decoder-Only Architecture (GPT-style)")
    st.markdown("""
    How do you build a model that is purely focused on generating text, one word at a time? The answer is surprisingly elegant: you only need half of the original Transformer.

    Models like GPT (Generative Pre-trained Transformer) are **Decoder-Only** models. They discard the Encoder stack entirely and use a modified version of the Decoder stack.
    """)

    st.subheader("🧠 The Method: Masked Self-Attention")
    st.markdown("""
    The most crucial component of a Decoder-Only model is **Masked Self-Attention**.
    
    In a normal self-attention block (like in an Encoder), every word can "look at" every other word, both before and after it. But for next-word prediction, this would be cheating! To predict the word "sat" in "the cat sat", the model should only be allowed to see "the" and "cat".

    The mask is a simple but brilliant trick that prevents the model from seeing "future" words. Before the softmax step in the attention calculation, the model adds a large negative number to the scores for all future positions. When the softmax is applied, these large negative scores become zero, effectively hiding the future words from the attention mechanism.
    """)

    st.image("https://i.imgur.com/sSGUa3d.png", caption="In Masked Self-Attention, a word can only attend to previous words (and itself). The mask blocks information from the future.")

# -------------------------------------------------------------------

# chapter6_2.py
import streamlit as st
import pandas as pd

def render_6_2():
    """Renders the Causal Language Modeling section."""
    st.subheader("6.2: The Training Objective: Causal Language Modeling")
    st.markdown("""
    How do you train such a massive model without millions of human-labeled examples? The training process is surprisingly simple and **self-supervised**: it's just **next-word prediction on a massive scale**.

    This specific task is often called **Causal Language Modeling** (CLM), because the goal is to predict the next token based only on the sequence of tokens that came before it (the "cause").
    """)

    st.subheader("🧠 The Method: Learning from Raw Text")
    st.markdown("""
    The model is given a huge amount of text from the internet (e.g., a Wikipedia article). Its only goal is to predict the next word at every single position in the text.

    This simple task, when performed on a vast and diverse dataset, forces the model to learn grammar, facts, reasoning, and world knowledge in order to get better at its one job. To accurately predict the word "French" in the sentence "The man from Paris speaks fluent...", the model must learn that Paris is in France and that people from France speak French.
    """)

    st.subheader("🛠️ Interactive Demo: Creating Training Examples")
    st.markdown("Enter a sentence to see the training pairs that would be generated from it for the model.")
    
    sentence = st.text_input("Enter a sentence:", "The cat sat on the mat")
    tokens = sentence.lower().split()
    
    if len(tokens) > 1:
        training_data = []
        for i in range(1, len(tokens)):
            context = " ".join(tokens[:i])
            target = tokens[i]
            training_data.append({"Input (Context)": f"`{context}`", "Output (Target)": f"`{target}`"})
        
        df = pd.DataFrame(training_data)
        st.dataframe(df)

# -------------------------------------------------------------------

# chapter6_3.py
import streamlit as st

def render_6_3():
    """Renders the In-Context Learning section."""
    st.subheader("6.3: The Emergence of In-Context Learning")
    st.markdown("""
    Training a model on the simple task of next-word prediction at a massive scale led to a surprising and magical new capability: **In-Context Learning**.

    This means the model can perform tasks it was never explicitly trained on, simply by being prompted correctly. It learns to recognize a pattern from the prompt and continue it.
    """)

    st.subheader("Zero-Shot vs. Few-Shot Learning")
    st.markdown("""
    -   **Zero-Shot Learning:** You ask the model to perform a task directly, without giving it any examples. The model uses its vast pre-trained knowledge to figure out what you want.
    -   **Few-Shot Learning:** You give the model a few examples of the task within the prompt itself. This gives the model a much clearer pattern to follow, dramatically improving its performance without any need to retrain or fine-tune the model's weights.
    """)
    
    st.subheader("🛠️ Interactive Demo: Zero-shot vs. Few-shot")
    st.markdown("See how a (simulated) model's response changes when you give it examples.")
    
    st.markdown("#### Zero-Shot Prompt")
    st.code("Classify this movie review as positive or negative.\n\nReview: The movie was a masterpiece!")
    st.success("Simulated Model Output: **Positive**")

    st.markdown("---")
    
    st.markdown("#### Few-Shot Prompt")
    st.code("""
Classify this movie review as positive or negative.

Review: The acting was terrible.
Classification: Negative

Review: I loved every minute of it.
Classification: Positive

Review: The movie was a masterpiece!
Classification:
    """)
    st.success("Simulated Model Output: **Positive**")
    st.caption("By providing examples, we make the task much clearer for the model, leading to more reliable results.")

# -------------------------------------------------------------------

# chapter6_4.py
import streamlit as st

def render_6_4():
    """Renders the interactive generation workbench."""
    st.subheader("6.4: Interactive Generation Workbench")
    st.markdown("Let's see a (very simplified) simulation of a generative model in action. Provide a prompt and see how the model might continue the text.")

    prompt = st.text_area("Enter your prompt:", "The best thing about living in the mountains is")
    
    if st.button("Generate Text"):
        # This is a highly simplified simulation. A real model's logic is far more complex.
        simulated_continuation = ""
        if "mountains" in prompt.lower():
            simulated_continuation = " the fresh air and the beautiful views. Every morning, you can wake up to the sight of..."
        elif "ocean" in prompt.lower():
            simulated_continuation = " the sound of the waves and the salty air. There is nothing more peaceful than..."
        else:
            simulated_continuation = " that you get to experience new things every day. For example, yesterday I learned how to..."
            
        st.markdown("#### Generated Continuation:")
        st.info(prompt + simulated_continuation)
