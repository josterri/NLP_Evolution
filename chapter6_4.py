import streamlit as st
import random
import time
import pandas as pd

def render_6_4():
    """Renders the interactive generation workbench."""
    st.subheader("6.4: Interactive Generation Workbench")
    
    st.subheader("Motivation: Seeing the Model Think")
    st.markdown("""
    This workbench brings together all the concepts from this chapter. We will simulate how a generative, decoder-only model uses its pre-trained knowledge to continue a piece of text you provide.

    The key thing to remember is that at each step, the model isn't just picking one word. It's calculating a probability score for *every single word* in its vocabulary. Our goal here is to peek inside that process.
    """)

    st.subheader("🧠 The Method: Sampling Strategies")
    st.markdown("""
    If the model always picked the single word with the highest probability (a method called **Greedy Search**), its text would be very boring, deterministic, and repetitive. To make the output more creative and human-like, we use sampling strategies:

    -   **Top-k Sampling:** Instead of just taking the top 1 word, we consider the top `k` (e.g., top 50) most likely words and choose one randomly from that smaller group.
    -   **Nucleus (Top-p) Sampling:** We choose from the smallest possible set of words whose cumulative probability exceeds a certain threshold `p`.

    For our demo, we'll use a simplified version of Top-k sampling to add a bit of randomness.
    """)

    st.subheader("🛠️ Interactive Demo: Generate Text Step-by-Step")
    st.markdown("Provide a prompt and see how the model generates the next words one by one.")

    prompt = st.text_area("Enter your prompt:", "The best thing about living in the mountains is")
    num_words = st.slider("Number of words to generate:", 1, 20, 5)

    if st.button("Generate Text"):
        generated_text = prompt
        placeholder = st.empty()

        for i in range(num_words):
            with placeholder.container():
                st.markdown(f"#### Generating word #{i+1}...")
                st.info(f"**Current Text:** {generated_text}")

                # This is a highly simplified simulation.
                # A real model's logic is far more complex.
                last_word = generated_text.split()[-1]
                
                # Simulate a vocabulary and probability distribution
                vocab = ["the", "is", "and", "a", "views", "air", "fresh", "beautiful", "peaceful", "sound", "of", "waves"]
                if "mountains" in generated_text.lower():
                    # Higher probability for mountain-related words
                    probabilities = [0.1, 0.1, 0.1, 0.1, 0.3, 0.4, 0.4, 0.3, 0.2, 0.05, 0.1, 0.05]
                elif "ocean" in generated_text.lower():
                    # Higher probability for ocean-related words
                    probabilities = [0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.2, 0.4, 0.1, 0.4]
                else:
                    probabilities = [1] * len(vocab) # Equal probability
                
                # Normalize probabilities
                probabilities = [p / sum(probabilities) for p in probabilities]

                # Get top 5 candidates
                top_k_indices = sorted(range(len(vocab)), key=lambda k: probabilities[k], reverse=True)[:5]
                top_k_words = [vocab[i] for i in top_k_indices]
                top_k_probs = [probabilities[i] for i in top_k_indices]

                st.write("**Top 5 Predicted Next Words (Simulated):**")
                df = pd.DataFrame({'Word': top_k_words, 'Probability': top_k_probs})
                st.dataframe(df.style.format({'Probability': '{:.2%}'}))

                # Choose the next word from the top candidates
                next_word = random.choice(top_k_words)
                st.success(f"**Word Chosen:** {next_word}")
                
                generated_text += " " + next_word
                time.sleep(1.5)

        st.markdown("---")
        st.subheader("Final Result")
        st.info(generated_text)

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Prompt Engineering:** How does changing the end of your prompt affect the generated text? For example, compare the output of "The best thing about mountains is" versus "The worst thing about mountains is".
    2.  **Greedy vs. Sampling:** If our simulation always picked the word with the highest probability (Greedy Search), how would the generated text change? Would it be more or less creative?
    3.  **Limitations:** Even with a real large language model, the generated text can sometimes be factually incorrect or nonsensical. Why do you think this happens, even if the model has been trained on a vast amount of factual text?
    """)

    st.subheader("🐍 The Python Behind the Generation Loop")
    with st.expander("Show the Python Code for Autoregressive Generation"):
        st.code("""
import random

def generate_text(model, tokenizer, prompt, num_words_to_generate):
    \"\"\"
    Generates text autoregressively using a pre-trained model.
    This is a conceptual example. Real implementation uses model-specific methods.
    \"\"\"
    # 1. Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt") # pt for PyTorch
    
    generated_ids = list(input_ids[0])

    for _ in range(num_words_to_generate):
        # 2. Get the model's prediction for the next token
        # The model outputs 'logits', which are raw scores for every word in the vocabulary
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :] # Get logits for the last token
        
        # 3. Apply a sampling strategy (e.g., top-k)
        # Get the top k most likely token IDs
        top_k_logits, top_k_indices = torch.topk(logits, k=50)
        
        # Convert logits to probabilities and sample from them
        probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
        predicted_token_index_in_top_k = torch.multinomial(probabilities, num_samples=1)
        predicted_token_id = top_k_indices[predicted_token_index_in_top_k]
        
        # 4. Append the new token and repeat
        generated_ids.append(predicted_token_id.item())
        input_ids = torch.tensor([generated_ids])

    # 5. Decode the final list of IDs back to a string
    return tokenizer.decode(generated_ids)

# --- Example ---
# This requires a real model and tokenizer from Hugging Face
# from transformers import AutoModelForCausalLM, AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("gpt2")
# generated_text = generate_text(model, tokenizer, "Hello, my name is", 10)
# print(generated_text)
        """, language='python')
