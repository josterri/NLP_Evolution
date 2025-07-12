import streamlit as st
import random
import time

def render_2_8():
    """Renders the word prediction demo section."""
    st.subheader("2.8: Demo - Using Embeddings to Predict a Sequence")
    st.markdown("""
    While Word2Vec is not a true language model (it doesn't understand grammar or long-range dependencies), we can use the learned vector space to perform a simple form of "generative" text prediction.

    The idea is to start with a sequence of words, look at the last word, and find other words in the vector space that are semantically close to it. We can then choose one of these similar words as the next word in the sequence. This method is a basic form of a **greedy search**, guided by semantic similarity.
    """)

    if 'w2v_model' not in st.session_state:
        st.warning("Please train a model in section '2.6: Demo - Word2Vec in Action' before using this demo.")
        return

    model = st.session_state.w2v_model
    st.success("Word2Vec model from section 2.6 is loaded and ready!")

    st.markdown("---")
    st.subheader("Generate a Sequence")
    
    # Provide a default seed from the model's vocabulary
    default_seed = " ".join(model.wv.index_to_key[:5]) if len(model.wv.index_to_key) >= 5 else ""
    
    seed_text = st.text_input("Enter the first five words to start the sequence:", value=default_seed)
    num_words_to_generate = st.slider("Number of words to generate:", 5, 50, 10)

    if st.button("Generate Next Words"):
        generated_words = seed_text.lower().split()
        
        st.markdown("---")
        st.subheader("Generation Process Visualization")
        st.markdown("Watch how the sequence is built step-by-step.")
        
        placeholder = st.empty()
        
        for i in range(num_words_to_generate):
            with placeholder.container():
                st.markdown(f"**Step {i+1} of {num_words_to_generate}**")
                cols = st.columns([3, 1, 2, 1])
                with cols[0]:
                    st.write("Current Sequence (last 5 words):")
                    st.info(f"...{' '.join(generated_words[-5:])}")
                
                try:
                    last_word = generated_words[-1]
                    with cols[1]:
                        st.write("Context Word:")
                        st.success(f"`{last_word}`")

                    if last_word in model.wv:
                        similar_words = model.wv.most_similar(last_word, topn=5)
                        with cols[2]:
                            st.write("Top Similar Candidates:")
                            st.json([word for word, sim in similar_words])
                        
                        next_word = random.choice([word for word, sim in similar_words])
                        generated_words.append(next_word)

                        with cols[3]:
                            st.write("Chosen Word:")
                            st.success(f"`{next_word}`")
                    else:
                        generated_words.append("[UNKNOWN]")
                        st.warning(f"Word '{last_word}' not in vocabulary. Stopping.")
                        break
                except Exception:
                    st.error("Could not find similar words. Stopping.")
                    break
                
                time.sleep(0.5) # Pause for visualization

        st.markdown("---")
        st.markdown("#### Final Generated Text:")
        st.info(" ".join(generated_words))

    st.markdown("---")
    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Repetitive Loops:** Try generating a long sequence (e.g., 50 words). Do you notice the text getting stuck in loops (e.g., "financial -> stability -> financial -> stability")? Why do you think this happens with our simple method?
    2.  **Impact of `topn`:** In our code, we choose from the top 5 similar words. How would the generated text change if we only chose from the single most similar word (`topn=1`)? What if we chose from the top 20 (`topn=20`)?
    3.  **Seed Importance:** Try starting with a different seed text. How drastically does the generated text change? What does this tell you about the limitations of this approach for coherent text generation?
    """)

    st.subheader("🐍 The Python Behind the Generation")
    with st.expander("Show the Python Code for Sequence Generation"):
        st.code("""
import random

def generate_sequence(model, seed_words, num_to_generate):
    generated_words = list(seed_words) # Start with the seed
    
    for _ in range(num_to_generate):
        try:
            last_word = generated_words[-1]
            if last_word in model.wv:
                # Find top 5 similar words
                similar_words = model.wv.most_similar(last_word, topn=5)
                # Choose one of the candidates randomly
                next_word = random.choice([word for word, sim in similar_words])
                generated_words.append(next_word)
            else:
                # Stop if we hit a word not in the vocabulary
                break
        except Exception:
            # Stop on any error (e.g., no similar words found)
            break
            
    return " ".join(generated_words)

# --- Example ---
# Assume 'model' is a trained Word2Vec model from section 2.6
# seed = ["financial", "stability", "is", "a", "goal"]
# full_text = generate_sequence(model, seed, 20)
# print(full_text)
        """, language='python')
