import streamlit as st

def render_3_2():
    """Renders content for section 3.2."""
    st.subheader("3.2 A Sequential Solution: ELMo")
    st.info("Content for this section is coming soon!")
    st.markdown("""
    The first major breakthrough in solving the context problem was a model called **ELMo (Embeddings from Language Models)**.

    **The Core Idea:** Instead of a fixed dictionary, ELMo uses a deep, two-layer Recurrent Neural Network (RNN) to process the sentence. It reads the sentence from left-to-right and from right-to-left. The final embedding for a word is a combination of its static embedding plus the "hidden states" from this RNN.

    This means the vector for "bank" is influenced by all the words that came before and after it, finally giving it the context it needs.
    """)
