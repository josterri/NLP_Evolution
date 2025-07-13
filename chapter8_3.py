# chapter8_0.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def render_8_0():
    """Renders the 'Cambrian Explosion' section."""
    st.subheader("8.0: The Cambrian Explosion - When Scale Creates Magic")
    st.markdown("""
    Our journey has taken us from simple counting to building a full Transformer. The final and most profound leap in this story is not a new algorithm, but a change in philosophy: **what happens when you scale these models to unimaginable sizes?**

    Researchers discovered that as you dramatically increase the scale (more data, more parameters, more compute), models don't just get incrementally better at next-word prediction. They begin to develop entirely new, unpredictable skills that they were never explicitly trained for. This is the concept of **emergent abilities**.
    """)

    st.subheader("The Analogy: From Ant to Anthill")
    st.markdown("""
    -   A single ant is a simple biological machine. It follows basic rules and has very limited intelligence.
    -   An anthill, a colony of millions of ants, exhibits incredibly complex, intelligent behavior. It can farm, build complex structures, and solve problems.

    This "colony intelligence" is an **emergent property**. It doesn't exist in any single ant; it emerges from the complex interactions of millions of simple components.

    Large Language Models are similar. The simple task of next-word prediction, when scaled up with billions of parameters and trillions of words, causes complex abilities like reasoning, translation, and even coding to emerge.
    """)

    # --- Visualization of Emergent Abilities ---
    st.subheader("Visualizing Emergent Abilities")
    fig, ax = plt.subplots()
    
    # Model scale
    scale = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # Performance on a simple task (e.g., grammar)
    performance_simple = 1 / (1 + np.exp(-2 * (scale - 3))) 
    # Performance on a complex task (e.g., reasoning)
    performance_complex = 1 / (1 + np.exp(-2 * (scale - 6)))

    ax.plot(scale, performance_simple, marker='o', linestyle='--', label='Simple Task (e.g., Grammar)')
    ax.plot(scale, performance_complex, marker='o', linestyle='--', label='Complex Task (e.g., Reasoning)')
    
    ax.set_title("Emergent Abilities vs. Model Scale")
    ax.set_xlabel("Model Scale (Parameters, Data, Compute)")
    ax.set_ylabel("Performance")
    ax.axvline(x=5.5, color='r', linestyle=':', label='Phase Transition Point')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.caption("Notice how performance on complex tasks is near zero for smaller models, and then suddenly 'takes off' at a certain scale. This is an emergent ability.")

# -------------------------------------------------------------------

# chapter8_1.py
import streamlit as st

def render_8_1():
    """Renders the Modern Landscape section."""
    st.subheader("8.1: The Modern Landscape")
    st.markdown("""
    The era of LLMs is characterized by a rapid proliferation of powerful models from various research labs and companies. While they are all based on the Transformer architecture, they have different strengths and specializations.
    """)

    st.subheader("Major Model Families")
    st.markdown("""
    -   **The GPT Family (OpenAI):** The models that brought LLMs into the public consciousness (GPT-3, GPT-3.5, GPT-4). They are known for their strong general reasoning, instruction following, and creative generation capabilities.
    -   **Gemini (Google):** A family of models designed from the ground up to be **multimodal**. This means they are not just trained on text, but on a vast dataset of text, images, audio, and video simultaneously. This allows them to natively understand and process information across different modalities.
    -   **Llama & Open Source Models (Meta, Mistral, etc.):** A growing movement of powerful open-source models. While often slightly smaller than the largest proprietary models, they provide a crucial platform for researchers and developers to build upon, inspect, and understand these complex systems.
    """)

    st.subheader("The Shift to Multimodality")
    st.markdown("""
    The most significant recent trend is the move towards **multimodality**. The world is not just text, and the next generation of models reflects this.
    
    A multimodal model can perform tasks like:
    -   Seeing a picture of a bridge and writing a poem about it.
    -   Listening to a recording of a meeting and generating a text summary.
    -   Looking at a chart and explaining the trends in plain English.

    This is achieved by training the model not just to predict the next word, but also to predict the next image patch or the next sound segment, all within a unified architecture.
    """)

# -------------------------------------------------------------------

# chapter8_2.py
import streamlit as st

def render_8_2():
    """Renders the Future & Frontiers section."""
    st.subheader("8.2: The Future & Frontiers of NLP")
    st.markdown("""
    Congratulations on completing this journey through the evolution of NLP! You now have the foundational knowledge to understand how we got here. But the journey is far from over. The field is moving at an incredible pace, and several major challenges and frontiers lie ahead.
    """)

    st.subheader("Current Challenges and Research Frontiers")
    st.markdown("""
    -   **Reasoning & Planning:** While LLMs are good at pattern matching, they still struggle with complex, multi-step reasoning. Improving their ability to plan, reason logically, and perform causal inference is a major area of research.
    -   **Hallucinations & Factuality:** LLMs are trained to generate plausible text, not necessarily truthful text. They can "hallucinate" facts, sources, and citations with great confidence. Developing methods to ground models in verifiable knowledge and reduce hallucination is a critical safety issue.
    -   **Efficiency & Cost:** The trend of "bigger is better" is not sustainable forever. A huge area of research is focused on **model distillation** and **quantization**—techniques to create smaller, faster, and cheaper models that retain the capabilities of their larger counterparts.
    -   **Personalization & Continual Learning:** How can a model continually learn from new information or a user's personal context without needing to be completely retrained? Developing models that can adapt and learn over time is a key frontier.
    -   **Explainability & Interpretability:** We have built incredibly powerful models, but we don't always fully understand *how* they arrive at their answers. The field of interpretability aims to peek inside the "black box" to understand the model's decision-making process.
    """)
    
    st.success("""
    **Conclusion:** You have witnessed the evolution from simple word counting to complex, emergent intelligence. The journey from N-grams to Transformers is a story of overcoming limitations and building on previous ideas. The same principles will undoubtedly drive the next generation of breakthroughs in artificial intelligence.
    """)
