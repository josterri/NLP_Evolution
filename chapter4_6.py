import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_4_6():
    """Renders the 'Why' behind the math section."""
    st.subheader("4.6: The 'Why' Behind the Math")
    st.markdown("""
    We've seen the steps of the attention calculation, but *why* does it work this way? Why use vectors and dot products? How does this complex process actually help us predict the next word? Let's break it down.
    """)

    st.subheader("Why Vectors? A Language of Directions")
    st.markdown("""
    As we saw in Chapter 2, vectors allow us to represent abstract concepts (words) in a geometric space. The power of this representation is that the **direction** of the vector holds meaning.
    - A vector pointing towards "royalty" is different from one pointing towards "animals".
    - The difference between the "king" and "queen" vectors captures the concept of gender.

    By using vectors for our Query, Key, and Value, we are giving the model a mathematical language to express and compare the nuanced meanings and relationships between words.
    """)

    st.subheader("Why the Dot Product? The Geometry of Similarity")
    st.markdown("""
    The dot product is the most important operation in the attention mechanism. Its power comes from its geometric interpretation:
    $$ A \cdot B = ||A|| \ ||B|| \cos(\theta) $$
    Where `θ` is the angle between the two vectors.

    - If two vectors `A` and `B` point in the **same direction**, the angle `θ` is 0, and `cos(0) = 1`. The dot product is at its maximum. This means they are **highly similar**.
    - If they are **perpendicular**, `θ` is 90°, and `cos(90) = 0`. The dot product is zero. This means they are **unrelated**.

    So, when we calculate `Score = Query • Key`, we are literally asking: **"How aligned is my question (Query) with this word's advertisement (Key)?"** The dot product gives us a natural, efficient way to measure this alignment or similarity.
    """)

    # --- Interactive Visualization of Dot Product ---
    st.subheader("🛠️ Interactive Demo: The Geometry of Dot Product")
    st.markdown("Use the sliders to change the angle of the two vectors and see how the dot product (their similarity score) changes.")

    c1, c2 = st.columns(2)
    with c1:
        angle_a = st.slider("Angle of Vector A (Query)", 0, 360, 45)
    with c2:
        angle_b = st.slider("Angle of Vector B (Key)", 0, 360, 60)

    # Convert angles to radians for numpy functions
    rad_a = np.deg2rad(angle_a)
    rad_b = np.deg2rad(angle_b)

    # Define vectors on a unit circle
    vec_a = np.array([np.cos(rad_a), np.sin(rad_a)])
    vec_b = np.array([np.cos(rad_b), np.sin(rad_b)])

    # Calculate dot product
    dot_product = np.dot(vec_a, vec_b)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Vector Alignment")
    ax.arrow(0, 0, vec_a[0], vec_a[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue', length_includes_head=True)
    ax.text(vec_a[0] * 1.2, vec_a[1] * 1.2, "A (Query)", color='blue', ha='center', va='center')
    ax.arrow(0, 0, vec_b[0], vec_b[1], head_width=0.1, head_length=0.1, fc='red', ec='red', length_includes_head=True)
    ax.text(vec_b[0] * 1.2, vec_b[1] * 1.2, "B (Key)", color='red', ha='center', va='center')
    
    # Draw unit circle for reference
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    ax.add_artist(circle)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    st.pyplot(fig)

    st.metric(label="Dot Product (Similarity Score)", value=f"{dot_product:.2f}")
    if dot_product > 0.7:
        st.success("The vectors are highly aligned, indicating high similarity.")
    elif dot_product < -0.7:
        st.error("The vectors are pointing in opposite directions, indicating dissimilarity.")
    else:
        st.info("The vectors are not strongly aligned, indicating low similarity.")


    st.subheader("Connecting it All to Next-Word Prediction")
    st.markdown("""
    This is the final, crucial link. All the complex attention calculations have one single goal: **to produce a better, context-aware vector for each word.**

    1.  **The Input:** We start with a simple, static embedding for a word like "bank". This vector is ambiguous.
    2.  **The Attention Process:** The attention mechanism acts like a powerful blender. It takes the **Value** vectors of all the words in the sentence and mixes them together. The amount of each "ingredient" it adds is determined by the attention weights we calculated. If "river" has a high attention weight, a lot of the "river" Value vector gets mixed into the final output.
    3.  **The Output:** The result is a new, context-rich vector for "bank" that is now heavily "flavored" with the meaning of "river".
    4.  **The Prediction:** This final, context-aware vector is then passed to a simple feed-forward neural network. This network's only job is to take this rich vector and predict the most likely next word.

    In short, **Attention creates a superior input for the final prediction step.** It enriches the word's representation so that the prediction network has a much easier job.
    """)
