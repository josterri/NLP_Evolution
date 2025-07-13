import streamlit as st

def render_6_3():
    """Renders the In-Context Learning section."""
    st.subheader("6.3: The Emergence of In-Context Learning")
    
    st.subheader("Motivation: A Surprising New Skill")
    st.markdown("""
    Training a model on the simple task of next-word prediction at a massive scale led to a surprising and magical new capability that researchers didn't explicitly design: **In-Context Learning**.

    This is the model's ability to perform tasks it was never specifically trained for, without any changes to its internal weights or parameters. It learns to recognize a pattern or task *from the prompt alone* and then continues that pattern to provide an answer. This was a revolutionary discovery that changed how we interact with language models.
    """)

    st.subheader("🧠 The Method: Pattern Recognition, Not Re-training")
    st.markdown("""
    When you give a large generative model a prompt, it isn't "learning" in the traditional sense of updating its neural network. Instead, it's using its vast pre-trained knowledge to perform sophisticated pattern matching.

    -   **Zero-Shot Learning:** You ask the model to perform a task directly, without giving it any examples. The model relies on patterns it has seen on the internet. For example, if it has seen thousands of web pages that have a title and then a summary, you can prompt it with "Summarize this text for me:" and it will recognize the pattern and generate a summary.

    -   **Few-Shot Learning:** This is even more powerful. You give the model a few examples of the task within the prompt itself. This gives the model a much clearer, more immediate pattern to follow, dramatically improving its performance. The model sees the examples, understands the "game" it's supposed to play, and applies that game to your new query.
    """)

    st.subheader("🛠️ Interactive Demo: Building a Few-Shot Prompt")
    st.markdown("Let's see how you would structure a prompt to teach a model a new, made-up task: classifying sentences as 'Formal' or 'Casual'.")

    st.markdown("#### Step 1: Define the Task")
    task_description = "Classify the tone of the following sentences as Formal or Casual."
    st.code(task_description, language='text')

    st.markdown("#### Step 2: Provide Examples (the 'Shots')")
    c1, c2 = st.columns(2)
    with c1:
        st.text("Example 1 Input:")
        st.info("hey what's up")
    with c2:
        st.text("Example 1 Label:")
        st.success("Casual")
    
    c1, c2 = st.columns(2)
    with c1:
        st.text("Example 2 Input:")
        st.info("per our previous correspondence")
    with c2:
        st.text("Example 2 Label:")
        st.success("Formal")

    st.markdown("#### Step 3: Provide the Query")
    query = st.text_input("Enter your sentence to classify:", "let's grab a bite to eat")
    st.code(f"Sentence: {query}\nClassification:")

    if st.button("Simulate Few-Shot Classification"):
        # Simple keyword-based simulation
        casual_words = {'hey', 'what\'s', 'up', 'grab', 'bite'}
        formal_words = {'correspondence', 'sincerely', 'regards'}
        
        is_casual = any(word in query.lower() for word in casual_words)
        is_formal = any(word in query.lower() for word in formal_words)

        st.markdown("---")
        st.subheader("Simulated Model Response")
        if is_casual and not is_formal:
            st.success("Casual")
            st.caption("The model recognized casual words from the query and followed the pattern from your examples.")
        elif is_formal and not is_casual:
            st.success("Formal")
            st.caption("The model recognized formal words from the query and followed the pattern from your examples.")
        else:
            st.warning("Uncertain")
            st.caption("The model did not find strong signals in the query to confidently match the pattern.")

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **One-Shot vs. Few-Shot:** What is the difference between "one-shot" and "few-shot" learning? How would you change the prompt above to be a one-shot prompt?
    2.  **Bad Examples:** What might happen if you provided bad or contradictory examples in a few-shot prompt? For instance, labeling "hey what's up" as "Formal"?
    3.  **Task Complexity:** Why is few-shot learning generally more effective for complex or unusual tasks compared to zero-shot learning?
    """)

    st.subheader("🐍 The Python Behind the Prompt")
    with st.expander("Show the Python Code for Constructing a Few-Shot Prompt"):
        st.code("""
def create_few_shot_prompt(task_description, examples, query):
    \"\"\"
    Constructs a text prompt for a few-shot learning task.
    \"\"\"
    prompt = task_description + "\\n\\n"
    
    # Add each example to the prompt
    for example in examples:
        prompt += f"Input: {example['input']}\\n"
        prompt += f"Output: {example['output']}\\n\\n"
        
    # Add the final query
    prompt += f"Input: {query}\\n"
    prompt += "Output:"
    
    return prompt

# --- Example ---
description = "Translate English to French."
example_pairs = [
    {"input": "hello", "output": "bonjour"},
    {"input": "goodbye", "output": "au revoir"}
]
final_query = "cat"

final_prompt = create_few_shot_prompt(description, example_pairs, final_query)
print(final_prompt)
# The model would receive this full text and its task is to predict
# the next word, which should be "chat".
        """, language='python')
