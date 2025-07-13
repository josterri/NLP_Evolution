import streamlit as st
import pandas as pd

def render_7_1():
    """Renders the Data and Tokenizer section."""
    st.subheader("7.1: Step 1 - The Data and the Tokenizer")
    
    st.subheader("Motivation: The Foundation of Everything")
    st.markdown("""
    Every language model, no matter how large or complex, begins with two fundamental components that form its foundation:
    1.  **The Data (Corpus):** The text that the model will learn from. This is its entire "worldview". The patterns, facts, biases, and style of the training data will be directly reflected in the model's behavior. The principle of "garbage in, garbage out" is paramount here.
    2.  **The Tokenizer:** A translator that converts raw text into a numerical format that the model can understand, and vice-versa. Models don't see words or characters; they only see numbers. The tokenizer is the crucial bridge between human language and the model's mathematical world.

    For our nano-GPT, we will keep things simple. Our "corpus" will be a small text file that you can provide, and our tokenizer will be a **character-level tokenizer**.
    """)

    st.subheader("🧠 The Method: Character-Level Tokenization")
    st.markdown("""
    Instead of treating words as the basic unit (like "cat", "sat"), we will treat individual characters as our "tokens" (like 'c', 'a', 't'). This has a major advantage for a simple model: the **vocabulary size is very small**. The vocabulary will consist of every unique character in our text file (e.g., 'a' through 'z', punctuation, spaces, etc.).

    **Pros of Character-level Tokenization:**
    -   **Small Vocabulary:** The number of unique characters is tiny compared to the number of unique words, making the model smaller and faster to train.
    -   **No "Unknown" Tokens:** The model can represent any word, even misspellings or new words, because they are just combinations of known characters. A word-level model would fail if it saw a word not in its vocabulary.

    **Cons of Character-level Tokenization:**
    -   **Longer Sequences:** The sequence of tokens for a sentence is much longer (e.g., "hello" is 5 character tokens but 1 word token). This requires more computational steps.
    -   **Less Inherent Meaning:** The model has to learn the concept of "words" from scratch by looking at patterns of characters, which is a harder task than starting with words.

    The process involves two key steps:
    1.  **Build the Vocabulary:** We read our text data and find every unique character that appears.
    2.  **Create Mappings:** We create two "lookup tables" or dictionaries:
        -   `char_to_int`: Maps each unique character to a unique integer (e.g., 'h' -> 4, 'e' -> 1, 'l' -> 2, 'o' -> 5).
        -   `int_to_char`: The inverse map, which maps each integer back to its character (e.g., 4 -> 'h', 1 -> 'e').

    The `encode` function will use `char_to_int` to convert a string into a list of integers. The `decode` function will use `int_to_char` to convert a list of integers back into a string.
    """)

    st.subheader("🛠️ Interactive Demo: Building a Tokenizer")
    st.markdown("Enter some text below to see how a character-level vocabulary and tokenizer are created from it.")

    text_input = st.text_area("Enter text to build the tokenizer from:", "hello world! this is a test.")
    
    if text_input:
        # Build vocabulary
        chars = sorted(list(set(text_input)))
        vocab_size = len(chars)
        
        # Create mappings
        char_to_int = {ch: i for i, ch in enumerate(chars)}
        int_to_char = {i: ch for i, ch in enumerate(chars)}

        st.write(f"**Vocabulary Size:** {vocab_size}")
        st.write(f"**Vocabulary:** `{''.join(chars)}`")

        st.markdown("---")
        st.write("Test the tokenizer:")
        string_to_test = st.text_input("Enter a string to encode/decode:", "hello")

        if string_to_test:
            # Encode
            try:
                encoded = [char_to_int[c] for c in string_to_test]
                st.write("**Encoded:**")
                st.code(str(encoded))

                # Decode
                decoded = "".join([int_to_char[i] for i in encoded])
                st.write("**Decoded:**")
                st.code(decoded)
            except KeyError as e:
                st.error(f"Error: The character '{e.args[0]}' in your test string is not in the vocabulary built from the text above. Please only use characters from the vocabulary.")

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **Case Sensitivity:** In our current setup, are 'H' and 'h' treated as the same token? How would you change the code to make them the same?
    2.  **Punctuation:** Is punctuation like '.' or '!' part of our vocabulary? What is the advantage or disadvantage of this?
    3.  **Word-level vs. Character-level:** If your corpus was "The cat sat. The dog sat.", what would the vocabulary size be for a word-level tokenizer versus a character-level tokenizer?
    """)

    st.subheader("🐍 The Python Code")
    st.markdown("Below is a simple, self-contained Python script that you can run. It reads a text file (`input.txt`), builds the tokenizer, and demonstrates the encode/decode functions.")
    
    with st.expander("Show the full Python code for the Tokenizer"):
        st.code("""
# We recommend creating a file named 'input.txt' in the same folder
# and pasting some text into it (e.g., a short story or poem).

# --- 1. Read the text data ---
try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Successfully loaded input.txt. Length of dataset in characters: {len(text)}")
except FileNotFoundError:
    print("'input.txt' not found. Please create this file.")
    text = "hello world" # Default text if file is not found

# --- 2. Build the Vocabulary ---
# Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary: {''.join(chars)}")
print(f"Vocabulary size: {vocab_size}")

# --- 3. Create the Mappings (Tokenizer) ---
# Create a mapping from characters to integers
char_to_int = {ch: i for i, ch in enumerate(chars)}
# Create a mapping from integers to characters
int_to_char = {i: ch for i, ch in enumerate(chars)}

# --- 4. Define Encode and Decode Functions ---
def encode(string):
    # Encoder: take a string, output a list of integers
    return [char_to_int[c] for c in string]

def decode(integer_list):
    # Decoder: take a list of integers, output a string
    return ''.join([int_to_char[i] for i in integer_list])

# --- 5. Demonstrate the Tokenizer ---
test_string = "hello there"
encoded_string = encode(test_string)
decoded_string = decode(encoded_string)

print(f"\\nOriginal string: {test_string}")
print(f"Encoded representation: {encoded_string}")
print(f"Decoded string: {decoded_string}")
        """, language='python')
