import streamlit as st
import pandas as pd
import numpy as np

def render_2_10():
    """Renders the simple analogy for Word2Vec."""
    st.subheader("2.10: A Simple Analogy for Word2Vec")
    st.markdown("""
    After all the technical details, let's put everything together with a simple story. How can a machine that doesn't understand language learn the meaning of words?
    """)

    st.subheader("The Starting Point: An Amnesiac Librarian")
    st.markdown("""
    Imagine a librarian who has lost their memory and doesn't know how to read. They are faced with two things:
    1.  **A Messy Library (The Training Data):** This is a massive, unorganized collection of books (our text corpus). The only useful information here is knowing which books were found next to which other books.
    2.  **A Big, Empty Shelf (The Vector Space):** This is where the books will be organized. To start, the librarian takes every unique book title from the messy library and places it at a completely random position on the empty shelf.

    The librarian's one and only instruction is this: **"Place books that are about similar topics close to each other on the shelf."**

    The librarian knows nothing about the topics themselves, but they *can* use the information from the messy library (which books were neighbors) to figure it out.
    """)

    st.subheader("A Step-by-Step Example")
    st.markdown("""
    Let's trace the process with a tiny library of just three words: **"king queen throne"**.
    - The librarian notes the neighboring pairs: `(king, queen)` and `(queen, throne)`.
    - They place the three books at random spots on the 2D shelf.

    **Initial Random Positions:**
    ```
    - king:   (x=9, y=2)
    - queen:  (x=1, y=8)
    - throne: (x=8, y=9)
    ```
    As you can see, the positions are meaningless. Now, the librarian begins their work.
    
    **Adjustment Step 1:**
    - **Pick a pair:** The librarian picks the first pair from their notes: `(king, queen)`.
    - **Find the error:** They see "king" is at `(9, 2)` and "queen" is at `(1, 8)`. They are very far apart!
    - **Nudge:** To fix this, they nudge both books slightly closer to each other. Let's say they move each book 10% of the way towards the other.
    
    The new positions might be:
    ```
    - king:   (x=8.2, y=2.6)  <-- Moved closer to queen
    - queen:  (x=1.8, y=7.4)  <-- Moved closer to king
    - throne: (x=8, y=9)      <-- Unchanged this step
    ```

    **Adjustment Step 2:**
    - **Pick the next pair:** The librarian picks `(queen, throne)`.
    - **Find the error:** They look at the *new* positions. "queen" is at `(1.8, 7.4)` and "throne" is at `(8, 9)`. Still far apart!
    - **Nudge:** They nudge "queen" and "throne" slightly closer to each other.

    The positions evolve again:
    ```
    - king:   (x=8.2, y=2.6)
    - queen:  (x=2.4, y=7.6)  <-- Nudged again, this time towards throne
    - throne: (x=7.4, y=8.8)  <-- Nudged towards queen
    ```
    After just two steps, you can already see that "queen" and "throne" have started to form a cluster, separate from "king". If we were to process the `(king, queen)` pair again, "king" would be pulled even closer to this new cluster. When this process is repeated millions of times with millions of books, a rich and meaningful structure emerges from the chaos.""")

    st.subheader("The Librarian's Method (The Skip-Gram Model)")
    st.markdown("""
    The librarian (our Word2Vec model) decides on a simple method to follow the instruction:
    1.  **Pick a Book:** They pick up a random book from the shelf, let's say one with the title "King".
    2.  **Look at its Neighbors (in the original text):** They consult their notes from the messy library to see which books were found next to "King". They see books titled "Queen", "Throne", and "Palace".
    3.  **Make an Adjustment:** The librarian thinks, "Okay, 'King' seems to belong near 'Queen', 'Throne', and 'Palace'." 
        
        **But how does this work?** The librarian looks at the current positions on the shelf. They see that the "King" book is far away from the "Queen" book (because all books started in random positions). This is a **prediction error**: based on its neighbors in the original text, "King" *should* be close to "Queen", but its current position on the shelf suggests it isn't. To fix this error, the librarian takes the "King" book and nudges it a tiny bit closer to where the "Queen" book is. They do the same for "Throne" and "Palace". This "nudging" is a physical representation of adjusting the word's vector to minimize the prediction error.

    4.  **Repeat, Millions of Times:** They repeat this process for every single book, millions of times. They pick up "Apple" and see it's near "Fruit" and "Orange", so they nudge "Apple" closer to them. They pick up "Dog" and see it's near "Puppy" and "Cat", so they nudge it in that direction.
    """)

    st.subheader("The Result: An Organized Shelf of Meaning")
    st.markdown("""
    After a very long time, something magical happens. The shelf is no longer random.
    - All the books about royalty (`king`, `queen`, `prince`) have ended up in one section.
    - All the books about fruits (`apple`, `orange`, `banana`) are in another section.
    - The *position* of a book on the shelf has come to represent its meaning.

    This position is the **word embedding**. It's just a set of coordinates (a vector), but it's incredibly powerful because it captures the relationships between all the words.
    """)
    
    st.success("""
    **The Key Takeaway:** Word2Vec doesn't learn the *definition* of a word. It learns a word's meaning by learning its *relationships* to all other words, based on the contexts in which it appears.
    """)

    st.subheader("✏️ Exercises")
    st.markdown("""
    1.  **The Analogy:** In our story, what do the following represent?
        - The Librarian
        - The Big, Empty Shelf
        - The Position of a Book
        - The "Nudging" Process
    2.  **A New Book:** If the librarian gets a new book titled "Corgi", where on the shelf would they likely place it first? Near "King" and "Queen", or near "Dog" and "Cat"? Why?
    """)
