import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from collections import defaultdict
import random

def render_chapter_0():
    """Renders Chapter 0: Before Neural Networks - The Pre-Neural Era of NLP."""
    st.header("Chapter 0: Before Neural Networks - The Pre-Neural Era")
    
    st.markdown("""
    Before the neural network revolution transformed NLP, the field was dominated by rule-based systems 
    and early statistical methods. Understanding this history helps us appreciate why neural approaches 
    were such a breakthrough.
    """)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏛️ Early Rule-Based Systems",
        "🎯 Statistical Revolution", 
        "🔍 Information Retrieval",
        "📊 Early ML Methods",
        "💡 Why Neural Networks?"
    ])
    
    with tab1:
        render_rule_based_systems()
    
    with tab2:
        render_statistical_revolution()
        
    with tab3:
        render_information_retrieval()
        
    with tab4:
        render_early_ml_methods()
        
    with tab5:
        render_why_neural_networks()

def render_rule_based_systems():
    """Section on early rule-based NLP systems."""
    st.subheader("🏛️ The Era of Rules and Patterns (1950s-1980s)")
    
    st.markdown("""
    The earliest NLP systems relied on carefully crafted rules and patterns. These systems were 
    labor-intensive to build but showed that computers could process human language.
    """)
    
    # Timeline of early systems
    st.markdown("### 📅 Timeline of Early NLP Systems")
    
    timeline_data = [
        {"Year": 1954, "System": "Georgetown-IBM Experiment", "Type": "Machine Translation", 
         "Description": "First public demo of MT: 60 Russian sentences to English"},
        {"Year": 1966, "System": "ELIZA", "Type": "Chatbot", 
         "Description": "Pattern matching therapist simulation"},
        {"Year": 1970, "System": "SHRDLU", "Type": "Understanding", 
         "Description": "Natural language understanding in blocks world"},
        {"Year": 1975, "System": "LUNAR", "Type": "Question Answering", 
         "Description": "Q&A about moon rock samples"},
        {"Year": 1980, "System": "MYCIN", "Type": "Expert System", 
         "Description": "Medical diagnosis using if-then rules"}
    ]
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Create a more robust timeline visualization
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df_timeline["Year"],
        y=df_timeline["Type"],
        mode='markers+text',
        text=df_timeline["System"],
        textposition="top center",
        marker=dict(size=15, color='blue', opacity=0.7),
        hovertemplate="<b>%{text}</b><br>" +
                      "Year: %{x}<br>" +
                      "Type: %{y}<br>" +
                      "<extra></extra>",
        name=""
    ))
    
    fig.update_layout(
        title="Early NLP Systems Timeline",
        xaxis_title="Year",
        yaxis_title="System Type",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ELIZA Demo
    st.markdown("### 🤖 ELIZA: The First Chatbot (1966)")
    st.markdown("""
    ELIZA used pattern matching and substitution to simulate a Rogerian psychotherapist. 
    Despite its simplicity, many users were convinced they were talking to a real therapist!
    """)
    
    with st.expander("Try a Simplified ELIZA"):
        # Simple ELIZA patterns
        eliza_patterns = [
            (r".*\b(feel|feeling)\b.*", ["Tell me more about your feelings.", "Why do you feel that way?"]),
            (r".*\bmother\b.*", ["Tell me about your mother.", "How does your mother make you feel?"]),
            (r".*\bfather\b.*", ["Tell me about your father.", "What is your relationship with your father like?"]),
            (r".*\b(yes|yeah|yep)\b.*", ["You seem quite certain.", "Can you elaborate on that?"]),
            (r".*\b(no|nope|not)\b.*", ["Why not?", "Are you sure?", "Can you explain why not?"]),
            (r".*\balways\b.*", ["Can you think of a specific example?", "Really, always?"]),
            (r".*\bnever\b.*", ["Never?", "Are you sure you've never...?"]),
            (r"I am (.*)", ["Why are you {0}?", "How long have you been {0}?"]),
            (r"I'm (.*)", ["Why are you {0}?", "What makes you {0}?"]),
            (r".*", ["Please go on.", "Tell me more.", "I see. Continue."])
        ]
        
        user_input = st.text_input("You:", placeholder="Tell me how you feel...")
        
        if user_input:
            # Find matching pattern
            response = None
            for pattern, responses in eliza_patterns:
                match = re.match(pattern, user_input.lower())
                if match:
                    response = random.choice(responses)
                    if "{0}" in response and match.groups():
                        response = response.format(match.group(1))
                    break
            
            if response:
                st.markdown(f"**ELIZA:** {response}")
    
    # Chomsky's contributions
    st.markdown("### 📚 Chomsky's Linguistic Theory")
    st.markdown("""
    Noam Chomsky revolutionized linguistics with his theory of generative grammar, arguing that 
    language has an underlying structure that can be described by formal rules.
    """)
    
    with st.expander("Context-Free Grammar Example"):
        st.code("""
        S → NP VP
        NP → Det N | Det Adj N
        VP → V | V NP
        Det → the | a
        N → cat | dog | ball
        Adj → big | small | red
        V → chased | caught | threw
        
        Example derivation:
        S → NP VP
          → Det N VP
          → the cat VP
          → the cat V NP
          → the cat chased NP
          → the cat chased Det N
          → the cat chased the ball
        """, language="text")
    
    # Limitations
    st.markdown("### ⚠️ Limitations of Rule-Based Approaches")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Challenges:**
        - 🔨 Labor-intensive rule creation
        - 🌍 Poor generalization
        - 🗣️ Couldn't handle real-world language variation
        - 📈 Didn't scale to large vocabularies
        """)
    
    with col2:
        st.markdown("""
        **Missing Capabilities:**
        - ❌ No learning from data
        - ❌ No handling of ambiguity
        - ❌ No robustness to errors
        - ❌ No semantic understanding
        """)

def render_statistical_revolution():
    """Section on the statistical revolution in NLP."""
    st.subheader("🎯 The Statistical Revolution (1980s-2000s)")
    
    st.markdown("""
    As computational power increased and annotated corpora became available, NLP shifted from 
    hand-crafted rules to data-driven statistical methods.
    """)
    
    # Hidden Markov Models
    st.markdown("### 🎲 Hidden Markov Models (HMMs)")
    st.markdown("""
    HMMs became the workhorse of early statistical NLP, particularly for sequence labeling tasks 
    like part-of-speech tagging and speech recognition.
    """)
    
    # Simple HMM visualization
    with st.expander("HMM for Part-of-Speech Tagging"):
        st.markdown("""
        An HMM models two types of probabilities:
        1. **Transition probabilities**: P(tag₂|tag₁)
        2. **Emission probabilities**: P(word|tag)
        """)
        
        # Example transition matrix
        tags = ["Noun", "Verb", "Det"]
        transitions = np.array([
            [0.3, 0.5, 0.2],  # From Noun
            [0.6, 0.1, 0.3],  # From Verb
            [0.9, 0.05, 0.05] # From Det
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=transitions,
            x=tags,
            y=tags,
            text=transitions,
            texttemplate="%{text:.2f}",
            colorscale="Blues"
        ))
        fig.update_layout(
            title="Transition Probabilities",
            xaxis_title="To Tag",
            yaxis_title="From Tag",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Maximum Entropy Models
    st.markdown("### 📊 Maximum Entropy Models")
    st.markdown("""
    MaxEnt models allowed incorporating multiple overlapping features, unlike the independence 
    assumptions of Naive Bayes.
    """)
    
    # CRFs
    st.markdown("### 🔗 Conditional Random Fields (CRFs)")
    st.markdown("""
    CRFs improved on HMMs by modeling the conditional probability of the entire sequence, 
    allowing for richer feature representations.
    """)
    
    # Show comparison
    comparison_data = {
        "Method": ["HMM", "MaxEnt", "CRF"],
        "Strengths": [
            "Simple, efficient, good for sequences",
            "Flexible features, no independence assumptions",
            "Global optimization, rich features"
        ],
        "Weaknesses": [
            "Strong independence assumptions",
            "Not inherently sequential",
            "Computationally expensive"
        ],
        "Best For": [
            "POS tagging, speech recognition",
            "Text classification",
            "Named entity recognition"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.table(df_comparison)
    
    # Penn Treebank
    st.markdown("### 📚 The Penn Treebank: Enabling Statistical NLP")
    st.markdown("""
    The Penn Treebank (1992) provided over 1 million words of annotated text, enabling 
    the training of statistical models at scale for the first time.
    """)
    
    with st.expander("Penn Treebank Annotation Example"):
        st.code("""
        Original: The cat sat on the mat.
        
        POS Tags: The/DT cat/NN sat/VBD on/IN the/DT mat/NN ./.
        
        Parse Tree:
        (S
          (NP (DT The) (NN cat))
          (VP (VBD sat)
            (PP (IN on)
              (NP (DT the) (NN mat))))
          (. .))
        """, language="text")

def render_information_retrieval():
    """Section on information retrieval methods."""
    st.subheader("🔍 Information Retrieval: Finding Needles in Haystacks")
    
    st.markdown("""
    Information Retrieval (IR) developed in parallel with NLP, focusing on finding relevant 
    documents in large collections. Many IR techniques became foundational for NLP.
    """)
    
    # TF-IDF
    st.markdown("### 📊 TF-IDF: The Workhorse of IR")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Term Frequency (TF):**
        - How often a term appears in a document
        - Assumption: Frequent terms are important
        """)
        st.latex(r"TF(t,d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}")
    
    with col2:
        st.markdown("""
        **Inverse Document Frequency (IDF):**
        - How rare a term is across all documents
        - Assumption: Rare terms are discriminative
        """)
        st.latex(r"IDF(t,D) = \log\frac{|D|}{|\{d \in D: t \in d\}|}")
    
    # Interactive TF-IDF demo
    st.markdown("### 🧮 Interactive TF-IDF Calculator")
    
    documents = [
        "The cat sat on the mat",
        "The dog sat on the log",
        "Cats and dogs are pets",
        "The mat was comfortable"
    ]
    
    query = st.text_input("Enter search query:", value="cat mat")
    
    if query:
        # Simple TF-IDF calculation
        # Use fallback implementation to avoid Python 3.13 sklearn compatibility issues
        from sklearn_fallbacks import TfidfVectorizer
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents + [query])
        
        # Get similarity scores
        query_vector = tfidf_matrix[-1]
        similarities = []
        for i in range(len(documents)):
            doc_vector = tfidf_matrix[i]
            # Handle both sparse matrices and numpy arrays
            if hasattr(query_vector, 'toarray'):
                similarity = (query_vector * doc_vector.T).toarray()[0][0]
            else:
                # For numpy arrays (our custom implementation)
                similarity = np.dot(query_vector, doc_vector.T).item()
            similarities.append(similarity)
        
        # Display results
        results_df = pd.DataFrame({
            "Document": documents,
            "Similarity Score": similarities
        }).sort_values("Similarity Score", ascending=False)
        
        st.dataframe(results_df)
    
    # BM25
    st.markdown("### 🎯 BM25: Probabilistic Retrieval")
    st.markdown("""
    BM25 improved on TF-IDF by adding document length normalization and term saturation, 
    becoming the standard baseline for IR tasks.
    """)
    
    with st.expander("BM25 Formula"):
        st.latex(r"""
        BM25(D,Q) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i,D) \cdot (k_1+1)}{f(q_i,D) + k_1 \cdot (1-b+b \cdot \frac{|D|}{avgdl})}
        """)
        st.markdown("""
        Where:
        - k₁ and b are tuning parameters
        - |D| is document length
        - avgdl is average document length
        """)

def render_early_ml_methods():
    """Section on early machine learning methods for NLP."""
    st.subheader("📊 Early Machine Learning in NLP")
    
    st.markdown("""
    Before deep learning, NLP relied on traditional machine learning algorithms with 
    carefully engineered features.
    """)
    
    # Feature Engineering
    st.markdown("### 🔧 Feature Engineering: The Art of NLP")
    
    st.markdown("""
    Success in early ML-based NLP heavily depended on clever feature engineering:
    """)
    
    feature_categories = {
        "Lexical Features": [
            "Bag of Words",
            "N-grams (bigrams, trigrams)",
            "Character n-grams",
            "Word shape (Xxxx, XXXX, etc.)"
        ],
        "Syntactic Features": [
            "Part-of-speech tags",
            "Dependency relations",
            "Parse tree features",
            "Chunk tags"
        ],
        "Semantic Features": [
            "WordNet synsets",
            "Named entity types",
            "Semantic role labels",
            "Brown clusters"
        ],
        "Other Features": [
            "Gazetteers (lists of entities)",
            "Regular expression matches",
            "Document metadata",
            "Position in document"
        ]
    }
    
    tabs = st.tabs(list(feature_categories.keys()))
    for i, (category, features) in enumerate(feature_categories.items()):
        with tabs[i]:
            for feature in features:
                st.markdown(f"• {feature}")
    
    # Algorithms
    st.markdown("### 🤖 Popular Algorithms")
    
    algo_comparison = {
        "Algorithm": ["Naive Bayes", "SVM", "MaxEnt/Logistic Regression", "Decision Trees"],
        "Strengths": [
            "Simple, fast, works with small data",
            "High accuracy, effective in high dimensions",
            "Probabilistic, handles overlapping features",
            "Interpretable, handles non-linear patterns"
        ],
        "Common Uses": [
            "Text classification, spam filtering",
            "Text classification, NER",
            "Various classification tasks",
            "Feature selection, ensemble methods"
        ]
    }
    
    df_algos = pd.DataFrame(algo_comparison)
    st.table(df_algos)
    
    # Interactive Naive Bayes demo
    st.markdown("### 🎮 Interactive: Naive Bayes Sentiment Classifier")
    
    with st.expander("Try Naive Bayes Classification"):
        # Simple training data
        training_data = [
            ("I love this movie", "positive"),
            ("This film is great", "positive"),
            ("Awesome performance", "positive"),
            ("I hate this movie", "negative"),
            ("Terrible acting", "negative"),
            ("Worst film ever", "negative")
        ]
        
        st.markdown("**Training Data:**")
        for text, label in training_data:
            st.markdown(f"- '{text}' → {label}")
        
        test_text = st.text_input("Enter text to classify:", value="I really enjoyed this film")
        
        if test_text:
            # Very simple Naive Bayes (just for demonstration)
            positive_words = set(["love", "great", "awesome", "enjoyed", "good", "excellent"])
            negative_words = set(["hate", "terrible", "worst", "bad", "awful", "horrible"])
            
            words = test_text.lower().split()
            pos_score = sum(1 for word in words if word in positive_words)
            neg_score = sum(1 for word in words if word in negative_words)
            
            if pos_score > neg_score:
                st.success(f"Predicted: POSITIVE (pos_score: {pos_score}, neg_score: {neg_score})")
            elif neg_score > pos_score:
                st.error(f"Predicted: NEGATIVE (pos_score: {pos_score}, neg_score: {neg_score})")
            else:
                st.warning(f"Predicted: NEUTRAL (pos_score: {pos_score}, neg_score: {neg_score})")

def render_why_neural_networks():
    """Section explaining why neural networks were needed."""
    st.subheader("💡 Why Neural Networks? The Limitations That Led to Revolution")
    
    st.markdown("""
    Despite significant progress, pre-neural NLP methods faced fundamental limitations that 
    neural networks would eventually address.
    """)
    
    # Key Limitations
    st.markdown("### 🚧 Key Limitations of Traditional Methods")
    
    limitations = {
        "🔤 Discrete Representations": {
            "Problem": "Words treated as atomic symbols with no notion of similarity",
            "Example": "'car' and 'automobile' as unrelated as 'car' and 'banana'",
            "Neural Solution": "Continuous word embeddings capture semantic similarity"
        },
        "🔧 Manual Feature Engineering": {
            "Problem": "Required expert knowledge and extensive manual work",
            "Example": "Designing features for each new task/domain",
            "Neural Solution": "Automatic feature learning from raw data"
        },
        "📊 Data Sparsity": {
            "Problem": "Couldn't generalize to unseen word combinations",
            "Example": "'green car' seen in training, but not 'emerald automobile'",
            "Neural Solution": "Compositional representations handle novel combinations"
        },
        "🧩 Limited Context": {
            "Problem": "Fixed context windows, no long-range dependencies",
            "Example": "Can't connect 'he' to 'John' mentioned 10 sentences ago",
            "Neural Solution": "RNNs and Transformers model arbitrary dependencies"
        },
        "🎯 Task-Specific Models": {
            "Problem": "Separate models for each task, no knowledge sharing",
            "Example": "POS tagger can't help NER model",
            "Neural Solution": "Transfer learning and multi-task learning"
        }
    }
    
    for limitation, details in limitations.items():
        with st.expander(limitation):
            st.markdown(f"**Problem:** {details['Problem']}")
            st.markdown(f"**Example:** {details['Example']}")
            st.markdown(f"**Neural Solution:** {details['Neural Solution']}")
    
    # The Promise of Neural Networks
    st.markdown("### 🚀 The Promise of Neural Networks")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **What Neural Networks Offered:**
        - 📈 Learn features automatically
        - 🔄 Handle variable-length inputs
        - 🧠 Capture complex patterns
        - 🔗 Share knowledge across tasks
        - 📊 Scale with data
        """)
    
    with col2:
        st.markdown("""
        **Enabling Factors (2000s-2010s):**
        - 💾 Large annotated datasets
        - 🖥️ GPU acceleration
        - 🔧 Better optimization techniques
        - 🌐 Open-source frameworks
        - 👥 Growing research community
        """)
    
    # Transition to Neural Era
    st.markdown("### 🌉 The Bridge to Neural NLP")
    
    st.info("""
    The stage was set for the neural revolution. The combination of:
    - Limitations of existing methods becoming clear
    - Computational resources becoming available
    - Key algorithmic breakthroughs (backpropagation, word2vec)
    - Success in computer vision inspiring NLP researchers
    
    ...would lead to the transformation we'll explore in the following chapters.
    """)
    
    # Visual summary
    st.markdown("### 📊 Evolution Summary")
    
    evolution_data = [
        {"Era": "Rule-Based", "Years": "1950s-1980s", "Complexity": 3, "Data Needs": 1, "Performance": 2},
        {"Era": "Statistical", "Years": "1980s-2000s", "Complexity": 5, "Data Needs": 4, "Performance": 4},
        {"Era": "Early ML", "Years": "2000s-2010s", "Complexity": 6, "Data Needs": 6, "Performance": 6},
        {"Era": "Neural", "Years": "2010s+", "Complexity": 8, "Data Needs": 9, "Performance": 9}
    ]
    
    df_evolution = pd.DataFrame(evolution_data)
    
    fig = go.Figure()
    
    metrics = ["Complexity", "Data Needs", "Performance"]
    colors = ["blue", "green", "red"]
    
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Scatter(
            x=df_evolution["Era"],
            y=df_evolution[metric],
            mode='lines+markers',
            name=metric,
            line=dict(color=color, width=2),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title="Evolution of NLP Approaches",
        xaxis_title="Era",
        yaxis_title="Level (1-10)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    🎯 **Ready for the Neural Revolution?**
    
    Now that we understand the historical context and limitations of traditional approaches, 
    we're ready to explore how neural networks transformed NLP, starting with the statistical 
    foundations in Chapter 1.
    """)