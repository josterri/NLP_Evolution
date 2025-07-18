import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
from utils import handle_errors
import random

@handle_errors
def render_section_4_11():
    """Section 4.11: BERT - Bidirectional Transformers"""
    st.header("4.11 BERT: Bidirectional Encoder Representations from Transformers")
    
    st.markdown("""
    While GPT showed the power of transformers for generation, BERT (Devlin et al., 2018) 
    revolutionized how we approach understanding tasks by using bidirectional attention and 
    introducing the masked language modeling objective.
    """)
    
    # Create tabs for different aspects
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔄 Bidirectional vs Unidirectional",
        "🎭 Masked Language Modeling",
        "🔗 Next Sentence Prediction", 
        "🏗️ BERT Architecture",
        "🎯 Fine-tuning BERT",
        "📊 BERT vs GPT"
    ])
    
    with tab1:
        render_bidirectional_attention()
    
    with tab2:
        render_masked_lm()
    
    with tab3:
        render_nsp()
    
    with tab4:
        render_bert_architecture()
    
    with tab5:
        render_fine_tuning()
    
    with tab6:
        render_bert_vs_gpt()

def render_bidirectional_attention():
    """Explain bidirectional vs unidirectional attention."""
    st.subheader("🔄 The Power of Bidirectional Attention")
    
    st.markdown("""
    The key innovation of BERT is using **bidirectional** self-attention, meaning each token 
    can attend to all other tokens in the sequence, not just previous ones.
    """)
    
    # Visual comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Unidirectional (GPT-style)")
        st.markdown("""
        ```
        The cat sat on the [MASK]
        
        Attention pattern:
        The     → [The]
        cat     → [The, cat]
        sat     → [The, cat, sat]
        on      → [The, cat, sat, on]
        the     → [The, cat, sat, on, the]
        [MASK]  → [The, cat, sat, on, the, MASK]
        ```
        
        ❌ Can't see future context
        ✅ Good for generation
        """)
    
    with col2:
        st.markdown("### Bidirectional (BERT-style)")
        st.markdown("""
        ```
        The cat sat on the [MASK]
        
        Attention pattern:
        Each token → [The, cat, sat, on, the, MASK]
        
        Example for "sat":
        sat → sees "cat" (subject)
        sat → sees "on" (preposition)
        sat → full context!
        ```
        
        ✅ Sees full context
        ❌ Can't generate autoregressively
        """)
    
    # Interactive attention visualization
    st.markdown("### 🎮 Interactive Attention Patterns")
    
    sentence = st.text_input("Enter a sentence:", value="The cat sat on the mat")
    tokens = sentence.split()
    
    if tokens:
        attention_type = st.radio("Attention Type:", ["Unidirectional", "Bidirectional"])
        selected_token = st.selectbox("Focus on token:", tokens)
        token_idx = tokens.index(selected_token)
        
        # Create attention matrix
        n_tokens = len(tokens)
        attention_matrix = np.zeros((n_tokens, n_tokens))
        
        if attention_type == "Unidirectional":
            # Causal mask
            for i in range(n_tokens):
                for j in range(i + 1):
                    attention_matrix[i, j] = 1.0 / (i + 1)
        else:
            # Full attention
            attention_matrix = np.ones((n_tokens, n_tokens)) / n_tokens
        
        # Highlight selected token
        attention_vector = attention_matrix[token_idx]
        
        # Visualization
        fig = go.Figure(data=go.Heatmap(
            z=[attention_vector],
            x=tokens,
            y=[selected_token],
            colorscale="Blues",
            text=[[f"{v:.2f}" for v in attention_vector]],
            texttemplate="%{text}",
            showscale=False
        ))
        
        fig.update_layout(
            title=f"{attention_type} Attention from '{selected_token}'",
            xaxis_title="Attending to",
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if attention_type == "Unidirectional":
            can_see = tokens[:token_idx + 1]
            cannot_see = tokens[token_idx + 1:] if token_idx < n_tokens - 1 else []
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ Can see: {', '.join(can_see)}")
            with col2:
                if cannot_see:
                    st.error(f"❌ Cannot see: {', '.join(cannot_see)}")

def render_masked_lm():
    """Explain masked language modeling."""
    st.subheader("🎭 Masked Language Modeling (MLM)")
    
    st.markdown("""
    BERT's training objective is to predict masked tokens in a sentence. This forces the model 
    to understand context from both directions.
    """)
    
    # MLM process
    st.markdown("### The MLM Process")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1. Input Sentence**
        ```
        The cat sat on the mat
        ```
        """)
    
    with col2:
        st.markdown("""
        **2. Random Masking (15%)**
        ```
        The cat [MASK] on the mat
        ```
        """)
    
    with col3:
        st.markdown("""
        **3. Predict Masked Token**
        ```
        Model predicts: "sat"
        ```
        """)
    
    # Interactive MLM demo
    st.markdown("### 🎮 Try Masked Language Modeling")
    
    sample_sentences = [
        "The cat sat on the mat",
        "Machine learning is transforming the world",
        "BERT revolutionized natural language processing",
        "The weather today is quite pleasant"
    ]
    
    sentence = st.selectbox("Choose a sentence:", sample_sentences)
    tokens = sentence.split()
    
    if st.button("Apply Random Masking"):
        # Randomly mask 15% of tokens
        n_masks = max(1, int(0.15 * len(tokens)))
        mask_indices = random.sample(range(len(tokens)), n_masks)
        
        masked_tokens = []
        original_tokens = []
        
        for i, token in enumerate(tokens):
            if i in mask_indices:
                masked_tokens.append("[MASK]")
                original_tokens.append(token)
            else:
                masked_tokens.append(token)
        
        st.markdown("**Masked Sentence:**")
        st.info(" ".join(masked_tokens))
        
        with st.expander("See Original Tokens"):
            for idx, orig in zip(mask_indices, original_tokens):
                st.markdown(f"Position {idx}: **{orig}**")
        
        # Simulate BERT prediction
        st.markdown("**BERT's Task:**")
        st.markdown("Predict the masked tokens using bidirectional context from all other tokens.")
        
        # Show attention flow
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Left Context →**")
            left_context = " ".join(masked_tokens[:mask_indices[0]])
            if left_context:
                st.code(left_context)
        
        with col2:
            st.markdown("**← Right Context**")
            right_context = " ".join(masked_tokens[mask_indices[0]+1:])
            if right_context:
                st.code(right_context)
    
    # Masking strategies
    st.markdown("### 🎲 BERT's Masking Strategy")
    
    st.info("""
    **The 80-10-10 Rule:**
    
    When BERT masks 15% of tokens, it doesn't always use [MASK]:
    - **80%**: Replace with [MASK] token
    - **10%**: Replace with a random token
    - **10%**: Keep the original token
    
    **Why this complexity?**
    - Prevents the model from only learning to predict [MASK]
    - Forces robust representations for all tokens
    - Reduces the gap between pre-training and fine-tuning
    """)

def render_nsp():
    """Explain next sentence prediction."""
    st.subheader("🔗 Next Sentence Prediction (NSP)")
    
    st.markdown("""
    BERT's second training objective is to predict whether two sentences appear consecutively 
    in the original text. This helps BERT understand relationships between sentences.
    """)
    
    # NSP examples
    st.markdown("### Examples of NSP")
    
    examples = [
        {
            "sent_a": "The cat sat on the mat.",
            "sent_b": "It was a comfortable place to rest.",
            "label": "Next Sentence",
            "explanation": "Sentence B logically follows A (pronoun 'it' refers to 'mat')"
        },
        {
            "sent_a": "The cat sat on the mat.",
            "sent_b": "Pizza is a popular Italian dish.",
            "label": "Random Sentence",
            "explanation": "Sentences are unrelated"
        },
        {
            "sent_a": "The weather is beautiful today.",
            "sent_b": "Let's go for a walk in the park.",
            "label": "Next Sentence",
            "explanation": "Sentence B is a logical response to A"
        }
    ]
    
    for i, example in enumerate(examples):
        with st.expander(f"Example {i+1}: {example['label']}"):
            st.markdown(f"**Sentence A:** {example['sent_a']}")
            st.markdown(f"**Sentence B:** {example['sent_b']}")
            
            if example['label'] == "Next Sentence":
                st.success(f"✅ {example['label']}")
            else:
                st.error(f"❌ {example['label']}")
            
            st.markdown(f"**Why:** {example['explanation']}")
    
    # Interactive NSP
    st.markdown("### 🎮 Try Next Sentence Prediction")
    
    sent_a = st.text_input("Sentence A:", value="The sun was setting over the horizon.")
    sent_b_options = [
        "The sky turned beautiful shades of orange and pink.",
        "Quantum computing uses quantum bits called qubits.",
        "It was time to head back home.",
        "The recipe calls for two cups of flour."
    ]
    
    sent_b = st.selectbox("Sentence B options:", sent_b_options)
    
    if st.button("Predict Relationship"):
        # Simple heuristic for demo
        related_pairs = {
            ("The sun was setting over the horizon.", "The sky turned beautiful shades of orange and pink."): True,
            ("The sun was setting over the horizon.", "It was time to head back home."): True,
        }
        
        is_next = related_pairs.get((sent_a, sent_b), False)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if is_next:
                st.success("✅ Next Sentence")
            else:
                st.error("❌ Random Sentence")
        
        with col2:
            if is_next:
                st.markdown("BERT detected semantic continuity between the sentences.")
            else:
                st.markdown("BERT found no logical connection between the sentences.")
    
    # Note about NSP
    st.warning("""
    ⚠️ **Note:** Later research (RoBERTa) showed that NSP might not be necessary and can 
    even hurt performance on some tasks. Many modern BERT variants skip NSP entirely.
    """)

def render_bert_architecture():
    """Explain BERT's architecture."""
    st.subheader("🏗️ BERT Architecture")
    
    st.markdown("""
    BERT uses the transformer encoder architecture with some key modifications for its 
    bidirectional nature.
    """)
    
    # Architecture comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### BERT Sizes")
        
        bert_sizes = {
            "Model": ["BERT-Base", "BERT-Large"],
            "Layers": [12, 24],
            "Hidden Size": [768, 1024],
            "Attention Heads": [12, 16],
            "Parameters": ["110M", "340M"],
            "Training Time": ["4 days on 16 TPUs", "4 days on 64 TPUs"]
        }
        
        df_sizes = pd.DataFrame(bert_sizes)
        st.table(df_sizes)
    
    with col2:
        st.markdown("### Key Components")
        st.markdown("""
        1. **Token Embeddings**: WordPiece tokenization
        2. **Segment Embeddings**: Distinguish sentences A/B
        3. **Position Embeddings**: Learned, not sinusoidal
        4. **[CLS] Token**: Classification tasks
        5. **[SEP] Token**: Sentence separator
        """)
    
    # Input representation
    st.markdown("### BERT Input Representation")
    
    example_input = "[CLS] The cat sat [SEP] on the mat [SEP]"
    tokens = example_input.split()
    
    # Create visualization
    fig = go.Figure()
    
    # Token embeddings
    fig.add_trace(go.Scatter(
        x=list(range(len(tokens))),
        y=[3] * len(tokens),
        mode='text+markers',
        text=tokens,
        textposition="top center",
        name="Tokens",
        marker=dict(size=40, color='lightblue')
    ))
    
    # Segment embeddings
    segment_a = ["A"] * 4  # [CLS] The cat sat
    segment_b = ["B"] * 4  # [SEP] on the mat [SEP]
    segments = segment_a + segment_b
    
    fig.add_trace(go.Scatter(
        x=list(range(len(tokens))),
        y=[2] * len(tokens),
        mode='text+markers',
        text=segments,
        textposition="top center",
        name="Segments",
        marker=dict(size=40, color=['lightgreen']*4 + ['lightcoral']*4)
    ))
    
    # Position embeddings
    fig.add_trace(go.Scatter(
        x=list(range(len(tokens))),
        y=[1] * len(tokens),
        mode='text+markers',
        text=[str(i) for i in range(len(tokens))],
        textposition="top center",
        name="Positions",
        marker=dict(size=40, color='lightyellow')
    ))
    
    fig.update_layout(
        title="BERT Input Embeddings",
        xaxis=dict(visible=False),
        yaxis=dict(
            visible=True,
            ticktext=["Position", "Segment", "Token"],
            tickvals=[1, 2, 3],
            range=[0.5, 3.5]
        ),
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **BERT's Input** = Token Embedding + Segment Embedding + Position Embedding
    
    This rich representation allows BERT to understand both the content and structure of the input.
    """)

def render_fine_tuning():
    """Explain BERT fine-tuning process."""
    st.subheader("🎯 Fine-tuning BERT")
    
    st.markdown("""
    BERT's pre-training creates powerful representations that can be fine-tuned for various 
    downstream tasks with minimal architecture changes.
    """)
    
    # Fine-tuning strategies
    tasks = {
        "Classification": {
            "description": "Use [CLS] token representation",
            "architecture": "[CLS] → Dense → Softmax",
            "examples": ["Sentiment Analysis", "Spam Detection", "Topic Classification"],
            "data_needed": "1K-100K examples"
        },
        "Token Classification": {
            "description": "Use each token's representation",
            "architecture": "Each token → Dense → Softmax",
            "examples": ["Named Entity Recognition", "Part-of-Speech Tagging"],
            "data_needed": "10K-100K examples"
        },
        "Question Answering": {
            "description": "Predict start/end positions",
            "architecture": "Token representations → Start/End classifiers",
            "examples": ["SQuAD", "Natural Questions"],
            "data_needed": "10K-100K examples"
        },
        "Sentence Pair": {
            "description": "Use [CLS] for relationship",
            "architecture": "[CLS] → Dense → Classification",
            "examples": ["Natural Language Inference", "Paraphrase Detection"],
            "data_needed": "10K-400K examples"
        }
    }
    
    task_type = st.selectbox("Select Task Type:", list(tasks.keys()))
    task_info = tasks[task_type]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**How it works:** {task_info['description']}")
        st.markdown(f"**Architecture:** `{task_info['architecture']}`")
        st.markdown(f"**Data needed:** {task_info['data_needed']}")
    
    with col2:
        st.markdown("**Example Applications:**")
        for example in task_info['examples']:
            st.markdown(f"- {example}")
    
    # Fine-tuning best practices
    st.markdown("### 📋 Fine-tuning Best Practices")
    
    with st.expander("See Best Practices"):
        st.markdown("""
        1. **Learning Rate**: Use small learning rate (2e-5 to 5e-5)
        2. **Epochs**: Usually 2-4 epochs is enough
        3. **Batch Size**: 16 or 32 (limited by GPU memory)
        4. **Warmup**: Use learning rate warmup (10% of steps)
        5. **Dropout**: Keep BERT's dropout (usually 0.1)
        6. **Layer Freezing**: Optional - freeze lower layers for small datasets
        7. **Data Augmentation**: Helpful for small datasets
        """)
    
    # Performance gains
    st.markdown("### 📊 Typical Performance Gains")
    
    performance_data = {
        "Task": ["Sentiment Analysis", "NER", "Question Answering", "Text Classification"],
        "Pre-BERT SOTA": ["85%", "91%", "81%", "88%"],
        "BERT": ["94%", "96%", "91%", "95%"],
        "Improvement": ["+9%", "+5%", "+10%", "+7%"]
    }
    
    df_perf = pd.DataFrame(performance_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Pre-BERT', x=df_perf['Task'], y=[85, 91, 81, 88]))
    fig.add_trace(go.Bar(name='BERT', x=df_perf['Task'], y=[94, 96, 91, 95]))
    
    fig.update_layout(
        title="BERT Performance Improvements",
        yaxis_title="Accuracy (%)",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_bert_vs_gpt():
    """Compare BERT and GPT approaches."""
    st.subheader("📊 BERT vs GPT: Two Paradigms")
    
    st.markdown("""
    BERT and GPT represent two different approaches to using transformers for NLP, each with 
    its own strengths and ideal use cases.
    """)
    
    # Detailed comparison
    comparison = {
        "Aspect": [
            "Pre-training Objective",
            "Attention Pattern",
            "Best For",
            "Model Size (Original)",
            "Training Data",
            "Fine-tuning",
            "Zero-shot Ability",
            "Generation Quality",
            "Understanding Quality"
        ],
        "BERT": [
            "MLM + NSP",
            "Bidirectional",
            "Understanding tasks",
            "110M-340M params",
            "BookCorpus + Wikipedia",
            "Required for all tasks",
            "❌ None",
            "❌ Cannot generate",
            "✅✅ Excellent"
        ],
        "GPT": [
            "Next token prediction",
            "Unidirectional (causal)",
            "Generation tasks",
            "117M params",
            "BookCorpus",
            "Optional",
            "✅ Some ability",
            "✅✅ Excellent",
            "✅ Good"
        ]
    }
    
    df_comp = pd.DataFrame(comparison)
    st.table(df_comp)
    
    # Visual comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### BERT Strengths")
        st.success("""
        ✅ Superior for classification tasks
        ✅ Better contextual understanding
        ✅ Excellent for structured prediction
        ✅ Smaller models, faster inference
        ✅ State-of-the-art on many benchmarks
        """)
    
    with col2:
        st.markdown("### GPT Strengths")
        st.info("""
        ✅ Natural text generation
        ✅ Few-shot learning ability
        ✅ Single model for many tasks
        ✅ No task-specific architecture
        ✅ Scales well with size
        """)
    
    # Evolution timeline
    st.markdown("### 📅 The Evolution Continues")
    
    timeline_data = [
        {"Year": 2018, "Model": "BERT", "Innovation": "Bidirectional pre-training"},
        {"Year": 2019, "Model": "RoBERTa", "Innovation": "Better training of BERT"},
        {"Year": 2019, "Model": "ALBERT", "Innovation": "Parameter sharing"},
        {"Year": 2019, "Model": "GPT-2", "Innovation": "Scale + zero-shot"},
        {"Year": 2020, "Model": "T5", "Innovation": "Unified text-to-text"},
        {"Year": 2020, "Model": "GPT-3", "Innovation": "Few-shot learning"},
        {"Year": 2022, "Model": "ChatGPT", "Innovation": "RLHF + instruction following"}
    ]
    
    df_timeline = pd.DataFrame(timeline_data)
    
    fig = px.scatter(df_timeline, x="Year", y="Model", size=[100]*len(df_timeline),
                     hover_data=["Innovation"], 
                     title="Evolution of Transformer Models",
                     color=["BERT-family", "BERT-family", "BERT-family", 
                           "GPT-family", "Unified", "GPT-family", "GPT-family"])
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Conclusion
    st.success("""
    🎯 **Key Takeaway**: BERT and GPT pioneered two complementary approaches:
    
    - **BERT** → Bidirectional understanding → Excellence at analysis tasks
    - **GPT** → Autoregressive generation → Excellence at synthesis tasks
    
    Modern models like T5 and GPT-4 have learned from both approaches, leading to models that 
    can both understand and generate with remarkable capability.
    """)