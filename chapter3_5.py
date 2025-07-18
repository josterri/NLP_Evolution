import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
from utils import handle_errors, display_loading_message

@handle_errors
def render_section_3_5():
    """Section 3.5: RNNs and LSTMs - Sequential Neural Networks"""
    st.header("3.5 RNNs and LSTMs: Neural Networks with Memory")
    
    st.markdown("""
    Before transformers revolutionized NLP, Recurrent Neural Networks (RNNs) and their variants 
    like LSTMs were the go-to architectures for sequential data. Let's understand how they work 
    and why they were eventually superseded.
    """)
    
    # Create tabs for different concepts
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Basic RNNs",
        "🧠 The LSTM Cell",
        "🔄 GRU: Simplified LSTM",
        "💻 Interactive Demo",
        "⚖️ RNN vs Transformer"
    ])
    
    with tab1:
        render_basic_rnn()
    
    with tab2:
        render_lstm_cell()
    
    with tab3:
        render_gru()
    
    with tab4:
        render_interactive_demo()
    
    with tab5:
        render_comparison()

def render_basic_rnn():
    """Explain basic RNN architecture."""
    st.subheader("📊 Understanding Basic RNNs")
    
    st.markdown("""
    ### The Core Idea
    
    RNNs process sequences by maintaining a **hidden state** that gets updated at each time step. 
    This hidden state acts as the network's "memory" of what it has seen so far.
    """)
    
    # RNN equations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**RNN Forward Pass:**")
        st.latex(r"h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)")
        st.latex(r"y_t = W_{hy} h_t + b_y")
        
        st.markdown("""
        Where:
        - $h_t$ = hidden state at time $t$
        - $x_t$ = input at time $t$
        - $W_{hh}, W_{xh}, W_{hy}$ = weight matrices
        - $b_h, b_y$ = bias vectors
        """)
    
    with col2:
        st.markdown("**Unrolled RNN Visualization:**")
        # Create a simple RNN unrolling diagram
        steps = 4
        fig = go.Figure()
        
        # Hidden states
        for i in range(steps):
            fig.add_shape(type="circle",
                x0=i*2-0.3, y0=0.7, x1=i*2+0.3, y1=1.3,
                fillcolor="lightblue", line_color="black")
            fig.add_annotation(x=i*2, y=1, text=f"h<sub>{i}</sub>",
                showarrow=False, font=dict(size=12))
        
        # Inputs
        for i in range(steps):
            fig.add_shape(type="rect",
                x0=i*2-0.2, y0=-0.5, x1=i*2+0.2, y1=-0.1,
                fillcolor="lightgreen", line_color="black")
            fig.add_annotation(x=i*2, y=-0.3, text=f"x<sub>{i}</sub>",
                showarrow=False, font=dict(size=12))
        
        # Connections
        for i in range(steps-1):
            fig.add_annotation(x=i*2, y=1, ax=(i+1)*2, ay=1,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2)
        
        for i in range(steps):
            fig.add_annotation(x=i*2, y=-0.1, ax=i*2, ay=0.7,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2)
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Vanishing gradient problem
    st.markdown("### 🚨 The Vanishing Gradient Problem")
    
    st.warning("""
    **The Fatal Flaw of Basic RNNs:**
    
    During backpropagation through time (BPTT), gradients get multiplied many times. Since the 
    derivative of tanh is bounded between 0 and 1, repeated multiplication causes gradients to 
    vanish exponentially with sequence length.
    """)
    
    # Visualization of vanishing gradients
    with st.expander("See Gradient Flow Visualization"):
        sequence_lengths = [5, 10, 20, 50]
        gradient_magnitudes = [0.9**n for n in sequence_lengths]
        
        fig = go.Figure(data=[
            go.Bar(x=[f"Length {n}" for n in sequence_lengths],
                   y=gradient_magnitudes,
                   text=[f"{g:.2e}" for g in gradient_magnitudes],
                   textposition='auto')
        ])
        
        fig.update_layout(
            title="Gradient Magnitude vs Sequence Length (gradient factor = 0.9)",
            yaxis_title="Gradient Magnitude",
            yaxis_type="log"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        With a gradient factor of 0.9, after just 50 time steps, the gradient is reduced to 
        0.005 of its original value! This makes it nearly impossible to learn long-range dependencies.
        """)

def render_lstm_cell():
    """Explain LSTM architecture in detail."""
    st.subheader("🧠 The LSTM Cell: A Solution to Vanishing Gradients")
    
    st.markdown("""
    Long Short-Term Memory (LSTM) networks, introduced by Hochreiter & Schmidhuber (1997), 
    solved the vanishing gradient problem through a clever gating mechanism.
    """)
    
    # LSTM architecture
    st.markdown("### The LSTM Cell Architecture")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **LSTM has three gates:**
        1. **Forget Gate** - What to forget from previous state
        2. **Input Gate** - What new information to store
        3. **Output Gate** - What to output based on state
        
        **Plus two states:**
        - **Cell State (C)** - Long-term memory
        - **Hidden State (h)** - Short-term memory/output
        """)
    
    with col2:
        # LSTM equations
        st.markdown("**LSTM Equations:**")
        st.latex(r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)")
        st.latex(r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)")
        st.latex(r"\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)")
        st.latex(r"C_t = f_t * C_{t-1} + i_t * \tilde{C}_t")
        st.latex(r"o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)")
        st.latex(r"h_t = o_t * \tanh(C_t)")
    
    # Interactive LSTM visualization
    st.markdown("### 🎮 Interactive LSTM Cell")
    
    with st.expander("Explore LSTM Gates"):
        # Input values
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_val = st.slider("Input (x_t)", -1.0, 1.0, 0.5)
            h_prev = st.slider("Previous Hidden (h_t-1)", -1.0, 1.0, 0.3)
        
        with col2:
            c_prev = st.slider("Previous Cell (C_t-1)", -1.0, 1.0, 0.7)
            
        # Simple gate calculations (simplified for demonstration)
        forget_gate = 1 / (1 + np.exp(-(0.5 * h_prev + 0.3 * x_val)))
        input_gate = 1 / (1 + np.exp(-(0.4 * h_prev + 0.6 * x_val)))
        candidate = np.tanh(0.7 * h_prev + 0.5 * x_val)
        output_gate = 1 / (1 + np.exp(-(0.6 * h_prev + 0.4 * x_val)))
        
        # New states
        c_new = forget_gate * c_prev + input_gate * candidate
        h_new = output_gate * np.tanh(c_new)
        
        with col3:
            st.markdown("**Gate Values:**")
            st.metric("Forget Gate", f"{forget_gate:.3f}")
            st.metric("Input Gate", f"{input_gate:.3f}")
            st.metric("Output Gate", f"{output_gate:.3f}")
        
        # Visualization
        st.markdown("**State Updates:**")
        
        fig = go.Figure()
        
        # Add bars for different components
        fig.add_trace(go.Bar(
            x=['Previous Cell', 'Forget Effect', 'Input Effect', 'New Cell'],
            y=[c_prev, forget_gate * c_prev, input_gate * candidate, c_new],
            text=[f"{v:.3f}" for v in [c_prev, forget_gate * c_prev, input_gate * candidate, c_new]],
            textposition='auto',
            marker_color=['lightblue', 'orange', 'green', 'red']
        ))
        
        fig.update_layout(
            title="Cell State Update Process",
            yaxis_title="Value",
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **Result:**
        - New Cell State: {c_new:.3f}
        - New Hidden State: {h_new:.3f}
        
        The LSTM successfully combines the previous memory ({c_prev:.3f}) with new input 
        information, using gates to control the flow.
        """)
    
    # Why LSTMs work
    st.markdown("### 💡 Why LSTMs Solve Vanishing Gradients")
    
    st.success("""
    **Key Innovation: The Cell State Highway**
    
    The cell state C_t acts as a "gradient highway" that allows gradients to flow unchanged 
    across many time steps. The addition operation (not multiplication) preserves gradient magnitude.
    
    - **Forget gate = 1**: Perfect memory retention
    - **Input gate = 0**: Ignore current input
    - **Linear transformation**: Gradients flow without vanishing
    """)

def render_gru():
    """Explain GRU architecture."""
    st.subheader("🔄 GRU: Gated Recurrent Unit")
    
    st.markdown("""
    The GRU (Cho et al., 2014) simplifies the LSTM architecture while maintaining most of its 
    benefits. It combines the forget and input gates into a single update gate.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **GRU Components:**
        - **Reset Gate (r)**: How much past to forget
        - **Update Gate (z)**: Balance between past and present
        - **Hidden State (h)**: Single state (no separate cell state)
        
        **Advantages:**
        - Fewer parameters (3 gates vs 4)
        - Often comparable performance
        - Faster training
        """)
    
    with col2:
        st.markdown("**GRU Equations:**")
        st.latex(r"z_t = \sigma(W_z \cdot [h_{t-1}, x_t])")
        st.latex(r"r_t = \sigma(W_r \cdot [h_{t-1}, x_t])")
        st.latex(r"\tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t])")
        st.latex(r"h_t = (1-z_t) * h_{t-1} + z_t * \tilde{h}_t")
    
    # Comparison table
    st.markdown("### LSTM vs GRU Comparison")
    
    comparison_data = {
        "Feature": ["Number of Gates", "Number of States", "Parameters", "Training Speed", "Performance"],
        "LSTM": ["3 (forget, input, output)", "2 (cell, hidden)", "More", "Slower", "Slightly better on complex tasks"],
        "GRU": ["2 (reset, update)", "1 (hidden)", "~25% fewer", "Faster", "Often comparable"]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.table(df_comparison)

def render_interactive_demo():
    """Interactive RNN/LSTM demo."""
    st.subheader("💻 Interactive Demo: Sequence Processing")
    
    st.markdown("""
    Let's see how RNNs and LSTMs process sequences differently. We'll use a simple task: 
    remembering and outputting a specific token after many time steps.
    """)
    
    # Demo setup
    sequence_length = st.slider("Sequence Length", 5, 20, 10)
    memory_position = st.slider("Position to Remember", 0, sequence_length-1, 2)
    
    # Generate sequence
    vocab = ['A', 'B', 'C', 'D', 'E']
    sequence = [vocab[i % len(vocab)] for i in range(sequence_length)]
    target_token = sequence[memory_position]
    
    st.markdown(f"**Task**: Remember the token at position {memory_position} ('{target_token}') and output it at the end.")
    st.markdown(f"**Sequence**: {' → '.join(sequence)}")
    
    # Simulate RNN vs LSTM behavior
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Basic RNN")
        
        # Simulate RNN memory decay
        rnn_memory = []
        decay_factor = 0.8
        
        for i, token in enumerate(sequence):
            if i == memory_position:
                memory_strength = 1.0
            else:
                memory_strength = decay_factor ** (i - memory_position) if i > memory_position else 0
            
            rnn_memory.append(memory_strength)
        
        fig_rnn = go.Figure(data=[
            go.Bar(x=list(range(sequence_length)), y=rnn_memory,
                   text=[f"{m:.2f}" for m in rnn_memory],
                   textposition='auto',
                   marker_color=['red' if i == memory_position else 'lightblue' 
                                for i in range(sequence_length)])
        ])
        
        fig_rnn.update_layout(
            title="RNN Memory Strength Over Time",
            xaxis_title="Position",
            yaxis_title="Memory Strength",
            height=300
        )
        
        st.plotly_chart(fig_rnn, use_container_width=True)
        
        final_memory_rnn = rnn_memory[-1]
        st.metric("Final Memory of Target", f"{final_memory_rnn:.3f}")
        
        if final_memory_rnn < 0.1:
            st.error("❌ RNN likely forgot the target token!")
        else:
            st.success("✅ RNN might remember the target token")
    
    with col2:
        st.markdown("### LSTM")
        
        # Simulate LSTM memory (with gates)
        lstm_memory = []
        cell_state = 0
        
        for i, token in enumerate(sequence):
            if i == memory_position:
                # Input gate opens, store in cell state
                cell_state = 1.0
                lstm_memory.append(1.0)
            else:
                # Forget gate mostly closed, preserve cell state
                forget_gate = 0.95  # LSTM can maintain memory better
                cell_state = cell_state * forget_gate
                lstm_memory.append(cell_state)
        
        fig_lstm = go.Figure(data=[
            go.Bar(x=list(range(sequence_length)), y=lstm_memory,
                   text=[f"{m:.2f}" for m in lstm_memory],
                   textposition='auto',
                   marker_color=['red' if i == memory_position else 'lightgreen' 
                                for i in range(sequence_length)])
        ])
        
        fig_lstm.update_layout(
            title="LSTM Memory Strength Over Time",
            xaxis_title="Position",
            yaxis_title="Memory Strength",
            height=300
        )
        
        st.plotly_chart(fig_lstm, use_container_width=True)
        
        final_memory_lstm = lstm_memory[-1]
        st.metric("Final Memory of Target", f"{final_memory_lstm:.3f}")
        
        if final_memory_lstm > 0.5:
            st.success("✅ LSTM successfully retained the memory!")
        else:
            st.warning("⚠️ LSTM memory is weakening")
    
    # Summary
    st.info(f"""
    **Key Observation**: 
    - RNN memory decayed to {final_memory_rnn:.1%} after {sequence_length - memory_position} steps
    - LSTM memory retained {final_memory_lstm:.1%} of the original signal
    
    This demonstrates why LSTMs are superior for long-range dependencies!
    """)

def render_comparison():
    """Compare RNNs with Transformers."""
    st.subheader("⚖️ RNNs vs Transformers: Why Transformers Won")
    
    st.markdown("""
    While LSTMs were a huge improvement over basic RNNs, they still had limitations that 
    transformers would eventually overcome.
    """)
    
    # Detailed comparison
    comparison = {
        "Aspect": [
            "Sequential Processing",
            "Parallelization", 
            "Long-range Dependencies",
            "Training Speed",
            "Memory Requirements",
            "Interpretability",
            "Position Encoding",
            "Computational Complexity"
        ],
        "RNN/LSTM": [
            "✅ Natural for sequences",
            "❌ Must process sequentially",
            "⚠️ Struggles beyond ~100 tokens",
            "❌ Slow (sequential bottleneck)",
            "✅ O(1) per step",
            "❌ Hidden states are opaque",
            "✅ Implicit through recurrence",
            "O(n) for sequence length n"
        ],
        "Transformer": [
            "❌ Requires position encoding",
            "✅ Fully parallel",
            "✅ Handles 1000s of tokens",
            "✅ Fast (parallel processing)",
            "❌ O(n²) for attention",
            "✅ Attention weights interpretable",
            "❌ Needs explicit encoding",
            "O(n²) for sequence length n"
        ]
    }
    
    df_comp = pd.DataFrame(comparison)
    st.dataframe(df_comp, use_container_width=True)
    
    # Visual comparison of processing
    st.markdown("### Processing Pattern Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**RNN: Sequential Processing**")
        st.image("https://via.placeholder.com/300x200/4A90E2/FFFFFF?text=RNN+Sequential", 
                caption="Each step depends on the previous one")
        st.markdown("""
        ```
        Step 1: Process token 1 → hidden state 1
        Step 2: Process token 2 + hidden state 1 → hidden state 2
        Step 3: Process token 3 + hidden state 2 → hidden state 3
        ...
        ```
        """)
    
    with col2:
        st.markdown("**Transformer: Parallel Processing**")
        st.image("https://via.placeholder.com/300x200/50C878/FFFFFF?text=Transformer+Parallel", 
                caption="All tokens processed simultaneously")
        st.markdown("""
        ```
        Step 1: Process ALL tokens at once
                ↓
        Attention connects every token to every other token
                ↓
        Output for all positions simultaneously
        ```
        """)
    
    # When to use what
    st.markdown("### 📝 When to Use Each Architecture")
    
    tab1, tab2 = st.tabs(["Use RNNs/LSTMs When", "Use Transformers When"])
    
    with tab1:
        st.markdown("""
        ✅ **Good for RNNs/LSTMs:**
        - Real-time/streaming applications (processing one token at a time)
        - Limited memory environments
        - Truly sequential data (time series, audio)
        - When sequence order is critical
        - Edge devices with limited compute
        
        **Example Applications:**
        - Speech recognition (streaming)
        - Real-time translation
        - Music generation
        - Simple chatbots
        """)
    
    with tab2:
        st.markdown("""
        ✅ **Good for Transformers:**
        - Large-scale language understanding
        - When you have GPU/TPU resources
        - Need to capture very long-range dependencies
        - Want interpretable attention patterns
        - Batch processing scenarios
        
        **Example Applications:**
        - Large language models (GPT, BERT)
        - Document understanding
        - Machine translation (non-streaming)
        - Question answering systems
        """)
    
    # Conclusion
    st.success("""
    🎯 **Key Takeaway**: While RNNs and LSTMs were crucial stepping stones in NLP's evolution, 
    transformers' ability to process sequences in parallel and capture long-range dependencies 
    more effectively made them the architecture of choice for modern NLP.
    
    However, the intuitions from RNNs—especially about maintaining state and gating mechanisms—
    continue to influence modern architectures like State Space Models and linear attention variants.
    """)