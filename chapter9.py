import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def render_chapter_9():
    """Renders Chapter 9: Course Completion and Future Directions."""
    st.header("Chapter 9: Course Completion & Your Journey Forward")
    
    st.markdown("""
    🎉 **Congratulations!** You've completed an incredible journey through the evolution of Natural Language Processing, 
    from simple statistical models to the foundations of modern AI.
    """)
    
    st.markdown("---")
    
    # Progress Summary
    st.subheader("📊 Your Learning Journey")
    st.markdown("""
    Let's review what you've accomplished:
    """)
    
    # Create a visual timeline of the journey
    timeline_data = [
        {"Chapter": "1", "Topic": "Statistical Era (N-grams)", "Era": "1990s-2000s", "Complexity": 2},
        {"Chapter": "2", "Topic": "Neural Networks & Embeddings", "Era": "2000s-2010s", "Complexity": 4},
        {"Chapter": "3", "Topic": "Sequential Models & Context", "Era": "2010s", "Complexity": 6},
        {"Chapter": "4", "Topic": "Transformer Revolution", "Era": "2017+", "Complexity": 8},
        {"Chapter": "5", "Topic": "Text Classification Applications", "Era": "2018+", "Complexity": 7},
        {"Chapter": "6", "Topic": "Generative Models", "Era": "2019+", "Complexity": 9},
        {"Chapter": "7", "Topic": "Build Your Own Model", "Era": "2020+", "Complexity": 10},
        {"Chapter": "8", "Topic": "Large Language Models", "Era": "2020+", "Complexity": 10}
    ]
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Create interactive timeline
    fig = px.scatter(df_timeline, x="Chapter", y="Complexity", 
                    size="Complexity", color="Era", hover_data=["Topic"],
                    title="Your NLP Learning Journey - Complexity Over Time")
    fig.update_layout(
        xaxis_title="Chapter",
        yaxis_title="Complexity Level",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Concepts Mastered
    st.subheader("🎯 Key Concepts You've Mastered")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Statistical Methods:**
        - N-gram models and Markov assumptions
        - Smoothing techniques
        - Probability calculations
        
        **Neural Approaches:**
        - Word embeddings (Word2Vec, GloVe)
        - Vector spaces and analogies
        - One-hot encoding limitations
        
        **Sequential Models:**
        - Context windows
        - Attention mechanisms
        - Query-Key-Value paradigm
        """)
    
    with col2:
        st.markdown("""
        **Transformer Architecture:**
        - Self-attention
        - Multi-head attention
        - Position encoding
        
        **Applications:**
        - Text classification
        - Sentiment analysis
        - Language generation
        
        **Modern AI:**
        - Large Language Models
        - In-context learning
        - Emergent capabilities
        """)
    
    st.markdown("---")
    
    # Skills Assessment
    st.subheader("📈 Skills Assessment")
    st.markdown("Rate your confidence in each area:")
    
    skills = [
        "Understanding N-gram models",
        "Working with word embeddings",
        "Explaining attention mechanisms",
        "Implementing basic transformers",
        "Text classification techniques",
        "Language generation concepts",
        "Modern LLM capabilities"
    ]
    
    confidence_scores = {}
    for skill in skills:
        confidence_scores[skill] = st.slider(
            skill, 1, 10, 7, 
            key=f"confidence_{skill.replace(' ', '_')}"
        )
    
    # Visualize skills
    if st.button("Visualize My Skills"):
        skills_df = pd.DataFrame(list(confidence_scores.items()), 
                               columns=['Skill', 'Confidence'])
        
        fig_skills = px.bar(skills_df, x='Confidence', y='Skill', 
                           orientation='h', title="Your NLP Skills Assessment")
        fig_skills.update_layout(height=400)
        st.plotly_chart(fig_skills, use_container_width=True)
    
    st.markdown("---")
    
    # Next Steps
    st.subheader("🚀 Your Next Steps in NLP")
    
    tab1, tab2, tab3 = st.tabs(["📚 Further Learning", "💻 Practice Projects", "📰 Stay Updated"])
    
    with tab1:
        st.markdown("""
        **Advanced Topics to Explore:**
        - **Transformer Variants:** BERT, GPT, T5, RoBERTa
        - **Multimodal Models:** Vision-Language models, CLIP
        - **Reinforcement Learning:** RLHF, PPO for language models
        - **Efficiency:** Model compression, quantization, distillation
        - **Ethics:** Bias, fairness, and responsible AI
        
        **Recommended Resources:**
        - 📄 "Attention Is All You Need" (Vaswani et al., 2017)
        - 📄 "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
        - 🎓 CS224N: Natural Language Processing with Deep Learning (Stanford)
        - 🎓 Hugging Face NLP Course
        - 📖 "Natural Language Processing with Python" (Bird, Klein, Loper)
        """)
    
    with tab2:
        st.markdown("""
        **Hands-on Projects to Build:**
        1. **Sentiment Analysis Tool:** Build a classifier for movie reviews or tweets
        2. **Question Answering System:** Create a chatbot for specific domains
        3. **Text Summarization:** Implement extractive and abstractive summarization
        4. **Language Translation:** Build a simple translation model
        5. **Content Generation:** Create a creative writing assistant
        6. **Information Extraction:** Build a named entity recognition system
        7. **Conversational AI:** Develop a task-oriented dialogue system
        
        **Development Platforms:**
        - 🤗 Hugging Face Transformers
        - 🔥 PyTorch/TensorFlow
        - ⚡ Lightning AI
        - 🎨 Streamlit/Gradio for demos
        """)
    
    with tab3:
        st.markdown("""
        **Stay Current in NLP:**
        - 📚 **ArXiv:** Latest research papers
        - 🐦 **Twitter:** Follow NLP researchers and practitioners
        - 📺 **YouTube:** Two Minute Papers, Yannic Kilcher
        - 🎙️ **Podcasts:** The TWIML AI Podcast, Practical AI
        - 👥 **Communities:** Reddit r/MachineLearning, Discord servers
        - 🎪 **Conferences:** ACL, EMNLP, NAACL, ICLR, NeurIPS
        
        **Key People to Follow:**
        - Christopher Manning (Stanford)
        - Yann LeCun (Meta)
        - Andrej Karpathy (OpenAI)
        - Hugging Face Team
        - Sebastian Ruder (research scientist)
        """)
    
    st.markdown("---")
    
    # Certificate/Badge Section
    st.subheader("🏆 Course Completion")
    
    if st.button("Generate Completion Certificate"):
        # Create a simple certificate
        st.success("🏆 Certificate of Completion")
        st.markdown(f"""
        <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; text-align: center;">
        <h2>🏆 Certificate of Completion</h2>
        <p><strong>This certifies that</strong></p>
        <h3>NLP Evolution Student</h3>
        <p><strong>has successfully completed</strong></p>
        <h3>"The Evolution of NLP: From Statistical Models to Modern AI"</h3>
        <p>Completed on: {datetime.now().strftime("%B %d, %Y")}</p>
        <p><em>You have demonstrated understanding of key concepts in Natural Language Processing, 
        from traditional statistical methods to modern transformer architectures.</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Final Encouragement
    st.subheader("💭 Final Thoughts")
    st.markdown("""
    You've just completed a comprehensive journey through the evolution of NLP. The field is moving 
    incredibly fast, with new breakthroughs happening regularly. The foundation you've built here 
    will serve you well as you continue to explore this exciting field.
    
    Remember:
    - **Stay curious** - The field is constantly evolving
    - **Practice regularly** - Build projects and experiment
    - **Join communities** - Learn from others and share your knowledge
    - **Keep learning** - There's always something new to discover
    
    **Thank you for taking this journey with us!** 🙏
    """)
    
    # Feedback section
    st.markdown("---")
    st.subheader("📝 Course Feedback")
    
    feedback_rating = st.slider("Rate this course:", 1, 5, 5)
    feedback_text = st.text_area("What did you like most? What could be improved?", 
                                placeholder="Your feedback helps us improve...")
    
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! 😊")
        # In a real application, you would save this to a database
        st.balloons()