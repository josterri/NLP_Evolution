"""
Interactive quiz system for NLP Evolution app.
Provides knowledge checks and assessments for each chapter.
"""

import streamlit as st
import pandas as pd
import random
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from utils import handle_errors

# Quiz questions organized by chapter
QUIZ_QUESTIONS = {
    "chapter0": {
        "title": "Chapter 0: Before Neural Networks",
        "questions": [
            {
                "id": "q0_1",
                "question": "Which was the first publicly demonstrated machine translation system?",
                "options": ["ELIZA", "Georgetown-IBM Experiment", "SHRDLU", "MYCIN"],
                "correct": 1,
                "explanation": "The Georgetown-IBM Experiment in 1954 was the first public demonstration of machine translation, translating 60 Russian sentences to English."
            },
            {
                "id": "q0_2",
                "question": "What was ELIZA designed to simulate?",
                "options": ["A chess player", "A psychotherapist", "A translator", "A mathematician"],
                "correct": 1,
                "explanation": "ELIZA was designed to simulate a Rogerian psychotherapist using pattern matching and substitution."
            },
            {
                "id": "q0_3",
                "question": "What is the main advantage of TF-IDF over simple word counts?",
                "options": ["Faster computation", "Handles rare words better", "Reduces vocabulary size", "Improves grammar"],
                "correct": 1,
                "explanation": "TF-IDF gives higher weights to rare, discriminative words while reducing the impact of common words."
            },
            {
                "id": "q0_4",
                "question": "Which of these was NOT a major limitation of rule-based NLP systems?",
                "options": ["Poor generalization", "Labor-intensive creation", "High computational cost", "Couldn't handle ambiguity"],
                "correct": 2,
                "explanation": "Rule-based systems were actually computationally efficient. Their main problems were poor generalization, manual effort, and inability to handle ambiguity."
            }
        ]
    },
    
    "chapter1": {
        "title": "Chapter 1: The Statistical Era",
        "questions": [
            {
                "id": "q1_1",
                "question": "What does the Markov assumption in n-gram models state?",
                "options": [
                    "All words are equally likely",
                    "The next word depends only on the previous n-1 words",
                    "Words are independent of each other",
                    "Longer sentences are more likely"
                ],
                "correct": 1,
                "explanation": "The Markov assumption states that the probability of the next word depends only on the previous n-1 words, not the entire history."
            },
            {
                "id": "q1_2",
                "question": "What is the main problem with n-gram models as n increases?",
                "options": ["Slower computation", "Data sparsity", "Memory usage", "All of the above"],
                "correct": 3,
                "explanation": "As n increases, n-gram models suffer from data sparsity (many n-grams never seen in training), require more memory, and become computationally expensive."
            },
            {
                "id": "q1_3",
                "question": "What is the purpose of smoothing in n-gram models?",
                "options": [
                    "Make text more readable",
                    "Reduce computation time",
                    "Handle unseen n-grams",
                    "Improve grammar"
                ],
                "correct": 2,
                "explanation": "Smoothing techniques help handle unseen n-grams by assigning small probabilities to word sequences not seen in training."
            }
        ]
    },
    
    "chapter2": {
        "title": "Chapter 2: Neural Networks & Embeddings",
        "questions": [
            {
                "id": "q2_1",
                "question": "What is the main advantage of word embeddings over one-hot encoding?",
                "options": [
                    "Smaller vocabulary size",
                    "Capture semantic similarity",
                    "Faster training",
                    "Better grammar"
                ],
                "correct": 1,
                "explanation": "Word embeddings capture semantic relationships between words, unlike one-hot encoding where all words are equidistant."
            },
            {
                "id": "q2_2",
                "question": "In Word2Vec, what does the context window determine?",
                "options": [
                    "The size of the vocabulary",
                    "The number of epochs",
                    "Which words are considered neighbors",
                    "The embedding dimension"
                ],
                "correct": 2,
                "explanation": "The context window determines how many words around a target word are considered as its neighbors for training."
            },
            {
                "id": "q2_3",
                "question": "What makes GloVe different from Word2Vec?",
                "options": [
                    "GloVe uses global word co-occurrence statistics",
                    "GloVe is faster to train",
                    "GloVe produces better embeddings",
                    "GloVe uses neural networks"
                ],
                "correct": 0,
                "explanation": "GloVe combines local context information with global word co-occurrence statistics, unlike Word2Vec which uses only local context."
            }
        ]
    },
    
    "chapter3": {
        "title": "Chapter 3: Sequential Models & Context",
        "questions": [
            {
                "id": "q3_1",
                "question": "What is the main problem that LSTMs solve compared to basic RNNs?",
                "options": [
                    "Faster training",
                    "Smaller model size",
                    "Vanishing gradient problem",
                    "Better accuracy"
                ],
                "correct": 2,
                "explanation": "LSTMs solve the vanishing gradient problem through their gating mechanisms and cell state, allowing them to learn long-range dependencies."
            },
            {
                "id": "q3_2",
                "question": "How many gates does an LSTM cell have?",
                "options": ["2", "3", "4", "5"],
                "correct": 1,
                "explanation": "LSTM has 3 gates: forget gate, input gate, and output gate. The cell state is updated using these gates."
            },
            {
                "id": "q3_3",
                "question": "What is the key innovation of attention mechanisms?",
                "options": [
                    "Faster computation",
                    "Smaller models",
                    "Focus on relevant parts of input",
                    "Better optimization"
                ],
                "correct": 2,
                "explanation": "Attention mechanisms allow models to focus on relevant parts of the input sequence when making predictions, rather than using a fixed representation."
            }
        ]
    },
    
    "chapter4": {
        "title": "Chapter 4: The Transformer Revolution",
        "questions": [
            {
                "id": "q4_1",
                "question": "In the attention mechanism, what do Query, Key, and Value represent?",
                "options": [
                    "Three types of neurons",
                    "Different learned transformations of the input",
                    "Separate neural networks",
                    "Different loss functions"
                ],
                "correct": 1,
                "explanation": "Query, Key, and Value are different learned linear transformations of the input that enable the attention mechanism to compute relevance scores."
            },
            {
                "id": "q4_2",
                "question": "What is the main advantage of self-attention over RNNs?",
                "options": [
                    "Lower memory usage",
                    "Better accuracy",
                    "Parallelizable computation",
                    "Simpler architecture"
                ],
                "correct": 2,
                "explanation": "Self-attention can be computed in parallel for all positions, unlike RNNs which must process sequentially."
            },
            {
                "id": "q4_3",
                "question": "What is the key difference between BERT and GPT?",
                "options": [
                    "BERT uses bidirectional attention, GPT uses unidirectional",
                    "BERT is larger than GPT",
                    "BERT is faster than GPT",
                    "BERT uses different activation functions"
                ],
                "correct": 0,
                "explanation": "BERT uses bidirectional attention (can see both past and future context), while GPT uses unidirectional attention (only past context) for autoregressive generation."
            }
        ]
    },
    
    "chapter5": {
        "title": "Chapter 5: Text Classification",
        "questions": [
            {
                "id": "q5_1",
                "question": "What is the main limitation of Bag of Words (BoW) representation?",
                "options": [
                    "It's too computationally expensive",
                    "It loses word order information",
                    "It can't handle large vocabularies",
                    "It requires pre-trained embeddings"
                ],
                "correct": 1,
                "explanation": "Bag of Words representation treats text as an unordered collection of words, losing all information about word order and context."
            },
            {
                "id": "q5_2",
                "question": "Which algorithm assumes feature independence in Naive Bayes classification?",
                "options": [
                    "The features are conditionally independent given the class",
                    "The classes are independent of each other",
                    "The training and test data are independent",
                    "The words are independent of their positions"
                ],
                "correct": 0,
                "explanation": "Naive Bayes assumes that features (words) are conditionally independent given the class label, which is often violated in practice but still works well."
            },
            {
                "id": "q5_3",
                "question": "What is the advantage of fine-tuning pre-trained models for classification?",
                "options": [
                    "They require less memory",
                    "They train faster from scratch",
                    "They leverage knowledge from large-scale pre-training",
                    "They don't need labeled data"
                ],
                "correct": 2,
                "explanation": "Fine-tuning pre-trained models leverages the general language understanding learned during pre-training, requiring less task-specific data for good performance."
            }
        ]
    },
    
    "chapter6": {
        "title": "Chapter 6: Generative Models",
        "questions": [
            {
                "id": "q6_1",
                "question": "What is the key difference between discriminative and generative models?",
                "options": [
                    "Discriminative models are faster",
                    "Generative models can create new samples",
                    "Discriminative models use more parameters",
                    "Generative models only work with text"
                ],
                "correct": 1,
                "explanation": "Generative models learn the joint probability distribution P(X,Y) and can generate new samples, while discriminative models only learn P(Y|X) for classification."
            },
            {
                "id": "q6_2",
                "question": "What is teacher forcing in sequence generation?",
                "options": [
                    "Using a larger model to teach a smaller one",
                    "Forcing the model to use specific tokens",
                    "Using ground truth tokens as input during training",
                    "Teaching the model with reinforcement learning"
                ],
                "correct": 2,
                "explanation": "Teacher forcing feeds the ground truth tokens as input during training instead of the model's own predictions, speeding up training but potentially causing exposure bias."
            },
            {
                "id": "q6_3",
                "question": "What is the purpose of temperature in text generation?",
                "options": [
                    "To control training speed",
                    "To adjust model size",
                    "To control randomness in sampling",
                    "To reduce memory usage"
                ],
                "correct": 2,
                "explanation": "Temperature scaling controls the randomness of sampling: lower temperature makes the model more deterministic, higher temperature increases diversity."
            }
        ]
    },
    
    "chapter7": {
        "title": "Chapter 7: Build Your Own Model",
        "questions": [
            {
                "id": "q7_1",
                "question": "What is the purpose of positional encoding in transformers?",
                "options": [
                    "To reduce model size",
                    "To inject position information into the model",
                    "To speed up attention computation",
                    "To normalize the input"
                ],
                "correct": 1,
                "explanation": "Since transformers don't have inherent position awareness like RNNs, positional encodings are added to give the model information about token positions."
            },
            {
                "id": "q7_2",
                "question": "What is gradient clipping used for in training?",
                "options": [
                    "To make training faster",
                    "To prevent exploding gradients",
                    "To reduce memory usage",
                    "To improve accuracy"
                ],
                "correct": 1,
                "explanation": "Gradient clipping prevents exploding gradients by limiting the maximum gradient norm, stabilizing training especially for RNNs and deep networks."
            },
            {
                "id": "q7_3",
                "question": "What is the typical vocabulary size for subword tokenization?",
                "options": [
                    "100-1,000 tokens",
                    "10,000-50,000 tokens",
                    "100,000-500,000 tokens",
                    "1-10 million tokens"
                ],
                "correct": 1,
                "explanation": "Subword tokenization typically uses 10K-50K tokens, balancing between character-level (too long sequences) and word-level (too large vocabulary)."
            }
        ]
    },
    
    "chapter8": {
        "title": "Chapter 8: Large Language Models",
        "questions": [
            {
                "id": "q8_1",
                "question": "What are emergent abilities in LLMs?",
                "options": [
                    "Abilities that emerge during fine-tuning",
                    "Capabilities that appear only at large scale",
                    "Features that emerge from data augmentation",
                    "Skills that emerge from multi-task training"
                ],
                "correct": 1,
                "explanation": "Emergent abilities are capabilities that appear suddenly at large model scales, like in-context learning, which smaller models cannot perform."
            },
            {
                "id": "q8_2",
                "question": "What is the key innovation of instruction tuning?",
                "options": [
                    "Training on code instead of text",
                    "Using larger batch sizes",
                    "Training to follow natural language instructions",
                    "Using more training data"
                ],
                "correct": 2,
                "explanation": "Instruction tuning trains models to follow natural language instructions, making them more helpful and aligned with user intent."
            },
            {
                "id": "q8_3",
                "question": "What is RLHF (Reinforcement Learning from Human Feedback)?",
                "options": [
                    "A new model architecture",
                    "A method to align models with human preferences",
                    "A data augmentation technique",
                    "A way to reduce model size"
                ],
                "correct": 1,
                "explanation": "RLHF uses reinforcement learning with rewards based on human feedback to align model outputs with human preferences and values."
            }
        ]
    },
    
    "chapter9": {
        "title": "Chapter 9: Future Directions",
        "questions": [
            {
                "id": "q9_1",
                "question": "What is a major challenge in multimodal AI?",
                "options": [
                    "Lack of compute power",
                    "Aligning representations across different modalities",
                    "Insufficient text data",
                    "Model architecture limitations"
                ],
                "correct": 1,
                "explanation": "The main challenge in multimodal AI is learning aligned representations that connect information across different modalities (text, image, audio, etc.)."
            },
            {
                "id": "q9_2",
                "question": "What is the goal of constitutional AI?",
                "options": [
                    "To make models run faster",
                    "To reduce training costs",
                    "To make AI systems more helpful, harmless, and honest",
                    "To improve model accuracy"
                ],
                "correct": 2,
                "explanation": "Constitutional AI aims to create AI systems that are helpful, harmless, and honest by training them to follow a set of principles or 'constitution'."
            },
            {
                "id": "q9_3",
                "question": "What is a potential risk of large language models?",
                "options": [
                    "They use too much electricity",
                    "They can generate misinformation or biased content",
                    "They are too slow for practical use",
                    "They require too much storage"
                ],
                "correct": 1,
                "explanation": "LLMs can generate convincing but false information and may perpetuate biases present in training data, requiring careful deployment and monitoring."
            }
        ]
    }
}

class QuizManager:
    """Manages quiz creation, scoring, and progress tracking."""
    
    def __init__(self):
        self.questions = QUIZ_QUESTIONS
        # Don't initialize session state in __init__ - do it lazily when needed
    
    def initialize_session_state(self):
        """Initialize session state for quiz tracking."""
        # Only initialize if streamlit context is available
        try:
            if "quiz_scores" not in st.session_state:
                st.session_state.quiz_scores = {}
            if "quiz_answers" not in st.session_state:
                st.session_state.quiz_answers = {}
            if "current_quiz" not in st.session_state:
                st.session_state.current_quiz = None
        except Exception as e:
            # Session state not available yet - will be initialized later
            pass
    
    @handle_errors
    def get_available_quizzes(self) -> Dict[str, str]:
        """Get list of available quizzes."""
        return {chapter: info["title"] for chapter, info in self.questions.items()}
    
    @handle_errors
    def start_quiz(self, chapter: str, randomize: bool = True) -> bool:
        """Start a quiz for a specific chapter."""
        self.initialize_session_state()  # Ensure session state is initialized
        if chapter not in self.questions:
            return False
        
        questions = self.questions[chapter]["questions"].copy()
        
        if randomize:
            random.shuffle(questions)
        
        st.session_state.current_quiz = {
            "chapter": chapter,
            "title": self.questions[chapter]["title"],
            "questions": questions,
            "current_question": 0,
            "user_answers": [],
            "start_time": datetime.now(),
            "completed": False
        }
        
        return True
    
    @handle_errors
    def get_current_question(self) -> Optional[Dict]:
        """Get the current question in the active quiz."""
        if not st.session_state.current_quiz:
            return None
        
        quiz = st.session_state.current_quiz
        if quiz["current_question"] >= len(quiz["questions"]):
            return None
        
        return quiz["questions"][quiz["current_question"]]
    
    @handle_errors
    def submit_answer(self, answer: int) -> bool:
        """Submit an answer for the current question."""
        if not st.session_state.current_quiz:
            return False
        
        quiz = st.session_state.current_quiz
        current_q = quiz["current_question"]
        
        if current_q >= len(quiz["questions"]):
            return False
        
        question = quiz["questions"][current_q]
        is_correct = answer == question["correct"]
        
        quiz["user_answers"].append({
            "question_id": question["id"],
            "user_answer": answer,
            "correct_answer": question["correct"],
            "is_correct": is_correct,
            "question_text": question["question"]
        })
        
        quiz["current_question"] += 1
        
        # Check if quiz is completed
        if quiz["current_question"] >= len(quiz["questions"]):
            quiz["completed"] = True
            quiz["end_time"] = datetime.now()
            self.save_quiz_results()
        
        return True
    
    @handle_errors
    def save_quiz_results(self):
        """Save quiz results to session state."""
        if not st.session_state.current_quiz or not st.session_state.current_quiz["completed"]:
            return
        
        quiz = st.session_state.current_quiz
        chapter = quiz["chapter"]
        
        # Calculate score
        correct_answers = sum(1 for answer in quiz["user_answers"] if answer["is_correct"])
        total_questions = len(quiz["questions"])
        score = (correct_answers / total_questions) * 100
        
        # Save results
        st.session_state.quiz_scores[chapter] = {
            "score": score,
            "correct": correct_answers,
            "total": total_questions,
            "completed_at": quiz["end_time"],
            "duration": quiz["end_time"] - quiz["start_time"]
        }
        
        st.session_state.quiz_answers[chapter] = quiz["user_answers"]
    
    @handle_errors
    def get_quiz_results(self, chapter: str) -> Optional[Dict]:
        """Get quiz results for a specific chapter."""
        self.initialize_session_state()  # Ensure session state is initialized
        return st.session_state.quiz_scores.get(chapter, None)
    
    @handle_errors
    def get_overall_progress(self) -> Dict:
        """Get overall quiz progress."""
        total_quizzes = len(self.questions)
        completed_quizzes = len(st.session_state.quiz_scores)
        
        if completed_quizzes == 0:
            return {"completion_rate": 0, "average_score": 0, "completed": 0, "total": total_quizzes}
        
        total_score = sum(result["score"] for result in st.session_state.quiz_scores.values())
        average_score = total_score / completed_quizzes
        
        return {
            "completion_rate": (completed_quizzes / total_quizzes) * 100,
            "average_score": average_score,
            "completed": completed_quizzes,
            "total": total_quizzes
        }

@handle_errors
def render_quiz_interface():
    """Render the main quiz interface."""
    st.subheader("🧠 Knowledge Check Quiz")
    
    quiz_manager = QuizManager()
    quiz_manager.initialize_session_state()  # Ensure session state is initialized
    
    # Check if there's an active quiz
    if st.session_state.current_quiz and not st.session_state.current_quiz["completed"]:
        render_active_quiz(quiz_manager)
    else:
        render_quiz_selection(quiz_manager)

@handle_errors
def render_quiz_selection(quiz_manager: QuizManager):
    """Render quiz selection interface."""
    st.markdown("Test your understanding of NLP concepts with interactive quizzes!")
    
    # Show overall progress
    progress = quiz_manager.get_overall_progress()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Quizzes Completed", f"{progress['completed']}/{progress['total']}")
    with col2:
        st.metric("Completion Rate", f"{progress['completion_rate']:.1f}%")
    with col3:
        st.metric("Average Score", f"{progress['average_score']:.1f}%")
    
    # Available quizzes
    st.markdown("### Available Quizzes")
    available_quizzes = quiz_manager.get_available_quizzes()
    
    for chapter, title in available_quizzes.items():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"**{title}**")
            
        with col2:
            # Show previous score if available
            results = quiz_manager.get_quiz_results(chapter)
            if results:
                st.success(f"✅ {results['score']:.1f}%")
            else:
                st.info("Not taken")
        
        with col3:
            if st.button(f"Start Quiz", key=f"start_{chapter}"):
                if quiz_manager.start_quiz(chapter):
                    st.rerun()
    
    # Quiz history
    if st.session_state.quiz_scores:
        st.markdown("### Quiz History")
        
        history_data = []
        for chapter, results in st.session_state.quiz_scores.items():
            history_data.append({
                "Quiz": available_quizzes[chapter],
                "Score": f"{results['score']:.1f}%",
                "Correct": f"{results['correct']}/{results['total']}",
                "Date": results['completed_at'].strftime("%Y-%m-%d %H:%M")
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)

@handle_errors
def render_active_quiz(quiz_manager: QuizManager):
    """Render active quiz interface."""
    quiz = st.session_state.current_quiz
    current_question = quiz_manager.get_current_question()
    
    if not current_question:
        render_quiz_results(quiz_manager)
        return
    
    # Progress bar
    progress = (quiz["current_question"] + 1) / len(quiz["questions"])
    st.progress(progress)
    
    # Question info
    st.markdown(f"**Question {quiz['current_question'] + 1} of {len(quiz['questions'])}**")
    st.markdown(f"### {current_question['question']}")
    
    # Answer options
    selected_answer = st.radio(
        "Choose your answer:",
        range(len(current_question["options"])),
        format_func=lambda x: current_question["options"][x],
        key=f"q_{current_question['id']}"
    )
    
    # Submit button
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Submit Answer", type="primary"):
            quiz_manager.submit_answer(selected_answer)
            
            # Show immediate feedback
            is_correct = selected_answer == current_question["correct"]
            if is_correct:
                st.success("✅ Correct!")
            else:
                st.error("❌ Incorrect")
                st.info(f"**Correct answer:** {current_question['options'][current_question['correct']]}")
            
            st.info(f"**Explanation:** {current_question['explanation']}")
            
            # Auto-advance after a delay
            if quiz["current_question"] < len(quiz["questions"]):
                if st.button("Next Question"):
                    st.rerun()
            else:
                if st.button("View Results"):
                    st.rerun()
    
    with col2:
        if st.button("Exit Quiz"):
            st.session_state.current_quiz = None
            st.rerun()

def render_quiz_results(quiz_manager: QuizManager):
    """Render quiz results."""
    quiz = st.session_state.current_quiz
    
    if not quiz or not quiz["completed"]:
        return
    
    st.markdown("## 🎉 Quiz Completed!")
    
    # Calculate final score
    correct_answers = sum(1 for answer in quiz["user_answers"] if answer["is_correct"])
    total_questions = len(quiz["questions"])
    score = (correct_answers / total_questions) * 100
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Final Score", f"{score:.1f}%")
    with col2:
        st.metric("Correct Answers", f"{correct_answers}/{total_questions}")
    with col3:
        duration = quiz["end_time"] - quiz["start_time"]
        st.metric("Time Taken", f"{duration.seconds // 60}:{duration.seconds % 60:02d}")
    
    # Performance feedback
    if score >= 80:
        st.success("🌟 Excellent work! You have a strong understanding of this topic.")
    elif score >= 60:
        st.info("👍 Good job! You understand most concepts but might want to review a few areas.")
    else:
        st.warning("📚 Consider reviewing this chapter's content to strengthen your understanding.")
    
    # Detailed results
    st.markdown("### Detailed Results")
    
    for i, answer in enumerate(quiz["user_answers"], 1):
        with st.expander(f"Question {i}: {'✅' if answer['is_correct'] else '❌'}"):
            st.write(f"**Question:** {answer['question_text']}")
            
            question_data = next(q for q in quiz["questions"] if q["id"] == answer["question_id"])
            options = question_data["options"]
            
            st.write(f"**Your answer:** {options[answer['user_answer']]}")
            st.write(f"**Correct answer:** {options[answer['correct_answer']]}")
            
            if not answer["is_correct"]:
                st.write(f"**Explanation:** {question_data['explanation']}")
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Retake Quiz"):
            quiz_manager.start_quiz(quiz["chapter"])
            st.rerun()
    
    with col2:
        if st.button("Back to Quiz Menu"):
            st.session_state.current_quiz = None
            st.rerun()

def render_quiz_widget(chapter: str):
    """Render a compact quiz widget for a specific chapter."""
    quiz_manager = QuizManager()
    quiz_manager.initialize_session_state()  # Ensure session state is initialized
    
    # Check if quiz exists for this chapter
    if chapter not in quiz_manager.questions:
        return
    
    st.markdown("### 🧠 Quick Knowledge Check")
    
    # Show previous score if available
    results = quiz_manager.get_quiz_results(chapter)
    if results:
        st.success(f"Previous score: {results['score']:.1f}% ({results['correct']}/{results['total']})")
    
    if st.button(f"Take Quiz for {quiz_manager.questions[chapter]['title']}", key=f"quiz_widget_{chapter}"):
        if quiz_manager.start_quiz(chapter):
            st.rerun()

def get_quiz_score(chapter: str) -> Optional[float]:
    """Get the quiz score for a specific chapter."""
    if "quiz_scores" not in st.session_state:
        return None
    
    results = st.session_state.quiz_scores.get(chapter, None)
    return results["score"] if results else None