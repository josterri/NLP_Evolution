"""
Utility functions for the NLP Evolution application.
Provides error handling, logging, and common functionality.
"""

import streamlit as st
import logging
import functools
import traceback
from typing import Any, Callable, Optional, Dict, List
import time
import os
import json
import psutil
from datetime import datetime
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nlp_evolution.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def handle_errors(func: Callable) -> Callable:
    """
    Decorator to handle errors gracefully in Streamlit functions.
    Shows user-friendly error messages while logging detailed errors.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"File not found in {func.__name__}: {e}")
            st.error(f"📁 File not found: {str(e)}")
            st.info("Please check that all required files are in the correct location.")
        except ImportError as e:
            logger.error(f"Import error in {func.__name__}: {e}")
            st.error(f"📦 Missing dependency: {str(e)}")
            st.info("Please install the required packages using: `pip install -r requirements.txt`")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            st.error(f"⚠️ An unexpected error occurred: {str(e)}")
            with st.expander("🔍 Technical Details"):
                st.code(traceback.format_exc())
            st.info("If this error persists, please check the logs or report an issue.")
        
        return None
    return wrapper

def log_user_interaction(action: str, details: Optional[str] = None):
    """Log user interactions for analytics and debugging."""
    logger.info(f"User action: {action}" + (f" - {details}" if details else ""))

def safe_file_operation(filepath: str, operation: str = "read") -> bool:
    """
    Safely check if file operations can be performed.
    
    Args:
        filepath: Path to the file
        operation: Type of operation ('read', 'write', 'exists')
    
    Returns:
        bool: True if operation is safe, False otherwise
    """
    try:
        if operation == "exists":
            return os.path.exists(filepath)
        elif operation == "read":
            return os.path.exists(filepath) and os.access(filepath, os.R_OK)
        elif operation == "write":
            dir_path = os.path.dirname(filepath)
            return os.access(dir_path, os.W_OK) if dir_path else True
        else:
            return False
    except Exception as e:
        logger.error(f"Error checking file operation {operation} for {filepath}: {e}")
        return False

def display_loading_message(message: str, duration: float = 2.0):
    """Display a loading message for a specified duration."""
    placeholder = st.empty()
    with placeholder.container():
        st.info(f"⏳ {message}")
        time.sleep(duration)
    placeholder.empty()

def validate_input(input_value: Any, input_type: str, min_length: int = 0, max_length: int = 1000) -> bool:
    """
    Validate user input.
    
    Args:
        input_value: The value to validate
        input_type: Type of input ('text', 'number', 'email', etc.)
        min_length: Minimum length for text inputs
        max_length: Maximum length for text inputs
    
    Returns:
        bool: True if input is valid, False otherwise
    """
    try:
        if input_type == "text":
            if not isinstance(input_value, str):
                return False
            return min_length <= len(input_value.strip()) <= max_length
        elif input_type == "number":
            try:
                float(input_value)
                return True
            except (ValueError, TypeError):
                return False
        elif input_type == "positive_number":
            try:
                return float(input_value) > 0
            except (ValueError, TypeError):
                return False
        else:
            return True
    except Exception as e:
        logger.error(f"Input validation error: {e}")
        return False

def create_download_link(content: str, filename: str, link_text: str = "Download") -> str:
    """
    Create a download link for text content.
    
    Args:
        content: The content to download
        filename: Name for the downloaded file
        link_text: Text to display for the link
    
    Returns:
        str: HTML for the download link
    """
    import base64
    
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def format_number(num: float, decimal_places: int = 2) -> str:
    """Format numbers for display."""
    if num >= 1e6:
        return f"{num/1e6:.{decimal_places}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{decimal_places}f}K"
    else:
        return f"{num:.{decimal_places}f}"

def get_system_info() -> dict:
    """Get system information for debugging."""
    import platform
    import sys
    
    return {
        "platform": platform.system(),
        "python_version": sys.version,
        "streamlit_version": st.__version__,
        "working_directory": os.getcwd()
    }

class ProgressTracker:
    """Simple progress tracking for user learning journey."""
    
    def __init__(self):
        if 'progress' not in st.session_state:
            st.session_state.progress = {
                'chapters_visited': set(),
                'sections_completed': set(),
                'exercises_attempted': set(),
                'start_time': time.time()
            }
    
    def mark_chapter_visited(self, chapter_id: str):
        """Mark a chapter as visited."""
        st.session_state.progress['chapters_visited'].add(chapter_id)
        log_user_interaction(f"Chapter visited: {chapter_id}")
    
    def mark_section_completed(self, section_id: str):
        """Mark a section as completed."""
        st.session_state.progress['sections_completed'].add(section_id)
        log_user_interaction(f"Section completed: {section_id}")
    
    def mark_exercise_attempted(self, exercise_id: str):
        """Mark an exercise as attempted."""
        st.session_state.progress['exercises_attempted'].add(exercise_id)
        log_user_interaction(f"Exercise attempted: {exercise_id}")
    
    def get_progress_stats(self) -> dict:
        """Get progress statistics."""
        progress = st.session_state.progress
        total_time = time.time() - progress['start_time']
        
        return {
            'chapters_visited': len(progress['chapters_visited']),
            'sections_completed': len(progress['sections_completed']),
            'exercises_attempted': len(progress['exercises_attempted']),
            'total_time_minutes': total_time / 60,
            'completion_percentage': min(100, len(progress['chapters_visited']) / 9 * 100)
        }

def display_progress_sidebar():
    """Display progress information in the sidebar."""
    tracker = ProgressTracker()
    stats = tracker.get_progress_stats()
    
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Your Progress")
        
        # Progress bar
        progress_percentage = stats['completion_percentage']
        st.progress(progress_percentage / 100)
        st.caption(f"{progress_percentage:.1f}% Complete")
        
        # Quick stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chapters", stats['chapters_visited'])
        with col2:
            st.metric("Sections", stats['sections_completed'])
        
        # Time spent
        time_spent = stats['total_time_minutes']
        if time_spent < 60:
            time_str = f"{time_spent:.0f} min"
        else:
            time_str = f"{time_spent/60:.1f} hours"
        st.caption(f"Time spent: {time_str}")

def initialize_session_state():
    """Initialize session state variables."""
    import time
    defaults = {
        'current_chapter': 'Chapter 1',
        'current_theme': 'light',
        'user_preferences': {
            'show_code': True,
            'show_exercises': True,
            'auto_advance': False
        },
        'models_cache': {},
        'data_cache': {},
        'app_start_time': time.time(),
        'current_time': time.time(),
        'completed_chapters': set(),
        'chapters_visited': set()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_data
def load_sample_data(filename: str) -> Optional[str]:
    """Load sample data files with caching."""
    try:
        if safe_file_operation(filename, "read"):
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            logger.warning(f"Sample data file not found: {filename}")
            return None
    except Exception as e:
        logger.error(f"Error loading sample data {filename}: {e}")
        return None

def display_system_info():
    """Display system information for debugging."""
    with st.expander("🔧 System Information"):
        info = get_system_info()
        for key, value in info.items():
            st.text(f"{key}: {value}")

def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'matplotlib', 
        'plotly', 'seaborn', 'nltk', 'torch'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.error(f"Missing required packages: {', '.join(missing_packages)}")
        st.code(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

# ===== Evaluation Metrics Module =====

class NLPMetrics:
    """
    A comprehensive collection of NLP evaluation metrics for various tasks.
    Includes metrics for classification, generation, and sequence labeling.
    """
    
    @staticmethod
    def accuracy(y_true: list, y_pred: list) -> float:
        """
        Calculate accuracy for classification tasks.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score between 0 and 1
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Length of y_true and y_pred must be equal")
        
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true) if y_true else 0.0
    
    @staticmethod
    def precision_recall_f1(y_true: list, y_pred: list, target_class=None):
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_class: If specified, calculate for this class (binary). Otherwise, macro average.
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        from collections import defaultdict
        
        if target_class is not None:
            # Binary classification metrics
            tp = sum(1 for true, pred in zip(y_true, y_pred) 
                    if true == target_class and pred == target_class)
            fp = sum(1 for true, pred in zip(y_true, y_pred) 
                    if true != target_class and pred == target_class)
            fn = sum(1 for true, pred in zip(y_true, y_pred) 
                    if true == target_class and pred != target_class)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        else:
            # Macro average for multiclass
            classes = set(y_true + y_pred)
            metrics = defaultdict(list)
            
            for cls in classes:
                cls_metrics = NLPMetrics.precision_recall_f1(y_true, y_pred, target_class=cls)
                for metric, value in cls_metrics.items():
                    metrics[metric].append(value)
            
            return {
                metric: sum(values) / len(values) if values else 0.0
                for metric, values in metrics.items()
            }
    
    @staticmethod
    def bleu_score(reference: str, hypothesis: str, n_gram: int = 4) -> float:
        """
        Calculate BLEU score for text generation tasks.
        Simplified implementation for demonstration.
        
        Args:
            reference: Reference text
            hypothesis: Generated text
            n_gram: Maximum n-gram size (default: 4)
            
        Returns:
            BLEU score between 0 and 1
        """
        import math
        from collections import Counter
        
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        # Calculate brevity penalty
        bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(hyp_tokens))
        
        # Calculate n-gram precision
        precisions = []
        
        for n in range(1, min(n_gram + 1, len(hyp_tokens) + 1)):
            ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
            hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1))
            
            overlap = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())
            
            precision = overlap / total if total > 0 else 0.0
            precisions.append(precision)
        
        # Calculate geometric mean
        if precisions and all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0.0
        
        return bp * geo_mean
    
    @staticmethod
    def perplexity(log_likelihoods: list) -> float:
        """
        Calculate perplexity for language modeling tasks.
        
        Args:
            log_likelihoods: List of log probabilities for each token
            
        Returns:
            Perplexity score (lower is better)
        """
        import math
        
        if not log_likelihoods:
            return float('inf')
        
        avg_log_likelihood = sum(log_likelihoods) / len(log_likelihoods)
        return math.exp(-avg_log_likelihood)
    
    @staticmethod
    def rouge_score(reference: str, hypothesis: str, rouge_type: str = "rouge-1") -> dict:
        """
        Calculate ROUGE score for summarization tasks.
        Simplified implementation.
        
        Args:
            reference: Reference text
            hypothesis: Generated text
            rouge_type: Type of ROUGE score ("rouge-1", "rouge-2", "rouge-l")
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if rouge_type == "rouge-1":
            ref_set = set(ref_tokens)
            hyp_set = set(hyp_tokens)
        elif rouge_type == "rouge-2":
            ref_set = set(zip(ref_tokens[:-1], ref_tokens[1:]))
            hyp_set = set(zip(hyp_tokens[:-1], hyp_tokens[1:]))
        elif rouge_type == "rouge-l":
            # Simplified LCS calculation
            lcs_length = NLPMetrics._lcs_length(ref_tokens, hyp_tokens)
            precision = lcs_length / len(hyp_tokens) if hyp_tokens else 0.0
            recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            return {"precision": precision, "recall": recall, "f1": f1}
        else:
            raise ValueError(f"Unknown rouge_type: {rouge_type}")
        
        overlap = len(ref_set & hyp_set)
        precision = overlap / len(hyp_set) if hyp_set else 0.0
        recall = overlap / len(ref_set) if ref_set else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    @staticmethod
    def _lcs_length(seq1: list, seq2: list) -> int:
        """Helper function to calculate longest common subsequence length."""
        if not seq1 or not seq2:
            return 0
        
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def ner_metrics(true_entities: list, pred_entities: list) -> dict:
        """
        Calculate metrics for Named Entity Recognition tasks.
        
        Args:
            true_entities: List of tuples (text, start, end, label)
            pred_entities: List of tuples (text, start, end, label)
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        true_set = set((e[1], e[2], e[3]) for e in true_entities)  # (start, end, label)
        pred_set = set((e[1], e[2], e[3]) for e in pred_entities)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": len(true_set)
        }

def display_metrics_comparison(metrics_dict: dict, title: str = "Metrics Comparison"):
    """
    Display a comparison of different metrics in Streamlit.
    
    Args:
        metrics_dict: Dictionary of metric names to values
        title: Title for the display
    """
    st.subheader(title)
    
    cols = st.columns(len(metrics_dict))
    for i, (metric_name, value) in enumerate(metrics_dict.items()):
        with cols[i]:
            if isinstance(value, dict):
                st.metric(metric_name, f"{value.get('f1', 0):.3f}")
                with st.expander("Details"):
                    for k, v in value.items():
                        st.text(f"{k}: {v:.3f}")
            else:
                st.metric(metric_name, f"{value:.3f}")

def evaluate_model_interactive():
    """
    Interactive model evaluation widget for Streamlit apps.
    Allows users to input predictions and see various metrics.
    """
    st.subheader("🎯 Interactive Model Evaluation")
    
    task_type = st.selectbox(
        "Select Task Type:",
        ["Classification", "Text Generation", "Named Entity Recognition"]
    )
    
    if task_type == "Classification":
        col1, col2 = st.columns(2)
        
        with col1:
            true_labels = st.text_area(
                "True Labels (comma-separated):",
                value="positive,negative,positive,negative,positive"
            ).split(",")
            
        with col2:
            pred_labels = st.text_area(
                "Predicted Labels (comma-separated):",
                value="positive,negative,negative,negative,positive"
            ).split(",")
        
        if st.button("Calculate Metrics"):
            true_labels = [label.strip() for label in true_labels]
            pred_labels = [label.strip() for label in pred_labels]
            
            if len(true_labels) == len(pred_labels):
                metrics = NLPMetrics()
                
                accuracy = metrics.accuracy(true_labels, pred_labels)
                prf = metrics.precision_recall_f1(true_labels, pred_labels)
                
                display_metrics_comparison({
                    "Accuracy": accuracy,
                    "Precision": prf["precision"],
                    "Recall": prf["recall"],
                    "F1 Score": prf["f1"]
                })
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                classes = sorted(set(true_labels + pred_labels))
                matrix = [[0] * len(classes) for _ in range(len(classes))]
                
                for true, pred in zip(true_labels, pred_labels):
                    true_idx = classes.index(true)
                    pred_idx = classes.index(pred)
                    matrix[true_idx][pred_idx] += 1
                
                import pandas as pd
                df_cm = pd.DataFrame(matrix, index=classes, columns=classes)
                st.dataframe(df_cm)
            else:
                st.error("Number of true labels and predicted labels must match!")
    
    elif task_type == "Text Generation":
        reference = st.text_area(
            "Reference Text:",
            value="The cat sat on the mat"
        )
        
        hypothesis = st.text_area(
            "Generated Text:",
            value="The cat was on the mat"
        )
        
        if st.button("Calculate Metrics"):
            metrics = NLPMetrics()
            
            bleu = metrics.bleu_score(reference, hypothesis)
            rouge = metrics.rouge_score(reference, hypothesis, "rouge-1")
            
            display_metrics_comparison({
                "BLEU-4": bleu,
                "ROUGE-1": rouge
            })
    
    elif task_type == "Named Entity Recognition":
        st.info("Enter entities as: text,start,end,label (one per line)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            true_entities_text = st.text_area(
                "True Entities:",
                value="Apple,0,5,ORG\nJobs,6,10,PER"
            )
            
        with col2:
            pred_entities_text = st.text_area(
                "Predicted Entities:",
                value="Apple,0,5,ORG\nSteve Jobs,0,10,PER"
            )
        
        if st.button("Calculate Metrics"):
            true_entities = []
            pred_entities = []
            
            for line in true_entities_text.strip().split('\n'):
                parts = line.split(',')
                if len(parts) == 4:
                    true_entities.append((parts[0], int(parts[1]), int(parts[2]), parts[3]))
            
            for line in pred_entities_text.strip().split('\n'):
                parts = line.split(',')
                if len(parts) == 4:
                    pred_entities.append((parts[0], int(parts[1]), int(parts[2]), parts[3]))
            
            metrics = NLPMetrics()
            ner_results = metrics.ner_metrics(true_entities, pred_entities)
            
            display_metrics_comparison({
                "NER Metrics": ner_results
            })


# ===== NEW SESSION STATE CLEANUP FUNCTIONS =====

def cleanup_session_state(keep_keys: Optional[List[str]] = None):
    """
    Clean up large objects from session state to prevent memory bloat.
    
    Args:
        keep_keys: List of keys to preserve during cleanup
    """
    if keep_keys is None:
        keep_keys = ['progress_tracker', 'current_chapter', 'quiz_scores', 
                     'exercise_progress', 'completed_chapters']
    
    # Keys that typically hold large objects
    large_object_keys = ['model_cache', 'embeddings_cache', 'large_datasets',
                        'trained_models', 'visualization_cache']
    
    cleaned_count = 0
    for key in list(st.session_state.keys()):
        if key in large_object_keys and key not in keep_keys:
            try:
                del st.session_state[key]
                cleaned_count += 1
                logger.info(f"Cleaned up session state key: {key}")
            except Exception as e:
                logger.error(f"Error cleaning up {key}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} items from session state")


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }


def monitor_memory_usage(threshold_mb: float = 500):
    """Monitor memory usage and log warnings if threshold exceeded."""
    memory_stats = get_memory_usage()
    
    if memory_stats['rss_mb'] > threshold_mb:
        logger.warning(f"High memory usage detected: {memory_stats['rss_mb']:.1f}MB")
        # Only show UI warnings if streamlit is available and properly initialized
        try:
            st.warning(f"⚠️ High memory usage: {memory_stats['rss_mb']:.1f}MB. Consider refreshing the page.")
            
            # Trigger automatic cleanup if very high
            if memory_stats['rss_mb'] > threshold_mb * 1.5:
                cleanup_session_state()
                st.info("🧹 Automatic cleanup performed to free memory.")
        except Exception as e:
            # If streamlit is not ready, just log the warning
            logger.warning(f"Could not display memory warning in UI: {e}")
    
    return memory_stats


# ===== PROGRESS INDICATORS =====

@contextmanager
def show_progress(message: str = "Processing...", show_time: bool = True):
    """
    Context manager for showing progress with optional time tracking.
    
    Usage:
        with show_progress("Loading model...") as progress:
            # Do work
            progress(0.5)  # Update to 50%
            # More work
            progress(1.0)  # Complete
    """
    # Initialize placeholders as None and create them only when first accessed
    progress_bar = None
    status_text = None
    start_time = time.time()
    
    def update_progress(value: float, text: Optional[str] = None):
        nonlocal progress_bar, status_text
        # Create streamlit components only when actually needed
        if progress_bar is None:
            progress_bar = st.progress(0)
        if status_text is None:
            status_text = st.empty()
            
        progress_bar.progress(value)
        if text:
            status_text.text(text)
        elif show_time:
            elapsed = time.time() - start_time
            status_text.text(f"{message} ({elapsed:.1f}s)")
        else:
            status_text.text(message)
    
    # Don't call update_progress(0) here - wait for first actual call
    
    try:
        yield update_progress
    finally:
        # Only try to clean up if components were actually created
        if progress_bar is not None:
            progress_bar.empty()
        if status_text is not None:
            status_text.empty()
        if show_time:
            elapsed = time.time() - start_time
            logger.info(f"Progress '{message}' completed in {elapsed:.1f}s")


# Export functionality removed - focusing on core learning experience


# ===== INPUT VALIDATION IMPROVEMENTS =====

def validate_text_input(text: str, min_length: int = 1, max_length: int = 1000,
                       allowed_chars: Optional[str] = None) -> tuple[bool, str]:
    """
    Validate text input with detailed error messages.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(text, str):
        return False, "Input must be a string"
    
    if len(text) < min_length:
        return False, f"Input must be at least {min_length} characters long"
    
    if len(text) > max_length:
        return False, f"Input must be no more than {max_length} characters long"
    
    if allowed_chars and not all(c in allowed_chars for c in text):
        return False, "Input contains invalid characters"
    
    return True, ""


def validate_numeric_input(value: Any, min_val: Optional[float] = None,
                         max_val: Optional[float] = None,
                         allow_negative: bool = True) -> tuple[bool, str]:
    """
    Validate numeric input with detailed error messages.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        return False, "Input must be a number"
    
    if not allow_negative and num_value < 0:
        return False, "Input must be non-negative"
    
    if min_val is not None and num_value < min_val:
        return False, f"Input must be at least {min_val}"
    
    if max_val is not None and num_value > max_val:
        return False, f"Input must be no more than {max_val}"
    
    return True, ""


# Help system removed - replaced with comprehensive glossary


# ===== LAZY LOADING =====

def lazy_load_module(module_name: str):
    """Lazy load a module only when needed."""
    import importlib
    
    try:
        if module_name not in st.session_state.get('loaded_modules', set()):
            module = importlib.import_module(module_name)
            
            if 'loaded_modules' not in st.session_state:
                st.session_state.loaded_modules = set()
            
            st.session_state.loaded_modules.add(module_name)
            logger.info(f"Lazy loaded module: {module_name}")
            return module
        else:
            return importlib.import_module(module_name)
    except ImportError as e:
        logger.error(f"Failed to lazy load {module_name}: {e}")
        raise


# ===== CACHING WITH TTL =====

@st.cache_data(ttl=3600)  # 1 hour cache
def cached_expensive_operation(operation_id: str, *args, **kwargs):
    """Generic caching wrapper for expensive operations."""
    logger.info(f"Executing cached operation: {operation_id}")
    # This would be replaced with actual operation
    return f"Cached result for {operation_id}"