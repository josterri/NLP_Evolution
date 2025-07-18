"""
Real dataset integration for NLP Evolution app.
Provides easy access to popular NLP datasets for hands-on learning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, List, Tuple, Optional
import random
from utils import handle_errors

# Sample datasets for immediate use (no external downloads required)
SAMPLE_DATASETS = {
    "imdb_reviews": {
        "name": "IMDB Movie Reviews (Sample)",
        "description": "Sentiment analysis dataset with movie reviews",
        "task": "sentiment_analysis",
        "size": 100,
        "data": [
            {"text": "This movie was absolutely fantastic! Great acting and storyline.", "label": "positive"},
            {"text": "Terrible film, waste of time. Poor acting and boring plot.", "label": "negative"},
            {"text": "Amazing cinematography and brilliant performances throughout.", "label": "positive"},
            {"text": "Completely disappointed. Expected much more from this movie.", "label": "negative"},
            {"text": "One of the best films I've seen this year. Highly recommend!", "label": "positive"},
            {"text": "Boring and predictable. Nothing new or exciting.", "label": "negative"},
            {"text": "Excellent direction and outstanding cast performance.", "label": "positive"},
            {"text": "Poorly written script with wooden acting.", "label": "negative"},
            {"text": "Masterpiece of cinema with incredible emotional depth.", "label": "positive"},
            {"text": "Overhyped and underwhelming. Not worth watching.", "label": "negative"}
        ]
    },
    
    "ag_news": {
        "name": "AG News (Sample)",
        "description": "News article classification dataset",
        "task": "text_classification",
        "size": 80,
        "classes": ["World", "Sports", "Business", "Technology"],
        "data": [
            {"text": "The stock market showed significant gains today as investors responded positively to earnings reports.", "label": "Business"},
            {"text": "Scientists have discovered a new exoplanet that may have conditions suitable for life.", "label": "Technology"},
            {"text": "The championship game ended with a thrilling overtime victory for the home team.", "label": "Sports"},
            {"text": "International leaders gathered for emergency talks following the diplomatic crisis.", "label": "World"},
            {"text": "New smartphone technology promises to revolutionize mobile computing.", "label": "Technology"},
            {"text": "The football season kicks off with high expectations for several teams.", "label": "Sports"},
            {"text": "Economic indicators suggest a strong recovery in the manufacturing sector.", "label": "Business"},
            {"text": "Humanitarian aid continues to flow to regions affected by natural disasters.", "label": "World"}
        ]
    },
    
    "customer_reviews": {
        "name": "Customer Reviews (Sample)",
        "description": "Product review sentiment analysis",
        "task": "sentiment_analysis",
        "size": 60,
        "data": [
            {"text": "Great product, exactly as described. Fast shipping too!", "label": "positive"},
            {"text": "Poor quality, broke after one week. Not recommended.", "label": "negative"},
            {"text": "Good value for money. Would buy again.", "label": "positive"},
            {"text": "Arrived damaged and customer service was unhelpful.", "label": "negative"},
            {"text": "Excellent quality and exceeded my expectations.", "label": "positive"},
            {"text": "Cheaply made, not worth the price.", "label": "negative"}
        ]
    },
    
    "spam_detection": {
        "name": "Email Spam Detection (Sample)",
        "description": "Email spam vs ham classification",
        "task": "binary_classification",
        "size": 50,
        "data": [
            {"text": "Congratulations! You've won $1000! Click here to claim your prize now!", "label": "spam"},
            {"text": "Meeting scheduled for tomorrow at 3 PM. Please confirm your attendance.", "label": "ham"},
            {"text": "URGENT: Your account will be closed! Update your info immediately!", "label": "spam"},
            {"text": "Thanks for your presentation today. The team was impressed.", "label": "ham"},
            {"text": "FREE MONEY! No strings attached! Act now before it's too late!", "label": "spam"},
            {"text": "Can you send me the report by Friday? Thanks!", "label": "ham"}
        ]
    },
    
    "question_answering": {
        "name": "Question Answering (Sample)",
        "description": "Reading comprehension dataset",
        "task": "question_answering",
        "size": 40,
        "data": [
            {
                "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
                "question": "Who is the Eiffel Tower named after?",
                "answer": "Gustave Eiffel"
            },
            {
                "context": "Python is a high-level programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability with its notable use of significant whitespace.",
                "question": "When was Python first released?",
                "answer": "1991"
            },
            {
                "context": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "question": "What is machine learning a subset of?",
                "answer": "artificial intelligence"
            }
        ]
    }
}

class DatasetManager:
    """Manages dataset loading and preprocessing for the NLP Evolution app."""
    
    def __init__(self):
        self.datasets = SAMPLE_DATASETS
        self.cache = {}
    
    @handle_errors
    def get_available_datasets(self) -> Dict[str, Dict]:
        """Get list of available datasets."""
        return {
            key: {
                "name": dataset["name"],
                "description": dataset["description"],
                "task": dataset["task"],
                "size": dataset["size"]
            }
            for key, dataset in self.datasets.items()
        }
    
    @handle_errors
    def load_dataset(self, dataset_name: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load a dataset by name."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset = self.datasets[dataset_name]
        data = dataset["data"].copy()
        
        # Sample data if requested
        if sample_size and sample_size < len(data):
            data = random.sample(data, sample_size)
        
        return pd.DataFrame(data)
    
    @handle_errors
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get detailed information about a dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset = self.datasets[dataset_name]
        df = pd.DataFrame(dataset["data"])
        
        info = {
            "name": dataset["name"],
            "description": dataset["description"],
            "task": dataset["task"],
            "total_size": len(dataset["data"]),
            "sample_size": len(df),
            "columns": list(df.columns),
        }
        
        # Add task-specific info
        if dataset["task"] in ["sentiment_analysis", "binary_classification"]:
            info["labels"] = df["label"].unique().tolist()
            info["label_distribution"] = df["label"].value_counts().to_dict()
        elif dataset["task"] == "text_classification":
            info["classes"] = dataset.get("classes", df["label"].unique().tolist())
            info["class_distribution"] = df["label"].value_counts().to_dict()
        
        return info
    
    @handle_errors
    def preview_dataset(self, dataset_name: str, n_samples: int = 5) -> pd.DataFrame:
        """Get a preview of the dataset."""
        df = self.load_dataset(dataset_name)
        return df.head(n_samples)
    
    @handle_errors
    def split_dataset(self, dataset_name: str, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train and test sets."""
        df = self.load_dataset(dataset_name)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split
        split_idx = int(len(df) * train_ratio)
        train_df = df[:split_idx]
        test_df = df[split_idx:]
        
        return train_df, test_df
    
    @handle_errors
    def get_class_examples(self, dataset_name: str, class_name: str, n_examples: int = 3) -> List[str]:
        """Get examples of a specific class from the dataset."""
        df = self.load_dataset(dataset_name)
        
        if "label" not in df.columns:
            return []
        
        class_examples = df[df["label"] == class_name]["text"].tolist()
        return class_examples[:n_examples]

@st.cache_data
def load_dataset_cached(dataset_name: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Cached version of dataset loading."""
    manager = DatasetManager()
    return manager.load_dataset(dataset_name, sample_size)

def render_dataset_explorer():
    """Render an interactive dataset explorer."""
    st.subheader("📊 Dataset Explorer")
    
    manager = DatasetManager()
    available_datasets = manager.get_available_datasets()
    
    # Dataset selection
    dataset_names = list(available_datasets.keys())
    selected_dataset = st.selectbox(
        "Choose a dataset to explore:",
        dataset_names,
        format_func=lambda x: available_datasets[x]["name"]
    )
    
    if selected_dataset:
        # Dataset info
        info = manager.get_dataset_info(selected_dataset)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", info["total_size"])
        with col2:
            st.metric("Task Type", info["task"].replace("_", " ").title())
        with col3:
            if "labels" in info:
                st.metric("Classes", len(info["labels"]))
        
        st.write(f"**Description:** {info['description']}")
        
        # Preview data
        st.markdown("### 📋 Data Preview")
        preview_df = manager.preview_dataset(selected_dataset, n_samples=10)
        st.dataframe(preview_df)
        
        # Label distribution
        if "label_distribution" in info or "class_distribution" in info:
            st.markdown("### 📊 Label Distribution")
            
            dist_data = info.get("label_distribution", info.get("class_distribution", {}))
            
            # Create bar chart
            labels = list(dist_data.keys())
            counts = list(dist_data.values())
            
            chart_data = pd.DataFrame({
                "Label": labels,
                "Count": counts
            })
            
            st.bar_chart(chart_data.set_index("Label"))
        
        # Sample by class
        if "labels" in info:
            st.markdown("### 🔍 Examples by Class")
            
            selected_class = st.selectbox("Select a class to see examples:", info["labels"])
            
            if selected_class:
                examples = manager.get_class_examples(selected_dataset, selected_class, n_examples=5)
                
                for i, example in enumerate(examples, 1):
                    with st.expander(f"Example {i}"):
                        st.write(example)
        
        # Download option
        st.markdown("### 💾 Use This Dataset")
        
        if st.button("Load Dataset for Analysis"):
            df = manager.load_dataset(selected_dataset)
            st.session_state.loaded_dataset = df
            st.session_state.dataset_name = selected_dataset
            st.success(f"Dataset '{info['name']}' loaded successfully! You can now use it in other sections.")
            
            # Show how to access it
            st.code(f"""
# Access the loaded dataset in your code:
df = st.session_state.loaded_dataset
print(f"Dataset shape: {{df.shape}}")
print(f"Columns: {{df.columns.tolist()}}")
            """)

def render_dataset_widget():
    """Render a compact dataset selection widget."""
    st.markdown("### 📊 Choose Dataset")
    
    manager = DatasetManager()
    available_datasets = manager.get_available_datasets()
    
    dataset_names = list(available_datasets.keys())
    selected_dataset = st.selectbox(
        "Dataset:",
        dataset_names,
        format_func=lambda x: available_datasets[x]["name"],
        key="dataset_widget"
    )
    
    if selected_dataset:
        sample_size = st.slider(
            "Sample size:",
            min_value=10,
            max_value=available_datasets[selected_dataset]["size"],
            value=min(50, available_datasets[selected_dataset]["size"]),
            key="sample_size_widget"
        )
        
        if st.button("Load Dataset", key="load_dataset_widget"):
            df = load_dataset_cached(selected_dataset, sample_size)
            st.session_state.current_dataset = df
            st.session_state.current_dataset_name = selected_dataset
            st.success(f"Loaded {len(df)} samples from {available_datasets[selected_dataset]['name']}")
            
            # Show preview
            st.dataframe(df.head())
    
    return selected_dataset

def get_current_dataset() -> Optional[pd.DataFrame]:
    """Get the currently loaded dataset."""
    return st.session_state.get("current_dataset", None)

def get_dataset_for_task(task_type: str) -> List[str]:
    """Get datasets suitable for a specific task."""
    suitable_datasets = []
    
    for dataset_name, dataset_info in SAMPLE_DATASETS.items():
        if dataset_info["task"] == task_type:
            suitable_datasets.append(dataset_name)
    
    return suitable_datasets