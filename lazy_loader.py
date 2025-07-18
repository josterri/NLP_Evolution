"""
Lazy loading system for NLP Evolution app modules.
Reduces initial memory footprint by loading chapters only when needed.
"""

import streamlit as st
import importlib
import logging
from typing import Dict, Callable, Optional
from utils import handle_errors

logger = logging.getLogger(__name__)

class LazyChapterLoader:
    """Manages lazy loading of chapter modules."""
    
    def __init__(self):
        self.loaded_modules: Dict[str, object] = {}
        self.chapter_mapping = {
            "Chapter 0: Before Neural Networks": "chapter0",
            "Chapter 1: The Statistical Era": "chapter1", 
            "Chapter 2: The Rise of Neural Networks & Embeddings": "chapter2",
            "Chapter 3: Sequential Models & The Power of Context": "chapter3",
            "Chapter 4: The Transformer Revolution": "chapter4",
            "Chapter 5: Applying the Foundations: Text Classification": "chapter5",
            "Chapter 6: The Rise of Generative Models": "chapter6",
            "Chapter 7: Build Your Own Generative Model": "chapter7",
            "Chapter 8: The Era of Large Language Models (LLMs)": "chapter8",
            "Chapter 9: Course Completion & Future Directions": "chapter9",
        }
        
        # Initialize session state for tracking loaded modules
        if 'lazy_loaded_modules' not in st.session_state:
            st.session_state.lazy_loaded_modules = set()
    
    @handle_errors
    def load_chapter(self, chapter_name: str) -> Optional[Callable]:
        """
        Lazy load a chapter module and return its render function.
        
        Args:
            chapter_name: Display name of the chapter
            
        Returns:
            Render function for the chapter, or None if loading fails
        """
        module_name = self.chapter_mapping.get(chapter_name)
        if not module_name:
            logger.error(f"Unknown chapter: {chapter_name}")
            return None
        
        # Check if already loaded
        if module_name in self.loaded_modules:
            render_func = getattr(self.loaded_modules[module_name], f"render_{module_name}", None)
            if render_func:
                return render_func
        
        # Load the module
        try:
            if st.session_state.get('show_loading_spinner', True):
                with st.spinner(f"Loading {chapter_name}..."):
                    module = importlib.import_module(module_name)
            else:
                module = importlib.import_module(module_name)
                
            self.loaded_modules[module_name] = module
            st.session_state.lazy_loaded_modules.add(module_name)
            
            # Get the render function
            render_func = getattr(module, f"render_{module_name}", None)
            if render_func:
                logger.info(f"Successfully lazy loaded: {module_name}")
                return render_func
            else:
                logger.error(f"No render function found in {module_name}")
                return None
                    
        except ImportError as e:
            logger.error(f"Failed to import {module_name}: {e}")
            st.error(f"Failed to load {chapter_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {module_name}: {e}")
            st.error(f"Error loading {chapter_name}: {str(e)}")
            return None
    
    @handle_errors
    def load_interactive_module(self, module_name: str) -> Optional[object]:
        """
        Lazy load interactive modules (quiz, exercises, etc.).
        
        Args:
            module_name: Name of the module to load
            
        Returns:
            Loaded module or None if loading fails
        """
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]
        
        try:
            with st.spinner(f"Loading {module_name}..."):
                module = importlib.import_module(module_name)
                self.loaded_modules[module_name] = module
                st.session_state.lazy_loaded_modules.add(module_name)
                logger.info(f"Successfully lazy loaded interactive module: {module_name}")
                return module
                
        except ImportError as e:
            logger.error(f"Failed to import {module_name}: {e}")
            st.error(f"Failed to load {module_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading {module_name}: {e}")
            st.error(f"Error loading {module_name}: {str(e)}")
            return None
    
    def get_loaded_modules(self) -> list:
        """Get list of currently loaded modules."""
        return list(st.session_state.get('lazy_loaded_modules', set()))
    
    def unload_module(self, module_name: str) -> bool:
        """
        Unload a module to free memory.
        
        Args:
            module_name: Name of module to unload
            
        Returns:
            True if successfully unloaded, False otherwise
        """
        try:
            if module_name in self.loaded_modules:
                del self.loaded_modules[module_name]
                
            if module_name in st.session_state.get('lazy_loaded_modules', set()):
                st.session_state.lazy_loaded_modules.remove(module_name)
                
            logger.info(f"Successfully unloaded module: {module_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading {module_name}: {e}")
            return False
    
    def clear_all_modules(self) -> int:
        """
        Clear all loaded modules to free memory.
        
        Returns:
            Number of modules cleared
        """
        count = len(self.loaded_modules)
        self.loaded_modules.clear()
        st.session_state.lazy_loaded_modules = set()
        logger.info(f"Cleared {count} lazy loaded modules")
        return count


class LazyFunctionLoader:
    """Lazy loader for expensive functions within modules."""
    
    def __init__(self):
        self.function_cache: Dict[str, Callable] = {}
    
    @handle_errors
    def lazy_function(self, module_name: str, function_name: str) -> Optional[Callable]:
        """
        Get a function from a module, loading it lazily if needed.
        
        Args:
            module_name: Name of the module
            function_name: Name of the function
            
        Returns:
            Function if found, None otherwise
        """
        cache_key = f"{module_name}.{function_name}"
        
        if cache_key in self.function_cache:
            return self.function_cache[cache_key]
        
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, function_name, None)
            
            if func:
                self.function_cache[cache_key] = func
                logger.info(f"Lazy loaded function: {cache_key}")
                return func
            else:
                logger.warning(f"Function {function_name} not found in {module_name}")
                return None
                
        except ImportError as e:
            logger.error(f"Failed to import {module_name} for function {function_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error loading function {cache_key}: {e}")
            return None


# Global instances
lazy_chapter_loader = LazyChapterLoader()
lazy_function_loader = LazyFunctionLoader()


def get_chapter_render_functions() -> Dict[str, Callable]:
    """
    Get all chapter render functions with lazy loading.
    
    Returns:
        Dictionary mapping chapter names to render functions
    """
    render_functions = {}
    
    for chapter_name in lazy_chapter_loader.chapter_mapping.keys():
        # Create a function that will lazy load and execute when called
        def create_lazy_render(name):
            def lazy_render():
                render_func = lazy_chapter_loader.load_chapter(name)
                if render_func:
                    return render_func()
                else:
                    st.error(f"Failed to load {name}")
            return lazy_render
        
        render_functions[chapter_name] = create_lazy_render(chapter_name)
    
    return render_functions


def get_interactive_render_functions() -> Dict[str, Callable]:
    """
    Get interactive module render functions with lazy loading.
    
    Returns:
        Dictionary mapping module names to render functions
    """
    interactive_modules = {
        "🧠 Knowledge Check Quizzes": ("quiz_system", "render_quiz_interface"),
        "💻 Interactive Code Exercises": ("code_exercises", "render_code_exercise_interface"),
        "📊 Dataset Explorer": ("datasets", "render_dataset_explorer"),
        "🔍 Search Content": ("search_functionality", "render_search_interface"),
        "📥 Export Progress": ("export_progress", "render_export_interface"),
    }
    
    render_functions = {}
    
    for display_name, (module_name, func_name) in interactive_modules.items():
        def create_lazy_function(mod_name, fn_name):
            def lazy_render():
                module = lazy_chapter_loader.load_interactive_module(mod_name)
                if module:
                    render_func = getattr(module, fn_name, None)
                    if render_func:
                        return render_func()
                    else:
                        st.error(f"Function {fn_name} not found in {mod_name}")
                else:
                    st.error(f"Failed to load module {mod_name}")
            return lazy_render
        
        render_functions[display_name] = create_lazy_function(module_name, func_name)
    
    return render_functions


def render_lazy_loading_stats():
    """Render statistics about lazy loading in the sidebar."""
    with st.sidebar.expander("🔄 Lazy Loading Stats"):
        loaded_modules = lazy_chapter_loader.get_loaded_modules()
        st.write(f"**Loaded Modules:** {len(loaded_modules)}")
        
        if loaded_modules:
            st.write("**Currently Loaded:**")
            for module in loaded_modules:
                st.write(f"• {module}")
        
        # Memory usage from utils
        from utils import get_memory_usage
        memory_stats = get_memory_usage()
        st.write(f"**Memory Usage:** {memory_stats['rss_mb']:.1f}MB")
        
        if st.button("Clear Loaded Modules", help="Free memory by unloading all modules"):
            count = lazy_chapter_loader.clear_all_modules()
            st.success(f"Cleared {count} modules")
            st.rerun()


if __name__ == "__main__":
    # Test the lazy loader
    st.title("Lazy Loading Test")
    
    loader = LazyChapterLoader()
    render_func = loader.load_chapter("Chapter 1: The Statistical Era")
    
    if render_func:
        st.success("Successfully loaded chapter 1")
        render_func()
    else:
        st.error("Failed to load chapter 1")
    
    render_lazy_loading_stats()