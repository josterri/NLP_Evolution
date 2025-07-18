"""
Theme management system for NLP Evolution app.
Provides dark mode, custom themes, and responsive design.
"""

import streamlit as st
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class ThemeManager:
    """Manages app themes and styling."""
    
    def __init__(self):
        self.themes = {
            "light": {
                "primary_color": "#1f77b4",
                "background_color": "#ffffff",
                "secondary_background_color": "#f0f2f6",
                "text_color": "#262730",
                "font": "sans serif"
            },
            "dark": {
                "primary_color": "#ff6b35",
                "background_color": "#0e1117",
                "secondary_background_color": "#262730",
                "text_color": "#fafafa",
                "font": "sans serif"
            },
            "high_contrast": {
                "primary_color": "#ffff00",
                "background_color": "#000000",
                "secondary_background_color": "#1a1a1a",
                "text_color": "#ffffff",
                "font": "monospace"
            },
            "academic": {
                "primary_color": "#2e86ab",
                "background_color": "#f8f9fa",
                "secondary_background_color": "#e9ecef",
                "text_color": "#212529",
                "font": "serif"
            }
        }
    
    def get_current_theme(self) -> Dict[str, str]:
        """Get the current theme configuration."""
        # Ensure theme is initialized
        if 'current_theme' not in st.session_state:
            st.session_state.current_theme = 'light'
        return self.themes.get(st.session_state.current_theme, self.themes['light'])
    
    def set_theme(self, theme_name: str) -> bool:
        """
        Set the current theme.
        
        Args:
            theme_name: Name of the theme to set
            
        Returns:
            True if theme was set successfully, False otherwise
        """
        if theme_name in self.themes:
            st.session_state.current_theme = theme_name
            logger.info(f"Theme changed to: {theme_name}")
            return True
        else:
            logger.warning(f"Unknown theme: {theme_name}")
            return False
    
    def inject_custom_css(self) -> None:
        """Inject custom CSS for the current theme."""
        theme = self.get_current_theme()
        
        css = f"""
        <style>
        /* Main theme colors */
        .stApp {{
            background-color: {theme['background_color']};
            color: {theme['text_color']};
            font-family: {theme['font']};
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            background-color: {theme['secondary_background_color']};
        }}
        
        /* Button styling */
        .stButton > button {{
            background-color: {theme['primary_color']};
            color: white;
            border: none;
            border-radius: 5px;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background-color: {self._darken_color(theme['primary_color'], 0.1)};
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        /* Metric styling */
        .metric-container {{
            background-color: {theme['secondary_background_color']};
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid {theme['primary_color']};
        }}
        
        /* Code block styling */
        .stCode {{
            background-color: {theme['secondary_background_color']};
            color: {theme['text_color']};
            border-radius: 8px;
        }}
        
        /* Quiz styling */
        .quiz-question {{
            background: linear-gradient(135deg, {theme['secondary_background_color']}, {theme['background_color']});
            padding: 20px;
            border-radius: 15px;
            margin: 15px 0;
            border: 2px solid {theme['primary_color']};
        }}
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {{
            background-color: {theme['primary_color']};
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: {theme['secondary_background_color']};
            color: {theme['text_color']};
        }}
        
        /* Dark mode specific adjustments */
        {self._get_dark_mode_css() if st.session_state.current_theme == 'dark' else ''}
        
        /* High contrast mode adjustments */
        {self._get_high_contrast_css() if st.session_state.current_theme == 'high_contrast' else ''}
        
        /* Mobile responsive design */
        @media (max-width: 768px) {{
            .main-content {{
                padding: 10px;
            }}
            
            .stColumns > div {{
                margin-bottom: 20px;
            }}
            
            .metric-container {{
                padding: 10px;
            }}
        }}
        
        /* Animation keyframes */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.5s ease-out;
        }}
        
        /* Accessibility improvements */
        .stButton > button:focus {{
            outline: 3px solid {theme['primary_color']};
            outline-offset: 2px;
        }}
        
        /* Loading spinner styling */
        .stSpinner > div {{
            border-top-color: {theme['primary_color']} !important;
        }}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {theme['secondary_background_color']};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {theme['primary_color']};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {self._darken_color(theme['primary_color'], 0.2)};
        }}
        </style>
        """
        
        st.markdown(css, unsafe_allow_html=True)
    
    def _darken_color(self, color: str, factor: float) -> str:
        """Darken a hex color by a given factor."""
        # Simple color darkening - in production, use a proper color library
        if color.startswith('#'):
            color = color[1:]
        
        try:
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            
            r = max(0, int(r * (1 - factor)))
            g = max(0, int(g * (1 - factor)))
            b = max(0, int(b * (1 - factor)))
            
            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return color
    
    def _get_dark_mode_css(self) -> str:
        """Get additional CSS for dark mode."""
        return """
        /* Dark mode specific styles */
        .stMarkdown {{
            color: #fafafa;
        }}
        
        .stSelectbox > div > div {{
            background-color: #262730;
            color: #fafafa;
        }}
        
        .stTextInput > div > div > input {{
            background-color: #262730;
            color: #fafafa;
            border: 1px solid #4a4a4a;
        }}
        
        .stTextArea > div > div > textarea {{
            background-color: #262730;
            color: #fafafa;
            border: 1px solid #4a4a4a;
        }}
        """
    
    def _get_high_contrast_css(self) -> str:
        """Get additional CSS for high contrast mode."""
        return """
        /* High contrast mode styles */
        .stMarkdown {{
            color: #ffffff;
            font-weight: bold;
        }}
        
        .stButton > button {{
            border: 2px solid #ffffff;
            font-weight: bold;
        }}
        
        a {{
            color: #ffff00 !important;
            text-decoration: underline !important;
        }}
        """
    
    def render_theme_selector(self) -> None:
        """Render the theme selector widget."""
        with st.sidebar.expander(" Theme Settings"):
            # Ensure theme is initialized
            if 'current_theme' not in st.session_state:
                st.session_state.current_theme = 'light'
            current_theme = st.session_state.current_theme
            
            new_theme = st.selectbox(
                "Choose Theme:",
                options=list(self.themes.keys()),
                index=list(self.themes.keys()).index(current_theme),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            if new_theme != current_theme:
                if self.set_theme(new_theme):
                    st.success(f"Theme changed to {new_theme.replace('_', ' ').title()}")
                    st.rerun()
            
            # Theme preview
            theme = self.get_current_theme()
            st.markdown("**Theme Preview:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div style="
                    background-color: {theme['primary_color']};
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    margin: 5px 0;
                ">Primary</div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="
                    background-color: {theme['secondary_background_color']};
                    color: {theme['text_color']};
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    margin: 5px 0;
                    border: 1px solid {theme['primary_color']};
                ">Secondary</div>
                """, unsafe_allow_html=True)
            
            # Accessibility options
            st.markdown("**Accessibility:**")
            
            reduce_animations = st.checkbox("Reduce animations", key="reduce_animations")
            large_text = st.checkbox("Large text", key="large_text")
            
            if large_text:
                self._inject_accessibility_css()
    
    def _inject_accessibility_css(self) -> None:
        """Inject accessibility-focused CSS."""
        st.markdown("""
        <style>
        .stMarkdown, .stText, .stSelectbox, .stButton {
            font-size: 1.2em !important;
        }
        </style>
        """, unsafe_allow_html=True)


# Keyboard shortcuts removed - focusing on simplified UI


# Global instances
theme_manager = ThemeManager()


def apply_theme() -> None:
    """Apply the current theme to the app."""
    theme_manager.inject_custom_css()


def render_theme_controls() -> None:
    """Render theme controls in the sidebar."""
    theme_manager.render_theme_selector()


if __name__ == "__main__":
    # Test the theme manager
    st.title("Theme Manager Test")
    
    apply_theme()
    render_theme_controls()
    
    st.write("This is a test of the theme manager.")
    st.button("Test Button")
    st.slider("Test Slider", 0, 100, 50)