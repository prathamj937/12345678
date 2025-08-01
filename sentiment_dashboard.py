#!/usr/bin/env python3
"""
MD&A Sentiment & Complexity Dashboard
Real-time NLP-powered analysis of financial disclosures
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
import torch
from transformers import pipeline
import re
import warnings
from typing import Dict, List, Tuple, Any
from collections import Counter
import textstat
from datetime import datetime
import io
from analyzer import FinancialSentimentAnalyzer

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# Import the sentiment analyzer (embedded for standalone functionality)
# NOTE: Replace this section with your actual analyzer import if running as separate modules
# from financial_sentiment_analyzer import FinancialSentimentAnalyzer

warnings.filterwarnings('ignore')
# ...existing imports...

class FinancialSentimentAnalyzer:
    def __init__(self):
        """Initialize the analyzer"""
        self.finbert_analyzer = None
        self.setup_models()
        self.define_valence_shifters()
        self.define_macro_keywords()
        self.create_sample_ml_lexicon()
        self.setup_lm_dictionary()

    @st.cache_resource
    def setup_models(_self):  # Changed 'self' to '_self' to make it hashable
        """Initialize FinBERT model for sentiment analysis"""
        print("Loading FinBERT model...")
        try:
            # First check if transformers is installed
            try:
                from transformers import pipeline
                import torch

                # Check if CUDA is available
                device = 0 if torch.cuda.is_available() else -1
                
                _self.finbert_analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=device
                )
                print("✓ FinBERT model loaded successfully")
                
            except ImportError:
                print("Error: transformers package not installed.")
                print("Please run: pip install transformers torch")
                _self.finbert_analyzer = None
                return
                
        except Exception as e:
            print(f"Critical error in setup_models: {str(e)}")
            _self.finbert_analyzer = None

        if _self.finbert_analyzer is None:
            print("Warning: FinBERT analyzer not initialized. Some functionality will be limited.")
        
    def create_sample_ml_lexicon(self):
        """Create sample ML lexicon"""
        sample_data = {
            'decline': {'pos_prob': 0.1, 'neg_prob': 0.8, 'neutral_prob': 0.1, 'intensity': 0.7, 'frequency': 0.3},
            'growth': {'pos_prob': 0.8, 'neg_prob': 0.1, 'neutral_prob': 0.1, 'intensity': 0.8, 'frequency': 0.4},
            'challenging': {'pos_prob': 0.2, 'neg_prob': 0.7, 'neutral_prob': 0.1, 'intensity': 0.6, 'frequency': 0.3},
            'strong': {'pos_prob': 0.7, 'neg_prob': 0.2, 'neutral_prob': 0.1, 'intensity': 0.7, 'frequency': 0.5},
            'significant': {'pos_prob': 0.3, 'neg_prob': 0.4, 'neutral_prob': 0.3, 'intensity': 0.6, 'frequency': 0.6},
            'improve': {'pos_prob': 0.8, 'neg_prob': 0.1, 'neutral_prob': 0.1, 'intensity': 0.7, 'frequency': 0.4},
            'loss': {'pos_prob': 0.1, 'neg_prob': 0.8, 'neutral_prob': 0.1, 'intensity': 0.8, 'frequency': 0.3},
            'revenue': {'pos_prob': 0.5, 'neg_prob': 0.2, 'neutral_prob': 0.3, 'intensity': 0.5, 'frequency': 0.8},
            'profitable': {'pos_prob': 0.9, 'neg_prob': 0.05, 'neutral_prob': 0.05, 'intensity': 0.9, 'frequency': 0.4},
            'uncertainty': {'pos_prob': 0.1, 'neg_prob': 0.8, 'neutral_prob': 0.1, 'intensity': 0.7, 'frequency': 0.4}
        }
        self.ml_lexicon = sample_data
    
    def setup_lm_dictionary(self):
        """Setup basic LM dictionary"""
        positive_words = [
            'able', 'abundance', 'accomplish', 'achievement', 'advance', 'advantage',
            'agree', 'attractive', 'benefit', 'best', 'better', 'brilliant', 'capable',
            'competitive', 'complete', 'confident', 'creative', 'effective', 'efficient',
            'enhance', 'excellent', 'exciting', 'expand', 'gain', 'good', 'great',
            'growing', 'growth', 'high', 'improve', 'increase', 'innovation', 'leading',
            'opportunities', 'optimal', 'outstanding', 'positive', 'profitable', 'progress',
            'strong', 'success', 'successful', 'superior', 'valuable', 'winning'
        ]
        
        negative_words = [
            'abandon', 'adverse', 'allegations', 'bankruptcy', 'breach', 'burden',
            'challenge', 'complaint', 'concern', 'conflict', 'crisis', 'critical',
            'damage', 'danger', 'decline', 'decrease', 'defeat', 'deficiency',
            'deteriorate', 'difficult', 'disappointing', 'disaster', 'dispute',
            'fail', 'failure', 'fear', 'force', 'harm', 'hurt', 'impossible',
            'inadequate', 'loss', 'negative', 'obstacle', 'problem', 'recession',
            'risk', 'serious', 'severe', 'threat', 'uncertain', 'unfavorable',
            'vulnerable', 'weakness', 'worse', 'worst'
        ]
        
        self.lm_positive = set(positive_words)
        self.lm_negative = set(negative_words)
    
    def define_valence_shifters(self):
        """Define valence shifters from the provided image"""
        # Based on the uploaded image
        self.valence_shifters = {
            'amplifiers': [
                'absolutely', 'acute', 'acutely', 'almost', 'certainly', 'considerably',
                'decidedly', 'deep', 'deeply', 'definite', 'enormous', 'especially',
                'extreme', 'extremely', 'few', 'greatly', 'haven\'t', 'heavily',
                'heavy', 'high', 'highly', 'however', 'huge', 'hugely', 'incredibly',
                'least', 'little', 'massive', 'massively', 'more', 'most', 'much',
                'particularly', 'partly', 'purpose', 'purposely', 'quite', 'rarely',
                'real', 'really', 'seldom', 'serious', 'seriously', 'severe',
                'severely', 'significant', 'significantly', 'slightly', 'somewhat',
                'sporadically', 'sure', 'totally', 'true', 'truly', 'uber',
                'vastly', 'very'
            ],
            'de_amplifiers': [
                'almost', 'fairly', 'few', 'incredibly', 'least', 'little',
                'only', 'quite', 'rarely', 'really', 'seldom', 'slightly',
                'somewhat', 'sporadically'
            ],
            'negators': [
                'although', 'but', 'cannot', 'can\'t', 'certain', 'decidedly',
                'doesn\'t', 'dont', 'enormous', 'extreme', 'haven\'t', 'neither',
                'never', 'no', 'nobody', 'none', 'nor', 'not', 'werent',
                'whereas', 'won\'t'
            ],
            'adversative_conjunctions': [
                'although', 'but', 'however', 'neither', 'whereas'
            ]
        }
        
        # Flatten all valence shifters
        self.all_valence_shifters = set()
        for category, words in self.valence_shifters.items():
            self.all_valence_shifters.update(words)
    
    def define_macro_keywords(self):
        """Define macro keywords"""
        self.macro_keywords = [
            'covid-19', 'covid', 'coronavirus', 'pandemic', 'lockdown',
            'inflation', 'recession', 'geopolitical', 'trade war', 'supply chain',
            'disruption', 'energy crisis', 'interest rates', 'federal reserve',
            'unemployment', 'economic uncertainty', 'market volatility'
        ]
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences"""
        sentences = nltk.sent_tokenize(text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words"""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words
    
    def analyze_finbert_sentiment(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze sentiment using FinBERT"""
        if not self.finbert_analyzer:
            # Fallback to simple sentiment for demo
            return self._fallback_sentiment_analysis(sentences)
        
        sentence_results = []
        scores = []
        
        for sentence in sentences:
            try:
                truncated = sentence[:512] if len(sentence) > 512 else sentence
                result = self.finbert_analyzer(truncated)[0]
                
                if result['label'] == 'positive':
                    score = result['score']
                elif result['label'] == 'negative':
                    score = -result['score']
                else:
                    score = 0
                
                sentence_results.append({
                    'sentence': sentence,
                    'label': result['label'],
                    'confidence': result['score'],
                    'normalized_score': score
                })
                scores.append(score)
                
            except Exception:
                sentence_results.append({
                    'sentence': sentence,
                    'label': 'neutral',
                    'confidence': 0.5,
                    'normalized_score': 0.0
                })
                scores.append(0.0)
        
        avg_score = np.mean(scores) if scores else 0
        pos_count = sum(1 for r in sentence_results if r['label'] == 'positive')
        neg_count = sum(1 for r in sentence_results if r['label'] == 'negative')
        neu_count = sum(1 for r in sentence_results if r['label'] == 'neutral')
        total_sentences = len(sentence_results)
        
        return {
            'sentence_results': sentence_results,
            'document_metrics': {
                'average_score': round(avg_score, 4),
                'positive_percent': round((pos_count / total_sentences) * 100, 2) if total_sentences > 0 else 0,
                'negative_percent': round((neg_count / total_sentences) * 100, 2) if total_sentences > 0 else 0,
                'neutral_percent': round((neu_count / total_sentences) * 100, 2) if total_sentences > 0 else 0,
                'total_sentences': total_sentences
            }
        }
    
    def _fallback_sentiment_analysis(self, sentences: List[str]) -> Dict[str, Any]:
        """Fallback sentiment analysis using LM dictionary"""
        sentence_results = []
        scores = []
        
        for sentence in sentences:
            words = self.tokenize_words(sentence)
            pos_count = sum(1 for word in words if word in self.lm_positive)
            neg_count = sum(1 for word in words if word in self.lm_negative)
            
            if pos_count > neg_count:
                label = 'positive'
                score = 0.6
            elif neg_count > pos_count:
                label = 'negative'
                score = -0.6
            else:
                label = 'neutral'
                score = 0.0
            
            sentence_results.append({
                'sentence': sentence,
                'label': label,
                'confidence': 0.6,
                'normalized_score': score
            })
            scores.append(score)
        
        avg_score = np.mean(scores) if scores else 0
        pos_count = sum(1 for r in sentence_results if r['label'] == 'positive')
        neg_count = sum(1 for r in sentence_results if r['label'] == 'negative')
        neu_count = sum(1 for r in sentence_results if r['label'] == 'neutral')
        total_sentences = len(sentence_results)
        
        return {
            'sentence_results': sentence_results,
            'document_metrics': {
                'average_score': round(avg_score, 4),
                'positive_percent': round((pos_count / total_sentences) * 100, 2) if total_sentences > 0 else 0,
                'negative_percent': round((neg_count / total_sentences) * 100, 2) if total_sentences > 0 else 0,
                'neutral_percent': round((neu_count / total_sentences) * 100, 2) if total_sentences > 0 else 0,
                'total_sentences': total_sentences
            }
        }
    
    def analyze_ml_lexicon_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using ML Lexicon"""
        words = self.tokenize_words(text)
        word_scores = {}
        total_score = 0
        total_weight = 0
        
        for word in words:
            if word in self.ml_lexicon:
                entry = self.ml_lexicon[word]
                pos_prob = entry.get('pos_prob', 0)
                neg_prob = entry.get('neg_prob', 0)
                intensity = entry.get('intensity', 1)
                frequency = entry.get('frequency', 1)
                
                polarity = pos_prob - neg_prob
                weighted_score = polarity * intensity * frequency
                
                if word in word_scores:
                    word_scores[word]['count'] += 1
                    word_scores[word]['total_contribution'] += weighted_score
                else:
                    word_scores[word] = {
                        'count': 1,
                        'polarity': polarity,
                        'weighted_score': weighted_score,
                        'total_contribution': weighted_score
                    }
                
                total_score += weighted_score
                total_weight += intensity * frequency
        
        normalized_score = total_score / total_weight if total_weight > 0 else 0
        
        return {
            'total_score': round(total_score, 4),
            'normalized_score': round(normalized_score, 4),
            'word_contributions': word_scores
        }
    
    def analyze_lm_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using LM dictionary"""
        words = self.tokenize_words(text)
        positive_count = sum(1 for word in words if word in self.lm_positive)
        negative_count = sum(1 for word in words if word in self.lm_negative)
        total_sentiment_words = positive_count + negative_count
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words if total_sentiment_words > 0 else 0
        
        return {
            'sentiment_score': round(sentiment_score, 4),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'total_words': len(words)
        }
    
    def calculate_semantic_complexity_index(self, sentences: List[str]) -> Dict[str, Any]:
        """Calculate SCI"""
        sentences_with_shifters = 0
        sentence_details = []
        
        for sentence in sentences:
            words = self.tokenize_words(sentence)
            shifters_found = [word for word in words if word in self.all_valence_shifters]
            has_shifters = len(shifters_found) > 0
            
            if has_shifters:
                sentences_with_shifters += 1
            
            sentence_details.append({
                'sentence': sentence,
                'has_valence_shifters': has_shifters,
                'shifters_found': shifters_found
            })
        
        sci_percentage = (sentences_with_shifters / len(sentences)) * 100 if sentences else 0
        
        return {
            'sci_percentage': round(sci_percentage, 2),
            'sentences_with_shifters': sentences_with_shifters,
            'sentence_details': sentence_details
        }
    
    def calculate_readability_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate readability metrics"""
        sentences = self.tokenize_sentences(text)
        words = self.tokenize_words(text)
        
        if not sentences or not words:
            return {'gunning_fog': 0, 'flesch_kincaid': 0}
        
        # Gunning Fog
        complex_words = sum(1 for word in words if self.count_syllables(word) >= 3)
        avg_sentence_length = len(words) / len(sentences)
        complex_word_ratio = (complex_words / len(words)) * 100
        gunning_fog = 0.4 * (avg_sentence_length + complex_word_ratio)
        
        # Flesch-Kincaid
        try:
            flesch_kincaid = textstat.flesch_kincaid_grade(text)
        except:
            flesch_kincaid = gunning_fog * 0.8  # Approximation
        
        return {
            'gunning_fog': round(gunning_fog, 2),
            'flesch_kincaid': round(flesch_kincaid, 2)
        }
    
    def count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def calculate_fox_index(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """Calculate Fox Index"""
        words = self.tokenize_words(text)
        
        if not sentences or not words:
            return {'fox_index': 0}
        
        avg_sentence_length = len(words) / len(sentences)
        length_score = min(100, max(0, (avg_sentence_length - 10) * 2.5))
        
        passive_indicators = ['was', 'were', 'been', 'being', 'be']
        passive_count = sum(1 for word in words if word in passive_indicators)
        passive_ratio = (passive_count / len(words)) * 100
        passive_score = min(100, passive_ratio * 10)
        
        complexity_words = [word for word in words if word in self.all_valence_shifters]
        complexity_ratio = (len(complexity_words) / len(words)) * 100
        complexity_score = min(100, complexity_ratio * 5)
        
        fox_index = (length_score * 0.4 + passive_score * 0.3 + complexity_score * 0.3)
        
        return {'fox_index': round(fox_index, 2)}
    
    def analyze_macro_context(self, sentences: List[str]) -> Dict[str, Any]:
        """Analyze macro context"""
        macro_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            macro_terms_found = [keyword for keyword in self.macro_keywords if keyword in sentence_lower]
            
            if macro_terms_found:
                macro_sentences.append({
                    'sentence': sentence,
                    'macro_terms': macro_terms_found
                })
        
        return {
            'macro_flagged_sentences': macro_sentences,
            'total_macro_sentences': len(macro_sentences)
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Main analysis method"""
        sentences = self.tokenize_sentences(text)
        
        results = {
            'finbert': self.analyze_finbert_sentiment(sentences),
            'ml_lexicon': self.analyze_ml_lexicon_sentiment(text),
            'lm_dictionary': self.analyze_lm_sentiment(text),
            'sci': self.calculate_semantic_complexity_index(sentences),
            'readability': self.calculate_readability_metrics(text),
            'fox_index': self.calculate_fox_index(text, sentences),
            'macro_context': self.analyze_macro_context(sentences)
        }
        
        return results


# Initialize the analyzer
# Initialize the analyzer
@st.cache_resource
def load_analyzer():
    analyzer = FinancialSentimentAnalyzer()
    # Ensure the model is loaded
    if analyzer.finbert_analyzer is None:
        print("Retrying model initialization...")
        analyzer.setup_models()
    return analyzer

# Configure Streamlit page
st.set_page_config(
    page_title="MD&A Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for blue theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4682b4;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .metric-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1976d2;
        margin: 0.5rem 0;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #1565c0;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        color: #0d47a1;
        font-weight: bold;
    }
    .analysis-section {
        background: #f8fbff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e3f2fd;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #e3f2fd 0%, #ffffff 100%);
    }
    .stButton > button {
        background: linear-gradient(90deg, #1976d2 0%, #42a5f5 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.6rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<p class="main-header">📊 MD&A Sentiment & Complexity Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time NLP-powered analysis of financial disclosures</p>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = load_analyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Analysis Controls")
        
        # Company/Year selector (dummy for now)
        st.selectbox("Company", ["Sample Company", "AAPL", "MSFT", "TSLA"], index=0)
        st.selectbox("Fiscal Year", ["2023", "2022", "2021", "2020"], index=0)
        
        st.divider()
        
        # Text input options
        st.subheader("📝 Input Method")
        input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"])
        
        text_input = ""
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload MD&A text file",
                type=['txt', 'md'],
                help="Upload a .txt or .md file containing MD&A text"
            )
            if uploaded_file is not None:
                text_input = str(uploaded_file.read(), "utf-8")
                st.success(f"File uploaded: {len(text_input)} characters")
        else:
            text_input = st.text_area(
                "Paste MD&A text here:",
                height=200,
                placeholder="Paste your MD&A section text here for analysis...",
                help="Paste the raw MD&A text from a 10-K filing"
            )
        
        # Analysis button
        analyze_button = st.button("🚀 Analyze Text", type="primary", use_container_width=True)
        
        if text_input:
            st.info(f"Text ready: {len(text_input.split())} words, {len(text_input)} characters")
    
    # Main content
    if analyze_button and text_input:
        with st.spinner("🔄 Analyzing sentiment and complexity..."):
            try:
                # Run analysis
                results = analyzer.analyze_text(text_input)
                
                # Store results in session state
                st.session_state['results'] = results
                st.session_state['text_input'] = text_input
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                return
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Metric Cards Section
        st.subheader("📊 Key Metrics")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            finbert_score = results['finbert']['document_metrics']['average_score']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🤖 FinBERT Score</div>
                <div class="metric-value">{finbert_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ml_score = results['ml_lexicon']['normalized_score']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📚 ML Lexicon</div>
                <div class="metric-value">{ml_score:+.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            sci_score = results['sci']['sci_percentage']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🔄 SCI (%)</div>
                <div class="metric-value">{sci_score:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            fog_score = results['readability']['gunning_fog']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🌫 Gunning Fog</div>
                <div class="metric-value">{fog_score:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            fk_score = results['readability']['flesch_kincaid']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">📖 F-K Grade</div>
                <div class="metric-value">{fk_score:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            fox_score = results['fox_index']['fox_index']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">🦊 Fox Index</div>
                <div class="metric-value">{fox_score:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Visualizations Section
        st.subheader("📈 Sentiment Analysis Visualizations")
        
        # Row 1: Sentiment breakdown and word contributions
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # FinBERT sentiment breakdown pie chart
            fig_pie = create_sentiment_pie_chart(results['finbert'])
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Top contributing words bar chart
            fig_words = create_word_contribution_chart(results['ml_lexicon'])
            st.plotly_chart(fig_words, use_container_width=True)
        
        # Row 2: Sentence-wise analysis and complexity
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sentence-wise sentiment heatmap
            fig_sentences = create_sentence_sentiment_chart(results['finbert'])
            st.plotly_chart(fig_sentences, use_container_width=True)
        
        with col2:
            # SCI and complexity metrics
            fig_complexity = create_complexity_chart(results)
            st.plotly_chart(fig_complexity, use_container_width=True)
        
        # Macro flags section
        # if results['macro_context']['total_macro_sentences'] > 0:
        #     st.subheader("🌍 Macroeconomic Context Flags")
            
        #     macro_badges = create_macro_badges(results['macro_context'])
        #     st.markdown(macro_badges, unsafe_allow_html=True)
        
        st.divider()
        
        # # Detailed Analysis Section
        # with st.expander("🔍 Detailed Sentence-Level Analysis", expanded=False):
        #     sentence_df = create_sentence_dataframe(results)
            
        #     # Add filtering options
        #     col1, col2, col3 = st.columns(3)
        #     with col1:
        #         sentiment_filter = st.selectbox(
        #             "Filter by sentiment:",
        #             ["All", "Positive", "Negative", "Neutral"]
        #         )
        #     with col2:
        #         valence_filter = st.selectbox(
        #             "Filter by valence shifters:",
        #             ["All", "With Shifters", "Without Shifters"]
        #         )
        #     with col3:
        #         macro_filter = st.selectbox(
        #             "Filter by macro terms:",
        #             ["All", "With Macro Terms", "Without Macro Terms"]
        #         )
            
        #     # Apply filters
        #     filtered_df = apply_filters(sentence_df, sentiment_filter, valence_filter, macro_filter)
            
        #     st.dataframe(
        #         filtered_df,
        #         use_container_width=True,
        #         height=400,
        #         column_config={
        #             "sentence": st.column_config.TextColumn("Sentence", width="large"),
        #             "finbert_score": st.column_config.NumberColumn("FinBERT Score", format="%.3f"),
        #             "ml_lexicon_score": st.column_config.NumberColumn("ML Lexicon Score", format="%.3f"),
        #             "has_valence_shifters": st.column_config.CheckboxColumn("Valence Shifters"),
        #             "has_macro_terms": st.column_config.CheckboxColumn("Macro Terms")
        #         }
        #     )
            
        #     # Download button for detailed results
        #     csv_data = filtered_df.to_csv(index=False)
        #     st.download_button(
        #         label="📥 Download Detailed Analysis (CSV)",
        #         data=csv_data,
        #         file_name=f"mda_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        #         mime="text/csv"
        #     )
        
        # Summary insights
        st.subheader("💡 Key Insights")
        insights = generate_insights(results)
        
        for insight in insights:
            if insight['type'] == 'positive':
                st.success(f"✅ {insight['text']}")
            elif insight['type'] == 'negative':
                st.error(f"⚠ {insight['text']}")
            else:
                st.info(f"ℹ {insight['text']}")
    
    else:
        # Welcome message when no analysis has been run
        st.markdown("""
        <div class="analysis-section">
            <h3 style="color: #1976d2;">👋 Welcome to the MD&A Sentiment Dashboard</h3>
            <p style="color: #424242; font-size: 1.1rem;">
                This dashboard provides comprehensive sentiment and complexity analysis of Management Discussion & Analysis (MD&A) sections from 10-K filings.
            </p>
            <h4 style="color: #1976d2;">🔍 What This Dashboard Analyzes:</h4>
            <ul style="color: #424242;">
                <li><strong>FinBERT Sentiment:</strong> AI-powered financial sentiment analysis</li>
                <li><strong>ML Lexicon Scoring:</strong> Weighted sentiment using financial dictionaries</li>
                <li><strong>Semantic Complexity Index (SCI):</strong> Measures linguistic complexity and hedging</li>
                <li><strong>Readability Metrics:</strong> Gunning Fog Index and Flesch-Kincaid Grade Level</li>
                <li><strong>Fox Index:</strong> Proxy for potential managerial obfuscation</li>
                <li><strong>Macroeconomic Context:</strong> Identifies and analyzes macro-related content</li>
            </ul>
            <p style="color: #1976d2; font-weight: bold; font-size: 1.1rem;">
                👈 Get started by pasting MD&A text or uploading a file in the sidebar!
            </p>
        </div>
        """, unsafe_allow_html=True)


def create_sentiment_pie_chart(finbert_results):
    """Create sentiment breakdown pie chart"""
    metrics = finbert_results['document_metrics']
    
    labels = ['Positive', 'Neutral', 'Negative']
    values = [
        metrics['positive_percent'],
        metrics['neutral_percent'],
        metrics['negative_percent']
    ]
    
    # Blue color scheme
    colors = ['#42a5f5', '#90caf9', '#1976d2']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=12,
        marker=dict(line=dict(color='#ffffff', width=2))
    )])
    
    fig.update_layout(
        title={
            'text': "FinBERT Sentiment Distribution",
            'x': 0.5,
            'font': {'size': 16, 'color': '#1976d2'}
        },
        showlegend=True,
        height=350,
        margin=dict(t=50, b=20, l=20, r=20),
        font=dict(color='#424242')
    )
    
    return fig


def create_word_contribution_chart(ml_lexicon_results):
    """Create word contribution bar chart"""
    word_contributions = ml_lexicon_results.get('word_contributions', {})
    
    if not word_contributions:
        # Create empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="No ML lexicon words found in text",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color='#666666')
        )
        fig.update_layout(
            title="Top Contributing Words (ML Lexicon)",
            height=350,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
    
    # Get top 10 words by absolute contribution
    sorted_words = sorted(
        word_contributions.items(),
        key=lambda x: abs(x[1]['total_contribution']),
        reverse=True
    )[:10]
    
    words = [item[0] for item in sorted_words]
    contributions = [item[1]['total_contribution'] for item in sorted_words]
    
    # Color based on positive/negative contribution
    colors = ['#42a5f5' if contrib >= 0 else '#1976d2' for contrib in contributions]
    
    fig = go.Figure(data=[go.Bar(
        x=contributions,
        y=words,
        orientation='h',
        marker_color=colors,
        text=[f"{contrib:.3f}" for contrib in contributions],
        textposition='auto',
        textfont=dict(color='white', size=10)
    )])
    
    fig.update_layout(
        title={
            'text': "Top Contributing Words (ML Lexicon)",
            'x': 0.5,
            'font': {'size': 16, 'color': '#1976d2'}
        },
        xaxis_title="Contribution Score",
        yaxis_title="Words",
        height=350,
        margin=dict(t=50, b=40, l=100, r=20),
        font=dict(color='#424242'),
        xaxis=dict(gridcolor='#e3f2fd'),
        yaxis=dict(gridcolor='#e3f2fd')
    )
    
    return fig


def create_sentence_sentiment_chart(finbert_results):
    """Create sentence-wise sentiment visualization"""
    sentence_results = finbert_results.get('sentence_results', [])
    
    if not sentence_results:
        fig = go.Figure()
        fig.add_annotation(
            text="No sentence data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
    
    # Limit to first 20 sentences for readability
    sentences_to_show = sentence_results[:20]
    
    sentence_nums = list(range(1, len(sentences_to_show) + 1))
    scores = [s['normalized_score'] for s in sentences_to_show]
    labels = [s['label'] for s in sentences_to_show]
    
    # Color mapping
    color_map = {'positive': '#42a5f5', 'neutral': '#90caf9', 'negative': '#1976d2'}
    colors = [color_map[label] for label in labels]
    
    fig = go.Figure(data=[go.Bar(
        x=sentence_nums,
        y=scores,
        marker_color=colors,
        text=[f"{score:.2f}" for score in scores],
        textposition='auto',
        textfont=dict(size=8),
        hovertemplate='<b>Sentence %{x}</b><br>Score: %{y:.3f}<br>Label: %{customdata}<extra></extra>',
        customdata=labels
    )])
    
    fig.update_layout(
        title={
            'text': "Sentence-wise Sentiment Scores (First 20 Sentences)",
            'x': 0.5,
            'font': {'size': 16, 'color': '#1976d2'}
        },
        xaxis_title="Sentence Number",
        yaxis_title="Sentiment Score",
        height=350,
        margin=dict(t=50, b=40, l=60, r=20),
        font=dict(color='#424242'),
        xaxis=dict(gridcolor='#e3f2fd'),
        yaxis=dict(gridcolor='#e3f2fd', zeroline=True, zerolinecolor='#666666')
    )
    
    return fig


def create_complexity_chart(results):
    """Create complexity metrics chart"""
    metrics = {
        'SCI (%)': results['sci']['sci_percentage'],
        'Gunning Fog': results['readability']['gunning_fog'],
        'F-K Grade': results['readability']['flesch_kincaid'],
        'Fox Index': results['fox_index']['fox_index']
    }
    
    # Normalize metrics for comparison (0-100 scale)
    normalized_metrics = {}
    for key, value in metrics.items():
        if key == 'SCI (%)':
            normalized_metrics[key] = value  # Already in percentage
        elif key in ['Gunning Fog', 'F-K Grade']:
            normalized_metrics[key] = min(100, value * 5)  # Scale up for visibility
        else:  # Fox Index
            normalized_metrics[key] = value  # Already 0-100
    
    fig = go.Figure(data=[go.Bar(
        x=list(normalized_metrics.keys()),
        y=list(normalized_metrics.values()),
        marker_color=['#1976d2', '#42a5f5', '#64b5f6', '#90caf9'],
        text=[f"{metrics[k]:.1f}" for k in normalized_metrics.keys()],
        textposition='auto',
        textfont=dict(color='white', size=12)
    )])
    
    fig.update_layout(
        title={
            'text': "Complexity Metrics",
            'x': 0.5,
            'font': {'size': 16, 'color': '#1976d2'}
        },
        yaxis_title="Normalized Score (0-100)",
        height=350,
        margin=dict(t=50, b=40, l=60, r=20),
        font=dict(color='#424242'),
        xaxis=dict(gridcolor='#e3f2fd'),
        yaxis=dict(gridcolor='#e3f2fd')
    )
    
    return fig


# def create_macro_badges(macro_context):
#     """Create macro context badges"""
#     macro_sentences = macro_context.get('macro_flagged_sentences', [])
    
#     if not macro_sentences:
#         return "<p style='color: #666666;'>No macroeconomic terms detected in the text.</p>"
    
#     # Collect all unique macro terms
#     all_terms = set()
#     for sentence in macro_sentences:
#         all_terms.update(sentence.get('macro_terms', []))
    
#     badges_html = "<div style='margin: 1rem 0;'>"
#     for term in sorted(all_terms):
#         badges_html += f"""
#         <span style='background: linear-gradient(90deg, #1976d2, #42a5f5); 
#                      color: white; 
#                      padding: 0.3rem 0.8rem; 
#                      border-radius: 15px; 
#                      margin: 0.2rem 0.3rem; 
#                      display: inline-block; 
#                      font-size: 0.9rem; 
#                      font-weight: bold;'>
#             {term.upper()}
#         </span>
#         """
#     badges_html += "</div>"
    
#     badges_html += f"<p style='color: #666666; margin-top: 1rem;'>Found {len(macro_sentences)} sentences with macroeconomic context.</p>"
    
#     return badges_html


def create_sentence_dataframe(results):
    """Create detailed sentence-level dataframe"""
    finbert_sentences = results['finbert'].get('sentence_results', [])
    sci_sentences = results['sci'].get('sentence_details', [])
    macro_sentences = results['macro_context'].get('macro_flagged_sentences', [])
    
    # Create macro terms mapping
    macro_map = {}
    for macro_sent in macro_sentences:
        sentence_key = macro_sent['sentence'][:50]  # Use first 50 chars as key
        macro_map[sentence_key] = macro_sent['macro_terms']
    
    sentence_data = []
    
    max_sentences = max(len(finbert_sentences), len(sci_sentences))
    
    for i in range(max_sentences):
        row = {}
        
        # FinBERT data
        if i < len(finbert_sentences):
            fb_sent = finbert_sentences[i]
            sentence_text = fb_sent['sentence']
            row['sentence'] = sentence_text[:100] + "..." if len(sentence_text) > 100 else sentence_text
            row['finbert_sentiment'] = fb_sent['label']
            row['finbert_score'] = round(fb_sent['normalized_score'], 3)
            row['finbert_confidence'] = round(fb_sent['confidence'], 3)
        else:
            row['sentence'] = "N/A"
            row['finbert_sentiment'] = "N/A"
            row['finbert_score'] = 0
            row['finbert_confidence'] = 0
        
        # SCI data
        if i < len(sci_sentences):
            sci_sent = sci_sentences[i]
            row['has_valence_shifters'] = sci_sent['has_valence_shifters']
            row['valence_shifters'] = ', '.join(sci_sent['shifters_found'][:3]) if sci_sent['shifters_found'] else ''
        else:
            row['has_valence_shifters'] = False
            row['valence_shifters'] = ''
        
        # Macro data (simplified matching)
        sentence_key = row['sentence'][:50] if row['sentence'] != "N/A" else ""
        matching_macro = None
        for key, terms in macro_map.items():
            if key in sentence_key or sentence_key in key:
                matching_macro = terms
                break
        
        row['has_macro_terms'] = matching_macro is not None
        row['macro_terms'] = ', '.join(matching_macro[:2]) if matching_macro else ''
        
        # Add a simple ML lexicon score (placeholder)
        row['ml_lexicon_score'] = 0.0  # Would need sentence-level ML lexicon analysis
        
        sentence_data.append(row)
    
    return pd.DataFrame(sentence_data)


def apply_filters(df, sentiment_filter, valence_filter, macro_filter):
    """Apply filters to sentence dataframe"""
    filtered_df = df.copy()
    
    # Sentiment filter
    if sentiment_filter != "All":
        if sentiment_filter == "Positive":
            filtered_df = filtered_df[filtered_df['finbert_sentiment'] == 'positive']
        elif sentiment_filter == "Negative":
            filtered_df = filtered_df[filtered_df['finbert_sentiment'] == 'negative']
        elif sentiment_filter == "Neutral":
            filtered_df = filtered_df[filtered_df['finbert_sentiment'] == 'neutral']
    
    # Valence shifter filter
    if valence_filter != "All":
        if valence_filter == "With Shifters":
            filtered_df = filtered_df[filtered_df['has_valence_shifters'] == True]
        elif valence_filter == "Without Shifters":
            filtered_df = filtered_df[filtered_df['has_valence_shifters'] == False]
    
    # Macro filter
    if macro_filter != "All":
        if macro_filter == "With Macro Terms":
            filtered_df = filtered_df[filtered_df['has_macro_terms'] == True]
        elif macro_filter == "Without Macro Terms":
            filtered_df = filtered_df[filtered_df['has_macro_terms'] == False]
    
    return filtered_df


def generate_insights(results):
    """Generate key insights from analysis results"""
    insights = []
    
    # FinBERT insights
    finbert_score = results['finbert']['document_metrics']['average_score']
    if finbert_score > 0.2:
        insights.append({
            'type': 'positive',
            'text': f"Overall sentiment is positive (FinBERT: {finbert_score:+.3f}), suggesting optimistic management tone."
        })
    elif finbert_score < -0.2:
        insights.append({
            'type': 'negative',
            'text': f"Overall sentiment is negative (FinBERT: {finbert_score:+.3f}), indicating cautious or pessimistic outlook."
        })
    else:
        insights.append({
            'type': 'neutral',
            'text': f"Sentiment is neutral (FinBERT: {finbert_score:+.3f}), suggesting balanced reporting tone."
        })
    
    # SCI insights
    sci_score = results['sci']['sci_percentage']
    if sci_score > 50:
        insights.append({
            'type': 'negative',
            'text': f"High Semantic Complexity Index ({sci_score:.1f}%) indicates extensive use of hedging language and qualifiers."
        })
    elif sci_score > 30:
        insights.append({
            'type': 'neutral',
            'text': f"Moderate Semantic Complexity Index ({sci_score:.1f}%) shows typical financial disclosure complexity."
        })
    else:
        insights.append({
            'type': 'positive',
            'text': f"Low Semantic Complexity Index ({sci_score:.1f}%) suggests clear, direct communication style."
        })
    
    # Readability insights
    fog_score = results['readability']['gunning_fog']
    if fog_score > 16:
        insights.append({
            'type': 'negative',
            'text': f"Very high Gunning Fog Index ({fog_score:.1f}) indicates text is difficult to read and may hinder investor comprehension."
        })
    elif fog_score > 12:
        insights.append({
            'type': 'neutral',
            'text': f"Gunning Fog Index ({fog_score:.1f}) shows college-level complexity, typical for financial documents."
        })
    
    # Fox Index insights
    fox_score = results['fox_index']['fox_index']
    if fox_score > 70:
        insights.append({
            'type': 'negative',
            'text': f"High Fox Index ({fox_score:.1f}) suggests potential obfuscation through complex language structure."
        })
    elif fox_score > 40:
        insights.append({
            'type': 'neutral',
            'text': f"Moderate Fox Index ({fox_score:.1f}) indicates some linguistic complexity but within normal range."
        })
    
    # Macro context insights
    macro_count = results['macro_context']['total_macro_sentences']
    if macro_count > 0:
        insights.append({
            'type': 'neutral',
            'text': f"Document contains {macro_count} sentences with macroeconomic context, showing awareness of external factors."
        })
    
    # ML Lexicon insights
    ml_score = results['ml_lexicon']['normalized_score']
    if abs(finbert_score - ml_score) > 0.3:
        insights.append({
            'type': 'negative',
            'text': f"Significant divergence between FinBERT ({finbert_score:+.3f}) and ML Lexicon ({ml_score:+.3f}) suggests potential sentiment complexity or mixed messaging."
        })
    
    return insights


if __name__ == "__main__":
    main()