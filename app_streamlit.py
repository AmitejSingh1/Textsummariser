import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from rouge_score import rouge_scorer
import re
import math
import time
from io import BytesIO
from datetime import datetime
import pandas as pd

# BERTScore import (optional)
try:
    from bert_score import score as bert_score_func
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    bert_score_func = None

# Visualization imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# PDF processing imports
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Advanced Text Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: left;
        margin-bottom: 0.5rem;
        margin-top: 0;
        padding: 0;
    }
    .subtitle {
        text-align: left;
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1rem;
        margin-top: 0;
        padding: 0;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 95%;
    }
    [data-testid="stHeader"] {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    [data-testid="stSidebar"] {
        padding-top: 1rem;
    }
    .element-container {
        margin-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #5568d3 0%, #653a91 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8faff 0%, #fef5ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #667eea;
        margin: 1rem 0;
    }
    .result-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        margin: 1rem 0;
    }
    .best-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        margin-left: 1rem;
    }
    .method-badge {
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .extractive-badge {
        background: #dbeafe;
        color: #1e40af;
    }
    .abstractive-badge {
        background: #f3e8ff;
        color: #6b21a8;
    }
    </style>
""", unsafe_allow_html=True)

# Download required NLTK data
@st.cache_data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()

# Try to import transformers (optional)
# Set environment variables to avoid TensorFlow/JAX issues
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TRANSFORMERS_OFFLINE'] = '0'

summarizer_bart = None
summarizer_t5 = None
transformers_available = False
transformers_error = None

# Try importing pipeline with better error handling
try:
    # Try to import pipeline - this may fail due to dependency issues
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        from transformers import pipeline
    transformers_available = True
except RuntimeError as e:
    transformers_available = False
    transformers_error = str(e)
    # Check if it's the numpy/JAX issue
    if 'numpy' in str(e).lower() or 'dtypes' in str(e).lower():
        transformers_error = "NumPy version compatibility issue detected"
except ImportError as e:
    transformers_available = False
    transformers_error = "Transformers not installed"
except Exception as e:
    transformers_available = False
    transformers_error = f"Error: {type(e).__name__}"

# Initialize models in session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.summarizer_bart = None
    st.session_state.summarizer_t5 = None

# Load abstractive models if available (only once)
if transformers_available and not st.session_state.models_loaded:
    # Use a placeholder to show loading only once
    placeholder = st.empty()
    with placeholder.container():
        st.info("🔄 Loading abstractive models on first run (this may take a few minutes)...")
    
    try:
        try:
            st.session_state.summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        except Exception as e:
            st.session_state.summarizer_bart = None
        
        try:
            st.session_state.summarizer_t5 = pipeline("summarization", model="t5-small", device=-1)
        except Exception as e:
            st.session_state.summarizer_t5 = None
    except Exception as e:
        st.session_state.summarizer_bart = None
        st.session_state.summarizer_t5 = None
    
    placeholder.empty()
    st.session_state.models_loaded = True

rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Initialize BERTScore model (lazy loading)
bertscore_model = None
bertscore_tokenizer = None
if BERTSCORE_AVAILABLE:
    try:
        # BERTScore will load model on first use
        pass
    except Exception:
        pass

class TextSummarizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def extractive_summarize_tfidf(self, text, num_sentences=3, return_scores=False):
        """Extractive summarization using TF-IDF"""
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        
        # Always summarize, but ensure we don't exceed total sentences
        num_sentences = min(num_sentences, total_sentences)
        
        # If we have very few sentences, still try to summarize
        if total_sentences <= 2:
            if return_scores:
                return text, {i: 1.0 for i in range(len(sentences))}
            return text
        
        cleaned_sentences = [re.sub(r'[^a-zA-Z0-9\s]', '', s.lower()) for s in sentences]
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
            
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            # Normalize scores to 0-1 range
            if sentence_scores.max() > 0:
                sentence_scores = sentence_scores / sentence_scores.max()
            
            sentence_score_dict = {i: float(sentence_scores[i]) for i in range(len(sentences))}
            
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)
            
            summary = ' '.join([sentences[i] for i in top_indices])
            
            # Ensure summary is actually shorter
            if len(summary.strip()) >= len(text.strip()) * 0.95:
                # If summary is almost as long, take fewer sentences
                num_sentences = max(1, num_sentences - 1)
                top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
                top_indices = sorted(top_indices)
                summary = ' '.join([sentences[i] for i in top_indices])
            
            if return_scores:
                return summary, sentence_score_dict
            return summary
        except Exception as e:
            # Fallback: return first N sentences
            if return_scores:
                return ' '.join(sentences[:num_sentences]), {i: 0.5 for i in range(len(sentences))}
            return ' '.join(sentences[:num_sentences])
    
    def extractive_summarize_textrank(self, text, num_sentences=3, return_scores=False):
        """Extractive summarization using TextRank algorithm"""
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        
        # Always summarize, but ensure we don't exceed total sentences
        num_sentences = min(num_sentences, total_sentences)
        
        # If we have very few sentences, still try to summarize
        if total_sentences <= 2:
            if return_scores:
                return text, {i: 1.0 for i in range(len(sentences))}
            return text
        
        cleaned_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [self.stemmer.stem(word) for word in words if word.isalnum() and word not in self.stop_words]
            cleaned_sentences.append(' '.join(words))
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            sentence_vectors = vectorizer.fit_transform(cleaned_sentences)
            similarity_matrix = cosine_similarity(sentence_vectors)
            
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph, max_iter=100)
            
            # Normalize scores to 0-1 range
            max_score = max(scores.values()) if scores else 1.0
            if max_score > 0:
                sentence_score_dict = {i: float(scores[i] / max_score) for i in range(len(sentences))}
            else:
                sentence_score_dict = {i: 0.5 for i in range(len(sentences))}
            
            ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
            top_indices = sorted([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
            
            summary = ' '.join([sentences[i] for i in top_indices])
            
            # Ensure summary is actually shorter
            if len(summary.strip()) >= len(text.strip()) * 0.95:
                # If summary is almost as long, take fewer sentences
                num_sentences = max(1, num_sentences - 1)
                top_indices = sorted([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
                summary = ' '.join([sentences[i] for i in top_indices])
            
            if return_scores:
                return summary, sentence_score_dict
            return summary
        except Exception as e:
            # Fallback: return first N sentences
            if return_scores:
                return ' '.join(sentences[:num_sentences]), {i: 0.5 for i in range(len(sentences))}
            return ' '.join(sentences[:num_sentences])
    
    def extractive_summarize_lsa(self, text, num_sentences=3, return_scores=False):
        """Extractive summarization using Latent Semantic Analysis"""
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        
        # Always summarize, but ensure we don't exceed total sentences
        num_sentences = min(num_sentences, total_sentences)
        
        # If we have very few sentences, still try to summarize
        if total_sentences <= 2:
            if return_scores:
                return text, {i: 1.0 for i in range(len(sentences))}
            return text
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            from sklearn.decomposition import TruncatedSVD
            n_components = min(num_sentences, total_sentences - 1, tfidf_matrix.shape[0] - 1)
            if n_components < 1:
                n_components = 1
            
            svd = TruncatedSVD(n_components=n_components)
            svd_matrix = svd.fit_transform(tfidf_matrix)
            
            sentence_scores = np.sum(svd_matrix ** 2, axis=1)
            # Normalize scores to 0-1 range
            if sentence_scores.max() > 0:
                sentence_scores = sentence_scores / sentence_scores.max()
            
            sentence_score_dict = {i: float(sentence_scores[i]) for i in range(len(sentences))}
            
            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)
            
            summary = ' '.join([sentences[i] for i in top_indices])
            
            # Ensure summary is actually shorter
            if len(summary.strip()) >= len(text.strip()) * 0.95:
                # If summary is almost as long, take fewer sentences
                num_sentences = max(1, num_sentences - 1)
                top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
                top_indices = sorted(top_indices)
                summary = ' '.join([sentences[i] for i in top_indices])
            
            if return_scores:
                return summary, sentence_score_dict
            return summary
        except Exception as e:
            # Fallback: return first N sentences
            if return_scores:
                return ' '.join(sentences[:num_sentences]), {i: 0.5 for i in range(len(sentences))}
            return ' '.join(sentences[:num_sentences])
    
    def abstractive_summarize_bart(self, text, max_length=130, min_length=30):
        """Abstractive summarization using BART"""
        if st.session_state.summarizer_bart is None:
            return "BART model not available"
        
        try:
            # Calculate appropriate length based on input
            input_words = len(word_tokenize(text))
            # Set max_length to be about 30% of input, but within model limits
            dynamic_max = max(min_length, min(max_length, int(input_words * 0.3)))
            dynamic_min = min(min_length, dynamic_max - 10)
            
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            summary = st.session_state.summarizer_bart(
                text, 
                max_length=dynamic_max, 
                min_length=dynamic_min, 
                do_sample=False
            )[0]['summary_text']
            
            # Ensure summary is actually shorter than original
            if len(summary.strip()) >= len(text.strip()) * 0.9:
                # Try with shorter length
                summary = st.session_state.summarizer_bart(
                    text,
                    max_length=max(min_length, dynamic_max - 20),
                    min_length=max(10, dynamic_min - 10),
                    do_sample=False
                )[0]['summary_text']
            
            return summary
        except Exception as e:
            return f"Error in BART summarization: {str(e)}"
    
    def abstractive_summarize_t5(self, text, max_length=130, min_length=30):
        """Abstractive summarization using T5"""
        if st.session_state.summarizer_t5 is None:
            return "T5 model not available"
        
        try:
            # Calculate appropriate length based on input
            input_words = len(word_tokenize(text))
            # Set max_length to be about 30% of input, but within model limits
            dynamic_max = max(min_length, min(max_length, int(input_words * 0.3)))
            dynamic_min = min(min_length, dynamic_max - 10)
            
            max_input_length = 512
            original_text = text
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            summary = st.session_state.summarizer_t5(
                text,
                max_length=dynamic_max,
                min_length=dynamic_min,
                do_sample=False
            )[0]['summary_text']
            
            # Ensure summary is actually shorter than original
            if len(summary.strip()) >= len(original_text.strip()) * 0.9:
                # Try with shorter length
                summary = st.session_state.summarizer_t5(
                    text,
                    max_length=max(min_length, dynamic_max - 20),
                    min_length=max(10, dynamic_min - 10),
                    do_sample=False
                )[0]['summary_text']
            
            return summary
        except Exception as e:
            return f"Error in T5 summarization: {str(e)}"
    
    def calculate_rouge_scores(self, reference, summary):
        """Calculate ROUGE scores for summary evaluation"""
        scores = rouge.score(reference, summary)
        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'f1': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'f1': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'f1': scores['rougeL'].fmeasure
            }
        }
    
    def calculate_compression_ratio(self, original, summary):
        """Calculate compression ratio"""
        original_words = len(word_tokenize(original))
        summary_words = len(word_tokenize(summary))
        if original_words == 0:
            return 0
        return summary_words / original_words
    
    def calculate_bertscore(self, reference, summary):
        """Calculate BERTScore for summary evaluation"""
        if not BERTSCORE_AVAILABLE:
            return None
        
        try:
            # BERTScore calculation
            # Using default model for faster computation (roberta-large or microsoft/deberta-xlarge-mnli)
            # lang='en' automatically selects the best model
            P, R, F1 = bert_score_func(
                [summary],
                [reference],
                lang='en',
                verbose=False,
                device='cpu'  # Use CPU to avoid GPU issues
            )
            
            return {
                'precision': float(P[0].item()),
                'recall': float(R[0].item()),
                'f1': float(F1[0].item())
            }
        except Exception as e:
            # If BERTScore fails, return None
            return None
    
    def calculate_summary_score(self, original, summary, rouge_scores, bertscore=None):
        """Calculate overall summary score (weighted combination)"""
        compression = self.calculate_compression_ratio(original, summary)
        
        # Base score from ROUGE
        base_score = (
            0.4 * rouge_scores['rouge1']['f1'] +
            0.3 * rouge_scores['rouge2']['f1'] +
            0.2 * rouge_scores['rougeL']['f1'] +
            0.1 * (1 - abs(compression - 0.3))
        )
        
        # If BERTScore is available, incorporate it (reduce ROUGE weight slightly)
        if bertscore is not None:
            # Weighted: 70% ROUGE-based, 20% BERTScore F1, 10% compression
            score = (
                0.35 * rouge_scores['rouge1']['f1'] +
                0.25 * rouge_scores['rouge2']['f1'] +
                0.15 * rouge_scores['rougeL']['f1'] +
                0.20 * bertscore['f1'] +
                0.05 * (1 - abs(compression - 0.3))
            )
        else:
            score = base_score
        
        return round(score * 100, 2)
    
    def hybrid_summarize(self, text, num_sentences=3, method_weights=None):
        """Novel: Hybrid summarization combining multiple methods"""
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        
        if total_sentences <= 2:
            return text, {}
        
        num_sentences = min(num_sentences, total_sentences)
        
        # Default weights for each method
        if method_weights is None:
            method_weights = {'tfidf': 0.4, 'textrank': 0.4, 'lsa': 0.2}
        
        # Get sentence scores from each method
        all_scores = {}
        
        try:
            # TF-IDF scores
            if 'tfidf' in method_weights and method_weights['tfidf'] > 0:
                _, tfidf_scores = self.extractive_summarize_tfidf(text, num_sentences, return_scores=True)
                for idx, score in tfidf_scores.items():
                    all_scores[idx] = all_scores.get(idx, 0) + score * method_weights['tfidf']
        except:
            pass
        
        try:
            # TextRank scores
            if 'textrank' in method_weights and method_weights['textrank'] > 0:
                _, textrank_scores = self.extractive_summarize_textrank(text, num_sentences, return_scores=True)
                for idx, score in textrank_scores.items():
                    all_scores[idx] = all_scores.get(idx, 0) + score * method_weights['textrank']
        except:
            pass
        
        try:
            # LSA scores
            if 'lsa' in method_weights and method_weights['lsa'] > 0:
                _, lsa_scores = self.extractive_summarize_lsa(text, num_sentences, return_scores=True)
                for idx, score in lsa_scores.items():
                    all_scores[idx] = all_scores.get(idx, 0) + score * method_weights['lsa']
        except:
            pass
        
        # Select top sentences based on combined scores
        if not all_scores:
            # Fallback to first N sentences
            top_indices = list(range(min(num_sentences, len(sentences))))
        else:
            ranked_sentences = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            top_indices = sorted([idx for idx, score in ranked_sentences[:num_sentences]])
        
        summary = ' '.join([sentences[i] for i in top_indices])
        sentence_scores = all_scores
        
        return summary, sentence_scores
    
    def recommend_method(self, text, num_sentences=3, quick_test=False):
        """Novel: Analyze text and recommend best summarization method"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        total_words = len(words)
        total_sentences = len(sentences)
        avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
        
        # Analyze text characteristics
        features = {
            'length': 'long' if total_words > 500 else 'medium' if total_words > 200 else 'short',
            'complexity': 'high' if avg_sentence_length > 20 else 'medium' if avg_sentence_length > 15 else 'low',
            'sentence_count': total_sentences,
            'word_count': total_words
        }
        
        recommendations = []
        
        # If quick_test is enabled, actually test methods to predict best one
        if quick_test and total_sentences > 3:
            try:
                # Quick test: Run all methods and see which scores highest
                test_results = {}
                
                # Test TF-IDF
                tfidf_summary = self.extractive_summarize_tfidf(text, num_sentences)
                tfidf_rouge = self.calculate_rouge_scores(text, tfidf_summary)
                test_results['TF-IDF'] = self.calculate_summary_score(text, tfidf_summary, tfidf_rouge)
                
                # Test TextRank
                textrank_summary = self.extractive_summarize_textrank(text, num_sentences)
                textrank_rouge = self.calculate_rouge_scores(text, textrank_summary)
                test_results['TextRank'] = self.calculate_summary_score(text, textrank_summary, textrank_rouge)
                
                # Test LSA
                lsa_summary = self.extractive_summarize_lsa(text, num_sentences)
                lsa_rouge = self.calculate_rouge_scores(text, lsa_summary)
                test_results['LSA'] = self.calculate_summary_score(text, lsa_summary, lsa_rouge)
                
                # Find best method
                best_method = max(test_results.items(), key=lambda x: x[1])
                sorted_methods = sorted(test_results.items(), key=lambda x: x[1], reverse=True)
                
                # Add recommendations based on actual test results
                for method, score in sorted_methods:
                    if method == best_method[0]:
                        recommendations.append({
                            'method': method,
                            'reason': f'Predicted best performer (test score: {score:.1f})',
                            'confidence': 0.95,
                            'test_score': score,
                            'is_best': True
                        })
                    else:
                        recommendations.append({
                            'method': method,
                            'reason': f'Test score: {score:.1f}',
                            'confidence': 0.70 + (score / best_method[1]) * 0.25,
                            'test_score': score,
                            'is_best': False
                        })
                
                # Always recommend hybrid
                recommendations.append({
                    'method': 'Hybrid (Ensemble)',
                    'reason': 'Combines multiple methods - often outperforms individual methods',
                    'confidence': 0.90,
                    'test_score': None,
                    'is_best': False
                })
                
                return features, recommendations, test_results
                
            except Exception as e:
                # If quick test fails, fall back to heuristic recommendations
                pass
        
        # Heuristic-based recommendations (fallback or when quick_test=False)
        # Improved heuristics based on empirical observations
        if features['length'] == 'short' and total_sentences <= 10:
            # For short texts, TF-IDF often works well
            recommendations.append({
                'method': 'TF-IDF',
                'reason': 'Typically performs well on short texts',
                'confidence': 0.80
            })
        elif features['length'] == 'long' and features['complexity'] == 'high':
            # For long complex texts, TextRank is often better
            recommendations.append({
                'method': 'TextRank',
                'reason': 'Often best for long, complex texts with many sentences',
                'confidence': 0.85
            })
        elif features['sentence_count'] > 20:
            # For many sentences, LSA can be good
            recommendations.append({
                'method': 'LSA',
                'reason': 'Good for documents with many sentences and multiple topics',
                'confidence': 0.75
            })
        else:
            # Default: TF-IDF and TextRank are usually good
            recommendations.append({
                'method': 'TF-IDF',
                'reason': 'Generally reliable for most text types',
                'confidence': 0.70
            })
            recommendations.append({
                'method': 'TextRank',
                'reason': 'Balanced performance for most text types',
                'confidence': 0.70
            })
        
        # Always recommend hybrid for comparison
        recommendations.append({
            'method': 'Hybrid (Ensemble)',
            'reason': 'Combines multiple methods - often produces best results',
            'confidence': 0.90
        })
        
        return features, recommendations, None
    
    def extract_key_phrases(self, text, top_n=10):
        """Novel: Extract key phrases from text using TF-IDF"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 2]
        
        # Get TF-IDF scores for words
        try:
            vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_words)])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top phrases
            phrase_scores = list(zip(feature_names, scores))
            phrase_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [phrase for phrase, score in phrase_scores[:top_n]]
        except:
            # Fallback: return most frequent words
            from collections import Counter
            word_freq = Counter(filtered_words)
            return [word for word, count in word_freq.most_common(top_n)]
    
    def build_semantic_network(self, text):
        """Novel: Build semantic similarity network of sentences"""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return None, None
        
        try:
            # Vectorize sentences
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            sentence_vectors = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(sentence_vectors)
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes (sentences)
            for i, sentence in enumerate(sentences):
                G.add_node(i, label=f"S{i+1}", text=sentence[:50] + "...")
            
            # Add edges (similarity > threshold)
            threshold = 0.1
            for i in range(len(sentences)):
                for j in range(i+1, len(sentences)):
                    similarity = similarity_matrix[i][j]
                    if similarity > threshold:
                        G.add_edge(i, j, weight=similarity)
            
            return G, similarity_matrix
        except:
            return None, None
    
    def generate_multi_level_summaries(self, text):
        """Novel: Generate summaries at different detail levels"""
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        
        if total_sentences < 3:
            return {
                'brief': text,
                'medium': text,
                'detailed': text
            }
        
        # Brief: ~20% of sentences
        brief_count = max(1, int(total_sentences * 0.2))
        # Medium: ~40% of sentences
        medium_count = max(2, int(total_sentences * 0.4))
        # Detailed: ~60% of sentences
        detailed_count = max(3, int(total_sentences * 0.6))
        
        # Use hybrid method for best results
        brief_summary, _ = self.hybrid_summarize(text, brief_count)
        medium_summary, _ = self.hybrid_summarize(text, medium_count)
        detailed_summary, _ = self.hybrid_summarize(text, detailed_count)
        
        return {
            'brief': brief_summary,
            'medium': medium_summary,
            'detailed': detailed_summary,
            'brief_count': brief_count,
            'medium_count': medium_count,
            'detailed_count': detailed_count
        }

# Initialize summarizer
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = TextSummarizer()

# Summary template functions
def format_executive_summary(summary_text, method_name):
    """Format summary as executive summary with bullet points"""
    sentences = sent_tokenize(summary_text)
    formatted = f"# Executive Summary - {method_name}\n\n"
    formatted += "## Key Points:\n\n"
    for i, sentence in enumerate(sentences, 1):
        formatted += f"• {sentence.strip()}\n"
    return formatted

def format_academic_summary(summary_text, method_name, metrics=None):
    """Format summary as academic summary with structured sections"""
    sentences = sent_tokenize(summary_text)
    formatted = f"# Academic Summary - {method_name}\n\n"
    formatted += "## Abstract\n\n"
    formatted += " ".join(sentences[:2]) + "\n\n"
    formatted += "## Summary\n\n"
    for i, sentence in enumerate(sentences[2:], 1):
        formatted += f"{i}. {sentence.strip()}\n\n"
    if metrics:
        formatted += "## Metrics\n\n"
        formatted += f"- ROUGE-1 F1: {metrics['rouge1']['f1']*100:.2f}%\n"
        formatted += f"- ROUGE-2 F1: {metrics['rouge2']['f1']*100:.2f}%\n"
        formatted += f"- ROUGE-L F1: {metrics['rougeL']['f1']*100:.2f}%\n"
        formatted += f"- Compression Ratio: {metrics.get('compression', 0)*100:.2f}%\n"
    return formatted

def format_news_summary(summary_text, method_name):
    """Format summary as news summary (5W format)"""
    sentences = sent_tokenize(summary_text)
    formatted = f"# News Summary - {method_name}\n\n"
    formatted += "## The 5 W's\n\n"
    
    # Try to extract 5W information (simplified)
    if len(sentences) >= 1:
        formatted += f"**What:** {sentences[0]}\n\n"
    if len(sentences) >= 2:
        formatted += f"**Why/How:** {sentences[1]}\n\n"
    if len(sentences) >= 3:
        formatted += f"**Details:** {' '.join(sentences[2:])}\n\n"
    
    formatted += "## Full Summary\n\n"
    for sentence in sentences:
        formatted += f"{sentence.strip()}\n\n"
    return formatted

def format_meeting_notes(summary_text, method_name):
    """Format summary as meeting notes with action items"""
    sentences = sent_tokenize(summary_text)
    formatted = f"# Meeting Notes Summary - {method_name}\n\n"
    formatted += "## Key Points\n\n"
    for i, sentence in enumerate(sentences, 1):
        formatted += f"{i}. {sentence.strip()}\n"
    formatted += "\n## Action Items\n\n"
    # Extract potential action items (sentences with action verbs)
    action_verbs = ['will', 'should', 'must', 'need', 'required', 'action', 'decide', 'implement']
    for sentence in sentences:
        if any(verb in sentence.lower() for verb in action_verbs):
            formatted += f"• {sentence.strip()}\n"
    return formatted

def apply_template(summary_text, method_name, template_type, metrics=None):
    """Apply the selected template to the summary"""
    if template_type == "Default":
        return summary_text
    elif template_type == "Executive Summary":
        return format_executive_summary(summary_text, method_name)
    elif template_type == "Academic Summary":
        return format_academic_summary(summary_text, method_name, metrics)
    elif template_type == "News Summary":
        return format_news_summary(summary_text, method_name)
    elif template_type == "Meeting Notes":
        return format_meeting_notes(summary_text, method_name)
    else:
        return summary_text

# PDF processing functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def create_summary_pdf(summaries, original_text_length, methods_used, output_buffer):
    """Create a PDF with summaries"""
    try:
        from reportlab.lib.colors import HexColor
        
        doc = SimpleDocTemplate(output_buffer, pagesize=A4, 
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#764ba2'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        )
        
        # Title
        elements.append(Paragraph("Text Summarization Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Metadata
        elements.append(Paragraph(f"<b>Generated on:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Paragraph(f"<b>Original text length:</b> {original_text_length} words", styles['Normal']))
        elements.append(Paragraph(f"<b>Methods used:</b> {', '.join(methods_used)}", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Add summaries
        for idx, (method_name, summary_data) in enumerate(summaries.items(), 1):
            elements.append(Paragraph(f"Summary {idx}: {summary_data['method']} ({summary_data['type']})", heading_style))
            
            # Summary text - format for reportlab (handles HTML automatically)
            summary_text = summary_data['summary'].replace('\n', '<br/>')
            elements.append(Paragraph(summary_text, normal_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Metrics
            metrics_text = (
                f"<b>Overall Score:</b> {summary_data['overall_score']:.1f} | "
                f"<b>ROUGE-1 F1:</b> {summary_data['rouge_scores']['rouge1']['f1']*100:.1f}% | "
                f"<b>ROUGE-2 F1:</b> {summary_data['rouge_scores']['rouge2']['f1']*100:.1f}% | "
                f"<b>ROUGE-L F1:</b> {summary_data['rouge_scores']['rougeL']['f1']*100:.1f}% | "
                f"<b>Compression:</b> {summary_data['compression_ratio']*100:.1f}%"
            )
            elements.append(Paragraph(metrics_text, styles['Normal']))
            elements.append(Spacer(1, 0.2*inch))
            
            if idx < len(summaries):
                elements.append(PageBreak())
        
        # Build PDF
        doc.build(elements)
        return True
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

# Helper function to display result card
def display_result_card(result, is_single=False, original_length=None):
    """Display a single result card with all metrics"""
    method_type_class = "extractive-badge" if result['type'] == 'extractive' else "abstractive-badge"
    
    # Header with badges
    col1, col2 = st.columns([3, 1])
    with col1:
        best_badge = "" if is_single else '<span class="best-badge">🏆 Best</span>'
        st.markdown(f"""
            <h2>
                {result['method']}
                <span class="method-badge {method_type_class}">{result['type']}</span>
                {best_badge}
            </h2>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Overall Score", f"{result['overall_score']:.1f}")
    
    # Summary
    st.subheader("📝 Summary")
    if 'formatted_summary' in result:
        st.markdown(result['formatted_summary'])
    else:
        st.info(result['summary'])
    
    # Novel: Interactive Sentence Selection (for extractive/hybrid methods with scores)
    if result.get('type') in ['extractive', 'hybrid'] and 'sentence_scores' in result:
        st.markdown("---")
        with st.expander("🔍 View Sentence Selection Explanation", expanded=False):
            st.markdown("**💡 How sentences were selected:**")
            sentences_list = result.get('original_sentences', [])
            if not sentences_list and 'original_text' in result:
                sentences_list = sent_tokenize(result['original_text'])
            elif not sentences_list:
                sentences_list = []
            
            sentence_scores_dict = result.get('sentence_scores', {})
            
            # Create a visualization of sentence scores
            if PLOTLY_AVAILABLE and sentence_scores_dict and sentences_list:
                score_values = [sentence_scores_dict.get(i, 0) for i in range(len(sentences_list))]
                sentence_indices = list(range(len(sentences_list)))
                
                fig_scores = go.Figure(data=go.Bar(
                    x=sentence_indices,
                    y=score_values,
                    marker=dict(
                        color=score_values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Importance Score")
                    ),
                    text=[f"Sentence {i+1}" for i in sentence_indices],
                    textposition='outside'
                ))
                fig_scores.update_layout(
                    title='Sentence Importance Scores',
                    xaxis_title='Sentence Number',
                    yaxis_title='Importance Score (0-1)',
                    height=300,
                    template='plotly_white'
                )
                st.plotly_chart(fig_scores, use_container_width=True)
            
            # Show sentences with their scores
            if sentences_list:
                st.markdown("**📊 Sentences with scores:**")
                summary_sentences = set(sent_tokenize(result['summary']))
                for idx, sentence in enumerate(sentences_list):
                    score = sentence_scores_dict.get(idx, 0)
                    score_color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                    is_selected = any(sentence.strip() in s or s in sentence.strip() for s in summary_sentences)
                    selected_indicator = " ✅ SELECTED" if is_selected else ""
                    st.markdown(f"{score_color} **Sentence {idx+1}** (Score: {score:.3f}){selected_indicator}")
                    st.caption(f'"{sentence[:100]}..."' if len(sentence) > 100 else f'"{sentence}"')
    
    # Metrics
    if result.get('bertscore') is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
    else:
        col1, col2, col3, col4 = st.columns(4)
        col5 = None
    
    with col1:
        st.metric("ROUGE-1 F1", f"{result['rouge_scores']['rouge1']['f1']*100:.1f}%")
    
    with col2:
        st.metric("ROUGE-2 F1", f"{result['rouge_scores']['rouge2']['f1']*100:.1f}%")
    
    with col3:
        st.metric("ROUGE-L F1", f"{result['rouge_scores']['rougeL']['f1']*100:.1f}%")
    
    with col4:
        st.metric("Compression", f"{result['compression_ratio']*100:.1f}%")
    
    if col5 and result.get('bertscore') is not None:
        with col5:
            st.metric("BERTScore F1", f"{result['bertscore']['f1']*100:.1f}%")
    
    # Detailed scores
    with st.expander("📈 Detailed Evaluation Scores"):
        if result.get('bertscore') is not None:
            col1, col2, col3, col4 = st.columns(4)
        else:
            col1, col2, col3, col4 = st.columns(3)
            col4 = None
        
        with col1:
            st.write("**ROUGE-1**")
            st.write(f"Precision: {result['rouge_scores']['rouge1']['precision']*100:.2f}%")
            st.write(f"Recall: {result['rouge_scores']['rouge1']['recall']*100:.2f}%")
            st.write(f"F1: {result['rouge_scores']['rouge1']['f1']*100:.2f}%")
        
        with col2:
            st.write("**ROUGE-2**")
            st.write(f"Precision: {result['rouge_scores']['rouge2']['precision']*100:.2f}%")
            st.write(f"Recall: {result['rouge_scores']['rouge2']['recall']*100:.2f}%")
            st.write(f"F1: {result['rouge_scores']['rouge2']['f1']*100:.2f}%")
        
        with col3:
            st.write("**ROUGE-L**")
            st.write(f"Precision: {result['rouge_scores']['rougeL']['precision']*100:.2f}%")
            st.write(f"Recall: {result['rouge_scores']['rougeL']['recall']*100:.2f}%")
            st.write(f"F1: {result['rouge_scores']['rougeL']['f1']*100:.2f}%")
        
        if col4 and result.get('bertscore') is not None:
            with col4:
                st.write("**BERTScore**")
                st.write(f"Precision: {result['bertscore']['precision']*100:.2f}%")
                st.write(f"Recall: {result['bertscore']['recall']*100:.2f}%")
                st.write(f"F1: {result['bertscore']['f1']*100:.2f}%")
        elif result.get('bertscore') is None and BERTSCORE_AVAILABLE == False:
            st.info("💡 Install bert-score to see BERTScore metrics: `pip install bert-score`")
    
    # Summary stats
    summary_length = len(word_tokenize(result['summary']))
    st.caption(f"Summary length: {summary_length} words | Compression ratio: {result['compression_ratio']*100:.1f}%")

# Main UI
st.markdown('<h1 class="main-header">📝 Advanced Text Summarizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Compare multiple summarization techniques and models</p>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    num_sentences = st.slider("Number of sentences (for extractive methods)", min_value=1, max_value=10, value=3)
    
    st.markdown("---")
    st.header("📋 Summary Template")
    template_type = st.selectbox(
        "Choose summary format:",
        ["Default", "Executive Summary", "Academic Summary", "News Summary", "Meeting Notes"],
        help="Select a template to format your summaries"
    )
    
    st.markdown("---")
    st.header("🎯 Select Methods")
    st.caption("Choose which summarization methods to use:")
    
    # Novel: Smart Method Recommendation
    if 'text_for_recommendation' in st.session_state and st.session_state.text_for_recommendation:
        with st.expander("🤖 Smart Recommendation", expanded=False):
            # Use quick_test for more accurate recommendations
            features, recommendations, test_results = st.session_state.summarizer.recommend_method(
                st.session_state.text_for_recommendation, 
                num_sentences, 
                quick_test=True
            )
            st.markdown("**📊 Text Analysis:**")
            st.write(f"- Length: {features['length'].title()} ({features['word_count']} words)")
            st.write(f"- Complexity: {features['complexity'].title()} ({features['sentence_count']} sentences)")
            
            if test_results:
                st.markdown("**🧪 Quick Test Results:**")
                st.caption("_Methods were tested to predict best performer_")
                test_df = pd.DataFrame([
                    {'Method': method, 'Predicted Score': f"{score:.1f}"}
                    for method, score in sorted(test_results.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(test_df, use_container_width=True, hide_index=True)
            
            st.markdown("**💡 Recommended Methods:**")
            for rec in recommendations:
                confidence_color = "🟢" if rec.get('is_best') else ("🟢" if rec['confidence'] > 0.8 else "🟡" if rec['confidence'] > 0.7 else "🔴")
                best_indicator = " ⭐ BEST PREDICTED" if rec.get('is_best') else ""
                if rec.get('test_score') is not None:
                    st.write(f"{confidence_color} **{rec['method']}** - {rec['reason']}{best_indicator}")
                else:
                    st.write(f"{confidence_color} **{rec['method']}** - {rec['reason']} (Confidence: {rec['confidence']*100:.0f}%)")
            
            st.caption("_Note: Recommendations are predictions. Actual results may vary. We recommend comparing multiple methods._")
    
    # Method selection checkboxes
    selected_methods = {}
    
    st.markdown("**📊 Extractive Methods:**")
    col1, col2 = st.columns(2)
    with col1:
        selected_methods['tfidf'] = st.checkbox("TF-IDF", value=True, help="Term Frequency-Inverse Document Frequency")
    with col2:
        selected_methods['textrank'] = st.checkbox("TextRank", value=True, help="Graph-based ranking algorithm")
    selected_methods['lsa'] = st.checkbox("LSA", value=True, help="Latent Semantic Analysis")
    
    # Novel: Hybrid/Ensemble Method
    st.markdown("**🌟 Novel Methods:**")
    selected_methods['hybrid'] = st.checkbox("🔮 Hybrid (Ensemble)", value=False, 
                                             help="Combines multiple methods intelligently for superior results")
    
    st.markdown("**🤖 Abstractive Methods:**")
    if transformers_available and (st.session_state.summarizer_bart or st.session_state.summarizer_t5):
        if st.session_state.summarizer_bart:
            selected_methods['bart'] = st.checkbox("BART", value=False, help="Facebook's BART abstractive model")
        if st.session_state.summarizer_t5:
            selected_methods['t5'] = st.checkbox("T5", value=False, help="Google's T5 abstractive model")
    else:
        st.caption("_(Abstractive methods not available)_")
        selected_methods['bart'] = False
        selected_methods['t5'] = False
    
    # Check if at least one method is selected
    methods_available = any(selected_methods.values())
    selected_count = sum(selected_methods.values())
    
    st.markdown("---")
    if methods_available:
        st.success(f"✅ **{selected_count} method(s) selected**")
    else:
        st.warning("⚠️ Please select at least one method")
    
    st.caption("💡 Tip: Select multiple methods to compare results!")

# Main input area
st.header("📄 Input")

# Input method selection
input_method = st.radio(
    "Choose input method:",
    ["Text Input", "PDF Upload"],
    horizontal=True,
    help="Select how you want to provide the text to summarize"
)

text_input = ""
pdf_file = None

if input_method == "Text Input":
    text_input = st.text_area(
        "Enter your text to summarize:",
        height=200,
        placeholder="Paste or type your text here...",
        help="Enter the text you want to summarize. The longer the text, the better the results."
    )
else:
    if not PDF_AVAILABLE:
        st.error("⚠️ PDF processing not available. Please install PyPDF2: `pip install pypdf2`")
    else:
        pdf_file = st.file_uploader(
            "Upload a PDF file",
            type=['pdf'],
            help="Upload a PDF file to extract and summarize text"
        )
        
        if pdf_file is not None:
            with st.spinner("Extracting text from PDF..."):
                text_input = extract_text_from_pdf(pdf_file)
                if text_input:
                    st.success(f"✅ Successfully extracted text from PDF!")
                    # Show extracted text in expander
                    with st.expander("📄 View extracted text"):
                        st.text_area("Extracted text:", text_input, height=200, disabled=True, key="extracted_text")
                else:
                    st.error("Failed to extract text from PDF. Please try another file.")

# Show text statistics and store for recommendation
if text_input:
    st.session_state.text_for_recommendation = text_input
    sentences = sent_tokenize(text_input)
    words = word_tokenize(text_input)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sentences", len(sentences))
    with col2:
        st.metric("Words", len(words))
    with col3:
        st.metric("Characters", len(text_input))
    
    # Warn if num_sentences is too high
    if len(sentences) > 0 and num_sentences >= len(sentences):
        st.warning(f"⚠️ You've selected {num_sentences} sentences, but your text only has {len(sentences)} sentences. The summary may include all sentences.")
else:
    if 'text_for_recommendation' in st.session_state:
        st.session_state.text_for_recommendation = None

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    summarize_button = st.button("✨ Summarize", type="primary", use_container_width=True)

# Check if we have results from a previous summarization
has_results = False
if 'summarization_results' in st.session_state:
    if st.session_state.summarization_results is not None:
        if isinstance(st.session_state.summarization_results, dict):
            if len(st.session_state.summarization_results) > 0:
                has_results = True

# Process summarization
if summarize_button:
    # Reset feature states when new summarization starts
    if 'multi_level_generated' in st.session_state:
        st.session_state.multi_level_generated = False
        st.session_state.multi_level_results = None
    if 'key_phrases_generated' in st.session_state:
        st.session_state.key_phrases_generated = False
        st.session_state.key_phrases_results = None
        st.session_state.highlighted_text_result = None
    if 'network_generated' in st.session_state:
        st.session_state.network_generated = False
        st.session_state.network_graph = None
        st.session_state.network_similarity_matrix = None
    
    if not text_input.strip():
        st.error("⚠️ Please enter some text to summarize.")
    elif not methods_available:
        st.error("⚠️ Please select at least one summarization method from the sidebar.")
    else:
        # Count selected methods for progress
        total_selected = sum(selected_methods.values())
        processed = 0
        
        with st.spinner(f"🔄 Processing your text with {total_selected} selected method(s)... This may take a moment."):
            results = {}
            
            # Process only selected methods
            if selected_methods.get('tfidf', False):
                with st.spinner("Processing TF-IDF..."):
                    results['tfidf'] = {
                        'summary': st.session_state.summarizer.extractive_summarize_tfidf(text_input, num_sentences),
                        'type': 'extractive',
                        'method': 'TF-IDF'
                    }
                    processed += 1
            
            if selected_methods.get('textrank', False):
                with st.spinner("Processing TextRank..."):
                    results['textrank'] = {
                        'summary': st.session_state.summarizer.extractive_summarize_textrank(text_input, num_sentences),
                        'type': 'extractive',
                        'method': 'TextRank'
                    }
                    processed += 1
            
            if selected_methods.get('lsa', False):
                with st.spinner("Processing LSA..."):
                    results['lsa'] = {
                        'summary': st.session_state.summarizer.extractive_summarize_lsa(text_input, num_sentences),
                        'type': 'extractive',
                        'method': 'LSA'
                    }
                    processed += 1
            
            # Novel: Hybrid/Ensemble method
            if selected_methods.get('hybrid', False):
                with st.spinner("🔮 Processing Hybrid (Ensemble) method - combining multiple methods..."):
                    hybrid_summary, hybrid_scores = st.session_state.summarizer.hybrid_summarize(text_input, num_sentences)
                    results['hybrid'] = {
                        'summary': hybrid_summary,
                        'type': 'hybrid',
                        'method': 'Hybrid (Ensemble)',
                        'sentence_scores': hybrid_scores,
                        'original_sentences': sent_tokenize(text_input)
                    }
                    processed += 1
            
            # Abstractive methods (only if selected)
            if selected_methods.get('bart', False) and st.session_state.summarizer_bart:
                with st.spinner("Processing BART (this may take longer)..."):
                    results['bart'] = {
                        'summary': st.session_state.summarizer.abstractive_summarize_bart(text_input),
                        'type': 'abstractive',
                        'method': 'BART'
                    }
                    processed += 1
            
            if selected_methods.get('t5', False) and st.session_state.summarizer_t5:
                with st.spinner("Processing T5 (this may take longer)..."):
                    results['t5'] = {
                        'summary': st.session_state.summarizer.abstractive_summarize_t5(text_input),
                        'type': 'abstractive',
                        'method': 'T5'
                    }
                    processed += 1
            
            # Calculate scores
            original_length = len(word_tokenize(text_input))
            
            for key, result in results.items():
                rouge_scores = st.session_state.summarizer.calculate_rouge_scores(text_input, result['summary'])
                result['rouge_scores'] = rouge_scores
                result['compression_ratio'] = round(st.session_state.summarizer.calculate_compression_ratio(text_input, result['summary']), 3)
                
                # Calculate BERTScore if available
                if BERTSCORE_AVAILABLE:
                    with st.spinner(f"Calculating BERTScore for {result['method']}..."):
                        bertscore = st.session_state.summarizer.calculate_bertscore(text_input, result['summary'])
                        result['bertscore'] = bertscore
                else:
                    result['bertscore'] = None
                
                result['overall_score'] = st.session_state.summarizer.calculate_summary_score(
                    text_input, 
                    result['summary'], 
                    rouge_scores, 
                    result.get('bertscore')
                )
            
            # Apply templates to summaries
            for key in results:
                if template_type != "Default":
                    results[key]['formatted_summary'] = apply_template(
                        results[key]['summary'],
                        results[key]['method'],
                        template_type,
                        results[key].get('rouge_scores')
                    )
                else:
                    results[key]['formatted_summary'] = results[key]['summary']
            
            # Sort by score
            sorted_results = sorted(results.items(), key=lambda x: x[1]['overall_score'], reverse=True)
            
            # Find actual best method
            actual_best_method = sorted_results[0][1]['method'] if sorted_results else None
            actual_best_score = sorted_results[0][1]['overall_score'] if sorted_results else 0
            
            # Store results in session state so they persist across reruns
            st.session_state.summarization_results = results
            st.session_state.sorted_results = sorted_results
            st.session_state.actual_best_method = actual_best_method
            st.session_state.actual_best_score = actual_best_score
            st.session_state.original_length = original_length
            st.session_state.current_text_input = text_input
            
            # Display results
            st.success(f"✅ Summarization complete! Processed {len(results)} method(s).")
            
            # Novel: Show actual best method (if multiple methods were run)
            if len(results) > 1:
                st.info(f"🏆 **Actual Best Method:** **{actual_best_method}** (Score: {actual_best_score:.1f})")
                st.caption("_💡 This is based on actual ROUGE scores. Compare this with the predicted recommendations above._")
            
            # Novel Feature: Multi-level Summary Generation
            st.markdown("---")
            with st.expander("🎚️ Multi-Level Summary Generator", expanded=True):
                st.markdown("**Generate summaries at different detail levels:**")
                
                # Initialize session state for multi-level summaries
                if 'multi_level_generated' not in st.session_state:
                    st.session_state.multi_level_generated = False
                if 'multi_level_results' not in st.session_state:
                    st.session_state.multi_level_results = None
                
                # Generate button
                if st.button("✨ Generate Multi-Level Summaries", use_container_width=True, key="generate_multi_level"):
                    with st.spinner("Generating summaries at different levels..."):
                        multi_level = st.session_state.summarizer.generate_multi_level_summaries(text_input)
                        st.session_state.multi_level_results = multi_level
                        st.session_state.multi_level_generated = True
                        st.rerun()
                
                # Display results if generated
                if st.session_state.multi_level_generated and st.session_state.multi_level_results:
                    multi_level = st.session_state.multi_level_results
                    level_tabs = st.tabs(["📄 Brief (~20%)", "📋 Medium (~40%)", "📑 Detailed (~60%)"])
                    
                    with level_tabs[0]:
                        st.markdown(f"**Brief Summary** ({multi_level['brief_count']} sentences)")
                        st.info(multi_level['brief'])
                        brief_words = len(word_tokenize(multi_level['brief']))
                        original_words = len(word_tokenize(text_input))
                        st.caption(f"Length: {brief_words} words | Compression: {brief_words/original_words*100:.1f}%")
                    
                    with level_tabs[1]:
                        st.markdown(f"**Medium Summary** ({multi_level['medium_count']} sentences)")
                        st.info(multi_level['medium'])
                        medium_words = len(word_tokenize(multi_level['medium']))
                        original_words = len(word_tokenize(text_input))
                        st.caption(f"Length: {medium_words} words | Compression: {medium_words/original_words*100:.1f}%")
                    
                    with level_tabs[2]:
                        st.markdown(f"**Detailed Summary** ({multi_level['detailed_count']} sentences)")
                        st.info(multi_level['detailed'])
                        detailed_words = len(word_tokenize(multi_level['detailed']))
                        original_words = len(word_tokenize(text_input))
                        st.caption(f"Length: {detailed_words} words | Compression: {detailed_words/original_words*100:.1f}%")
            
            # Novel Feature: Key Phrase Heatmap
            st.markdown("---")
            with st.expander("🔥 Key Phrase Heatmap", expanded=False):
                st.markdown("**Important phrases highlighted in the original text:**")
                
                # Initialize session state for key phrases
                if 'key_phrases_generated' not in st.session_state:
                    st.session_state.key_phrases_generated = False
                if 'key_phrases_results' not in st.session_state:
                    st.session_state.key_phrases_results = None
                if 'highlighted_text_result' not in st.session_state:
                    st.session_state.highlighted_text_result = None
                
                # Extract button
                if st.button("✨ Extract Key Phrases", use_container_width=True, key="extract_key_phrases"):
                    with st.spinner("Extracting key phrases..."):
                        key_phrases = st.session_state.summarizer.extract_key_phrases(text_input, top_n=15)
                        st.session_state.key_phrases_results = key_phrases
                        
                        # Highlight phrases in text
                        highlighted_text = text_input
                        for phrase in key_phrases[:10]:  # Highlight top 10
                            # Escape special regex characters
                            escaped_phrase = re.escape(phrase)
                            # Highlight phrase (case-insensitive)
                            highlighted_text = re.sub(
                                f'({escaped_phrase})',
                                r'<mark style="background-color: #fef08a; padding: 2px 4px; border-radius: 3px;">\1</mark>',
                                highlighted_text,
                                flags=re.IGNORECASE
                            )
                        st.session_state.highlighted_text_result = highlighted_text
                        st.session_state.key_phrases_generated = True
                        st.rerun()
                
                # Display results if generated
                if st.session_state.key_phrases_generated and st.session_state.key_phrases_results:
                    key_phrases = st.session_state.key_phrases_results
                    
                    st.markdown("**Top Key Phrases:**")
                    # Display phrases in a grid
                    cols = st.columns(3)
                    for idx, phrase in enumerate(key_phrases):
                        with cols[idx % 3]:
                            st.markdown(f"🔑 **{phrase}**")
                    
                    # Highlight phrases in text
                    st.markdown("---")
                    st.markdown("**Original Text with Highlighted Phrases:**")
                    st.markdown(st.session_state.highlighted_text_result, unsafe_allow_html=True)
            
            # Novel Feature: Semantic Similarity Network
            st.markdown("---")
            with st.expander("🕸️ Semantic Similarity Network", expanded=False):
                st.markdown("**Visualize how sentences relate to each other:**")
                
                # Initialize session state for network
                if 'network_generated' not in st.session_state:
                    st.session_state.network_generated = False
                if 'network_graph' not in st.session_state:
                    st.session_state.network_graph = None
                if 'network_similarity_matrix' not in st.session_state:
                    st.session_state.network_similarity_matrix = None
                
                # Generate button
                if st.button("✨ Generate Network Graph", use_container_width=True, key="generate_network"):
                    with st.spinner("Building semantic network..."):
                        G, similarity_matrix = st.session_state.summarizer.build_semantic_network(text_input)
                        st.session_state.network_graph = G
                        st.session_state.network_similarity_matrix = similarity_matrix
                        st.session_state.network_generated = True
                        st.rerun()
                
                # Display results if generated
                if st.session_state.network_generated and st.session_state.network_graph and PLOTLY_AVAILABLE:
                    G = st.session_state.network_graph
                    similarity_matrix = st.session_state.network_similarity_matrix
                    
                    if G:
                        sentences = sent_tokenize(text_input)
                        
                        # Get node positions using spring layout
                        pos = nx.spring_layout(G, k=1, iterations=50)
                        
                        # Extract node and edge information
                        node_x = [pos[node][0] for node in G.nodes()]
                        node_y = [pos[node][1] for node in G.nodes()]
                        node_text = [f"Sentence {i+1}: {sentences[i][:50]}..." for i in G.nodes()]
                        
                        # Create edge traces
                        edge_x = []
                        edge_y = []
                        edge_weights = []
                        for edge in G.edges(data=True):
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                            edge_weights.append(edge[2]['weight'])
                        
                        # Create network graph
                        fig = go.Figure()
                        
                        # Add edges
                        fig.add_trace(go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=1, color='#888'),
                            hoverinfo='none',
                            mode='lines'
                        ))
                        
                        # Add nodes
                        fig.add_trace(go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            marker=dict(
                                size=20,
                                color='#667eea',
                                line=dict(width=2, color='white')
                            ),
                            text=[f"S{i+1}" for i in G.nodes()],
                            textposition="middle center",
                            textfont=dict(size=10, color='white'),
                            hovertext=node_text,
                            hoverinfo='text',
                            name='Sentences'
                        ))
                        
                        fig.update_layout(
                            title='Semantic Similarity Network<br><sub>Nodes = Sentences, Edges = Similarity</sub>',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[
                                dict(
                                    text="Connected sentences are semantically similar",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.005, y=-0.002,
                                    xanchor="left", yanchor="bottom",
                                    font=dict(color="#888", size=12)
                                )
                            ],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=500,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show similarity matrix
                        st.markdown("**Similarity Matrix:**")
                        if similarity_matrix is not None:
                            similarity_df = pd.DataFrame(
                                similarity_matrix,
                                index=[f"S{i+1}" for i in range(len(sentences))],
                                columns=[f"S{i+1}" for i in range(len(sentences))]
                            )
                            # Create heatmap
                            fig_heatmap = px.imshow(
                                similarity_matrix,
                                labels=dict(x="Sentence", y="Sentence", color="Similarity"),
                                x=[f"S{i+1}" for i in range(len(sentences))],
                                y=[f"S{i+1}" for i in range(len(sentences))],
                                color_continuous_scale='Viridis',
                                aspect="auto"
                            )
                            fig_heatmap.update_layout(title="Sentence Similarity Heatmap", height=400)
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                        else:
                            st.warning("Network graph generation requires plotly. Install: `pip install plotly`")
            
            st.markdown("---")
            st.header("📊 Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Text Length", f"{original_length} words")
            with col2:
                st.metric("Methods Processed", len(results))
            with col3:
                if REPORTLAB_AVAILABLE:
                    # Create PDF download button
                    pdf_buffer = BytesIO()
                    methods_used = [result['method'] for result in results.values()]
                    if create_summary_pdf(results, original_length, methods_used, pdf_buffer):
                        pdf_buffer.seek(0)
                        st.download_button(
                            label="📥 Download Summary PDF",
                            data=pdf_buffer,
                            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    st.info("PDF download not available. Install reportlab: `pip install reportlab`")
            
            # Visual Comparison Dashboard (only if multiple methods)
            if len(results) > 1 and PLOTLY_AVAILABLE:
                st.markdown("---")
                st.header("📈 Visual Comparison Dashboard")
                
                # Create comparison charts
                dashboard_tabs = st.tabs(["📊 Metrics Comparison", "🎯 Quality Scores", "☁️ Word Clouds"])
                
                with dashboard_tabs[0]:
                    # Metrics Comparison Chart
                    methods = [result['method'] for result in results.values()]
                    rouge1_scores = [result['rouge_scores']['rouge1']['f1']*100 for result in results.values()]
                    rouge2_scores = [result['rouge_scores']['rouge2']['f1']*100 for result in results.values()]
                    rougeL_scores = [result['rouge_scores']['rougeL']['f1']*100 for result in results.values()]
                    overall_scores = [result['overall_score'] for result in results.values()]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='ROUGE-1 F1', x=methods, y=rouge1_scores, marker_color='#667eea'))
                    fig.add_trace(go.Bar(name='ROUGE-2 F1', x=methods, y=rouge2_scores, marker_color='#764ba2'))
                    fig.add_trace(go.Bar(name='ROUGE-L F1', x=methods, y=rougeL_scores, marker_color='#f093fb'))
                    # Add BERTScore if available
                    if any(result.get('bertscore') is not None for result in results.values()):
                        bertscore_scores = [
                            result.get('bertscore', {}).get('f1', 0)*100 if result.get('bertscore') else 0 
                            for result in results.values()
                        ]
                        fig.add_trace(go.Bar(name='BERTScore F1', x=methods, y=bertscore_scores, marker_color='#10b981'))
                    fig.add_trace(go.Bar(name='Overall Score', x=methods, y=overall_scores, marker_color='#4facfe'))
                    
                    fig.update_layout(
                        title='Metrics Comparison Across Methods',
                        xaxis_title='Method',
                        yaxis_title='Score (%)',
                        barmode='group',
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Compression Ratio Chart
                    compression_ratios = [result['compression_ratio']*100 for result in results.values()]
                    fig2 = go.Figure(data=go.Bar(x=methods, y=compression_ratios, marker_color='#10b981'))
                    fig2.update_layout(
                        title='Compression Ratio by Method',
                        xaxis_title='Method',
                        yaxis_title='Compression Ratio (%)',
                        height=300,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                with dashboard_tabs[1]:
                    # Radar Chart for Quality Scores
                    # Check if BERTScore is available
                    has_bertscore = any(result.get('bertscore') is not None for result in results.values())
                    if has_bertscore:
                        categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 'Compression', 'Overall']
                    else:
                        categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Compression', 'Overall']
                    
                    fig_radar = go.Figure()
                    
                    for method_name, result in results.items():
                        if has_bertscore:
                            bertscore_val = result.get('bertscore', {}).get('f1', 0)*100 if result.get('bertscore') else 0
                            values = [
                                result['rouge_scores']['rouge1']['f1']*100,
                                result['rouge_scores']['rouge2']['f1']*100,
                                result['rouge_scores']['rougeL']['f1']*100,
                                bertscore_val,
                                result['compression_ratio']*100,
                                result['overall_score']
                            ]
                        else:
                            values = [
                                result['rouge_scores']['rouge1']['f1']*100,
                                result['rouge_scores']['rouge2']['f1']*100,
                                result['rouge_scores']['rougeL']['f1']*100,
                                result['compression_ratio']*100,
                                result['overall_score']
                            ]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name=result['method']
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=True,
                        title="Quality Scores Radar Chart",
                        height=500,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
                    # Overall Score Comparison
                    methods_list = [result['method'] for result in results.values()]
                    overall_scores_list = [result['overall_score'] for result in results.values()]
                    fig_bar = go.Figure(data=go.Bar(
                        x=methods_list,
                        y=overall_scores_list,
                        marker=dict(
                            color=overall_scores_list,
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=[f"{score:.1f}" for score in overall_scores_list],
                        textposition='outside'
                    ))
                    fig_bar.update_layout(
                        title='Overall Score Comparison',
                        xaxis_title='Method',
                        yaxis_title='Overall Score',
                        height=400,
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with dashboard_tabs[2]:
                    if WORDCLOUD_AVAILABLE:
                        st.subheader("Word Clouds for Each Summary")
                        cols = st.columns(min(3, len(results)))
                        
                        for idx, (method_key, result) in enumerate(results.items()):
                            col = cols[idx % len(cols)]
                            with col:
                                try:
                                    # Generate word cloud
                                    wordcloud = WordCloud(
                                        width=400,
                                        height=300,
                                        background_color='white',
                                        colormap='viridis'
                                    ).generate(result['summary'])
                                    
                                    # Display word cloud
                                    fig_wc, ax = plt.subplots(figsize=(5, 4))
                                    ax.imshow(wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    ax.set_title(result['method'], fontsize=12, fontweight='bold')
                                    st.pyplot(fig_wc)
                                    plt.close(fig_wc)
                                except Exception as e:
                                    st.error(f"Error generating word cloud for {result['method']}: {str(e)}")
                    else:
                        st.info("Word cloud generation not available. Install wordcloud: `pip install wordcloud matplotlib`")
                
                # Comparison Table
                st.markdown("---")
                st.subheader("📋 Detailed Comparison Table")
                comparison_data = {
                    'Method': [result['method'] for result in results.values()],
                    'Type': [result['type'] for result in results.values()],
                    'Overall Score': [f"{result['overall_score']:.2f}" for result in results.values()],
                    'ROUGE-1 F1': [f"{result['rouge_scores']['rouge1']['f1']*100:.2f}%" for result in results.values()],
                    'ROUGE-2 F1': [f"{result['rouge_scores']['rouge2']['f1']*100:.2f}%" for result in results.values()],
                    'ROUGE-L F1': [f"{result['rouge_scores']['rougeL']['f1']*100:.2f}%" for result in results.values()],
                }
                # Add BERTScore if available for at least one result
                if any(result.get('bertscore') is not None for result in results.values()):
                    comparison_data['BERTScore F1'] = [
                        f"{result.get('bertscore', {}).get('f1', 0)*100:.2f}%" if result.get('bertscore') else "N/A" 
                        for result in results.values()
                    ]
                comparison_data['Compression'] = [f"{result['compression_ratio']*100:.2f}%" for result in results.values()]
                comparison_data['Summary Length'] = [f"{len(word_tokenize(result['summary']))} words" for result in results.values()]
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            elif len(results) > 1 and not PLOTLY_AVAILABLE:
                st.info("📈 Visual comparison charts not available. Install plotly: `pip install plotly`")
            
            # Novel Feature: Interactive Summary Builder
            st.markdown("---")
            with st.expander("🛠️ Interactive Summary Builder", expanded=False):
                st.markdown("**Build your own custom summary by selecting sentences:**")
                sentences_list = sent_tokenize(text_input)
                
                if len(sentences_list) > 1:
                    st.markdown(f"**Select sentences to include in your custom summary:**")
                    
                    # Get sentence scores if available (from any extractive method)
                    sentence_scores = {}
                    for result in results.values():
                        if result.get('type') in ['extractive', 'hybrid'] and 'sentence_scores' in result:
                            sentence_scores = result.get('sentence_scores', {})
                            break
                    
                    # Create checkboxes for each sentence
                    selected_sentence_indices = []
                    cols = st.columns(2)
                    
                    for idx, sentence in enumerate(sentences_list):
                        col = cols[idx % 2]
                        with col:
                            # Show score if available
                            score = sentence_scores.get(idx, 0)
                            score_display = f" (Score: {score:.2f})" if score > 0 else ""
                            score_color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴" if score > 0 else ""
                            
                            if st.checkbox(
                                f"{score_color} Sentence {idx+1}{score_display}",
                                key=f"custom_sentence_{idx}",
                                value=False
                            ):
                                selected_sentence_indices.append(idx)
                            
                            # Show sentence preview
                            st.caption(f'"{sentence[:80]}..."' if len(sentence) > 80 else f'"{sentence}"')
                    
                    # Build custom summary
                    if selected_sentence_indices:
                        selected_sentence_indices.sort()  # Maintain original order
                        custom_summary = ' '.join([sentences_list[i] for i in selected_sentence_indices])
                        
                        st.markdown("---")
                        st.markdown("**Your Custom Summary:**")
                        st.success(custom_summary)
                        
                        # Calculate metrics for custom summary
                        custom_rouge = st.session_state.summarizer.calculate_rouge_scores(text_input, custom_summary)
                        custom_compression = st.session_state.summarizer.calculate_compression_ratio(text_input, custom_summary)
                        # Calculate BERTScore if available
                        custom_bertscore = None
                        if BERTSCORE_AVAILABLE:
                            custom_bertscore = st.session_state.summarizer.calculate_bertscore(text_input, custom_summary)
                        custom_score = st.session_state.summarizer.calculate_summary_score(
                            text_input, custom_summary, custom_rouge, custom_bertscore
                        )
                        
                        if custom_bertscore is not None:
                            col1, col2, col3, col4, col5 = st.columns(5)
                        else:
                            col1, col2, col3, col4 = st.columns(4)
                            col5 = None
                        with col1:
                            st.metric("Overall Score", f"{custom_score:.1f}")
                        with col2:
                            st.metric("ROUGE-1 F1", f"{custom_rouge['rouge1']['f1']*100:.1f}%")
                        with col3:
                            st.metric("ROUGE-2 F1", f"{custom_rouge['rouge2']['f1']*100:.1f}%")
                        with col4:
                            st.metric("Compression", f"{custom_compression*100:.1f}%")
                        if col5 and custom_bertscore is not None:
                            with col5:
                                st.metric("BERTScore F1", f"{custom_bertscore['f1']*100:.1f}%")
                        
                        # Compare with best method
                        if len(results) > 0:
                            best_score = max([r['overall_score'] for r in results.values()])
                            if custom_score >= best_score * 0.9:
                                st.success(f"🎉 Your custom summary performs well! (Score: {custom_score:.1f} vs Best: {best_score:.1f})")
                            else:
                                st.info(f"💡 Your custom summary score: {custom_score:.1f} (Best method: {best_score:.1f})")
                    else:
                        st.info("👆 Select sentences above to build your custom summary")
                else:
                    st.info("Text needs at least 2 sentences for interactive builder")
            
            st.markdown("---")
            st.header("📄 Individual Summaries")
            
            # Find best method (only if multiple methods)
            best_method = sorted_results[0][0] if len(sorted_results) > 1 else None
            
            # Display results
            if len(sorted_results) == 1:
                # Single result - show directly without tabs
                method_key, result = sorted_results[0]
                display_result_card(result, best_method is None, original_length)
            else:
                # Multiple results - show in tabs
                tab_names = [f"{result[1]['method']} ({result[1]['overall_score']:.1f})" for result in sorted_results]
                tabs = st.tabs(tab_names)
                
                for idx, (tab, (method_key, result)) in enumerate(zip(tabs, sorted_results)):
                    with tab:
                        is_best = method_key == best_method
                        
                        # Header with badges
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            method_type_class = "extractive-badge" if result['type'] == 'extractive' else "abstractive-badge"
                            st.markdown(f"""
                                <h2>
                                    {result['method']}
                                    <span class="method-badge {method_type_class}">{result['type']}</span>
                                    {'<span class="best-badge">🏆 Best</span>' if is_best else ''}
                                </h2>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Overall Score", f"{result['overall_score']:.1f}")
                        
                        # Summary
                        st.subheader("📝 Summary")
                        if 'formatted_summary' in result and template_type != "Default":
                            st.markdown(result['formatted_summary'])
                        else:
                            st.info(result['summary'])
                        
                        # Novel: Interactive Sentence Selection (for extractive/hybrid methods with scores)
                        if result.get('type') in ['extractive', 'hybrid'] and 'sentence_scores' in result:
                            st.markdown("---")
                            with st.expander("🔍 View Sentence Selection Explanation", expanded=False):
                                st.markdown("**💡 How sentences were selected:**")
                                sentences_list = result.get('original_sentences', [])
                                if not sentences_list and 'original_text' in result:
                                    sentences_list = sent_tokenize(result['original_text'])
                                elif not sentences_list:
                                    sentences_list = []
                                
                                sentence_scores_dict = result.get('sentence_scores', {})
                                
                                # Create a visualization of sentence scores
                                if PLOTLY_AVAILABLE and sentence_scores_dict and sentences_list:
                                    score_values = [sentence_scores_dict.get(i, 0) for i in range(len(sentences_list))]
                                    sentence_indices = list(range(len(sentences_list)))
                                    
                                    fig_scores = go.Figure(data=go.Bar(
                                        x=sentence_indices,
                                        y=score_values,
                                        marker=dict(
                                            color=score_values,
                                            colorscale='Viridis',
                                            showscale=True,
                                            colorbar=dict(title="Importance Score")
                                        ),
                                        text=[f"Sentence {i+1}" for i in sentence_indices],
                                        textposition='outside'
                                    ))
                                    fig_scores.update_layout(
                                        title='Sentence Importance Scores',
                                        xaxis_title='Sentence Number',
                                        yaxis_title='Importance Score (0-1)',
                                        height=300,
                                        template='plotly_white'
                                    )
                                    st.plotly_chart(fig_scores, use_container_width=True)
                                
                                # Show sentences with their scores
                                if sentences_list:
                                    st.markdown("**📊 Sentences with scores:**")
                                    summary_sentences = set(sent_tokenize(result['summary']))
                                    for idx, sentence in enumerate(sentences_list):
                                        score = sentence_scores_dict.get(idx, 0)
                                        score_color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                                        is_selected = any(sentence.strip() in s or s in sentence.strip() for s in summary_sentences)
                                        selected_indicator = " ✅ SELECTED" if is_selected else ""
                                        st.markdown(f"{score_color} **Sentence {idx+1}** (Score: {score:.3f}){selected_indicator}")
                                        st.caption(f'"{sentence[:100]}..."' if len(sentence) > 100 else f'"{sentence}"')
                        
                        # Metrics
                        if result.get('bertscore') is not None:
                            col1, col2, col3, col4, col5 = st.columns(5)
                        else:
                            col1, col2, col3, col4 = st.columns(4)
                            col5 = None
                        
                        with col1:
                            st.metric("ROUGE-1 F1", f"{result['rouge_scores']['rouge1']['f1']*100:.1f}%")
                        
                        with col2:
                            st.metric("ROUGE-2 F1", f"{result['rouge_scores']['rouge2']['f1']*100:.1f}%")
                        
                        with col3:
                            st.metric("ROUGE-L F1", f"{result['rouge_scores']['rougeL']['f1']*100:.1f}%")
                        
                        with col4:
                            st.metric("Compression", f"{result['compression_ratio']*100:.1f}%")
                        
                        if col5 and result.get('bertscore') is not None:
                            with col5:
                                st.metric("BERTScore F1", f"{result['bertscore']['f1']*100:.1f}%")
                        
                        # Detailed scores
                        with st.expander("📈 Detailed Evaluation Scores"):
                            if result.get('bertscore') is not None:
                                col1, col2, col3, col4 = st.columns(4)
                            else:
                                col1, col2, col3, col4 = st.columns(3)
                                col4 = None
                            
                            with col1:
                                st.write("**ROUGE-1**")
                                st.write(f"Precision: {result['rouge_scores']['rouge1']['precision']*100:.2f}%")
                                st.write(f"Recall: {result['rouge_scores']['rouge1']['recall']*100:.2f}%")
                                st.write(f"F1: {result['rouge_scores']['rouge1']['f1']*100:.2f}%")
                            
                            with col2:
                                st.write("**ROUGE-2**")
                                st.write(f"Precision: {result['rouge_scores']['rouge2']['precision']*100:.2f}%")
                                st.write(f"Recall: {result['rouge_scores']['rouge2']['recall']*100:.2f}%")
                                st.write(f"F1: {result['rouge_scores']['rouge2']['f1']*100:.2f}%")
                            
                            with col3:
                                st.write("**ROUGE-L**")
                                st.write(f"Precision: {result['rouge_scores']['rougeL']['precision']*100:.2f}%")
                                st.write(f"Recall: {result['rouge_scores']['rougeL']['recall']*100:.2f}%")
                                st.write(f"F1: {result['rouge_scores']['rougeL']['f1']*100:.2f}%")
                            
                            if col4 and result.get('bertscore') is not None:
                                with col4:
                                    st.write("**BERTScore**")
                                    st.write(f"Precision: {result['bertscore']['precision']*100:.2f}%")
                                    st.write(f"Recall: {result['bertscore']['recall']*100:.2f}%")
                                    st.write(f"F1: {result['bertscore']['f1']*100:.2f}%")
                            elif result.get('bertscore') is None and not BERTSCORE_AVAILABLE:
                                st.info("💡 Install bert-score to see BERTScore metrics: `pip install bert-score`")
                        
                        # Summary stats
                        summary_length = len(word_tokenize(result['summary']))
                        st.caption(f"Summary length: {summary_length} words | Compression ratio: {result['compression_ratio']*100:.1f}%")

# Display stored results if they exist (even if summarize button wasn't just clicked)
elif has_results:
    results = st.session_state.summarization_results
    sorted_results = st.session_state.sorted_results
    actual_best_method = st.session_state.actual_best_method
    actual_best_score = st.session_state.actual_best_score
    original_length = st.session_state.original_length
    text_input = st.session_state.current_text_input
    
    st.success(f"✅ Showing previous summarization results ({len(results)} method(s)).")
    
    # Novel: Show actual best method (if multiple methods were run)
    if len(results) > 1:
        st.info(f"🏆 **Actual Best Method:** **{actual_best_method}** (Score: {actual_best_score:.1f})")
        st.caption("_💡 This is based on actual ROUGE scores._")
    
    # Novel Feature: Multi-level Summary Generation
    st.markdown("---")
    with st.expander("🎚️ Multi-Level Summary Generator", expanded=True):
        st.markdown("**Generate summaries at different detail levels:**")
        
        # Initialize session state for multi-level summaries
        if 'multi_level_generated' not in st.session_state:
            st.session_state.multi_level_generated = False
        if 'multi_level_results' not in st.session_state:
            st.session_state.multi_level_results = None
        
        # Generate button
        if st.button("✨ Generate Multi-Level Summaries", use_container_width=True, key="generate_multi_level_stored"):
            with st.spinner("Generating summaries at different levels..."):
                multi_level = st.session_state.summarizer.generate_multi_level_summaries(text_input)
                st.session_state.multi_level_results = multi_level
                st.session_state.multi_level_generated = True
                st.rerun()
        
        # Display results if generated
        if st.session_state.multi_level_generated and st.session_state.multi_level_results:
            multi_level = st.session_state.multi_level_results
            level_tabs = st.tabs(["📄 Brief (~20%)", "📋 Medium (~40%)", "📑 Detailed (~60%)"])
            
            with level_tabs[0]:
                st.markdown(f"**Brief Summary** ({multi_level['brief_count']} sentences)")
                st.info(multi_level['brief'])
                brief_words = len(word_tokenize(multi_level['brief']))
                original_words = len(word_tokenize(text_input))
                st.caption(f"Length: {brief_words} words | Compression: {brief_words/original_words*100:.1f}%")
            
            with level_tabs[1]:
                st.markdown(f"**Medium Summary** ({multi_level['medium_count']} sentences)")
                st.info(multi_level['medium'])
                medium_words = len(word_tokenize(multi_level['medium']))
                original_words = len(word_tokenize(text_input))
                st.caption(f"Length: {medium_words} words | Compression: {medium_words/original_words*100:.1f}%")
            
            with level_tabs[2]:
                st.markdown(f"**Detailed Summary** ({multi_level['detailed_count']} sentences)")
                st.info(multi_level['detailed'])
                detailed_words = len(word_tokenize(multi_level['detailed']))
                original_words = len(word_tokenize(text_input))
                st.caption(f"Length: {detailed_words} words | Compression: {detailed_words/original_words*100:.1f}%")
    
    # Novel Feature: Key Phrase Heatmap
    st.markdown("---")
    with st.expander("🔥 Key Phrase Heatmap", expanded=False):
        st.markdown("**Important phrases highlighted in the original text:**")
        
        # Initialize session state for key phrases
        if 'key_phrases_generated' not in st.session_state:
            st.session_state.key_phrases_generated = False
        if 'key_phrases_results' not in st.session_state:
            st.session_state.key_phrases_results = None
        if 'highlighted_text_result' not in st.session_state:
            st.session_state.highlighted_text_result = None
        
        # Extract button
        if st.button("✨ Extract Key Phrases", use_container_width=True, key="extract_key_phrases_stored"):
            with st.spinner("Extracting key phrases..."):
                key_phrases = st.session_state.summarizer.extract_key_phrases(text_input, top_n=15)
                st.session_state.key_phrases_results = key_phrases
                
                # Highlight phrases in text
                highlighted_text = text_input
                for phrase in key_phrases[:10]:  # Highlight top 10
                    # Escape special regex characters
                    escaped_phrase = re.escape(phrase)
                    # Highlight phrase (case-insensitive)
                    highlighted_text = re.sub(
                        f'({escaped_phrase})',
                        r'<mark style="background-color: #fef08a; padding: 2px 4px; border-radius: 3px;">\1</mark>',
                        highlighted_text,
                        flags=re.IGNORECASE
                    )
                st.session_state.highlighted_text_result = highlighted_text
                st.session_state.key_phrases_generated = True
                st.rerun()
        
        # Display results if generated
        if st.session_state.key_phrases_generated and st.session_state.key_phrases_results:
            key_phrases = st.session_state.key_phrases_results
            
            st.markdown("**Top Key Phrases:**")
            # Display phrases in a grid
            cols = st.columns(3)
            for idx, phrase in enumerate(key_phrases):
                with cols[idx % 3]:
                    st.markdown(f"🔑 **{phrase}**")
            
            # Highlight phrases in text
            st.markdown("---")
            st.markdown("**Original Text with Highlighted Phrases:**")
            st.markdown(st.session_state.highlighted_text_result, unsafe_allow_html=True)
    
    # Novel Feature: Semantic Similarity Network
    st.markdown("---")
    with st.expander("🕸️ Semantic Similarity Network", expanded=False):
        st.markdown("**Visualize how sentences relate to each other:**")
        
        # Initialize session state for network
        if 'network_generated' not in st.session_state:
            st.session_state.network_generated = False
        if 'network_graph' not in st.session_state:
            st.session_state.network_graph = None
        if 'network_similarity_matrix' not in st.session_state:
            st.session_state.network_similarity_matrix = None
        
        # Generate button
        if st.button("✨ Generate Network Graph", use_container_width=True, key="generate_network_stored"):
            with st.spinner("Building semantic network..."):
                G, similarity_matrix = st.session_state.summarizer.build_semantic_network(text_input)
                st.session_state.network_graph = G
                st.session_state.network_similarity_matrix = similarity_matrix
                st.session_state.network_generated = True
                st.rerun()
        
        # Display results if generated
        if st.session_state.network_generated and st.session_state.network_graph and PLOTLY_AVAILABLE:
            G = st.session_state.network_graph
            similarity_matrix = st.session_state.network_similarity_matrix
            
            if G:
                sentences = sent_tokenize(text_input)
                
                # Get node positions using spring layout
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Extract node and edge information
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                node_text = [f"Sentence {i+1}: {sentences[i][:50]}..." for i in G.nodes()]
                
                # Create edge traces
                edge_x = []
                edge_y = []
                edge_weights = []
                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_weights.append(edge[2]['weight'])
                
                # Create network graph
                fig = go.Figure()
                
                # Add edges
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                ))
                
                # Add nodes
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color='#667eea',
                        line=dict(width=2, color='white')
                    ),
                    text=[f"S{i+1}" for i in G.nodes()],
                    textposition="middle center",
                    textfont=dict(size=10, color='white'),
                    hovertext=node_text,
                    hoverinfo='text',
                    name='Sentences'
                ))
                
                fig.update_layout(
                    title='Semantic Similarity Network<br><sub>Nodes = Sentences, Edges = Similarity</sub>',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[
                        dict(
                            text="Connected sentences are semantically similar",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor="left", yanchor="bottom",
                            font=dict(color="#888", size=12)
                        )
                    ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show similarity matrix
                st.markdown("**Similarity Matrix:**")
                if similarity_matrix is not None:
                    # Create heatmap
                    fig_heatmap = px.imshow(
                        similarity_matrix,
                        labels=dict(x="Sentence", y="Sentence", color="Similarity"),
                        x=[f"S{i+1}" for i in range(len(sentences))],
                        y=[f"S{i+1}" for i in range(len(sentences))],
                        color_continuous_scale='Viridis',
                        aspect="auto"
                    )
                    fig_heatmap.update_layout(title="Sentence Similarity Heatmap", height=400)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            elif not PLOTLY_AVAILABLE:
                st.warning("Network graph generation requires plotly. Install: `pip install plotly`")
            else:
                st.warning("Could not generate network graph. Please try again.")
    
    st.markdown("---")
    st.header("📊 Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Text Length", f"{original_length} words")
    with col2:
        st.metric("Methods Processed", len(results))
    with col3:
        if REPORTLAB_AVAILABLE:
            # Create PDF download button
            pdf_buffer = BytesIO()
            methods_used = [result['method'] for result in results.values()]
            if create_summary_pdf(results, original_length, methods_used, pdf_buffer):
                pdf_buffer.seek(0)
                st.download_button(
                    label="📥 Download Summary PDF",
                    data=pdf_buffer,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.info("PDF download not available. Install reportlab: `pip install reportlab`")
    
    # Visual Comparison Dashboard (only if multiple methods)
    if len(results) > 1 and PLOTLY_AVAILABLE:
        st.markdown("---")
        st.header("📈 Visual Comparison Dashboard")
        
        # Create comparison charts
        dashboard_tabs = st.tabs(["📊 Metrics Comparison", "🎯 Quality Scores", "☁️ Word Clouds"])
        
        with dashboard_tabs[0]:
            # Metrics Comparison Chart
            methods = [result['method'] for result in results.values()]
            rouge1_scores = [result['rouge_scores']['rouge1']['f1']*100 for result in results.values()]
            rouge2_scores = [result['rouge_scores']['rouge2']['f1']*100 for result in results.values()]
            rougeL_scores = [result['rouge_scores']['rougeL']['f1']*100 for result in results.values()]
            overall_scores = [result['overall_score'] for result in results.values()]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='ROUGE-1 F1', x=methods, y=rouge1_scores, marker_color='#667eea'))
            fig.add_trace(go.Bar(name='ROUGE-2 F1', x=methods, y=rouge2_scores, marker_color='#764ba2'))
            fig.add_trace(go.Bar(name='ROUGE-L F1', x=methods, y=rougeL_scores, marker_color='#f093fb'))
            # Add BERTScore if available
            if any(result.get('bertscore') is not None for result in results.values()):
                bertscore_scores = [
                    result.get('bertscore', {}).get('f1', 0)*100 if result.get('bertscore') else 0 
                    for result in results.values()
                ]
                fig.add_trace(go.Bar(name='BERTScore F1', x=methods, y=bertscore_scores, marker_color='#10b981'))
            fig.add_trace(go.Bar(name='Overall Score', x=methods, y=overall_scores, marker_color='#4facfe'))
            
            fig.update_layout(
                title='Metrics Comparison Across Methods',
                xaxis_title='Method',
                yaxis_title='Score (%)',
                barmode='group',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Compression Ratio Chart
            compression_ratios = [result['compression_ratio']*100 for result in results.values()]
            fig2 = go.Figure(data=go.Bar(x=methods, y=compression_ratios, marker_color='#10b981'))
            fig2.update_layout(
                title='Compression Ratio by Method',
                xaxis_title='Method',
                yaxis_title='Compression Ratio (%)',
                height=300,
                template='plotly_white'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with dashboard_tabs[1]:
            # Radar Chart for Quality Scores
            # Check if BERTScore is available
            has_bertscore = any(result.get('bertscore') is not None for result in results.values())
            if has_bertscore:
                categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore', 'Compression', 'Overall']
            else:
                categories = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Compression', 'Overall']
            
            fig_radar = go.Figure()
            
            for method_name, result in results.items():
                if has_bertscore:
                    bertscore_val = result.get('bertscore', {}).get('f1', 0)*100 if result.get('bertscore') else 0
                    values = [
                        result['rouge_scores']['rouge1']['f1']*100,
                        result['rouge_scores']['rouge2']['f1']*100,
                        result['rouge_scores']['rougeL']['f1']*100,
                        bertscore_val,
                        result['compression_ratio']*100,
                        result['overall_score']
                    ]
                else:
                    values = [
                        result['rouge_scores']['rouge1']['f1']*100,
                        result['rouge_scores']['rouge2']['f1']*100,
                        result['rouge_scores']['rougeL']['f1']*100,
                        result['compression_ratio']*100,
                        result['overall_score']
                    ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=result['method']
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Quality Scores Radar Chart",
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Overall Score Comparison
            methods_list = [result['method'] for result in results.values()]
            overall_scores_list = [result['overall_score'] for result in results.values()]
            fig_bar = go.Figure(data=go.Bar(
                x=methods_list,
                y=overall_scores_list,
                marker=dict(
                    color=overall_scores_list,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f"{score:.1f}" for score in overall_scores_list],
                textposition='outside'
            ))
            fig_bar.update_layout(
                title='Overall Score Comparison',
                xaxis_title='Method',
                yaxis_title='Overall Score',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with dashboard_tabs[2]:
            if WORDCLOUD_AVAILABLE:
                st.subheader("Word Clouds for Each Summary")
                cols = st.columns(min(3, len(results)))
                
                for idx, (method_key, result) in enumerate(results.items()):
                    col = cols[idx % len(cols)]
                    with col:
                        try:
                            # Generate word cloud
                            wordcloud = WordCloud(
                                width=400,
                                height=300,
                                background_color='white',
                                colormap='viridis'
                            ).generate(result['summary'])
                            
                            # Display word cloud
                            fig_wc, ax = plt.subplots(figsize=(5, 4))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            ax.set_title(result['method'], fontsize=12, fontweight='bold')
                            st.pyplot(fig_wc)
                            plt.close(fig_wc)
                        except Exception as e:
                            st.error(f"Error generating word cloud for {result['method']}: {str(e)}")
            else:
                st.info("Word cloud generation not available. Install wordcloud: `pip install wordcloud matplotlib`")
        
        # Comparison Table
        st.markdown("---")
        st.subheader("📋 Detailed Comparison Table")
        comparison_data = {
            'Method': [result['method'] for result in results.values()],
            'Type': [result['type'] for result in results.values()],
            'Overall Score': [f"{result['overall_score']:.2f}" for result in results.values()],
            'ROUGE-1 F1': [f"{result['rouge_scores']['rouge1']['f1']*100:.2f}%" for result in results.values()],
            'ROUGE-2 F1': [f"{result['rouge_scores']['rouge2']['f1']*100:.2f}%" for result in results.values()],
            'ROUGE-L F1': [f"{result['rouge_scores']['rougeL']['f1']*100:.2f}%" for result in results.values()],
            'Compression': [f"{result['compression_ratio']*100:.2f}%" for result in results.values()],
            'Summary Length': [f"{len(word_tokenize(result['summary']))} words" for result in results.values()]
        }
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    elif len(results) > 1 and not PLOTLY_AVAILABLE:
        st.info("📈 Visual comparison charts not available. Install plotly: `pip install plotly`")
    
    # Novel Feature: Interactive Summary Builder
    st.markdown("---")
    with st.expander("🛠️ Interactive Summary Builder", expanded=False):
        st.markdown("**Build your own custom summary by selecting sentences:**")
        sentences_list = sent_tokenize(text_input)
        
        if len(sentences_list) > 1:
            st.markdown(f"**Select sentences to include in your custom summary:**")
            
            # Get sentence scores if available (from any extractive method)
            sentence_scores = {}
            for result in results.values():
                if result.get('type') in ['extractive', 'hybrid'] and 'sentence_scores' in result:
                    sentence_scores = result.get('sentence_scores', {})
                    break
            
            # Create checkboxes for each sentence
            selected_sentence_indices = []
            cols = st.columns(2)
            
            for idx, sentence in enumerate(sentences_list):
                col = cols[idx % 2]
                with col:
                    # Show score if available
                    score = sentence_scores.get(idx, 0)
                    score_display = f" (Score: {score:.2f})" if score > 0 else ""
                    score_color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴" if score > 0 else ""
                    
                    if st.checkbox(
                        f"{score_color} Sentence {idx+1}{score_display}",
                        key=f"custom_sentence_stored_{idx}",
                        value=False
                    ):
                        selected_sentence_indices.append(idx)
                    
                    # Show sentence preview
                    st.caption(f'"{sentence[:80]}..."' if len(sentence) > 80 else f'"{sentence}"')
            
            # Build custom summary
            if selected_sentence_indices:
                selected_sentence_indices.sort()  # Maintain original order
                custom_summary = ' '.join([sentences_list[i] for i in selected_sentence_indices])
                
                st.markdown("---")
                st.markdown("**Your Custom Summary:**")
                st.success(custom_summary)
                
                # Calculate metrics for custom summary
                custom_rouge = st.session_state.summarizer.calculate_rouge_scores(text_input, custom_summary)
                custom_compression = st.session_state.summarizer.calculate_compression_ratio(text_input, custom_summary)
                # Calculate BERTScore if available
                custom_bertscore = None
                if BERTSCORE_AVAILABLE:
                    custom_bertscore = st.session_state.summarizer.calculate_bertscore(text_input, custom_summary)
                custom_score = st.session_state.summarizer.calculate_summary_score(
                    text_input, custom_summary, custom_rouge, custom_bertscore
                )
                
                if custom_bertscore is not None:
                    col1, col2, col3, col4, col5 = st.columns(5)
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    col5 = None
                with col1:
                    st.metric("Overall Score", f"{custom_score:.1f}")
                with col2:
                    st.metric("ROUGE-1 F1", f"{custom_rouge['rouge1']['f1']*100:.1f}%")
                with col3:
                    st.metric("ROUGE-2 F1", f"{custom_rouge['rouge2']['f1']*100:.1f}%")
                with col4:
                    st.metric("Compression", f"{custom_compression*100:.1f}%")
                if col5 and custom_bertscore is not None:
                    with col5:
                        st.metric("BERTScore F1", f"{custom_bertscore['f1']*100:.1f}%")
                
                # Compare with best method
                if len(results) > 0:
                    best_score = max([r['overall_score'] for r in results.values()])
                    if custom_score >= best_score * 0.9:
                        st.success(f"🎉 Your custom summary performs well! (Score: {custom_score:.1f} vs Best: {best_score:.1f})")
                    else:
                        st.info(f"💡 Your custom summary score: {custom_score:.1f} (Best method: {best_score:.1f})")
            else:
                st.info("👆 Select sentences above to build your custom summary")
        else:
            st.info("Text needs at least 2 sentences for interactive builder")
    
    st.markdown("---")
    st.header("📄 Individual Summaries")
    
    # Find best method (only if multiple methods)
    best_method = sorted_results[0][0] if len(sorted_results) > 1 else None
    
    # Display results
    if len(sorted_results) == 1:
        # Single result - show directly without tabs
        method_key, result = sorted_results[0]
        display_result_card(result, best_method is None, original_length)
    else:
        # Multiple results - show in tabs
        tab_names = [f"{result[1]['method']} ({result[1]['overall_score']:.1f})" for result in sorted_results]
        tabs = st.tabs(tab_names)
        
        for idx, (tab, (method_key, result)) in enumerate(zip(tabs, sorted_results)):
            with tab:
                is_best = method_key == best_method
                
                # Header with badges
                col1, col2 = st.columns([3, 1])
                with col1:
                    method_type_class = "extractive-badge" if result['type'] == 'extractive' else "abstractive-badge"
                    st.markdown(f"""
                        <h2>
                            {result['method']}
                            <span class="method-badge {method_type_class}">{result['type']}</span>
                            {'<span class="best-badge">🏆 Best</span>' if is_best else ''}
                        </h2>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Overall Score", f"{result['overall_score']:.1f}")
                
                # Summary
                st.subheader("📝 Summary")
                if 'formatted_summary' in result and template_type != "Default":
                    st.markdown(result['formatted_summary'])
                else:
                    st.info(result['summary'])
                
                # Novel: Interactive Sentence Selection (for extractive/hybrid methods with scores)
                if result.get('type') in ['extractive', 'hybrid'] and 'sentence_scores' in result:
                    st.markdown("---")
                    with st.expander("🔍 View Sentence Selection Explanation", expanded=False):
                        st.markdown("**💡 How sentences were selected:**")
                        sentences_list = result.get('original_sentences', [])
                        if not sentences_list and 'original_text' in result:
                            sentences_list = sent_tokenize(result['original_text'])
                        elif not sentences_list:
                            sentences_list = []
                        
                        sentence_scores_dict = result.get('sentence_scores', {})
                        
                        # Create a visualization of sentence scores
                        if PLOTLY_AVAILABLE and sentence_scores_dict and sentences_list:
                            score_values = [sentence_scores_dict.get(i, 0) for i in range(len(sentences_list))]
                            sentence_indices = list(range(len(sentences_list)))
                            
                            fig_scores = go.Figure(data=go.Bar(
                                x=sentence_indices,
                                y=score_values,
                                marker=dict(
                                    color=score_values,
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="Importance Score")
                                ),
                                text=[f"Sentence {i+1}" for i in sentence_indices],
                                textposition='outside'
                            ))
                            fig_scores.update_layout(
                                title='Sentence Importance Scores',
                                xaxis_title='Sentence Number',
                                yaxis_title='Importance Score (0-1)',
                                height=300,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_scores, use_container_width=True)
                        
                        # Show sentences with their scores
                        if sentences_list:
                            st.markdown("**📊 Sentences with scores:**")
                            summary_sentences = set(sent_tokenize(result['summary']))
                            for idx, sentence in enumerate(sentences_list):
                                score = sentence_scores_dict.get(idx, 0)
                                score_color = "🟢" if score > 0.7 else "🟡" if score > 0.4 else "🔴"
                                is_selected = any(sentence.strip() in s or s in sentence.strip() for s in summary_sentences)
                                selected_indicator = " ✅ SELECTED" if is_selected else ""
                                st.markdown(f"{score_color} **Sentence {idx+1}** (Score: {score:.3f}){selected_indicator}")
                                st.caption(f'"{sentence[:100]}..."' if len(sentence) > 100 else f'"{sentence}"')
                
                # Metrics
                if result.get('bertscore') is not None:
                    col1, col2, col3, col4, col5 = st.columns(5)
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    col5 = None
                
                with col1:
                    st.metric("ROUGE-1 F1", f"{result['rouge_scores']['rouge1']['f1']*100:.1f}%")
                
                with col2:
                    st.metric("ROUGE-2 F1", f"{result['rouge_scores']['rouge2']['f1']*100:.1f}%")
                
                with col3:
                    st.metric("ROUGE-L F1", f"{result['rouge_scores']['rougeL']['f1']*100:.1f}%")
                
                with col4:
                    st.metric("Compression", f"{result['compression_ratio']*100:.1f}%")
                
                if col5 and result.get('bertscore') is not None:
                    with col5:
                        st.metric("BERTScore F1", f"{result['bertscore']['f1']*100:.1f}%")
                
                # Detailed scores
                with st.expander("📊 Detailed Evaluation Scores", expanded=False):
                    if result.get('bertscore') is not None:
                        col1, col2, col3, col4 = st.columns(4)
                    else:
                        col1, col2, col3, col4 = st.columns(3)
                        col4 = None
                    
                    with col1:
                        st.write("**ROUGE-1**")
                        st.write(f"Precision: {result['rouge_scores']['rouge1']['precision']*100:.2f}%")
                        st.write(f"Recall: {result['rouge_scores']['rouge1']['recall']*100:.2f}%")
                        st.write(f"F1: {result['rouge_scores']['rouge1']['f1']*100:.2f}%")
                    
                    with col2:
                        st.write("**ROUGE-2**")
                        st.write(f"Precision: {result['rouge_scores']['rouge2']['precision']*100:.2f}%")
                        st.write(f"Recall: {result['rouge_scores']['rouge2']['recall']*100:.2f}%")
                        st.write(f"F1: {result['rouge_scores']['rouge2']['f1']*100:.2f}%")
                    
                    with col3:
                        st.write("**ROUGE-L**")
                        st.write(f"Precision: {result['rouge_scores']['rougeL']['precision']*100:.2f}%")
                        st.write(f"Recall: {result['rouge_scores']['rougeL']['recall']*100:.2f}%")
                        st.write(f"F1: {result['rouge_scores']['rougeL']['f1']*100:.2f}%")
                    
                    if col4 and result.get('bertscore') is not None:
                        with col4:
                            st.write("**BERTScore**")
                            st.write(f"Precision: {result['bertscore']['precision']*100:.2f}%")
                            st.write(f"Recall: {result['bertscore']['recall']*100:.2f}%")
                            st.write(f"F1: {result['bertscore']['f1']*100:.2f}%")
                    elif result.get('bertscore') is None and not BERTSCORE_AVAILABLE:
                        st.info("💡 Install bert-score to see BERTScore metrics: `pip install bert-score`")
                
                # Summary stats
                summary_length = len(word_tokenize(result['summary']))
                st.caption(f"Summary length: {summary_length} words | Compression ratio: {result['compression_ratio']*100:.1f}%")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; padding: 2rem;'>"
    "Built with ❤️ using Streamlit | Advanced Text Summarization with Multiple Techniques"
    "</div>",
    unsafe_allow_html=True
)

