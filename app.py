from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app)

# Try to import transformers (optional - for abstractive models)
summarizer_bart = None
summarizer_t5 = None
transformers_available = False

try:
    from transformers import pipeline
    transformers_available = True
    print("Transformers library loaded successfully.")
except (ImportError, RuntimeError, Exception) as e:
    print(f"Warning: Transformers library not available. Abstractive models will be disabled.")
    print(f"Error: {type(e).__name__}: {e}")
    print("The app will still work with extractive summarization methods (TF-IDF, TextRank, LSA).")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize summarization models (only if transformers is available)
if transformers_available:
    print("Loading abstractive summarization models...")
    try:
        print("Loading BART model (this may take a few minutes on first run)...")
        summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        print("BART model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load BART model: {e}")
        summarizer_bart = None
    
    try:
        print("Loading T5 model (this may take a few minutes on first run)...")
        summarizer_t5 = pipeline("summarization", model="t5-small", device=-1)
        print("T5 model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load T5 model: {e}")
        summarizer_t5 = None
else:
    print("Skipping abstractive model loading (transformers not available).")

rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

class TextSummarizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def extractive_summarize_tfidf(self, text, num_sentences=3):
        """Extractive summarization using TF-IDF"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Clean sentences
        cleaned_sentences = [re.sub(r'[^a-zA-Z0-9\s]', '', s.lower()) for s in sentences]
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
        
        # Calculate sentence scores
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Get top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices)
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    
    def extractive_summarize_textrank(self, text, num_sentences=3):
        """Extractive summarization using TextRank algorithm"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Clean and tokenize sentences
        cleaned_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [self.stemmer.stem(word) for word in words if word.isalnum() and word not in self.stop_words]
            cleaned_sentences.append(' '.join(words))
        
        # Create similarity matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        sentence_vectors = vectorizer.fit_transform(cleaned_sentences)
        similarity_matrix = cosine_similarity(sentence_vectors)
        
        # Build graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Get top sentences
        ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)
        top_indices = sorted([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    
    def extractive_summarize_lsa(self, text, num_sentences=3):
        """Extractive summarization using Latent Semantic Analysis"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Apply SVD
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=min(num_sentences, len(sentences)-1))
        svd_matrix = svd.fit_transform(tfidf_matrix)
        
        # Calculate sentence importance
        sentence_scores = np.sum(svd_matrix ** 2, axis=1)
        
        # Get top sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        top_indices = sorted(top_indices)
        
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    
    def abstractive_summarize_bart(self, text, max_length=130, min_length=30):
        """Abstractive summarization using BART"""
        if summarizer_bart is None:
            return "BART model not available"
        
        try:
            # Truncate if too long
            max_input_length = 1024
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            summary = summarizer_bart(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            return f"Error in BART summarization: {str(e)}"
    
    def abstractive_summarize_t5(self, text, max_length=130, min_length=30):
        """Abstractive summarization using T5"""
        if summarizer_t5 is None:
            return "T5 model not available"
        
        try:
            # Truncate if too long
            max_input_length = 512
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            summary = summarizer_t5(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
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
    
    def calculate_summary_score(self, original, summary, rouge_scores):
        """Calculate overall summary score (weighted combination)"""
        compression = self.calculate_compression_ratio(original, summary)
        
        # Weighted score: 40% ROUGE-1 F1, 30% ROUGE-2 F1, 20% ROUGE-L F1, 10% compression ratio
        score = (
            0.4 * rouge_scores['rouge1']['f1'] +
            0.3 * rouge_scores['rouge2']['f1'] +
            0.2 * rouge_scores['rougeL']['f1'] +
            0.1 * (1 - abs(compression - 0.3))  # Prefer ~30% compression
        )
        return round(score * 100, 2)

summarizer = TextSummarizer()

@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        text = data.get('text', '')
        num_sentences = data.get('num_sentences', 3)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        results = {}
        
        # Extractive methods
        results['tfidf'] = {
            'summary': summarizer.extractive_summarize_tfidf(text, num_sentences),
            'type': 'extractive',
            'method': 'TF-IDF'
        }
        
        results['textrank'] = {
            'summary': summarizer.extractive_summarize_textrank(text, num_sentences),
            'type': 'extractive',
            'method': 'TextRank'
        }
        
        results['lsa'] = {
            'summary': summarizer.extractive_summarize_lsa(text, num_sentences),
            'type': 'extractive',
            'method': 'LSA'
        }
        
        # Abstractive methods (only if available)
        if summarizer_bart is not None:
            results['bart'] = {
                'summary': summarizer.abstractive_summarize_bart(text),
                'type': 'abstractive',
                'method': 'BART'
            }
        
        if summarizer_t5 is not None:
            results['t5'] = {
                'summary': summarizer.abstractive_summarize_t5(text),
                'type': 'abstractive',
                'method': 'T5'
            }
        
        # Calculate scores for each summary
        for key, result in results.items():
            rouge_scores = summarizer.calculate_rouge_scores(text, result['summary'])
            result['rouge_scores'] = rouge_scores
            result['compression_ratio'] = round(summarizer.calculate_compression_ratio(text, result['summary']), 3)
            result['overall_score'] = summarizer.calculate_summary_score(text, result['summary'], rouge_scores)
        
        # Sort by overall score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        
        return jsonify({
            'original_length': len(word_tokenize(text)),
            'results': {k: v for k, v in sorted_results}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'models_loaded': summarizer_bart is not None})

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)

