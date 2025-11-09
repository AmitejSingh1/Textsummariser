# Advanced Text Summarizer Web App

A comprehensive text summarization web application that uses multiple techniques and models to generate summaries, with scoring and comparison capabilities.

## 🚀 Quick Start (Streamlit - Recommended)

The easiest way to use this app is with Streamlit, which provides a beautiful, modern UI out of the box.

### Installation

1. **Install dependencies:**
   
   **Option A: Full installation (includes abstractive models):**
   ```bash
   pip install -r requirements-streamlit.txt
   ```
   
   **Option B: Basic installation (extractive methods only - recommended if you have TensorFlow issues):**
   ```bash
   pip install -r requirements-streamlit-basic.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app_streamlit.py
   ```

3. **The app will automatically open in your browser!** 🎉

## Features

- **Multiple Summarization Techniques:**
  - **Extractive Methods:**
    - TF-IDF based summarization
    - TextRank algorithm
    - Latent Semantic Analysis (LSA)
  - **Abstractive Methods:**
    - BART (Facebook's BART-large-cnn)
    - T5 (Google's T5-small)

- **Scoring and Comparison:**
  - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
  - Compression ratio
  - Overall weighted score for comparison
  - Visual ranking of best summaries

- **Modern Streamlit Interface:**
  - Beautiful, responsive design
  - Interactive sidebar with settings
  - Tabbed results view
  - Real-time progress indicators
  - Detailed metrics and scores
  - Automatic model detection

## Streamlit App Features

- ✅ Clean, modern UI with gradients and animations
- ✅ Sidebar with adjustable settings
- ✅ Real-time processing with progress bars
- ✅ Tabbed interface for comparing all methods
- ✅ Detailed metrics and ROUGE scores
- ✅ Automatic best method highlighting
- ✅ Works seamlessly with or without abstractive models

## Usage

1. **Start the Streamlit app:**
   ```bash
   streamlit run app_streamlit.py
   ```

2. **Enter your text** in the main text area

3. **Adjust settings** in the sidebar (number of sentences for extractive methods)

4. **Click "✨ Summarize"** button

5. **View results** in the tabbed interface - all methods are sorted by score, with the best one highlighted

## Flask App (Alternative)

If you prefer the Flask version with a custom frontend:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or for basic installation:
   ```bash
   pip install -r requirements-basic.txt
   ```

2. **Run the Flask server:**
   ```bash
   python app.py
   ```

3. **Open your browser to:**
   ```
   http://localhost:5000
   ```

## Troubleshooting

### TensorFlow DLL Error (Windows)
If you encounter a TensorFlow DLL error when importing transformers:

1. **Use basic installation** (recommended for quick start):
   ```bash
   pip install -r requirements-streamlit-basic.txt
   ```
   The app will work perfectly with extractive methods only.

2. **Fix TensorFlow installation** (if you need abstractive models):
   - Install Microsoft Visual C++ Redistributable
   - Or use a different Python version (3.9 or 3.11 often work better)
   - Or install PyTorch separately: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Streamlit Issues

- **Port already in use:** Streamlit will automatically try the next available port
- **Models not loading:** Ensure you have a stable internet connection for the first run (models are downloaded from Hugging Face)
- **Memory errors:** Try reducing the input text length or using only extractive methods

## Scoring System

The overall score is calculated using:
- 40% ROUGE-1 F1 score
- 30% ROUGE-2 F1 score
- 20% ROUGE-L F1 score
- 10% Compression ratio (prefers ~30% compression)

## Requirements

- Python 3.8+
- Streamlit (for the recommended app)
- Flask (for the alternative Flask app)
- PyTorch (optional, for abstractive models)
- Transformers (optional, for abstractive models)
- NLTK
- scikit-learn
- networkx
- rouge-score

## Notes

- The transformer models (BART and T5) require significant memory. If you encounter memory issues, consider using a smaller model or running on a GPU.
- Extractive methods work well for longer texts, while abstractive methods can generate more natural summaries but may require more processing time.
- The app uses CPU by default. For better performance with transformer models, consider using a GPU if available.
- **Streamlit is recommended** for the best user experience and easiest setup.

## License

This project is open source and available for educational purposes.
