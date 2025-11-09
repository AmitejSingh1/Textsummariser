# Why Extractive Methods Need "Number of Sentences"

## How Extractive Summarization Works

Extractive summarization works by:
1. **Analyzing** all sentences in your text
2. **Scoring** each sentence based on importance/relevance
3. **Selecting** the top N most important sentences
4. **Combining** them into a summary

## Why We Need the Number Parameter

The "Number of sentences" parameter tells the algorithm:
- **How many sentences to select** from the original text
- **How long the summary should be**

### Example:

**Original Text (8 sentences):**
```
1. Artificial Intelligence (AI) has rapidly evolved...
2. From voice assistants like Siri and Alexa...
3. The surge in computational power...
4. However, this growth has also raised concerns...
5. Issues like data privacy...
6. Many experts argue that...
7. In the future, AI is expected to play...
8. But its success will largely depend...
```

**If you set "Number of sentences" to 3:**
- The algorithm scores all 8 sentences
- Selects the 3 most important ones
- Returns those 3 sentences as the summary

**If you set "Number of sentences" to 5:**
- Same process, but selects top 5 sentences
- Longer, more detailed summary

## Different Methods Use It Differently

### 1. TF-IDF Method
- Scores sentences based on important words (TF-IDF scores)
- Selects top N sentences with highest scores

### 2. TextRank Method  
- Uses graph-based ranking (like Google's PageRank)
- Scores sentences based on similarity to other sentences
- Selects top N sentences with highest importance scores

### 3. LSA Method
- Uses Latent Semantic Analysis to find key concepts
- Scores sentences based on semantic importance
- Selects top N sentences that capture main topics

## Why Not Abstractive Methods?

**Abstractive methods (BART, T5)** don't need this parameter because:
- They **generate new sentences** (don't extract existing ones)
- They use **word/token limits** instead (e.g., "max 130 words")
- They create summaries that may not contain any original sentences

## Real-World Example

**Original Text:** A 500-word article (20 sentences)

**Setting "Number of sentences" to 3:**
- Gets 3 key sentences (~75-100 words)
- Very concise summary
- Good for quick overview

**Setting "Number of sentences" to 7:**
- Gets 7 sentences (~200-250 words)  
- More detailed summary
- Better for comprehensive understanding

## Best Practices

- **Short texts (5-10 sentences):** Use 2-3 sentences
- **Medium texts (10-30 sentences):** Use 3-5 sentences
- **Long texts (30+ sentences):** Use 5-10 sentences
- **Very long texts (100+ sentences):** Use 10-15% of original sentences

## In the Code

Looking at the TF-IDF method:

```python
def extractive_summarize_tfidf(self, text, num_sentences=3):
    sentences = sent_tokenize(text)  # Split into sentences
    # ... score all sentences ...
    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]  # Get top N
    summary = ' '.join([sentences[i] for i in top_indices])  # Select those N sentences
    return summary
```

The `num_sentences` parameter directly controls how many sentences are selected!

## Summary

**Without the "Number of sentences" parameter:**
- The algorithm wouldn't know when to stop
- It might select too many or too few sentences
- You'd have no control over summary length

**With the parameter:**
- You control summary length precisely
- You can adjust based on your needs
- The algorithm knows exactly how many sentences to return

This is why extractive methods need this parameter - it's essential for their operation!

