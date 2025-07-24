# 📜 Bible Verse Vault Builder

A powerful semantic search system for KJV Bible verses using AI embeddings and ChromaDB.

## 🌟 Features
- **Semantic Search**: Find verses by meaning, not just keywords
- **31,000+ Verses**: Complete KJV Bible coverage
- **AI-Powered**: Uses sentence transformers for intelligent matching
- **Fast & Efficient**: GPU support with optimized batch processing

## 🚀 Quick Start
1. Install requirements: `pip install pandas sentence-transformers chromadb torch`
2. Add your `KJV_Verses.csv` file to the directory
3. Run: `python bible_verse_vault.py`
4. Start searching! 🔍

## 📖 How It Works
Open `visual_guide.html` in your browser for a beginner-friendly explanation!

## 🔍 Example Usage
```python
# Search for verses about forgiveness
results = semantic_search(collection, "What does the Bible say about forgiveness?", n_results=5)
display_search_results(results)
