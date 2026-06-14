# Information Retrieval System - Streamlit Application

## Course
Information Retrieval (AIML CZG537/DSECLZG537) S2-25 — Assignment 1

## Overview
An end-to-end Information Retrieval system built with Streamlit demonstrating:
- Text preprocessing (tokenization, stopwords, stemming, lemmatization)
- Inverted index construction
- Stemming vs Lemmatization comparison with cosine similarity
- Phrase query processing (Biword Index & Positional Index)
- Dictionary search (BST vs B-Tree) with performance benchmarking
- Tolerant retrieval (Wildcard, Edit Distance, K-gram, Soundex)

## Live Deployment

The application is deployed and accessible online:

🔗 **Live App:** [https://irassignmentgroup29.streamlit.app/](https://irassignmentgroup29.streamlit.app/)

- **Hosted on:** Streamlit Community Cloud
- **Source Repository:** [github.com/merriumjohn](https://github.com/merriumjohn)
- **Deployment Account:** merriumjohn@gmail.com

No installation required — simply open the link above to use the application.

---

## Local Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Installation & Running

### Step 1: Create Virtual Environment
```bash
python3 -m venv venv
```

### Step 2: Activate Virtual Environment
```bash
# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Usage Instructions

1. **Section A:** Click "Load Sample Documents" or upload your own .txt files
2. **Section B:** Select a document and view all preprocessing steps + inverted index
3. **Section C:** Enter queries and compare stemming vs lemmatization with similarity scores
4. **Section D:** Test phrase queries, view biword/positional indexes, see false positive analysis
5. **Section E:** Compare BST and B-Tree with multiple queries, view performance table
6. **Section F:** Test wildcard queries, spelling correction, edit distance matrix, k-gram, Soundex
7. **Section G:** Read inference and discussion for all experiments

## File Structure
```
Assignment/
├── app.py              # Main Streamlit application (complete IR system)
├── requirements.txt    # Python dependencies
├── README.md           # This file (setup instructions)
├── REPORT.md           # Assignment report with experimental results
└── sample_docs/        # Sample document collection (10 .txt files)
    ├── doc1.txt
    ├── doc2.txt
    ├── ...
    └── doc10.txt
```

After running the app (`streamlit run app.py`) , open browser → Navigate to http://localhost:8501


## Dependencies
- `streamlit` — Web application framework
- `nltk` — Natural Language Toolkit (tokenization, stemming, lemmatization, stopwords)

## Troubleshooting

**NLTK data not found:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

**Port 8501 in use:**
```bash
streamlit run app.py --server.port 8502
```

**Virtual environment not working:**
```bash
python3 -m pip install --user streamlit nltk
python3 -m streamlit run app.py
```
