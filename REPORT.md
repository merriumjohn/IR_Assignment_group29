# Information Retrieval System - Assignment Report

**Course:** Information Retrieval (AIML CZG537/DSECLZG537) S2-25  
**Assignment:** Assignment 1  
**Topic:** End-to-End IR System using Streamlit  

---

## 1. Introduction

This report presents the design and implementation of an end-to-end Information Retrieval (IR) system built using Streamlit. The system provides an interactive front-end where users can upload documents, enter queries, select different retrieval techniques, and observe outputs for preprocessing, indexing, querying, and tolerant retrieval.

**Technology Stack:**
- Python 3.12
- Streamlit (Web UI framework)
- NLTK (Natural Language Processing)
- Pandas (Data display)

---

## 2. Dataset Description

The system uses a curated collection of **10 text documents** focused on Information Retrieval and related Computer Science topics:

| Document | Topic | Words |
|----------|-------|-------|
| doc01.txt | Information Retrieval fundamentals | ~65 |
| doc02.txt | Natural Language Processing | ~55 |
| doc03.txt | Machine Learning & Deep Learning | ~50 |
| doc04.txt | Data Mining | ~50 |
| doc05.txt | Vector Space Model | ~50 |
| doc06.txt | Boolean Retrieval Model | ~55 |
| doc07.txt | Text Preprocessing | ~50 |
| doc08.txt | Inverted Index | ~55 |
| doc09.txt | Tolerant Retrieval | ~50 |
| doc10.txt | Index Compression | ~50 |

The dataset was chosen to demonstrate diverse IR concepts while maintaining vocabulary overlap for meaningful retrieval experiments.

---

## 3. Implementation Details

### 3.1 Streamlit-Based End-to-End Workflow (Section A)

The application provides:
- **File upload** interface supporting .txt files
- **Sample document loading** with one click
- **Document viewer** with expandable sections showing content and word count
- **Sidebar navigation** for easy section switching
- **Status indicator** showing document load status

All interactions happen through the Streamlit frontend — no backend code or static outputs.

### 3.2 Text Preprocessing (Section B)

The preprocessing pipeline applies the following steps sequentially:

1. **Tokenization** — Using NLTK's `word_tokenize()` to split text into tokens
2. **Lowercasing** — Converting all tokens to lowercase for normalization
3. **Stop Word Removal** — Removing English stopwords using NLTK's stopword corpus (179 words)
4. **Hyphen Handling** — Splitting hyphenated terms (e.g., "text-mining" → "text", "mining", "textmining")
5. **Stemming** — Porter Stemmer for aggressive suffix removal
6. **Lemmatization** — WordNet Lemmatizer for dictionary-based normalization

**Inverted Index:** Built across the entire collection, mapping each term to its posting list (set of document IDs).

### 3.3 Stemming vs Lemmatization (Section C)

**Comparison Method:** Cosine similarity between query vector and document vectors, computed separately using stemmed and lemmatized forms.

**Experimental Setup:**
- Query: "information retrieval systems"
- Stemmed: ['inform', 'retriev', 'system']
- Lemmatized: ['information', 'retrieval', 'system']

**Key Finding:** Stemming produces higher recall (more documents matched) because it groups more word variants. Lemmatization produces more precise matches but may miss valid variations.

### 3.4 Phrase Query Processing (Section D)

**Biword Index:**
- Maps consecutive word pairs to documents
- Fast lookup, minimal storage
- Limitation: False positives for 3+ word phrases

**Positional Index:**
- Maps each term to (document, position) pairs
- Verifies exact consecutive positions for phrase queries
- Guarantees accurate results

**False Positive Demonstration:** For "information retrieval system":
- Biword checks: ("information retrieval" ∈ doc) AND ("retrieval system" ∈ doc) — independently
- A document with these biwords at different positions is a false positive
- Positional index verifies positions p, p+1, p+2 — no false positives possible

### 3.5 BST vs B-Tree (Section E)

**Implementation:**
- BST: Standard unbalanced binary search tree
- B-Tree: Order t=3 (max 5 keys per node, min 2)

**Metrics Compared:**
- Number of comparisons per search
- Average search time (over 100+ iterations)
- Tree height
- Build time

**Result:** B-Tree consistently requires fewer comparisons due to:
- Lower height (higher branching factor)
- Guaranteed balance (all leaves at same level)

### 3.6 Tolerant Retrieval (Section F)

Five techniques implemented:

1. **Wildcard Queries** — Pattern matching using k-gram index + regex filtering
2. **Spelling Correction** — Edit distance-based candidate generation
3. **Edit Distance** — Full Levenshtein DP matrix visualization
4. **K-gram Index** — Bigram index for efficient wildcard expansion
5. **Phonetic Correction** — Soundex algorithm for sound-alike matching

---

## 4. Experimental Results

### 4.1 Preprocessing Impact

| Step | Tokens Before | Tokens After | Reduction |
|------|:---:|:---:|:---:|
| Tokenization | (raw text) | ~65 | — |
| Lowercasing | 65 | 65 | 0% |
| Stopword Removal | 65 | ~30 | ~54% |
| Stemming | 30 | 30 | Normalized forms |
| Lemmatization | 30 | 30 | Normalized forms |

### 4.2 Stemming vs Lemmatization

| Metric | Stemming | Lemmatization |
|--------|:---:|:---:|
| Documents Retrieved | Higher | Lower |
| Avg Cosine Similarity | Higher | Lower |
| Technique | Aggressive | Conservative |
| Best For | Recall | Precision |

### 4.3 Phrase Query Comparison

| Query | Biword Results | Positional Results | False Positives |
|-------|:---:|:---:|:---:|
| "information retrieval" | 3 docs | 3 docs | 0 |
| "machine learning" | 2 docs | 2 docs | 0 |
| "information retrieval system" | Varies | Varies | Possible |
| "document retrieval systems" | Varies | Varies | Possible |

### 4.4 BST vs B-Tree Performance

| Metric | BST | B-Tree |
|--------|:---:|:---:|
| Tree Height | ~10-15 | ~2-3 |
| Avg Comparisons | ~7-10 | ~4-6 |
| Build Time | Faster | Slightly slower |
| Search Worst Case | O(n) | O(log_t n) |
| Balance | Not guaranteed | Always balanced |

### 4.5 Tolerant Retrieval Results

| Technique | Example Input | Output | Effectiveness |
|-----------|:---:|:---:|:---:|
| Wildcard | inform* | information | ✓ High |
| Edit Distance | informaton | information (dist=1) | ✓ High |
| K-gram | *tion | collection, information, ... | ✓ High |
| Soundex | informasion | information (I516) | ✓ Medium |

---

## 5. Inferences and Discussion

### 5.1 Best Preprocessing Technique
Stop word removal had the highest impact on retrieval quality by eliminating ~54% of tokens that add noise without contributing to relevance.

### 5.2 Stemming vs Lemmatization Verdict
**Stemming is more suitable** for this dataset because:
- The IR-domain documents use many morphological variants of the same concepts
- Recall is more important than precision in exploratory search
- The aggressive normalization captures relevant documents that lemmatization misses

### 5.3 More Accurate Phrase Index
**Positional Index** is definitively more accurate — it guarantees zero false positives by verifying exact term positions.

### 5.4 Faster Tree Structure
**B-Tree** is faster for dictionary lookup with fewer comparisons on average, guaranteed O(log_t n) search, and is the industry standard for database indexes.

### 5.5 Tolerant Retrieval Assessment
The system handles multiple types of query imperfections effectively. The combination of wildcard, edit distance, k-gram, and Soundex provides robust coverage.

### 5.6 Limitations
- Small corpus limits statistical significance
- Memory-based (not scalable to millions of docs)
- English-only; no ranked retrieval
- BST is not self-balancing

### 5.7 Improvements
- Add BM25 ranking, query expansion, relevance feedback
- Support multiple file formats (PDF, DOCX)
- Implement disk-based indexing with compression
- Add semantic search using word embeddings

---

## 6. How to Run

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

## 7. Screenshots and Recording

Section A: Documents uploaded and visible
![Section A: Documents uploaded and visible](image-16.png)
Section B: Preprocessing steps output
![Section B: Preprocessing steps output](image.png)
Section B: Preprocessing steps output
![Section B: Preprocessing steps output1](image-1.png)
Section B: Preprocessing steps output
![Section B: Preprocessing steps output](image-2.png)
Section C: Stemming vs Lemmatization comparison table
![Section C: Stemming vs Lemmatization comparison table](image-4.png)
Section C: Cosine similarity scores
![Section C: Cosine similarity scores](image-3.png)
Section D: Biword index index representation
![Section D: Biword index index representation](image-6.png)
Section D: Positional index representation
![Section D: Positional index representation](image-7.png)
Section D: Phrase query results comparison
![Section D: Phrase query results comparison](image-5.png)
Section E: BST vs B-Tree experimental results table
![Section E: BST vs B-Tree experimental results table](image-9.png)
Section E: Summary statistics
![Section E: Summary statistics](image-8.png)
Section F: Wildcard query results
![Section F: Wildcard query results](image-10.png)
Section F: Spelling correction results
![Section F: Spelling correction results](image-11.png)
ection F: Edit distance matrix
![Section F: Edit distance matrix](image-12.png)
Section F: K-gram index display
![Section F: K-gram index display](image-13.png)
Section F: Soundex phonetic matching
![Section F: Soundex phonetic matching](image-14.png)
Section G: Inference and discussion page
![Section G: Inference and discussion page](image-15.png)

[Screen Recording](https://wilpbitspilaniacin0-my.sharepoint.com/:v:/g/personal/2024dc04258_wilp_bits-pilani_ac_in/IQBBR4e3rkMiRZ6r7q_p-xkyASutfeUxh1kcKSpLAONCmhQ?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=94lFBV) 

---

## 8. Conclusion

This assignment successfully implements a complete end-to-end Information Retrieval system with an interactive Streamlit interface. The system demonstrates all major IR concepts including preprocessing, indexing (inverted, biword, positional), tree-based dictionary search (BST, B-Tree), and tolerant retrieval (wildcard, edit distance, k-gram, Soundex). All experimental results are displayed interactively on the frontend with supporting inferences.
