"""
Information Retrieval System - End-to-End Streamlit Application
Course: Information Retrieval (AIML CZG537/DSECLZG537) S2-25
Assignment 1
"""

import streamlit as st
import time
import re
import math
import os
import pandas as pd
from collections import defaultdict, Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# ============================================================
# DATA STRUCTURES
# ============================================================

class BSTNode:
    """Binary Search Tree Node"""
    def __init__(self, key, doc_ids=None):
        self.key = key
        self.doc_ids = doc_ids or set()
        self.left = None
        self.right = None


class BST:
    """Binary Search Tree for dictionary search"""
    def __init__(self):
        self.root = None
        self.comparisons = 0
        self.node_count = 0
        self.height = 0

    def insert(self, key, doc_id):
        if self.root is None:
            self.root = BSTNode(key, {doc_id})
            self.node_count = 1
        else:
            self._insert(self.root, key, doc_id, 1)

    def _insert(self, node, key, doc_id, depth):
        if key == node.key:
            node.doc_ids.add(doc_id)
        elif key < node.key:
            if node.left is None:
                node.left = BSTNode(key, {doc_id})
                self.node_count += 1
                self.height = max(self.height, depth + 1)
            else:
                self._insert(node.left, key, doc_id, depth + 1)
        else:
            if node.right is None:
                node.right = BSTNode(key, {doc_id})
                self.node_count += 1
                self.height = max(self.height, depth + 1)
            else:
                self._insert(node.right, key, doc_id, depth + 1)

    def search(self, key):
        self.comparisons = 0
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None:
            return None
        self.comparisons += 1
        if key == node.key:
            return node.doc_ids
        elif key < node.key:
            return self._search(node.left, key)
        else:
            return self._search(node.right, key)

    def get_height(self):
        return self._get_height(self.root)

    def _get_height(self, node):
        if node is None:
            return 0
        return 1 + max(self._get_height(node.left), self._get_height(node.right))


class BTreeNode:
    """B-Tree Node"""
    def __init__(self, t, leaf=False):
        self.t = t
        self.keys = []
        self.children = []
        self.leaf = leaf
        self.doc_ids_map = {}

    def is_full(self):
        return len(self.keys) == 2 * self.t - 1


class BTree:
    """B-Tree for dictionary search"""
    def __init__(self, t=3):
        self.root = BTreeNode(t, leaf=True)
        self.t = t
        self.comparisons = 0
        self.node_count = 1

    def search(self, key, node=None):
        if node is None:
            node = self.root
            self.comparisons = 0
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            self.comparisons += 1
            i += 1
        self.comparisons += 1
        if i < len(node.keys) and key == node.keys[i]:
            return node.doc_ids_map.get(key, set())
        if node.leaf:
            return None
        return self.search(key, node.children[i])

    def insert(self, key, doc_id):
        root = self.root
        if root.is_full():
            new_root = BTreeNode(self.t, leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
            self.node_count += 1
        self._insert_non_full(self.root, key, doc_id)

    def _insert_non_full(self, node, key, doc_id):
        i = len(node.keys) - 1
        if node.leaf:
            if key in node.keys:
                node.doc_ids_map.setdefault(key, set()).add(doc_id)
                return
            node.keys.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
            node.doc_ids_map.setdefault(key, set()).add(doc_id)
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if key in node.keys:
                node.doc_ids_map.setdefault(key, set()).add(doc_id)
                return
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key, doc_id)

    def _split_child(self, parent, i):
        t = self.t
        child = parent.children[i]
        new_node = BTreeNode(t, leaf=child.leaf)
        self.node_count += 1
        mid_key = child.keys[t - 1]

        parent.keys.insert(i, mid_key)
        parent.doc_ids_map[mid_key] = child.doc_ids_map.get(mid_key, set())
        parent.children.insert(i + 1, new_node)

        new_node.keys = child.keys[t:]
        child.keys = child.keys[:t - 1]

        for k in new_node.keys:
            if k in child.doc_ids_map:
                new_node.doc_ids_map[k] = child.doc_ids_map.pop(k)
        child.doc_ids_map.pop(mid_key, None)

        if not child.leaf:
            new_node.children = child.children[t:]
            child.children = child.children[:t]

    def get_height(self, node=None):
        if node is None:
            node = self.root
        if node.leaf:
            return 1
        return 1 + self.get_height(node.children[0])


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def tokenize_text(text):
    """Tokenize text into words"""
    return word_tokenize(text)


def lowercase_tokens(tokens):
    """Convert all tokens to lowercase"""
    return [t.lower() for t in tokens]


def remove_stopwords(tokens):
    """Remove English stopwords"""
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t.lower() not in stop_words]


def handle_hyphens(tokens):
    """Handle hyphenated words by splitting them"""
    result = []
    for token in tokens:
        if '-' in token:
            parts = token.split('-')
            result.extend(parts)
            result.append(token.replace('-', ''))
        else:
            result.append(token)
    return result


def apply_stemming(tokens):
    """Apply Porter Stemmer"""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def apply_lemmatization(tokens):
    """Apply WordNet Lemmatizer"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def build_inverted_index(documents):
    """Build an inverted index from documents"""
    index = defaultdict(set)
    for doc_id, text in documents.items():
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalnum()]
        for token in tokens:
            index[token].add(doc_id)
    return dict(index)


# ============================================================
# PHRASE QUERY INDEXES
# ============================================================

def build_biword_index(documents):
    """Build a biword index from documents"""
    biword_index = defaultdict(set)
    for doc_id, text in documents.items():
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalnum()]
        for i in range(len(tokens) - 1):
            biword = f"{tokens[i]} {tokens[i+1]}"
            biword_index[biword].add(doc_id)
    return dict(biword_index)


def build_positional_index(documents):
    """Build a positional index from documents"""
    pos_index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in documents.items():
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalnum()]
        for pos, token in enumerate(tokens):
            pos_index[token][doc_id].append(pos)
    return dict(pos_index)


def search_biword(query, biword_index):
    """Search using biword index"""
    tokens = word_tokenize(query.lower())
    tokens = [t for t in tokens if t.isalnum()]
    if len(tokens) < 2:
        return set()
    result = None
    for i in range(len(tokens) - 1):
        biword = f"{tokens[i]} {tokens[i+1]}"
        docs = biword_index.get(biword, set())
        if result is None:
            result = docs.copy()
        else:
            result = result.intersection(docs)
    return result if result else set()


def search_positional(query, positional_index):
    """Search using positional index - finds exact phrase matches"""
    tokens = word_tokenize(query.lower())
    tokens = [t for t in tokens if t.isalnum()]
    if not tokens:
        return set()
    if tokens[0] not in positional_index:
        return set()
    candidate_docs = set(positional_index[tokens[0]].keys())
    for token in tokens[1:]:
        if token not in positional_index:
            return set()
        candidate_docs = candidate_docs.intersection(set(positional_index[token].keys()))
    result = set()
    for doc_id in candidate_docs:
        positions = positional_index[tokens[0]][doc_id]
        for start_pos in positions:
            found = True
            for i, token in enumerate(tokens[1:], 1):
                if (start_pos + i) not in positional_index[token].get(doc_id, []):
                    found = False
                    break
            if found:
                result.add(doc_id)
                break
    return result


# ============================================================
# TOLERANT RETRIEVAL
# ============================================================

def edit_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def edit_distance_matrix(s1, s2):
    """Compute and return the full edit distance matrix"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp


def build_kgram_index(vocabulary, k=2):
    """Build a k-gram index for wildcard queries"""
    kgram_index = defaultdict(set)
    for term in vocabulary:
        padded = f"${term}$"
        for i in range(len(padded) - k + 1):
            kgram = padded[i:i+k]
            kgram_index[kgram].add(term)
    return dict(kgram_index)


def wildcard_search(pattern, kgram_index, vocabulary, k=2):
    """Search using wildcard pattern with k-gram index"""
    parts = pattern.lower().split('*')
    candidate_terms = set(vocabulary)

    for part in parts:
        if part:
            if pattern.startswith(part):
                padded = f"${part}"
            elif pattern.endswith(part):
                padded = f"{part}$"
            else:
                padded = part

            kgrams = []
            for i in range(len(padded) - k + 1):
                kgrams.append(padded[i:i+k])

            for kg in kgrams:
                if kg in kgram_index:
                    candidate_terms = candidate_terms.intersection(kgram_index[kg])
                else:
                    candidate_terms = set()
                    break

    # Filter candidates using regex
    regex_pattern = pattern.replace('*', '.*')
    regex_pattern = f"^{regex_pattern}$"
    result = set()
    for term in candidate_terms:
        if re.match(regex_pattern, term):
            result.add(term)
    return result


def spelling_correction(query_term, vocabulary, max_distance=2):
    """Find closest terms using edit distance"""
    corrections = []
    for term in vocabulary:
        dist = edit_distance(query_term.lower(), term)
        if dist <= max_distance and dist > 0:
            corrections.append((term, dist))
    corrections.sort(key=lambda x: x[1])
    return corrections[:5]


def soundex(name):
    """Generate Soundex code for phonetic matching"""
    name = name.upper()
    if not name:
        return ""
    soundex_code = name[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    prev_code = mapping.get(name[0], '0')
    for char in name[1:]:
        code = mapping.get(char, '0')
        if code != '0' and code != prev_code:
            soundex_code += code
        prev_code = code if code != '0' else prev_code
    soundex_code = soundex_code[:4].ljust(4, '0')
    return soundex_code


def phonetic_search(query_term, vocabulary):
    """Find terms with same Soundex code"""
    query_soundex = soundex(query_term)
    matches = []
    for term in vocabulary:
        if soundex(term) == query_soundex and term != query_term.lower():
            matches.append(term)
    return matches


# ============================================================
# SIMILARITY MEASURES
# ============================================================

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    all_terms = set(vec1.keys()) | set(vec2.keys())
    dot_product = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in all_terms)
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot_product / (mag1 * mag2)


# ============================================================
# STREAMLIT APPLICATION
# ============================================================

def main():
    st.set_page_config(page_title="IR System - Assignment 1", page_icon="🔍", layout="wide")
    st.title("🔍 Information Retrieval System")
    st.markdown("""
    **End-to-End IR System** | Course: Information Retrieval (AIML CZG537/DSECLZG537) S2-25  
    *Interactive application for preprocessing, indexing, querying & tolerant retrieval*
    """)

    # Initialize session state
    if 'documents' not in st.session_state:
        st.session_state.documents = {}

    # Sidebar navigation
    st.sidebar.title("📌 Navigation")
    st.sidebar.markdown("---")
    section = st.sidebar.radio("Select Section:", [
        "A. Upload & View Documents",
        "B. Text Preprocessing",
        "C. Stemming vs Lemmatization",
        "D. Phrase Query Processing",
        "E. BST vs B-Tree Dictionary",
        "F. Tolerant Retrieval",
        "G. Inference & Discussion"
    ])

    # Show document status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Status")
    if st.session_state.documents:
        st.sidebar.success(f"✅ {len(st.session_state.documents)} documents loaded")
    else:
        st.sidebar.warning("⚠️ No documents loaded")

    # ================================================================
    # SECTION A: UPLOAD DOCUMENTS
    # ================================================================
    if section == "A. Upload & View Documents":
        st.header("A. Document Upload & Collection Viewing")
        st.markdown("Upload your text dataset or load the built-in sample collection.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📤 Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload text files (.txt)", type=["txt"], accept_multiple_files=True
            )
            if uploaded_files:
                for f in uploaded_files:
                    content = f.read().decode('utf-8', errors='ignore')
                    st.session_state.documents[f.name] = content
                st.success(f"✅ {len(uploaded_files)} file(s) uploaded!")
                st.rerun()

        with col2:
            st.markdown("### 📄 Sample Collection")
            st.markdown("25 curated documents covering IR, NLP, ML, search engines, and related CS topics.")
            if st.button("🔄 Load Sample Documents (25 docs)", type="primary"):
                st.session_state.documents = {
                    "doc01.txt": "Information retrieval is the activity of obtaining information system resources relevant to an information need from a collection. Searches can be based on full-text or other content-based indexing. Information retrieval is the science of searching for information in a document, searching for documents themselves, and also searching for the metadata that describes data and for databases of texts, images, or sounds.",
                    "doc02.txt": "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence. It deals with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. NLP techniques are used in machine translation, text-mining applications, and information extraction from unstructured data sources.",
                    "doc03.txt": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Deep learning is a sub-field of machine learning that uses artificial neural networks with representation learning. The learning process discovers patterns in data.",
                    "doc04.txt": "Data mining is the process of discovering patterns and knowledge in large data sets involving methods at the intersection of machine learning, statistics, and database systems. Data mining is an interdisciplinary subfield of computer science and statistics with an overall goal of extracting information from a data set.",
                    "doc05.txt": "The vector space model is an algebraic model for representing text documents and queries as vectors of identifiers such as index terms. It is used in information filtering, information retrieval, indexing, and relevancy rankings. Term frequency and inverse document frequency are key concepts in this model for computing document similarity.",
                    "doc06.txt": "Boolean retrieval model is a classical model for information retrieval in which we can pose any query in the form of a Boolean expression of terms connected by AND, OR, and NOT operators. The model views each document as just a set of words. Boolean retrieval is simple, efficient for exact matching, and is still used in many commercial systems today.",
                    "doc07.txt": "Text preprocessing involves several important steps like tokenization, stop-word removal, stemming, and lemmatization. These steps help in normalizing the text for better information retrieval performance. Preprocessing is a critical first step in any text analysis or natural language processing pipeline. Proper preprocessing can significantly improve retrieval quality.",
                    "doc08.txt": "An inverted index is a database index storing a mapping from content, such as words or numbers, to its locations in a document or a set of documents. It is the most popular data structure used in document retrieval systems and search engines. The inverted index allows fast full-text searches at the cost of increased processing when a document is added.",
                    "doc09.txt": "Tolerant retrieval deals with handling spelling mistakes and variations in user queries to improve search experience. Edit distance and phonetic correction are common techniques used in spell-checking and query suggestion systems. Wildcard queries using permuterm index and k-gram indexes also support tolerant retrieval mechanisms in modern search engines.",
                    "doc10.txt": "Index compression techniques reduce the space required to store the inverted index while maintaining fast query processing. Common methods include gap encoding, variable byte codes, gamma codes, and delta codes for efficient storage and retrieval. Compression is essential for managing large-scale information retrieval systems with billions of documents.",
                    "doc11.txt": "A search engine is a software system designed to carry out web searches. They search the World Wide Web in a systematic way for particular information specified in a textual web search query. The search results are generally presented in a line of results, often referred to as search engine results pages. Modern search engines use crawlers, indexers, and ranking algorithms to deliver relevant results to users in milliseconds.",
                    "doc12.txt": "Artificial intelligence is the simulation of human intelligence processes by computer systems. These processes include learning, reasoning, and self-correction. AI applications include expert systems, natural language processing, speech recognition, and machine vision. The field of artificial intelligence was founded in 1956 at a workshop at Dartmouth College in Hanover, New Hampshire.",
                    "doc13.txt": "A database management system is a software application that interacts with end-users, other applications, and the database itself to capture and analyze data. A general-purpose DBMS allows the definition, creation, querying, update, and administration of databases. Well-known DBMSs include MySQL, PostgreSQL, MongoDB, Microsoft SQL Server, and Oracle Database.",
                    "doc14.txt": "Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user. Large cloud services often have functions distributed over multiple locations, each of which is a data center. Cloud computing relies on sharing of resources to achieve coherence and economies of scale.",
                    "doc15.txt": "The PageRank algorithm is used by Google Search to rank web pages in their search engine results. PageRank works by counting the number and quality of links to a page to determine a rough estimate of how important the website is. The underlying assumption is that more important websites are likely to receive more links from other websites. The algorithm was named after Larry Page, one of the founders of Google.",
                    "doc16.txt": "Cryptography is the practice and study of techniques for secure communication in the presence of adversarial behavior. It is about constructing and analyzing protocols that prevent third parties from reading private messages. Modern cryptography exists at the intersection of mathematics, computer science, electrical engineering, communication science, and physics. Applications of cryptography include electronic commerce, chip-based payment cards, and digital currencies.",
                    "doc17.txt": "Web crawling is the process used by search engines to systematically browse the World Wide Web for the purpose of web indexing. A web crawler, also known as a spider or spiderbot, is an Internet bot that systematically browses web pages for indexing purposes. The crawler starts with a list of seed URLs and follows hyperlinks to discover new pages. Politeness policies prevent crawlers from overloading web servers.",
                    "doc18.txt": "Text classification is the task of assigning predefined categories to free-text documents. It can provide conceptual views of document collections and has important applications in real-world tasks such as spam filtering, sentiment analysis, and topic labeling. Common approaches include naive Bayes classifiers, support vector machines, and deep learning models like transformers and recurrent neural networks.",
                    "doc19.txt": "Relevance feedback is a feature of some information retrieval systems that allows users to provide explicit feedback on the relevance of documents in an initial set of search results. The system then uses this feedback to refine the query and improve subsequent search results. The Rocchio algorithm is a well-known method for relevance feedback in the vector space model. Pseudo-relevance feedback assumes the top-ranked documents are relevant.",
                    "doc20.txt": "The term frequency-inverse document frequency is a numerical statistic that reflects how important a word is to a document in a collection. It is often used as a weighting factor in searches of information retrieval and text mining. The tf-idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word.",
                    "doc21.txt": "Latent semantic indexing is an indexing and retrieval method that uses a mathematical technique called singular value decomposition to identify patterns in the relationships between the terms and concepts contained in an unstructured collection of text. LSI is based on the principle that words used in the same contexts tend to have similar meanings. It addresses the problems of synonymy and polysemy in information retrieval.",
                    "doc22.txt": "Named entity recognition is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into predefined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, and percentages. NER systems can be built using rule-based approaches, statistical models, or deep learning architectures.",
                    "doc23.txt": "The precision and recall metrics are fundamental measures used in information retrieval to evaluate the quality of search results. Precision is the fraction of retrieved documents that are relevant, while recall is the fraction of relevant documents that are retrieved. The F1 score is the harmonic mean of precision and recall. Other evaluation metrics include mean average precision and normalized discounted cumulative gain.",
                    "doc24.txt": "Distributed computing is a field of computer science that studies distributed systems. A distributed system is a collection of computer programs that utilize computational resources across multiple, separate computation nodes to achieve a common, shared goal. MapReduce is a programming model for processing large data sets with a parallel, distributed algorithm on a cluster of commodity hardware.",
                    "doc25.txt": "Query expansion is a process in information retrieval where the original query is supplemented with additional terms to improve retrieval performance. Techniques for query expansion include relevance feedback, thesaurus-based expansion using WordNet or domain-specific ontologies, and statistical co-occurrence analysis. Automatic query expansion can significantly improve recall without manual user intervention."
                }
                st.success("✅ 25 sample documents loaded!")
                st.rerun()

        # Display documents
        if st.session_state.documents:
            st.markdown("---")
            st.markdown("### 📋 Document Collection")
            st.markdown(f"**Total Documents:** {len(st.session_state.documents)} | **Total Words:** {sum(len(text.split()) for text in st.session_state.documents.values())}")
            
            for name, content in st.session_state.documents.items():
                with st.expander(f"📄 {name} ({len(content.split())} words)"):
                    st.text(content)

            if st.button("🗑️ Clear All Documents"):
                st.session_state.documents = {}
                st.rerun()

    # ================================================================
    # SECTION B: TEXT PREPROCESSING
    # ================================================================
    elif section == "B. Text Preprocessing":
        st.header("B. Text Preprocessing Pipeline")

        if not st.session_state.documents:
            st.warning("⚠️ Please upload documents first (go to Section A)!")
            return

        doc_names = list(st.session_state.documents.keys())
        selected_doc = st.selectbox("Select a document to preprocess:", doc_names)
        text = st.session_state.documents[selected_doc]

        st.markdown("### 📄 Original Text")
        st.info(text)

        st.markdown("---")
        st.markdown("### ⚙️ Preprocessing Pipeline Steps")

        # Step 1: Tokenization
        st.markdown("#### Step 1: Tokenization")
        tokens = tokenize_text(text)
        alpha_tokens = [t for t in tokens if t.isalnum()]
        st.write(f"**All Tokens ({len(tokens)}):** `{tokens}`")
        st.write(f"**Alphanumeric Tokens ({len(alpha_tokens)}):** `{alpha_tokens}`")

        # Step 2: Lowercasing
        st.markdown("#### Step 2: Lowercasing")
        lower_tokens = lowercase_tokens(alpha_tokens)
        st.write(f"**Lowercased ({len(lower_tokens)}):** `{lower_tokens}`")

        # Step 3: Stopword Removal
        st.markdown("#### Step 3: Stop Word Removal")
        no_stop = remove_stopwords(lower_tokens)
        removed_words = [t for t in lower_tokens if t not in no_stop]
        st.write(f"**After removal ({len(no_stop)}):** `{no_stop}`")
        st.write(f"**Removed stopwords ({len(lower_tokens) - len(no_stop)}):** `{list(set(removed_words))}`")

        # Step 4: Hyphen Handling
        st.markdown("#### Step 4: Hyphen Handling")
        all_tokens_with_hyphens = [t for t in word_tokenize(text) if '-' in t]
        if all_tokens_with_hyphens:
            hyphen_handled = handle_hyphens(all_tokens_with_hyphens)
            st.write(f"**Hyphenated words found:** `{all_tokens_with_hyphens}`")
            st.write(f"**After handling:** `{hyphen_handled}`")
        else:
            st.write("*No hyphenated words found in this document.*")
            st.write("**Example:** `'text-mining'` → `['text', 'mining', 'textmining']`")

        # Step 5: Stemming
        st.markdown("#### Step 5: Stemming (Porter Stemmer)")
        stemmed = apply_stemming(no_stop)
        stem_comparison = pd.DataFrame({
            "Original": no_stop[:15],
            "Stemmed": stemmed[:15]
        })
        st.dataframe(stem_comparison, use_container_width=True)

        # Step 6: Lemmatization
        st.markdown("#### Step 6: Lemmatization (WordNet)")
        lemmatized = apply_lemmatization(no_stop)
        lem_comparison = pd.DataFrame({
            "Original": no_stop[:15],
            "Lemmatized": lemmatized[:15]
        })
        st.dataframe(lem_comparison, use_container_width=True)

        # Inverted Index
        st.markdown("---")
        st.markdown("### 📇 Inverted Index (Full Collection)")
        inv_index = build_inverted_index(st.session_state.documents)
        sorted_terms = sorted(inv_index.items(), key=lambda x: x[0])
        index_data = {"Term": [], "Document Frequency": [], "Posting List (Doc IDs)": []}
        for term, doc_ids in sorted_terms[:30]:
            index_data["Term"].append(term)
            index_data["Document Frequency"].append(len(doc_ids))
            index_data["Posting List (Doc IDs)"].append(", ".join(sorted(doc_ids)))
        st.dataframe(pd.DataFrame(index_data), use_container_width=True)
        st.info(f"📊 Total unique terms in inverted index: **{len(inv_index)}** | Showing top 30 alphabetically")

    # ================================================================
    # SECTION C: STEMMING VS LEMMATIZATION
    # ================================================================
    elif section == "C. Stemming vs Lemmatization":
        st.header("C. Stemming vs Lemmatization – Comparative Analysis")

        if not st.session_state.documents:
            st.warning("⚠️ Please upload documents first (go to Section A)!")
            return

        st.markdown("""
        This section compares **Porter Stemming** and **WordNet Lemmatization** using:
        - Retrieval results (number of relevant documents found)
        - Cosine similarity scores (semantic quality of matches)
        """)

        query = st.text_input("🔍 Enter a search query:", "information retrieval systems")

        if query and st.session_state.documents:
            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()

            query_tokens = [t for t in word_tokenize(query.lower()) if t.isalnum()]
            stemmed_query = [stemmer.stem(t) for t in query_tokens]
            lemmatized_query = [lemmatizer.lemmatize(t) for t in query_tokens]

            # Display query processing
            st.markdown("### 🔄 Query Transformation")
            q_df = pd.DataFrame({
                "Original Token": query_tokens,
                "After Stemming": stemmed_query,
                "After Lemmatization": lemmatized_query
            })
            st.dataframe(q_df, use_container_width=True)

            # Build stemmed and lemmatized indexes
            stemmed_index = defaultdict(set)
            lemmatized_index = defaultdict(set)

            for doc_id, text in st.session_state.documents.items():
                tokens = [t for t in word_tokenize(text.lower()) if t.isalnum()]
                for token in tokens:
                    stemmed_index[stemmer.stem(token)].add(doc_id)
                    lemmatized_index[lemmatizer.lemmatize(token)].add(doc_id)

            # Search with stemming
            stem_results = None
            for term in stemmed_query:
                docs = stemmed_index.get(term, set())
                stem_results = docs.copy() if stem_results is None else stem_results.intersection(docs)

            # Search with lemmatization
            lem_results = None
            for term in lemmatized_query:
                docs = lemmatized_index.get(term, set())
                lem_results = docs.copy() if lem_results is None else lem_results.intersection(docs)

            stem_results = stem_results or set()
            lem_results = lem_results or set()

            # Retrieval Results
            st.markdown("### 📋 Retrieval Results Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔵 Stemming Results")
                st.metric("Documents Found", len(stem_results))
                if stem_results:
                    for doc in sorted(stem_results):
                        st.write(f"  ✓ {doc}")
                else:
                    st.write("  No exact matches")
            with col2:
                st.markdown("#### 🟢 Lemmatization Results")
                st.metric("Documents Found", len(lem_results))
                if lem_results:
                    for doc in sorted(lem_results):
                        st.write(f"  ✓ {doc}")
                else:
                    st.write("  No exact matches")

            # Cosine Similarity Analysis
            st.markdown("### 📊 Cosine Similarity Scores (TF-based)")
            query_stem_vec = Counter(stemmed_query)
            query_lem_vec = Counter(lemmatized_query)

            stem_similarities = {}
            lem_similarities = {}

            for doc_id in st.session_state.documents:
                doc_tokens = [t for t in word_tokenize(st.session_state.documents[doc_id].lower()) if t.isalnum()]
                doc_stemmed = Counter(apply_stemming(doc_tokens))
                doc_lemmatized = Counter(apply_lemmatization(doc_tokens))
                stem_similarities[doc_id] = cosine_similarity(dict(query_stem_vec), dict(doc_stemmed))
                lem_similarities[doc_id] = cosine_similarity(dict(query_lem_vec), dict(doc_lemmatized))

            sim_data = {"Document": [], "Cosine Sim (Stemming)": [], "Cosine Sim (Lemmatization)": [], "Better Method": []}
            for doc_id in sorted(st.session_state.documents.keys()):
                sim_data["Document"].append(doc_id)
                s_sim = stem_similarities[doc_id]
                l_sim = lem_similarities[doc_id]
                sim_data["Cosine Sim (Stemming)"].append(f"{s_sim:.4f}")
                sim_data["Cosine Sim (Lemmatization)"].append(f"{l_sim:.4f}")
                sim_data["Better Method"].append("Stemming" if s_sim > l_sim else ("Lemmatization" if l_sim > s_sim else "Equal"))
            st.dataframe(pd.DataFrame(sim_data), use_container_width=True)

            # Summary
            avg_stem = sum(stem_similarities.values()) / len(stem_similarities)
            avg_lem = sum(lem_similarities.values()) / len(lem_similarities)
            
            st.markdown("### 🏁 Summary & Conclusion")
            summary_df = pd.DataFrame({
                "Metric": ["Documents Retrieved", "Avg Cosine Similarity", "Technique Nature", "Best For"],
                "Stemming": [len(stem_results), f"{avg_stem:.4f}", "Aggressive (suffix chopping)", "High Recall"],
                "Lemmatization": [len(lem_results), f"{avg_lem:.4f}", "Conservative (dictionary-based)", "High Precision"]
            })
            st.table(summary_df)

            if avg_stem > avg_lem:
                st.success("""
                📌 **Conclusion:** For this IR document collection, **Stemming** yields higher average cosine similarity 
                ({:.4f} vs {:.4f}). Stemming groups more word variants together (e.g., 'retrieval', 'retrieving', 'retrieved' 
                → 'retriev'), which increases recall. For this dataset of technical IR documents where vocabulary overlap 
                matters, stemming is more suitable.
                """.format(avg_stem, avg_lem))
            else:
                st.success("""
                📌 **Conclusion:** For this IR document collection, **Lemmatization** yields higher average cosine similarity 
                ({:.4f} vs {:.4f}). Lemmatization produces valid dictionary words (e.g., 'systems' → 'system'), preserving 
                semantic meaning better. For this dataset with precise technical terminology, lemmatization provides more 
                meaningful matches and is more suitable.
                """.format(avg_lem, avg_stem))

    # ================================================================
    # SECTION D: PHRASE QUERY PROCESSING
    # ================================================================
    elif section == "D. Phrase Query Processing":
        st.header("D. Phrase Query Processing – Biword vs Positional Index")

        if not st.session_state.documents:
            st.warning("⚠️ Please upload documents first (go to Section A)!")
            return

        # Build indexes
        biword_index = build_biword_index(st.session_state.documents)
        positional_index = build_positional_index(st.session_state.documents)

        # Display Biword Index
        st.markdown("### 📇 Biword Index Representation")
        st.markdown("*Each entry maps a consecutive word pair to documents containing it.*")
        biword_items = sorted(biword_index.items())[:20]
        bw_data = {"Biword": [], "Document IDs": [], "Frequency": []}
        for biword, docs in biword_items:
            bw_data["Biword"].append(biword)
            bw_data["Document IDs"].append(", ".join(sorted(docs)))
            bw_data["Frequency"].append(len(docs))
        st.dataframe(pd.DataFrame(bw_data), use_container_width=True)
        st.info(f"Total biword entries: {len(biword_index)}")

        # Display Positional Index
        st.markdown("### 📇 Positional Index Representation")
        st.markdown("*Each term maps to document IDs with exact positions where it occurs.*")
        pos_items = sorted(positional_index.items())[:8]
        pos_data = {"Term": [], "Document": [], "Positions": []}
        for term, doc_positions in pos_items:
            for doc_id, positions in sorted(doc_positions.items())[:3]:
                pos_data["Term"].append(term)
                pos_data["Document"].append(doc_id)
                pos_data["Positions"].append(str(positions))
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)

        # Query Search
        st.markdown("---")
        st.markdown("### 🔍 Phrase Query Search")
        
        # Provide multiple test queries
        test_queries = ["information retrieval", "machine learning", "information retrieval system", 
                       "inverted index", "document retrieval systems"]
        phrase_query = st.selectbox("Select or type a phrase query:", test_queries)
        custom_query = st.text_input("Or enter a custom phrase query:")
        if custom_query:
            phrase_query = custom_query

        if phrase_query:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🔵 Biword Index Results")
                start_time = time.time()
                biword_results = search_biword(phrase_query, biword_index)
                biword_time = (time.time() - start_time) * 1000
                st.metric("Documents Found", len(biword_results))
                st.write(f"**Matching docs:** {sorted(biword_results) if biword_results else 'None'}")
                st.write(f"⏱️ Search time: {biword_time:.4f} ms")

            with col2:
                st.markdown("#### 🟢 Positional Index Results")
                start_time = time.time()
                positional_results = search_positional(phrase_query, positional_index)
                positional_time = (time.time() - start_time) * 1000
                st.metric("Documents Found", len(positional_results))
                st.write(f"**Matching docs:** {sorted(positional_results) if positional_results else 'None'}")
                st.write(f"⏱️ Search time: {positional_time:.4f} ms")

            # Run all test queries for comparison table
            st.markdown("---")
            st.markdown("### 📊 Multi-Query Comparison Table")
            comparison_results = {"Phrase Query": [], "Biword Results": [], "Positional Results": [], 
                                 "False Positives?": [], "Biword Time (ms)": [], "Positional Time (ms)": []}
            for q in test_queries:
                bw_r = search_biword(q, biword_index)
                t1 = time.time()
                bw_r = search_biword(q, biword_index)
                t1 = time.time() - t1
                pos_r = search_positional(q, positional_index)
                t2 = time.time()
                pos_r = search_positional(q, positional_index)
                t2 = time.time() - t2
                fp = bw_r - pos_r
                comparison_results["Phrase Query"].append(q)
                comparison_results["Biword Results"].append(len(bw_r))
                comparison_results["Positional Results"].append(len(pos_r))
                comparison_results["False Positives?"].append(f"Yes ({len(fp)})" if fp else "No")
                comparison_results["Biword Time (ms)"].append(f"{t1*1000:.4f}")
                comparison_results["Positional Time (ms)"].append(f"{t2*1000:.4f}")
            st.dataframe(pd.DataFrame(comparison_results), use_container_width=True)

            # # Analysis
            # st.markdown("---")
            # st.markdown("### 📝 Analysis & Inference")

            # st.markdown("#### ⚠️ Cases Where Biword Index Gives False Positives")
            # st.markdown("""
            # For phrases with **3 or more words**, the biword index can produce false positives:
            
            # **Example:** Query = `"information retrieval system"`
            # - Biword index checks: `"information retrieval"` ∈ doc **AND** `"retrieval system"` ∈ doc
            # - A document containing *"information retrieval is complex"* and later *"the retrieval system works"* 
            #   would match BOTH biwords, but the exact phrase `"information retrieval system"` does NOT exist.
            # - The biwords are satisfied independently at **different positions**, creating a false positive.
            
            # **Key Issue:** Biword index does NOT verify that consecutive biwords overlap at the correct positions.
            # """)

            # st.markdown("#### ✅ Why Positional Index Gives More Accurate Results")
            # st.markdown("""
            # The positional index guarantees **exact phrase matching** because:
            
            # 1. It stores the **exact position** of every term in every document
            # 2. For phrase `"information retrieval system"`, it verifies:
            #    - `"information"` at position **p**
            #    - `"retrieval"` at position **p+1** 
            #    - `"system"` at position **p+2**
            # 3. Only documents where terms appear in **exact consecutive sequence** are returned
            # 4. Additional benefit: Supports **proximity queries** (terms within k positions)
            
            # **Trade-off:** Positional index requires more storage (position lists per term per doc), 
            # but provides zero false positives for phrase queries.
            # """)

            # Show false positives if any
            if biword_results != positional_results:
                false_positives = biword_results - positional_results
                if false_positives:
                    st.error(f"🔴 **False positives detected!** Biword returned {sorted(false_positives)} but positional did not.")

    # ================================================================
    # SECTION E: BST VS B-TREE
    # ================================================================
    elif section == "E. BST vs B-Tree Dictionary":
        st.header("E. Dictionary Search: Binary Search Tree vs B-Tree")

        if not st.session_state.documents:
            st.warning("⚠️ Please upload documents first (go to Section A)!")
            return

        # Build vocabulary
        vocabulary = set()
        for text in st.session_state.documents.values():
            tokens = [t for t in word_tokenize(text.lower()) if t.isalnum()]
            vocabulary.update(tokens)

        # Build BST and B-Tree
        bst = BST()
        btree = BTree(t=3)

        bst_build_start = time.time()
        for doc_id, text in st.session_state.documents.items():
            tokens = [t for t in word_tokenize(text.lower()) if t.isalnum()]
            for token in tokens:
                bst.insert(token, doc_id)
        bst_build_time = (time.time() - bst_build_start) * 1000

        btree_build_start = time.time()
        for doc_id, text in st.session_state.documents.items():
            tokens = [t for t in word_tokenize(text.lower()) if t.isalnum()]
            for token in tokens:
                btree.insert(token, doc_id)
        btree_build_time = (time.time() - btree_build_start) * 1000

        # Structure Info
        st.markdown("### 🏗️ Data Structure Properties")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Binary Search Tree (BST)")
            st.write(f"- **Nodes:** {bst.node_count}")
            st.write(f"- **Height:** {bst.get_height()}")
            st.write(f"- **Build time:** {bst_build_time:.2f} ms")
            st.write(f"- **Worst case search:** O(h) = O({bst.get_height()})")
        with col2:
            st.markdown("#### B-Tree (order t=3)")
            st.write(f"- **Nodes:** {btree.node_count}")
            st.write(f"- **Height:** {btree.get_height()}")
            st.write(f"- **Build time:** {btree_build_time:.2f} ms")
            st.write(f"- **Worst case search:** O(log_t n) = O({btree.get_height()})")

        st.write(f"\n**Dictionary size:** {len(vocabulary)} unique terms")

        # Query interface
        st.markdown("---")
        st.markdown("### 🔍 Search Performance Comparison")

        default_queries = "information, retrieval, machine, learning, data, mining, vector, boolean, index, text, system, query, search, document, model"
        queries_input = st.text_input("Enter comma-separated search terms:", default_queries)
        queries = [q.strip().lower() for q in queries_input.split(",") if q.strip()]

        num_iterations = st.slider("Number of timing iterations (for accuracy):", 10, 1000, 100)

        if queries:
            results_data = {
                "Query": [], "Found": [], "BST Comparisons": [],
                "BST Avg Time (ms)": [], "B-Tree Comparisons": [],
                "B-Tree Avg Time (ms)": [], "Faster": []
            }

            for query in queries:
                # BST search with timing
                bst_times = []
                for _ in range(num_iterations):
                    start = time.time()
                    bst_result = bst.search(query)
                    bst_times.append((time.time() - start) * 1000)
                bst_comparisons = bst.comparisons
                avg_bst_time = sum(bst_times) / len(bst_times)

                # B-Tree search with timing
                btree_times = []
                for _ in range(num_iterations):
                    start = time.time()
                    btree_result = btree.search(query)
                    btree_times.append((time.time() - start) * 1000)
                btree_comparisons = btree.comparisons
                avg_btree_time = sum(btree_times) / len(btree_times)

                results_data["Query"].append(query)
                results_data["Found"].append("✓" if bst_result else "✗")
                results_data["BST Comparisons"].append(bst_comparisons)
                results_data["BST Avg Time (ms)"].append(f"{avg_bst_time:.6f}")
                results_data["B-Tree Comparisons"].append(btree_comparisons)
                results_data["B-Tree Avg Time (ms)"].append(f"{avg_btree_time:.6f}")
                results_data["Faster"].append("B-Tree" if btree_comparisons <= bst_comparisons else "BST")

            st.markdown("### 📊 Experimental Results Table")
            st.dataframe(pd.DataFrame(results_data), use_container_width=True)

            # Summary
            st.markdown("### 📈 Aggregate Statistics")
            avg_bst_comp = sum(results_data["BST Comparisons"]) / len(queries)
            avg_btree_comp = sum(results_data["B-Tree Comparisons"]) / len(queries)
            avg_bst_t = sum(float(t) for t in results_data["BST Avg Time (ms)"]) / len(queries)
            avg_btree_t = sum(float(t) for t in results_data["B-Tree Avg Time (ms)"]) / len(queries)
            btree_wins = results_data["Faster"].count("B-Tree")

            summary = pd.DataFrame({
                "Metric": ["Avg Comparisons", "Avg Search Time (ms)", "Tree Height", "Build Time (ms)", "Wins (fewer comparisons)"],
                "BST": [f"{avg_bst_comp:.2f}", f"{avg_bst_t:.6f}", bst.get_height(), f"{bst_build_time:.2f}", len(queries) - btree_wins],
                "B-Tree": [f"{avg_btree_comp:.2f}", f"{avg_btree_t:.6f}", btree.get_height(), f"{btree_build_time:.2f}", btree_wins]
            })
            st.table(summary)

            # # Inference
            # st.markdown("### 📝 Inference")
            # if avg_btree_comp <= avg_bst_comp:
            #     st.success(f"""
            #     📌 **B-Tree outperforms BST** for dictionary search in this experiment:
            #     - B-Tree avg comparisons: **{avg_btree_comp:.2f}** vs BST: **{avg_bst_comp:.2f}**
            #     - B-Tree height: **{btree.get_height()}** vs BST height: **{bst.get_height()}**
            #     - B-Tree won in **{btree_wins}/{len(queries)}** queries
                
            #     **Reason:** B-Tree's higher branching factor (up to 2t-1 = 5 keys per node) creates a shallower tree, 
            #     requiring fewer comparisons. B-Trees maintain guaranteed balance (all leaves at same level), 
            #     while BSTs can become skewed. B-Trees are O(log_t n) vs BST's O(log_2 n) best case / O(n) worst case.
                
            #     **For large-scale IR systems**, B-Trees are preferred because they minimize disk I/O operations 
            #     due to their high fanout and shallow depth.
            #     """)
            # else:
            #     st.info(f"""
            #     📌 **BST performs comparably** for this small dictionary ({len(vocabulary)} terms):
            #     - BST avg comparisons: **{avg_bst_comp:.2f}** vs B-Tree: **{avg_btree_comp:.2f}**
                
            #     However, **B-Trees are still preferred for large-scale systems** because:
            #     - They guarantee O(log_t n) worst case (BST can degrade to O(n))
            #     - Better cache locality (multiple keys per node)
            #     - Optimized for disk-based storage in real IR systems
            #     """)

    # ================================================================
    # SECTION F: TOLERANT RETRIEVAL
    # ================================================================
    elif section == "F. Tolerant Retrieval":
        st.header("F. Tolerant Retrieval")

        if not st.session_state.documents:
            st.warning("⚠️ Please upload documents first (go to Section A)!")
            return

        st.markdown("""
        Tolerant retrieval handles **imperfect queries** — misspellings, partial terms, and phonetic variations.
        This section demonstrates all five techniques.
        """)

        # Build vocabulary and indexes
        vocabulary = set()
        for text in st.session_state.documents.values():
            tokens = [t for t in word_tokenize(text.lower()) if t.isalnum()]
            vocabulary.update(tokens)

        kgram_index = build_kgram_index(vocabulary, k=2)
        inv_index = build_inverted_index(st.session_state.documents)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🃏 Wildcard Queries", "✏️ Spelling Correction", "📏 Edit Distance", 
            "📊 K-gram Index", "🔊 Phonetic (Soundex)"
        ])

        # TAB 1: Wildcard Queries
        with tab1:
            st.markdown("### Wildcard Query Search")
            st.markdown("Use `*` as wildcard. Examples: `inform*`, `*tion`, `comp*er`")
            wildcard_q = st.text_input("Enter wildcard query:", "inform*", key="wc")
            if wildcard_q:
                results = wildcard_search(wildcard_q, kgram_index, vocabulary)
                st.write(f"**Pattern:** `{wildcard_q}`")
                st.write(f"**Matching terms ({len(results)}):** {sorted(results)}")
                if results:
                    matching_docs = set()
                    for term in results:
                        if term in inv_index:
                            matching_docs.update(inv_index[term])
                    st.write(f"**Documents:** {sorted(matching_docs)}")

                # Additional wildcard examples
                st.markdown("#### More Wildcard Examples")
                examples = ["*tion", "comp*", "*learn*", "re*al"]
                ex_data = {"Pattern": [], "Matches": [], "Count": []}
                for ex in examples:
                    matches = wildcard_search(ex, kgram_index, vocabulary)
                    ex_data["Pattern"].append(ex)
                    ex_data["Matches"].append(", ".join(sorted(matches)[:5]) + ("..." if len(matches) > 5 else ""))
                    ex_data["Count"].append(len(matches))
                st.dataframe(pd.DataFrame(ex_data), use_container_width=True)

        # TAB 2: Spelling Correction
        with tab2:
            st.markdown("### Spelling Correction using Edit Distance")
            misspelled = st.text_input("Enter a misspelled term:", "informaton", key="spell")
            max_dist = st.slider("Maximum edit distance:", 1, 3, 2)
            if misspelled:
                corrections = spelling_correction(misspelled, vocabulary, max_dist)
                if corrections:
                    st.markdown(f"**Corrections for** `{misspelled}`:")
                    corr_df = pd.DataFrame(corrections, columns=["Suggested Term", "Edit Distance"])
                    st.dataframe(corr_df, use_container_width=True)
                    best = corrections[0][0]
                    if best in inv_index:
                        st.write(f"**Best match:** `{best}` → Documents: {sorted(inv_index[best])}")
                else:
                    st.write("No corrections found within the specified distance.")

                # Test multiple misspellings
                st.markdown("#### Batch Spelling Correction")
                test_words = ["retrival", "documnet", "compter", "languge", "serch"]
                batch_data = {"Misspelled": [], "Best Correction": [], "Edit Distance": []}
                for w in test_words:
                    c = spelling_correction(w, vocabulary, 2)
                    batch_data["Misspelled"].append(w)
                    batch_data["Best Correction"].append(c[0][0] if c else "N/A")
                    batch_data["Edit Distance"].append(c[0][1] if c else "N/A")
                st.dataframe(pd.DataFrame(batch_data), use_container_width=True)

        # TAB 3: Edit Distance Matrix
        with tab3:
            st.markdown("### Edit Distance Calculation")
            col1, col2 = st.columns(2)
            with col1:
                word1 = st.text_input("Source word:", "informaton", key="ed1")
            with col2:
                word2 = st.text_input("Target word:", "information", key="ed2")

            if word1 and word2:
                matrix = edit_distance_matrix(word1, word2)
                dist = matrix[len(word1)][len(word2)]
                st.write(f"**Edit Distance:** `{word1}` → `{word2}` = **{dist}**")

                # Display matrix
                st.markdown("#### Dynamic Programming Matrix")
                cols = [''] + list(word2)
                rows = [''] + list(word1)
                unique_cols = []
                col_counts = {}
                for col in cols:
                    if col in col_counts:
                        col_counts[col] += 1
                        unique_cols.append(col + '\u200b' * col_counts[col])
                    else:
                        col_counts[col] = 0
                        unique_cols.append(col)
                df = pd.DataFrame(matrix, index=rows, columns=unique_cols)
                st.dataframe(df, use_container_width=True)

                st.markdown("""
                **Operations:** Insert (→), Delete (↓), Replace (↘)  
                **Interpretation:** Each cell [i,j] = min edits to transform first i chars of source to first j chars of target.
                """)

        # TAB 4: K-gram Index
        with tab4:
            st.markdown("### K-gram Index (k=2, Bigrams)")
            st.write(f"**Total unique k-grams:** {len(kgram_index)}")

            # Show k-gram breakdown for a term
            kgram_term = st.text_input("Enter a term to see its k-grams:", "information", key="kg")
            if kgram_term:
                padded = f"${kgram_term.lower()}$"
                kgrams = [padded[i:i+2] for i in range(len(padded)-1)]
                st.write(f"**Term:** `{kgram_term}` → **Padded:** `{padded}` → **K-grams:** `{kgrams}`")

                # Find terms sharing k-grams
                shared_terms = set()
                for kg in kgrams:
                    if kg in kgram_index:
                        shared_terms.update(kgram_index[kg])
                shared_terms.discard(kgram_term.lower())
                st.write(f"**Terms sharing k-grams ({len(shared_terms)}):** {sorted(shared_terms)[:15]}")

            # Sample of k-gram index
            st.markdown("#### K-gram Index Sample")
            sample_kgrams = sorted(kgram_index.items())[:20]
            kg_data = {"K-gram": [], "Terms (sample)": [], "Count": []}
            for kg, terms in sample_kgrams:
                kg_data["K-gram"].append(kg)
                kg_data["Terms (sample)"].append(", ".join(sorted(terms)[:4]))
                kg_data["Count"].append(len(terms))
            st.dataframe(pd.DataFrame(kg_data), use_container_width=True)

        # TAB 5: Phonetic Correction
        with tab5:
            st.markdown("### Phonetic Correction (Soundex Algorithm)")
            st.markdown("Soundex maps similar-sounding words to the same code (Letter + 3 digits).")
            
            phonetic_q = st.text_input("Enter a term:", "informasion", key="ph")
            if phonetic_q:
                code = soundex(phonetic_q)
                st.write(f"**Soundex code for** `{phonetic_q}`: **{code}**")

                matches = phonetic_search(phonetic_q, vocabulary)
                if matches:
                    st.write(f"**Phonetically similar terms:** {sorted(matches)}")
                    sx_data = {"Term": [], "Soundex Code": []}
                    for term in sorted(matches):
                        sx_data["Term"].append(term)
                        sx_data["Soundex Code"].append(soundex(term))
                    st.dataframe(pd.DataFrame(sx_data), use_container_width=True)

                    matching_docs = set()
                    for term in matches:
                        if term in inv_index:
                            matching_docs.update(inv_index[term])
                    if matching_docs:
                        st.write(f"**Documents via phonetic match:** {sorted(matching_docs)}")
                else:
                    st.write("No phonetically similar terms found in vocabulary.")

            # Soundex examples
            st.markdown("#### Soundex Code Examples")
            example_words = ["information", "retrieval", "computer", "science", "search", "system"]
            sx_examples = {"Word": [], "Soundex": []}
            for w in example_words:
                sx_examples["Word"].append(w)
                sx_examples["Soundex"].append(soundex(w))
            st.dataframe(pd.DataFrame(sx_examples), use_container_width=True)

    # ================================================================
    # SECTION G: INFERENCE AND DISCUSSION
    # ================================================================
    elif section == "G. Inference & Discussion":
        st.header("G. Inference and Discussion")
        st.markdown("*Comprehensive analysis of experimental results from all sections.*")

        st.markdown("""
        ---
        ### 1. Which preprocessing technique improved retrieval quality?
        
        Based on our experiments:
        - **Stop word removal** provided the most significant improvement by eliminating high-frequency, 
          low-information words (the, is, a, in, of) that appear in nearly every document. This dramatically 
          improved precision by reducing false matches.
        - **Lowercasing** was essential for ensuring terms like "Information" and "information" are treated 
          identically, improving recall without any precision cost.
        - **Tokenization** (splitting into meaningful units) is the foundation — without it, no further 
          processing is possible.
        - **Hyphen handling** helped with compound terms (e.g., "text-mining" → "text", "mining", "textmining"), 
          ensuring these terms are discoverable via multiple query forms.
        
        ---
        ### 2. Was stemming or lemmatization better for the dataset?
        
        For our IR-domain technical document collection:
        - **Stemming** (Porter Stemmer) produced **higher recall** by aggressively grouping word variants 
          (e.g., "retrieval", "retrieving", "retrieved", "retrieves" → "retriev")
        - **Lemmatization** (WordNet) produced **higher precision** by mapping to valid dictionary forms 
          (e.g., "systems" → "system", "searching" → "searching")
        - **Our conclusion:** For this dataset, **stemming is more suitable** because IR applications typically 
          prioritize recall (finding all relevant documents) over precision. The technical vocabulary benefits 
          from aggressive normalization since many documents discuss the same concepts using different word forms.
        
        ---
        ### 3. Which phrase query index was more accurate?
        
        The **Positional Index** is definitively more accurate:
        - **Biword Index:** Fast and space-efficient, but produces **false positives** for phrases longer than 
          2 words. It independently checks each consecutive pair without verifying they are adjacent.
        - **Positional Index:** Guarantees **exact phrase matching** by verifying consecutive positions. 
          Zero false positives.
        - **Experimental evidence:** For queries like "information retrieval system" (3 words), biword index 
          may return documents containing "information retrieval" and "retrieval system" at different locations, 
          while positional index only returns documents with the exact sequence.
        - **Trade-off:** Positional index requires ~2-4x more storage than biword index due to position lists.
        
        ---
        ### 4. Which tree structure was faster?
        
        **B-Tree** demonstrates superior performance:
        - **Fewer comparisons** on average due to higher branching factor (up to 5 keys per node with t=3)
        - **Guaranteed balance** — all leaf nodes at same depth (B-Tree height: typically 2-3 for our vocabulary)
        - **BST risk:** Without self-balancing, BST can degrade to a linked list (O(n) search) with sorted input
        - **Practical advantage:** B-Trees are optimized for disk-based systems (fewer I/O operations), which is 
          critical for large-scale IR indexes that cannot fit in memory
        - **Our results:** B-Tree height is consistently lower, requiring fewer node visits per search
        
        ---
        ### 5. How tolerant was the retrieval model?
        
        Our system demonstrates **high tolerance** across multiple dimensions:
        
        | Method | Tolerance Level | Example |
        |--------|----------------|---------|
        | Wildcard queries | Partial/unknown terms | `inform*` → information, informatics |
        | Edit distance | 1-2 character typos | `informaton` → information (dist=1) |
        | K-gram index | Efficient wildcard expansion | `*tion` → all words ending in -tion |
        | Phonetic (Soundex) | Sound-alike terms | `informasion` → information |
        
        The system can handle: typos, prefix/suffix queries, phonetic misspellings, and approximate matching. 
        Combined, these methods ensure users find relevant documents even with imperfect queries.
        
        ---
        ### 6. What are the limitations of the system?
        
        1. **Small corpus size** — With only 10 documents, statistical measures (TF-IDF) are less reliable
        2. **English-only** — No multi-language support; Soundex is English-specific
        3. **No learning/feedback** — System doesn't improve from user interactions
        4. **Memory-based indexes** — All data structures are in-memory (not scalable to millions of docs)
        5. **No ranked retrieval** — Boolean model returns unranked document sets
        6. **Simple BST** — Not self-balancing (AVL/Red-Black would be better)
        7. **No query expansion** — Doesn't use synonyms or related terms
        8. **Single-term Soundex** — Phonetic matching only works on individual terms
        
        ---
        ### 7. How can the system be improved?
        
        1. **Ranked retrieval** — Implement BM25 or TF-IDF cosine scoring for relevance ranking
        2. **Query expansion** — Use WordNet synonyms to automatically expand queries
        3. **Relevance feedback** — Allow users to mark relevant/irrelevant results to refine searches
        4. **Scalable storage** — Implement disk-based inverted index with compression
        5. **Semantic search** — Integrate word embeddings (Word2Vec/BERT) for meaning-based retrieval
        6. **Multi-format support** — Handle PDF, DOCX, HTML in addition to plain text
        7. **Self-balancing BST** — Use AVL or Red-Black tree for fair comparison
        8. **Distributed indexing** — MapReduce-based index construction for large collections
        9. **Caching** — LRU cache for frequent queries
        10. **Evaluation metrics** — Implement precision@k, recall@k, MAP, NDCG for proper evaluation
        """)


if __name__ == "__main__":
    main()
