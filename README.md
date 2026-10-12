# AI Course — Notes & Implementation Index

Welcome to the **AI Course** repository. This repository contains notes, code implementations, Jupyter notebooks, and conceptual guides covering foundational deep learning concepts, Streamlit app development, and Natural Language Processing (NLP) from classical machine learning to recurrent neural networks.

---

## 📚 Course Index & Modules

Below is the directory index with links to each module's detailed documentation and study notes:

### 0. Prerequisites
- **[Artificial Neural Networks (ANNs)](./pre_requisites/ann/README.md)**
  - *Overview:* Core building blocks of neural networks mimicking biological neurons.
  - *Topics covered:*
    - Neurons & Node Components (Inputs $X_i$, Weights $W_i$, Biases $B_i$, Output $O_i$)
    - Network Architecture: Input, Hidden, and Output Layers
    - Layer Connections and Learned Representations
    - Activation Functions & Non-linearity

---

### Module 9: Streamlit with Python
- **[Chapter 9: Introduction to Streamlit](./9_streamlit_with_python/README.md)**
  - *Overview:* Building interactive web dashboards and Machine Learning web apps in Python.
  - *Key files & implementations:*
    - Basic Streamlit app: [`app.py`](./9_streamlit_with_python/app.py)
    - UI Widgets demonstration: [`widgets.py`](./9_streamlit_with_python/widgets.py)
    - Interactive ML classification app: [`classification.py`](./9_streamlit_with_python/ml_project/classification.py)
    - Notebook walkthrough: [`streamlit.ipynb`](./9_streamlit_with_python/streamlit.ipynb)

---

### Module 10: Machine Learning for NLP
- **[Chapter 10: Machine Learning for NLP](./10_ml_for_nlp/README.md)**
  - *Overview:* Comprehensive guide to classical text preprocessing, tokenisation, feature extraction, and word embeddings.
  - *Topics & Notebooks covered:*
    - **Text Preprocessing:**
      - Tokenisation (Corpus, Documents, Vocabulary, Words): [`tokenisation.ipynb`](./10_ml_for_nlp/tokenisation.ipynb)
      - Stemming: [`stemming.ipynb`](./10_ml_for_nlp/stemming.ipynb)
      - Lemmatisation: [`lemmatization.ipynb`](./10_ml_for_nlp/lemmatization.ipynb)
      - Stopwords Removal: [`stopwords.ipynb`](./10_ml_for_nlp/stopwords.ipynb)
      - Parts of Speech (POS) Tagging: [`parts_of_speech_tagging.ipynb`](./10_ml_for_nlp/parts_of_speech_tagging.ipynb)
      - Named Entity Recognition (NER): [`named_entity_recognition.ipynb`](./10_ml_for_nlp/named_entity_recognition.ipynb)
    - **Vectorization & Feature Extraction:**
      - One-Hot Encoding (OHE) — Intuition, Advantages & Limitations
      - Bag of Words (BOW) — Binary & Non-binary representations
      - N-Grams (Monograms, Bigrams, Trigrams)
      - TF-IDF (Term Frequency - Inverse Document Frequency)
    - **Word Embeddings:**
      - Word2Vec Intuition (Cosine Similarity, Semantic Distance)
      - Continuous Bag of Words (CBOW) vs. Skip-grams
      - Word2Vec from scratch: [`word2vec_from_scratch.ipynb`](./10_ml_for_nlp/word2vec_from_scratch.ipynb)
      - Pre-trained Word2Vec: [`pretrained_word2vec.ipynb`](./10_ml_for_nlp/pretrained_word2vec.ipynb)
      - Average Word2Vec for document representations
    - **Applied Project:**
      - SMS / Text Spam Classification: [`spam-classification.ipynb`](./10_ml_for_nlp/spam-classification.ipynb)

---

### Module 11: Deep Learning for NLP
- **[Chapter 11: Deep Learning for NLP](./11_nlp_in_deep_learning/README.md)**
  - *Overview:* Transitioning from classical ML to deep learning architectures for sequential data.
  - *Topics covered:*
    - Limitations of feedforward ANNs for text and sequential data
    - Preserving word order and contextual sequences
    - ANN vs. RNN structural comparisons
    - Recurrent Neural Network (RNN) feedback loops and hidden state formulation:
      $$h_t = \tanh(W_{ih} X_t + W_{hh} h_{t-1} + b)$$

---

### Module 12: Recurrent Neural Networks (RNN) In-depth
- **[Chapter 12: Simple RNN In-depth Intuition](./12_rnn_indepth_intuition/README.md)**
  - *Overview:* Step-by-step mathematical and visual intuition of RNN forward propagation, backpropagation through time (BPTT), and the fundamental challenges with vanilla RNNs.
  - *Key Documents:*
    - Primary Lecture Notes: [`README.md`](./12_rnn_indepth_intuition/README.md)
    - Comprehensive BPTT Guide & Formula Reference: [`README-AI.md`](./12_rnn_indepth_intuition/README-AI.md)
  - *Topics covered:*
    - **RNN Forward Propagation with Time:**
      - Sequential feeding of word vectors ($X_{11}, X_{12}, \dots$)
      - Time-unfolded forward propagation and neuron activations:
        $$O_n = f(X_{in} \cdot W_n + O_{n-1} \cdot W_h + b_n)$$
      - Context aggregation across time steps via recurrent connections
    - **RNN Backward Propagation with Time (BPTT):**
      - Output prediction ($\hat{y}$) and loss calculation ($\text{Loss} = y - \hat{y}$)
      - Gradient descent minimization to reach global minima
      - Weight updates across time for input layer ($W_i$), hidden recurrent loop ($W_h$), and output layer ($W_o$)
    - **Limitations of Vanilla RNNs & Vanishing Gradients:**
      - The **Vanishing Gradient Problem** in long sequences and text generation
      - Inability to capture long-term dependencies (decay of weight update signals toward zero)
      - Recency bias (late words dominating learning while early context is lost)
    - **Mitigations & Gateway to Advanced Architectures:**
      - Alternative activations (`ReLU`, `Leaky ReLU`) and gradient clipping
      - Introduction to gated architectures: **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit)

---

## 📁 Repository Structure

```plaintext
AI_course/
├── README.md                          # Repository Index (this file)
├── requirements.txt                   # Python package dependencies
├── pre_requisites/
│   └── ann/
│       └── README.md                  # Prerequisites: Artificial Neural Networks
├── 9_streamlit_with_python/
│   ├── README.md                      # Chapter 9: Streamlit intro & UI demo
│   ├── app.py                         # Starter Streamlit application
│   ├── widgets.py                     # Streamlit widget demonstrations
│   ├── streamlit.ipynb                # Streamlit walkthrough notebook
│   └── ml_project/
│       └── classification.py          # Machine learning dashboard app
├── 10_ml_for_nlp/
│   ├── README.md                      # Chapter 10: Machine Learning for NLP
│   ├── Notes.pdf                      # Handwritten / lecture notes
│   ├── tokenisation.ipynb
│   ├── stemming.ipynb
│   ├── lemmatization.ipynb
│   ├── stopwords.ipynb
│   ├── parts_of_speech_tagging.ipynb
│   ├── named_entity_recognition.ipynb
│   ├── word2vec_from_scratch.ipynb
│   ├── pretrained_word2vec.ipynb
│   └── spam-classification.ipynb
├── 11_nlp_in_deep_learning/
│   └── README.md                      # Chapter 11: Deep Learning for NLP
├── 12_rnn_indepth_intuition/
│   ├── README.md                      # Chapter 12: Simple RNN Intuition & Vanishing Gradients
│   └── README-AI.md                   # Extended mathematical guide & BPTT formula reference
├── course_images/                     # Diagram assets referenced across chapters
└── datasets/
    └── spam.csv                       # SMS Spam Collection dataset
```

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone git@github.com:vedanshdwivedi/AI_course.git
   cd AI_course
   ```

2. **Set up a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
