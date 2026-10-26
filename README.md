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

### Module 13: End-to-End ANN Project Implementation
- **[Chapter 13: ANN Project Implementation & Streamlit Deployment](./13_ann_project_implementation/README.md)**
  - *Overview:* End-to-end practical deep learning pipelines covering dataset preprocessing, feature engineering, model training, hyperparameter tuning with Keras Tuner, serialization, and interactive Streamlit web applications.
  - *Core Pipeline Flow:*
    $$\text{Load Data} \longrightarrow \text{Feature Engineering} \longrightarrow \text{ANN Training} \longrightarrow \text{Regularization / EarlyStopping} \longrightarrow \text{Hyperparameter Tuning} \longrightarrow \text{Serialization} \longrightarrow \text{Streamlit UI}$$
  - *Projects & Applications:*
    - **Customer Churn Classification (Binary Classification):**
      - Notebook training & validation: [`churn_classifier_trainer.ipynb`](./13_ann_project_implementation/churn_classifier_trainer.ipynb)
      - Offline inference & testing: [`churn_classifier_predictor.ipynb`](./13_ann_project_implementation/churn_classifier_predictor.ipynb)
      - Interactive Streamlit app: [`app.py`](./13_ann_project_implementation/app.py)
      - Key topics: Sigmoid activation, Binary Cross-Entropy loss, Dropout, EarlyStopping callbacks (patience=20), TensorBoard tracking in `logs/fit`.
      - Serialized artifacts: `best_model.keras`, `best_model.h5`, `model.h5`, `label_encoder_gender.pkl`, `one_hot_encoder_geography.pkl`, `sclaer.pkl`
    - **Customer Salary Estimation (ANN Regression):**
      - Regression training & Keras Tuner architecture search: [`salary_regression_training.ipynb`](./13_ann_project_implementation/salary_regression_training.ipynb)
      - Interactive Streamlit regression app: [`regressor_app.py`](./13_ann_project_implementation/regressor_app.py)
      - Key topics: Linear activation for regression, Mean Absolute Error (MAE), automated hyperparameter search via `keras_tuner.RandomSearch` (tuning hidden layers, neuron units, learning rate).
      - Serialized artifacts: `salary_estimate_model.keras`, `regressor_scaler.pkl`

---

### Module 14: RNN Project Implementation & Word Embeddings
- **[Chapter 14: RNN Project Implementation & Word Embeddings](./14_rnn_project_implementation/README.md)**
  - *Overview:* Practical implementation of neural word embeddings, text sequence preprocessing, and recurrent neural network classification for sentiment analysis using TensorFlow/Keras on the 50,000 IMDB Movie Reviews dataset.
  - *Core Pipeline Flow:*
    $$\text{Raw Text Sentences} \longrightarrow \text{One-Hot / Tokenizer Encoding} \longrightarrow \text{Sequence Padding (pad\_sequences)} \longrightarrow \text{Keras Embedding Layer} \longrightarrow \text{SimpleRNN / LSTM} \longrightarrow \text{Dense Classification} \longrightarrow \text{Streamlit UI}$$
  - *Key Concepts & Techniques:*
    - **Tokenization & Vocabulary Mapping:** Mapping words to bounded integer indices using `Tokenizer(num_words=10000, oov_token='<OOV>')`.
    - **Sequence Normalization & Padding:** Equalizing review text lengths to uniform vectors ($L = 200$) using `pad_sequences()` with `padding='post'`.
    - **Feature Representation with Keras Embedding Layer:** Creating continuous dense vector representations of words where each discrete word index is mapped to a trainable embedding vector ($D = 64$ or $128$).
    - **Hyperparameter Optimization with Keras Tuner:** Automated architecture search tuning embedding dimensions, recurrent units, dropout regularization, and learning rates.
    - **Interactive Streamlit Deployment:** Real-time sentiment prediction dashboard featuring cached neural network inference, confidence score calculations, and intuitive visual feedback.
  - *Notebooks & Files:*
    - Word embedding fundamentals: [`embedding.ipynb`](./14_rnn_project_implementation/embedding.ipynb)
    - RNN training & Keras Tuner architecture search: [`rnn_project.ipynb`](./14_rnn_project_implementation/rnn_project.ipynb)
    - Interactive Streamlit sentiment app: [`streamlit_app.py`](./14_rnn_project_implementation/streamlit_app.py)
    - Serialized model: `imdb_review_classification.keras`
    - Serialized tokenizer: `tokenizer.pickle`
    - Target dataset: 50,000 IMDB Movie Reviews Dataset ([`IMDB Dataset.csv`](./datasets/IMDB%20Dataset.csv))

---

### Module 15: LSTM RNN In-depth Intuition
- **[Chapter 15: LSTM RNN In-depth Intuition](./15_lstm_rnn_indepth_intuition/)**
  - *Overview:* In-depth mathematical intuition and architecture walkthrough of Long Short-Term Memory (LSTM) networks, engineered to resolve the vanishing gradient problem in sequential deep learning models.
  - *Key Concepts & Architecture:*
    - **Why Simple RNNs Fail on Long Sequences:** Inability to retain early context due to vanishing gradient dynamics during backpropagation through time (BPTT).
    - **Dual Memory State System:**
      - **Cell State ($C_t$):** The constant long-term memory conveyor belt carrying essential information across arbitrary time steps with minimal linear interactions.
      - **Hidden State ($h_t$):** The short-term working memory vector driving step-level predictions and gating decisions.
    - **The Three Gating Mechanisms:**
      - **Forget Gate ($f_t$):** Decides what information to discard from the previous cell state using a sigmoid activation:
        $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
      - **Input Gate ($i_t$) & Candidate Memory ($\tilde{C}_t$):** Decides what new information to add to the cell state:
        $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i), \quad \tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$
        $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
      - **Output Gate ($o_t$):** Regulates the filtered cell state to output the next hidden state:
        $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(C_t)$$
    - **Architecture Comparisons:** Comparing parameter efficiency, training stability, and gradient flow between Simple RNNs, LSTMs, and GRUs (Gated Recurrent Units).

---

## 📁 Repository Structure

```plaintext
AI_course/
├── README.md                          # Repository Index (this file)
├── requirements.txt                   # Python package dependencies
├── pre_requisites/
│   └── ann/
│       ├── README.md                  # Prerequisites: Artificial Neural Networks
│       └── image.png                  # Biological neuron vs. artificial perceptron diagram
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
│   └── README.md                      # Chapter 12: Simple RNN Intuition & Vanishing Gradients
├── 13_ann_project_implementation/
│   ├── README.md                      # Chapter 13 Overview & pipeline flow
│   ├── app.py                         # Customer churn prediction Streamlit app
│   ├── regressor_app.py               # Salary estimation regression Streamlit app
│   ├── churn_classifier_trainer.ipynb # Churn ANN training & TensorBoard logging
│   ├── churn_classifier_predictor.ipynb# Offline inference walkthrough
│   ├── salary_regression_training.ipynb# Salary regression & Keras Tuner optimization
│   ├── best_model.keras               # Best trained churn classification model
│   ├── best_model.h5                  # HDF5 model format
│   ├── model.h5                       # Trained baseline classification model
│   ├── salary_estimate_model.keras    # Best trained salary regression model
│   ├── label_encoder_gender.pkl       # LabelEncoder for Gender feature
│   ├── one_hot_encoder_geography.pkl  # OneHotEncoder for Geography feature
│   ├── sclaer.pkl                     # ColumnTransformer for churn classifier
│   └── regressor_scaler.pkl           # ColumnTransformer for salary regressor
├── 14_rnn_project_implementation/
│   ├── README.md                      # Chapter 14: RNN Project & Word Embeddings Overview
│   ├── embedding.ipynb                # Keras Embedding layer & sequence padding walkthrough
│   ├── rnn_project.ipynb              # IMDB sentiment classification & Keras Tuner search
│   ├── streamlit_app.py               # Interactive Streamlit sentiment analyzer app
│   ├── imdb_review_classification.keras # Best tuned Keras RNN model
│   └── tokenizer.pickle               # Fitted word-to-index Tokenizer instance
├── 15_lstm_rnn_indepth_intuition/     # Chapter 15: LSTM & Gated Recurrent Architectures
├── course_images/                     # Diagram assets referenced across chapters
│   ├── image_9_1.png to 9_3.png       # Streamlit UI demos
│   ├── image_10_1.png to 10_3.png     # NLP & ANN/RNN comparison
│   ├── image_12_1.png, 12_2.jpeg      # RNN forward/backward propagation
│   ├── image_13_1.png                 # ANN architecture & flow
│   └── word_embedding.png             # Word embedding continuous representation
└── datasets/
    ├── Churn_Modelling.csv            # Bank customer churn & salary dataset
    ├── IMDB Dataset.csv               # 50,000 IMDB Movie Reviews dataset for NLP/RNN
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

4. **Running Interactive Web Applications:**
   - **Customer Churn Predictor (Module 13):**
     ```bash
     streamlit run 13_ann_project_implementation/app.py
     ```
   - **Customer Salary Estimator (Module 13):**
     ```bash
     streamlit run 13_ann_project_implementation/regressor_app.py
     ```
   - **IMDB Movie Review Sentiment Analyzer (Module 14):**
     ```bash
     streamlit run 14_rnn_project_implementation/streamlit_app.py
     ```

5. **Visualizing Training with TensorBoard:**
   ```bash
   tensorboard --logdir 13_ann_project_implementation/logs/fit
   ```
   Open [http://localhost:6006](http://localhost:6006) in your browser to view loss, accuracy, and layer histograms.

6. **Running Word Embedding & RNN Notebooks:**
   ```bash
   jupyter notebook 14_rnn_project_implementation/embedding.ipynb
   jupyter notebook 14_rnn_project_implementation/rnn_project.ipynb
   ```


