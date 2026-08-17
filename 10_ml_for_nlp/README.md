# Chapter 10: Machine Learning for NLP

## Practical uses of NLP

1. Grammar/Spelling correction in word editors (in email or other places)
2. Automated reply recommendations on Linkedin
3. Language Translation in Google Translate or See Translation features in Social Media
4. Search Engines heavily rely on NLP
5. Smart assistants like Alexa or Google agent

## Tokenisation in NLP

### Topics to Cover

1. Corpus - A paragraph usually called a corpus
2. Documents - A document usually consists of sentences.
3. Vocabulary - All the _unique words_ present in the paragraph
4. Words

---

Let us consider the following corpus:

"My name is Vedansh, and I am currently studying tokenisation, which is a very important topic for NLP. Apart from this I love to travel."

Tokenisation is the process where we convert corpus or documents into tokens. So, if we perform tokenisation on the above corpus it becomes,

Tokenisation(corpus) => sentences => ["My name is Vedansh, and I am currently studying tokenisation, which is a very important topic for NLP", "Apart from this I love to travel"]

When we tokenise a corpus, we get sentences. And when we tokenise sentences, we get words.

Tokenisation(sentences) => words => ["My", "name", "is", "Vedansh", "and", "I", "am", "currently", "studying", "tokenisation", "which", "is", "a", "very", "important", "topic", "for", "NLP", "Apart", "from", "this", "I", "love", "to", "travel"]

---

Let us consider another example:

corpus = "I like to drink apple juice. My friend like mango juice"
sentences = tokenisation(corpus) = ["I like to drink apple juice", "My friend like mango juice"]
words = tokenisation(sentences) = ["I","like","to","drink","apple","juice","My","friend","like","mango","juice"]

count(words) = 11
count(unique_words) = 9 => this is called vocabulary

## Stemming in NLP

Stemming is the process of reducing a word to it's word's stem. For instance, stemming(eating) = eat, stemming(dancing) = dance
This is usually important in problem statements, where we let's say want to perform sentiment analysis of a movie review, and we want to treat "love" and "loving" as the same word.

## Lemmatisation in NLP

When we perform stemming, for some words, we do not get the expected results (the result may change the meaning of the word entirely or in some cases give meaningless results, for example: 'history' becomes 'histori'). Lemmatization is very much like stemming. The outputs that we get from lemmatisation is called 'lemma' while stemming yields 'word stems'. Lemma is the root word and not the root stem. Lemmatisation guarantees that we get valid words that have the same meaning.

Compared to Stemming, Lemmatisation is slower.

## Stopwords in NLP

Certain words like I, are, is, that, their etc do not play any significant role in determining the meaning of the sentence. For example, in the sentence: "I need a DevOps engineer to make an existing Swiss healthcare app run entirely without Supabase Cloud." These needs to be removed from the texts to keep the significant words only (depends on usecase).

## Parts of Speech Tagging in NLP

In the lemmatisation, we observed that parts of speech tagging is a very crucial process, since the output got affected based on the part of the speech we used to provide in the WordNetLemmatizer (as the pos argument).

## Named entity recognition

Named Entity Recognition (NER) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as persons, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.

Named entity recognition (NER) is the process of identifying and categorizing key information (entities) in text.

## Steps for solving NLP Problems

Lets say we want to solve a sentiment analysis problem. To solve it we would have lets say a corpus. We would take the following steps

#### Text Preprocessing

- Tokenisation
- Lowercase conversion
- Regular Expression
- Stemming
- Lemmatisation
- Stopwords
- Convert text to vectors
- Train the ML model

## Text to vector conversions

We need to convert text data to vector data in order to feed it to Machine Learning models. These are some of the ways:

1. One-Hot Encoding
2. Bag of words (BOW)
3. TF-IDF
4. word2vec
5. Average word2vec

## One Hot Encoding

One-Hot Encoding is no longer being used in NLP usecases, but it is important to understand

So let us consider three statements:

S1 -> The food is good
S2 -> The food is bad
S3 -> Pizza is amazing

Now, the unique vocabulary in the above statements are

<!-- Vocabulary  -> The    food    is  good    bad     Pizza   amazing
The         ->  1       0       0      0       0           0       0
Food        ->  0       1       0      0       0           0       0
is          ->  0       0       1      0       0           0       0
Good        ->  0       0       0      1       0           0       0
Bad         ->  0       0       0      0       1           0       0
Pizza       ->  0       0       0      0       0           1       0
Amazing     ->  0       0       0      0       0           0       1 -->

Now, the Statements can be represented as

S1 -> [1, 1, 1, 1, 0, 0, 0]
S2 -> [1, 1, 1, 0, 1, 0, 0]
S3 -> [0, 0, 1, 0, 0, 1, 1]

This is how we convert words to vector using one-hot encoding.
