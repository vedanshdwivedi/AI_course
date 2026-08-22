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

<!-- Voca.       The     food    is    good     bad     Pizza   amazing
The           ->  1       0       0      0       0           0       0
Food          ->  0       1       0      0       0           0       0
is            ->  0       0       1      0       0           0       0
Good          ->  0       0       0      1       0           0       0
Bad           ->  0       0       0      0       1           0       0
Pizza         ->  0       0       0      0       0           1       0
Amazing       ->  0       0       0      0       0           0       1 -->

Now, the Statements can be represented as

S1 -> [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
S2 -> [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0]]
S3 -> [[0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0]]

This is how we convert words to vector using one-hot encoding.

### Advantages and Disadvantages of OHE

#### Advantages

1. Easy to implement in Python
2. Since we have fixed the size of the vectors (based on the vocabulary), we can easily store the data in a 2D matrix or DataFrame

#### Disadvantages

1. Size of vector increases with the number of documents (vocabulary size) [Sparse Matrix gets created which results in overfitting]
2. No semantic meaning is captured. For example, if we add "pizza" in our corpus, the vector for "pizza" will be completely different from "burger", even though they are similar
3. If we add new words to the corpus, we need to regenerate the entire vector space

## Bag of Words (BOW)

This is a simple technique that can be used for small-text problem statements. Some of the popular usecases are spam-classification, review analysis, etc. To understand this technique let us consider the following statements:

S1 -> He is a good boy.
S2 -> She is a good girl.
S3 -> Boy and girl are good.

Step 1: We will make the text lowercase.
Step 2: Eliminate Stopwords

This makes the statements as follows:

S1 -> [good, boy]
S2 -> [good, girl]
S3 -> [boy, girl, good]

Now our vocabulary has 3 words. We can create a frequency distribution table (sorted in descending order of frequency) as follows:

good -> 3
boy -> 2
girl -> 2

In a bigger dataset, we can have more words in the vocabulary. There can be some words which are present only once, we can select the top 10-20 words based on the dataset. Now statements can be represented as follows:

S1 -> [1, 1, 0]
S2 -> [1, 0, 1]
S3 -> [1, 1, 1]

Here the vector is representing [good, boy, girl]
If we compare the implementation with that of One-hot Encoding, we would see how simple it was to convert statements to vectors (the size is also small). Also, the BOW (Bag of words) can be either binary or non-binary in nature. In binary-BOW, the vectors will have 0-1 depending on presence of the word, whereas, in binary-BOW, the vector will contain frequency of the occurance of the word in the sentence.

#### Advantages

1. Easy to implement.
2. Output is of fixed-size (Good for ML algorithms).
3. Captures frequency of the words (Better than OHE).

#### Disadvantages

1. Sparse Matrix problem (similar to OHE) that causes overfitting.
2. Does not capture semantic meaning of the words. Also the vectors that gets created may have different order of words.
3. A lot of words are going to get rejected in real-life scenarios.
4. The out-of-vocabulary issue is present in this as well.

## N-Grams (bigrams, trigrams etc)

Let us take the following sentences as examples
S1 -> The food is good
S2 -> The food is not good

Both the sentences S1 and S2 are completely different from each other.

<!-- vocabulary -> [food not good]

S1 -> [1 0 1]
S2 -> [1 1 1] -->

With respect to vectors S1 and S2 are very similar. If we make use of combinations, lets say combinations of 2 words (bigrams) or 3 words (trigrams) etc. My vocabulary will become [food, good, not, food not, not good, not good]. With this new vocabulary, the sentences S1 and S2 will become S1 -> [0, 1, 1, 0, 1, 0], S2 -> [0, 1, 1, 1, 0, 0].
This will capture more context/semantics for the sentences and improve the accuracy of the model.
The tuple is read as (x, y) where x = starting point and y = ending point (inclusive)
(1,1) -> monograms only
(1,2) -> monograms and bigrams
(1,3) -> monogram, bigram and trigrams etc
(2,3) -> bigrams, trigrams
(3,3) -> trigrams only
