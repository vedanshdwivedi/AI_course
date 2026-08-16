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
