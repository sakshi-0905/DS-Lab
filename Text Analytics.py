# Import libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

# Download all required resources (FINAL FIXES)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')  # <-- FINAL FIX
nltk.download('wordnet')

# Sample text
text = "Text analytics is the process of analyzing text data to extract useful information."

print("Original Text:\n", text)

# Tokenization
tokens = word_tokenize(text)
print("\nTokens:\n", tokens)

# POS Tagging
pos_tags = pos_tag(tokens)
print("\nPOS Tags:\n", pos_tags)

# Stopwords Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
print("\nAfter Stopwords Removal:\n", filtered_tokens)

# Stemming
ps = PorterStemmer()
stemmed = [ps.stem(w) for w in filtered_tokens]
print("\nStemmed Words:\n", stemmed)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered_tokens]
print("\nLemmatized Words:\n", lemmatized)

# TF-IDF
documents = [
    "Text analytics extracts useful information from text",
    "Text mining and text analytics are related fields"
]

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)

print("\nTF-IDF Features:\n", vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix:\n", tfidf.toarray())