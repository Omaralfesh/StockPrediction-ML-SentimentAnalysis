### References

# 1. **re Module (Regular Expressions)**
#    - Tutorial: [Regular Expression Operations](https://docs.python.org/3/library/re.html)
#      - Reference: Python Software Foundation. "Regular Expression Operations." *Python 3 Documentation*. Latest version. Python Software Foundation.

# 2. **NLTK (Natural Language Toolkit)**
#    - Tutorial: [NLTK Documentation](https://www.nltk.org/)
#      - Reference: Steven Bird, Edward Loper, and Ewan Klein. "Natural Language Processing with Python." *NLTK Documentation*. Latest version. NLTK Project.

# 3. **Scikit-learn**
#    - Tutorial: [Feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
#      - Reference: Pedregosa et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825-2830, 2011.

# 4. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
#    - Tutorial: [VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text](https://github.com/cjhutto/vaderSentiment)
#      - Reference: Hutto, C.J. & Gilbert, E.E. (2014). "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." *Eighth International Conference on Weblogs and Social Media (ICWSM-14)*.

# 5. **TextBlob**
#    - Tutorial: [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)
#      - Reference: Steven Loria. "TextBlob: Simplified Text Processing." *TextBlob Documentation*. Latest version. TextBlob.

# 6. **Hugging Face Transformers**
#    - Tutorial: [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
#      - Reference: Hugging Face Inc. "Transformers: State-of-the-Art Natural Language Processing." *Hugging Face Transformers Documentation*. Latest version. Hugging Face Inc.





import re  
import nltk  
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize  
from nltk.sentiment.vader import SentimentIntensityAnalyzer  
from textblob import TextBlob  
import string  # Importing string module for string operations
from transformers import BertTokenizer, BertForSequenceClassification, pipeline  # Importing BERT-based sentiment analysis tools

# Loading FinBERT model and tokenizer
finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

# Creating a pipeline for sentiment analysis using FinBERT
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

# Downloading required nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text data
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in set(stopwords.words("english"))]  # Remove stopwords
    
    lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
    porter = PorterStemmer()  # Initialize stemmer
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    processed_text = " ".join(tokens)  # Join tokens into a single string

    return processed_text

# Function to get sentiment label using FinBERT
def get_finbert_label(text):
    try:
        return nlp(text)[0]["label"]  # Get sentiment label using FinBERT
    except:
        return "NULL"

sid = SentimentIntensityAnalyzer()  # Initialize Vader sentiment analyzer

# Function to get sentiment label using Vader
def get_vader_label(text):
    score = sid.polarity_scores(text)["compound"]  # Get sentiment score using Vader
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

# Function to get sentiment score using Vader
def get_vader_sentiment_score(text):
    return sid.polarity_scores(text)["compound"]  # Get sentiment score using Vader

# Function to get sentiment label using TextBlob
def get_textblob_label(text):
    analysis = TextBlob(text)  # Perform sentiment analysis using TextBlob
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"
