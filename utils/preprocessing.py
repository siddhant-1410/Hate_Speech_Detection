# utils/preprocessing.py
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Use local nltk_data
nltk.data.path.append('./nltk_data')

# Setup
stop_words = set(stopwords.words('english'))
stop_words.add("rt")
lemmatizer = WordNetLemmatizer()

def remove_entity(text):
    return re.sub(r"&[^\s;]+;", "", text)

def change_user(text):
    return re.sub(r"@([^ ]+)", "user", text)

def remove_url(text):
    url_regex = r"http\S+|www\S+|https\S+"
    return re.sub(url_regex, '', text)

def remove_noise_symbols(text):
    return re.sub(r'[^\w\s]', '', text)

def lemmatize_token(token):
    return lemmatizer.lemmatize(token, pos='v')

def preprocess_text(text):
    text = change_user(text)
    text = remove_entity(text)
    text = remove_url(text)
    text = remove_noise_symbols(text)
    
    tokens = word_tokenize(text)
    filtered = [lemmatize_token(w.lower()) for w in tokens if w.lower() not in stop_words and w.isalpha()]
    
    return " ".join(filtered)
