import re 
import pandas as pd
import numpy as np
import sklearn.feature_extraction.text as ft 

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from src.preprocessing.ctfidf import CTFIDFVectorizer


def expand_contractions(string: str) -> str:
    """
    Expand common contractions (I've -> I have) based on predefined mapping. 
        (subset of https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions)
    The function expects lowercase input, but should also work with upper/mixed case.
    There could be other expansions (i.e. he's -> he is/ he does/ he has), but only one is assumed here.
    Arguments:
        string - string to expand
    Returns:
        expanded - expanded string
    """
    contractions = { 
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that had",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    for cont, expnd in contractions.items():
        string = re.sub(r'\b' + cont + r'\b', expnd, string, flags=re.IGNORECASE)

    return string

def tokenize(string: str) -> list[str]:
    """
    Tokenize string into a list of tokens. Used for further preprocessing.
    """
    tokens = word_tokenize(string)
    return tokens 

def remove_punctuation(tokens: list[str]) -> list[str]:
    """
    Remove punctuation marks from tokens list.
    """
    return [token for token in tokens if token.isalpha()]

def lemmatize(tokens: list[str]) -> list[str]:
    """
    Lemmatize tokens to reduce dictionary size.
    """
    lemmatizer = WordNetLemmatizer() #with no additional info, it assumes all tokens to be nouns TODO: evaluate simpler stemming approaches
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens 

def normalize_text(string: str) -> str:
    """
    Normalize string by putting it to lowercase, expanding contractions, tokenizing, 
    removing punctuation, lemmatizing and joining back into a string for sklearn methods.
    """
    string = string.lower()
    expanded = expand_contractions(string)
    tokens = tokenize(expanded)
    tokens = remove_punctuation(tokens)
    lemmas = lemmatize(tokens)
    normalized = ' '.join(lemmas)
    return normalized


def get_n_most_important_words_cftfidf(df: pd.DataFrame, n_words: int) -> dict[str, list[str]]:
    """
    Get n_words most important words per each class according to cftfidf.
    Arguments:
        df - preprocessed dataframe that has "Text" of intereset and "Label" for each class
        n_words - number of words to pick
    Returns:
        words_per_class - dictionary with list of n_words for each class
    """
    text_per_class = df.groupby(['Label'], as_index=False).agg({'Text': ' '.join})
    count_vectorizer = ft.CountVectorizer().fit(text_per_class['Text'])
    count = count_vectorizer.transform(text_per_class['Text'])
    words = count_vectorizer.get_feature_names_out()
    print(f'{len(words)} unique words.')

    ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=df.shape[0]).toarray()

    labels = text_per_class['Label'].unique()

    words_per_class = {label: [words[index] for index in ctfidf[label].argsort()[-n_words:]] for label in labels}

    return words_per_class

def filter_unimportant_words(df: pd.DataFrame, important_words: dict[str, list[str]]) -> pd.DataFrame:
    """
    Filter out words that are not included in important_words dict.
    Arguments:
        df - preprocessed dataframe
        important_words - dictionary with important words per class
    Returns:
        filtered_df - dataframe with filtered out unimportant words
    """

    def _filter_words(text_string: str) -> str:
        word_list = text_string.split(' ')
        word_list = [word for word in word_list if word in all_important_words]
        return ' '.join(word_list)

    all_important_words = []
    for _, words in important_words.items():
        all_important_words.extend(words)

    df['Text'] = df['Text'].apply(_filter_words)

    return df


