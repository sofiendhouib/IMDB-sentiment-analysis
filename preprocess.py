""" module for data loading and cleaning
Several functions were adopted from
https://towardsdatascience.com/elegant-s-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb8

"""

import os
import pandas as pd

# For s processing
import re # regular expression handling
from bs4 import BeautifulSoup # to remove html tags
from textblob import TextBlob # for spell checking, deprecated, for example "corrects" dedication to education
from spellchecker import SpellChecker
from unidecode import unidecode # to remove accents
import contractions # to expand contractions
import string
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet') # to use the WordNet Lemmatizer and for stopwords
nltk.download('punkt') # to use the word tokenizer
nltk.download('averaged_perceptron_tagger') 

def remove_html_tags(s):
    return BeautifulSoup(s, 'html.parser').get_text()

def replace_diacritics(s):
    return unidecode(s, errors="preserve")

def remove_numbers(s):
    return re.sub(r'\d+', '', s)

def replace_dots_with_spaces(s):
    return s.replace(".", " ")

punctuation_except_period = string.punctuation.replace(".","")
def remove_punctuations_except_periods(s):
    return re.sub(r"[{%s}]"%punctuation_except_period,"",s)

def remove_all_punctuations(s):
    return re.sub('[%s]' %re.escape(string.punctuation), '' , s)

def remove_double_spaces(s):
    return re.sub(r'\s+', ' ', s)

spell = SpellChecker()

def correct_or_let(w):
    res = spell.correction(w)
    if res is None: 
        return w
    else:
        return res
def fix_typos(s):
    #return TextBlob(s).correct().string
    corrected_words = [correct_or_let(w) for w in s.split()]
    return " ".join(corrected_words)

def remove_repeating_chars(s):
    # remove alphabetical characters repeating at least 3 times
    s = re.sub(r"(.)\1{3,}", r"\1\1", s) # limit sequences of repeated characters to 2
    s = re.sub(r"\b(.)\1{2,}\b", "", s)
    return s

lemmatizer = nltk.WordNetLemmatizer()
pos2wornet_dict = {
                    "J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV
                }

def lemmatize_text(s):
    token_pos_tuples = nltk.pos_tag(nltk.word_tokenize(s)) # (token, pos) tuples of the sentence
    token_lemmatized_list = [] # list of lemmatized words
    for token, pos_tag in token_pos_tuples:
        # obtain the tag compatible with wordnet, default to noun if unknown
        wordnet_tag = pos2wornet_dict.get(pos_tag[0], wordnet.NOUN) 
        # lemmatize the token using the obtained tag
        token_lemmatized = lemmatizer.lemmatize(token, wordnet_tag) 
        token_lemmatized_list.append(token_lemmatized) # update the list of lemmatized words
    return " ".join(token_lemmatized_list)

default_cleaning_steps = [
                remove_html_tags,
                replace_diacritics,
                contractions.fix, # contractions package can handle a whole sentence
                replace_dots_with_spaces,
                remove_numbers,
                lambda s: s.lower(),
                remove_repeating_chars,
                remove_double_spaces,
                fix_typos,
                remove_punctuations_except_periods,
                lemmatize_text,
                lambda s: s.replace(".", "") # remove period
            ]
def clean_text(s, cleaning_steps = default_cleaning_steps):
    res = s
    for f in cleaning_steps:
        res = f(res)
    return res

def load_clean_data(raw_file_path, clean_file_path):
    # test if the clean data has already been created
    if os.path.exists(clean_file_path):
        df = pd.read_csv(clean_file_path)
    else:
        # creating the clean data
        df = pd.read_csv(raw_file_path)
        df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x.startswith("p") else 0)
        df["review"] = df["review"].apply(clean_text)
       
        df.to_csv(clean_file_path)
    return df

def df_sample_stratified(df, stratify, **kwargs):
    """
        Generate a sample from a dataframe, with same proportions given a column
        specified by stratify
    """
    to_concat = []
    for val in df[stratify].unique():
        to_concat.append(df[df[stratify] == val].sample(**kwargs))

    return pd.concat(to_concat, axis= 0)