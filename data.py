""" module for data loading and cleaning
Several functions were adopted from
https://towardsdatascience.com/elegant-s-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb8

"""

import os
import pandas as pd

# For s processing
import re # regular expression handling
from bs4 import BeautifulSoup # to remove html tags
from textblob import TextBlob # for spell checking
from unidecode import unidecode # to remove accents
import contractions # to expand contractions
import string

class DataCleaner:
    pass
    

def remove_html_tags(s):
    return BeautifulSoup(s, 'html.parser').get_text()

def replace_diacritics(s):
    return unidecode(s, errors="preserve")

# def to_lower(s):
#     s.X = np.apply_along_axis(lambda x: x.lower(), s.X)
#     

def expand_contractions(s):
    s = " ".join([contractions.fix(w) for w in s.split()])
    s = s.replace("n't", " not")
    return s

def remove_numbers(s):
    return re.sub(r'\d+', '', s)

def replace_dots_with_spaces(s):
    return re.sub("[.]", " ", s)

def remove_punctuations_except_periods(s):
    return re.sub('[%s]' %re.escape(s.remove_punctuations), '' , s)

def remove_all_punctuations(s):
    return re.sub('[%s]' %re.escape(string.punctuation), '' , s)

def remove_double_spaces(s):
    return re.sub(' +', ' ', s)

def fix_typos(s):
    return str(TextBlob(s).correct())

def remove_repeating_chars(s):
    # remove alphabetical characters repeating at least 3 times
    s = re.sub(r'([A-Z])\1\1+', r'\1', s)
    s = re.sub(r'([a-z])\1\1+', r'\1', s)
    # remove symbols
    s = re.sub(r'([[-`])\1', '', s)
    s = re.sub(r'(["-/])\1', '', s)
    s = re.sub(r'([:->])\1', '', s)
    # remove isolated characters repeated 2 times 
    s = re.sub(r" ([A-z])\1+ ", '', s)
    return s
# def remove_stopwords(s):
#     # remove stop words from token list in each column
#     lambda s: " ".join([ word for word in s.split() if word not in s.sw_nltk]) )

# def lemmatize(s):
#     lemmatizer = WordNetLemmatizer()
#     return lemmatize_pos_tagged_text(
#                             x, lemmatizer, s.post_tag_dict))
    

# def detect_quote(string):
#     quote_starters = ["\"", "`", "'"]
#     return any([string.startswith(c) for c in quote_starters])

def remove_proper_names(text):
    text_split = text.split()
    for i, s in enumerate(text_split[:-1]):
        first, second = s, text_split[i+1]
        if all([s[0].isupper() and s[-1:].islower() for s in [first,second]]):
            text_split[i] = text_split[i+1] = ""
    return " ".join(text_split)

def clean_data(df, cleaning_steps):
    """_summary_

    Parameters
    ----------
    df : pandas.core.series.Series

        Series of texts
    cleaning_list : list
        a list of functions with string input and ouput

    Returns
    -------
    pandas.core.series.Series
        Series of string after treatment
    """
    # TODO see if automatically removed via feature selection or feature effect etc
    # is there an advantage to compose all functions and then use map one time ?
    for f in cleaning_steps:
        df["review"] = df["review"].map(f)

    return df

def load_clean_data(raw_file_path, clean_file_path):
    if os.path.exists(clean_file_path):
        df = pd.read_csv(clean_file_path, sep="\t")
    else:
        df = pd.read_csv(raw_file_path, sep="\t")
        df = clean_data(
            df,
            cleaning_steps = [
                remove_html_tags,
                replace_dots_with_spaces,
                remove_double_spaces,
                remove_repeating_chars,
                replace_diacritics,
                expand_contractions,
                remove_numbers,
                # fix_typos, # takes a lot of time
            ])
        if "sentiment" not in df.columns:
            df["sentiment"] = df["id"].map(lambda s: int(int(s.split("_")[-1]) >= 7))
        df.to_csv(clean_file_path, sep = "\t", index= False)
    return df



# class NltkPreprocessingSteps:
#     """
#     Adopter from
#     https://towardsdatascience.com/elegant-s-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb8
#     """
#     def __init__(self, X):
#         self.X = X
#         download_if_non_existent('~/nltk_data/corpora/stopwords', 'stopwords')
#         download_if_non_existent('~/nltk_data/tokenizers/punkt', 'punkt')
#         download_if_non_existent('~/nltk_data/taggers/averaged_perceptron_tagger',
#                                     'averaged_perceptron_tagger')
#         download_if_non_existent('~/nltk_data/corpora/wordnet', 'wordnet')
#         download_if_non_existent('~/nltk_data/corpora/omw-1.4', 'omw-1.4')

#         self.sw_nltk = stopwords.words('english')
#         new_stopwords = ['<*>']
#         self.sw_nltk.extend(new_stopwords)
#         self.sw_nltk.remove('not')

#         self.pos_tag_dict = {"J": wordnet.ADJ,
#                         "N": wordnet.NOUN,
#                         "V": wordnet.VERB,
#                         "R": wordnet.ADV}

#         # '!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~' 32 punctuations in python
#         # we dont want to replace . first time around
#         self.remove_punctuations = string.punctuation.replace('.','')

#     def remove_html_tags(self):
#         self.X = self.X.apply(
#                 lambda x: BeautifulSoup(x, 'html.parser').get_text())
#         return self

#     def replace_diacritics(self):
#         self.X = self.X.apply(
#                 lambda x: unidecode(x, errors="preserve"))
#         return self

#     def to_lower(self):
#         self.X = np.apply_along_axis(lambda x: x.lower(), self.X)
#         return self

#     def expand_contractions(self):
#         self.X = self.X.apply(
#                 lambda x: " ".join([contractions.fix(expanded_word) 
#                             for expanded_word in x.split()]))
#         return self

#     def remove_numbers(self):
#         self.X = self.X.apply(lambda x: re.sub(r'\d+', '', x))
#         return self

#     def replace_dots_with_spaces(self):
#         self.X = self.X.apply(lambda x: re.sub("[.]", " ", x))
#         return self

#     def remove_punctuations_except_periods(self):
#         self.X = self.X.apply(
#                         lambda x: re.sub('[%s]' %
#                         re.escape(self.remove_punctuations), '' , x))
#         return self

#     def remove_all_punctuations(self):
#         self.X = self.X.apply(lambda x: re.sub('[%s]' %re.escape(string.punctuation), '' , x))
#         return self

#     def remove_double_spaces(self):
#         self.X = self.X.apply(lambda x: re.sub(' +', ' ', x))
#         return self

#     def fix_typos(self):
#         self.X = self.X.apply(lambda x: str(TextBlob(x).correct()))
#         return self

#     def remove_stopwords(self):
#         # remove stop words from token list in each column
#         self.X = self.X.apply(
#                 lambda x: " ".join([ word for word in x.split() 
#                             if word not in self.sw_nltk]) )
#         return self

#     def lemmatize(self):
#         lemmatizer = WordNetLemmatizer()
#         self.X = self.X.apply(lambda x: lemmatize_pos_tagged_text(
#                                 x, lemmatizer, self.post_tag_dict))
#         return self

#     def get_processed_text(self):
#         return self.X

def df_sample_stratified(df, stratify, **kwargs):
    """
        Generate a sample from a dataframe, with same proportions given a column
        specified by stratify
    """
    to_concat = []
    for val in df[stratify].unique():
        to_concat.append(df[df[stratify] == val].sample(**kwargs))

    return pd.concat(to_concat, axis= 0)