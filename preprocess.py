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
from nltk.corpus import wordnet, stopwords
nltk.download('wordnet') # to use the WordNet Lemmatizer and for stopwords
nltk.download('punkt') # to use the word tokenizernltk.download('averaged_perceptron_tagger') 
nltk.download('punkt_tab')
stop_words = stopwords.words('english')

for w in ["no", "not", "while", "nor", "too"]:
    stop_words.remove(w)
for w in stop_words:
    if w.endswith("n't"): stop_words.remove(w)
stop_words = set(stop_words)

html_tags_re = re.compile(r'<.*?>')
numbers_re = re.compile(r'\d+')

#def remove_html_tags(s):
    #return BeautifulSoup(s, 'html.parser').get_text()
    # return html_tags_re.sub(" ", s)

def remove_control_chars(s):
    # XXX solution provided by following link does not work for \x85
    "https://www.geeksforgeeks.org/python-program-to-remove-all-control-characters/"
    return re.sub(r"\W", " ", s) # kind of a radical solution

def replace_diacritics(s):
    # XXX can remove control characters such as \x97
    # s = re.sub(r"\\x\d+", " ", r"%s"%s) # replaces control characters with space
    return unidecode(s, errors="preserve")

# def remove_numbers(s):
#     return numbers_re.sub(" ", s)

# def replace_dots_with_spaces(s):
#     return s.replace(".", " ")

punctuation_except_period = string.punctuation.replace(".","")
punctuation_except_period_re = re.compile(f"[{re.escape(punctuation_except_period)}]")
# def remove_punctuation_except_period(s):
    # XXX Replace with space otherwise some words will be attached irreversibly
    # example: online-sources --> onlinesources
    # return punctuation_except_period_re(" ", s)

def remove_antislash_x_characters(s):
    return re.sub(r"\x\d+")

def remove_all_punctuations(s):
    return re.sub('[%s]' %re.escape(string.punctuation), ' ' , s)

double_spaces_re = re.compile(r'\s+')
# def remove_double_spaces(s):
#     return double_spaces_re.sub(' ', s)

spell_checker = SpellChecker()

def correct_or_let(w):
    res = spell_checker.correction(w)
    # if res is None: 
    #     return w
    # else:
    #     return res
    return w if res is None else res
    
def fix_typos(s, unknown_words):
    #return TextBlob(s).correct().string
    corrected_words = []
    for w in s.split: # might be accelerated by correcting the unknown words and creating a dictionary
        if w in unknown_words:
            corr_res = spell_checker.correction(w)
            corrected_word = w if corr_res is None else corr_res
        else: corrected_word= w
        corrected_word.append()
    return " ".join(corrected_words)
repeated_at_least_3_re = re.compile(r"(.)\1{3,}")
repeated_isolated_re = re.compile(r"\b(.)\1{2,}\b")
# def remove_repeating_chars(s):
#     # remove alphabetical characters repeating at least 3 times
#     s = repeated_at_least_3_re.sub(r"\1\1", s) # limit sequences of repeated characters to 2
#     s = repeated_isolated_re.sub(" ", s) # remove isolated repeated sequences
#     return s

isolated_characters_re = re.compile(r"\b[b-hj-z]\b")
# def remove_isolated_chars(s):
#     return isolated_characters_re.sub(" ",s)

lemmatizer = nltk.WordNetLemmatizer()
pos2wordnet_dict = {
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
        wordnet_tag = pos2wordnet_dict.get(pos_tag[0], wordnet.NOUN) 
        # lemmatize the token using the obtained tag
        token_lemmatized = lemmatizer.lemmatize(token, wordnet_tag) 
        token_lemmatized_list.append(token_lemmatized) # update the list of lemmatized words
    return " ".join(token_lemmatized_list)

proper_noun_re = re.compile(r"([^\.])[A-Z][a-z]+") # FIXME  Word starting by a capital letter, coming after a space
proper_noun_2_re = re.compile(r"([A-Z][a-z]+ *){2,}") # # succession or 2 or more words starting by an upper case letter followed by lower case letters

clean_steps_before_normalize = [
                lambda s: html_tags_re.sub(" ", s),#remove_html_tags,
                contractions.fix, # placed here so that apostrophes that remain are removed later
                remove_control_chars,
                replace_diacritics,
                #replace_dots_with_spaces,
                lambda s: numbers_re.sub(" ", s),#remove_numbers,
                lambda s: proper_noun_2_re.sub(" ", s),
                lambda s: proper_noun_re.sub(r"\1", s),
                lambda s: s.lower(),
                lambda s: punctuation_except_period_re.sub(" ", s),#remove_punctuation_except_period,
                lambda s: repeated_at_least_3_re.sub(r"\1\1", s),#remove_repeating_chars,
                lambda s: repeated_isolated_re.sub(" ", s),
                lambda s: isolated_characters_re.sub(" ", s),
                lambda s: double_spaces_re.sub(" ", s),#remove_double_spaces,
                ]

def clean_text(s, steps = clean_steps_before_normalize, 
               check_spell= True, normalize= None,
               rm_stop_words= True):
    steps = steps[:] # pass by value
    if rm_stop_words:
        steps.append(lambda s: " ".join([w for w in s.split() if not w in stop_words]))
    
    if check_spell:
        steps.append(fix_typos)
        #cleaning_steps.append(remove_punctuations_except_periods)
    
    if not normalize in {"lemmatize", "stem", None}:
        raise ValueError("normalize argument should be either 'lemmatize', 'stem' or None ") 
    if normalize == "lemmatize":
        steps.append(lemmatize_text)
    
    elif normalize == "stem":
        raise NotImplementedError("Stemming is not yet implemented")
    steps.append(lambda s: s.replace(".", " "))
    res = s
    for f in steps:
        res = f(res)
    return res

def load_clean_data(raw_file_path, clean_file_path):
    # test if the clean data has already been created
    if os.path.exists(clean_file_path):
        df = pd.read_csv(clean_file_path)
    else:
        # creating the clean data
        df = pd.read_csv(raw_file_path)
        df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x.startswith("p") else 0) #FIXME
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