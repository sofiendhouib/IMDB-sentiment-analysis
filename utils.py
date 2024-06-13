import nltk
from nltk.corpus import stopwords
from datetime import datetime
from joblib import dump

def download_if_non_existent(res_path, res_name):
    """
        Adopted from
        https://towardsdatascience.com/elegant-text-pre-processing-with-nltk-in-sklearn-pipeline-d6fe18b91eb8
    """
    try:
        nltk.data.find(res_path)
    except LookupError:
        print(f'resource {res_path} not found. Downloading now...')
        nltk.download(res_name)

download_if_non_existent('~/nltk_data/corpora/stopwords', 'stopwords')

sw_nltk = stopwords.words('english')
for w in ["no", "not", "while", "nor", "too"]:
    sw_nltk.remove(w)
for w in sw_nltk:
    if w.endswith("n't"): sw_nltk.remove(w)

def print_long(string_arg, line_length = 10):
    """ Given a long text or a list of words or ngrams, write the corresponding
    tests splitted into multiple lines. The list is assumed to be a splitted text.
    """
    if isinstance(string_arg, list):
        string_list = string_arg
    if isinstance(string_arg, str):
        string_list = string_arg.split(" ")
    line = [] # initizalize the line
    for i, s in enumerate(string_list):
        if i % line_length == 0:
            print(" ".join(line))
            line = [] # re_init
        line.append(s.split()[0])
    ngram_remain = string_list[i].split(" ")[1:]
    line.extend(ngram_remain)
    print(" ".join(line))
    #print(line) # last remaining portion
    return None



def shorten_cv_results(cv_results, sort_key= "test"):
    """ Produce a dataframe containing only essential metrics from the result 
    of a cross validation
    """
    assert sort_key in ["train", "test"]
    select_cols = [col for col in cv_results.columns if col.startswith("param_")]
    select_cols.extend(["mean_train_score", "mean_test_score"])
    cv_results["train_test_diff"] = (cv_results["mean_train_score"] - cv_results["mean_test_score"]).abs()
    select_cols.append("train_test_diff")
    return cv_results[select_cols].sort_values(by= f"mean_{sort_key}_score", ascending = False)


# TODO add directory and choose whether to add time
# TODO: strip the beyond second from time
def save_obj(obj, name):
    filename = f"{name}_{str(datetime.now()).replace(" ", "_")}.joblib"
    dump(obj, f"./cv_results/{filename}")
    return None

def replace_head(pipeline, step):
    pipeline.steps.pop()
    pipeline.steps.append(step)
    return pipeline

