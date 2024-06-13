# %%
from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV
from sklearn import linear_model
import pandas as pd
import joblib
from itertools import chain
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import utils, postprocessing, data, tuning

run_cross_valid = True
load_grid_results = False


# %%  # TODO put in a function
df_train = data.load_clean_data(
    "./input/labeledTrainData.tsv", "./input/labeledTrainDataClean.tsv")

df_train_sub = data.df_sample_stratified(
    df_train, stratify="sentiment", frac=0.4, random_state=0)
# df_train_sub = data.²clean_data(df_train_sub, cleaning_steps= [data.fix_typos])


# %%

# %%
model = Pipeline([
    # ("preproc", preprocessor),
    ("tfidf", text.TfidfVectorizer(use_idf=True, ngram_range=(1, 2),
                                   min_df=0.0, max_df=1.0,
                                   stop_words=utils.sw_nltk)),
    # ("clf", linear_model.SGDClassifier(alpha=1e-5, penalty='l1', loss= 'log_loss',
    #                                    early_stopping= True, validation_fraction= 0.2)),
    # ("clf", RandomForestClassifier(n_estimators= 100, 
    #                                max_samples= 0.2, n_jobs= -1, verbose= 1)),
    ("clf", GradientBoostingClassifier(learning_rate= 1e-2, n_estimators= 1000, 
                                       max_depth= 1, max_features= "sqrt",
                                       verbose = 1)),
],
    memory=joblib.Memory(location="cachedir", verbose=0))


if run_cross_valid:

    model = utils.replace_head(model, ("clf", "passthrough"))

    grid_searcher = HalvingGridSearchCV(#GridSearchCV(
        estimator=model,
        param_grid=[
            {
                'tfidf__ngram_range': [(1, 1), (1, 2)],

                "clf": list(chain.from_iterable(tuning.model_on_param_grid(*(model, grid)) for model, grid in 
                tuning.models_param_grid_zip)),
               
            },

        ],
        cv=5, n_jobs=-1, verbose=1,
        scoring='accuracy',
        return_train_score=True,
        refit=False,
    )
    grid_searcher.fit(df_train_sub["review"], df_train_sub["sentiment"])

    utils.save_obj(grid_searcher, "grid_search")
    cv_res = pd.DataFrame(grid_searcher.cv_results_)
    utils.save_obj(cv_res, "cv_results")
elif load_grid_results:
    # joblib.load("./cv_results/")
    cv_res = pd.read_csv("./cv_results/cv_results_2024-01-2210:48:59.csv")
    
    

# %%


if run_cross_valid or load_grid_results:
    cv_res_short = utils.shorten_cv_results(cv_res)
    ind_best = tuning.choose_best_model(cv_res_short, prefer_low_gap=False, thd=0.8)
    model = tuning.model_set_params(model, cv_res, ind_best)



# %%
model.fit(df_train["review"],df_train["sentiment"])

#%%
feature_effects= postprocessing.compute_feature_effects(model, df_train["review"], k= 20)

#%%
pd.Series(feature_effects).plot.barh()
plt.show()
disp, report = postprocessing.metric_analysis(model, df_train["review"], df_train["sentiment"])
print(report)

# %%
df_test = data.load_clean_data(
    "./input/testData.tsv", "./input/testDataClean.tsv")

feature_effects= postprocessing.compute_feature_effects(model, df_test["review"], k= 20)

#%%
pd.Series(feature_effects).plot.barh()
plt.show()
disp, report = postprocessing.metric_analysis(model, df_test["review"], df_test["sentiment"])
print(report)

# %%
