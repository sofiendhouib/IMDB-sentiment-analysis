import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
import pandas as pd

def compute_feature_effects(model, X_raw, k= 5, average= True):
    """
    Initially adopted from 
    https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html
    with modifications
    """
    # TODO works for 2 classes and positive features, but must be modified otherwise
    # and specialized per class
    # learned coefficients weighted by frequency of appearance
    clf = model.steps[-1][1]
    if   hasattr(clf, "coef_"):
        feature_effects = clf.coef_.flatten()
    elif hasattr(clf, "feature_importances_"):
        feature_effects = clf.feature_importances_
    elif hasattr(clf, "feature_log_prob_"):
        feature_effects = np.dot([-1,1], clf.feature_log_prob_)


    if average:
        transformer = Pipeline(steps= model.steps[:-1])
        X_train = transformer.transform(X_raw) 
        feature_effects *= np.asarray(X_train.mean(axis=0)).ravel()

    top_k_inds = np.argsort(np.abs(feature_effects))[-k:]#[::-1]
    
    feature_effects = feature_effects[top_k_inds]
    feature_names= model.steps[0][1].get_feature_names_out()
    predictive_tokens = [feature_names[i] for i in  top_k_inds]

    return dict(zip(predictive_tokens, feature_effects))

def metric_analysis(model, X_raw, y_true):
    """
        plot a confusion matrix and generate a classification report
    """
    y_train_pred = model.predict(X_raw)
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_train_pred, 
                                                normalize= "all", values_format= ".2%")
    
    report = metrics.classification_report(y_true, y_train_pred, output_dict= True)
    return disp, pd.DataFrame(report)

def results_analysis(model, X_raw, y_true, k= 20):
    feature_effects= compute_feature_effects(model, X_raw, k= k)
    pd.Series(feature_effects).plot.barh()
    disp, report = metric_analysis(model, X_raw, y_true)
    return disp, report, feature_effects


class postprocessor():
    # must store a view on the transformed data
    
    pass