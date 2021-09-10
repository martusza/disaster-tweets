import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import click
from hyperopt import hp, tpe, Trials, space_eval, fmin, STATUS_OK

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

seed = 42
np.random.seed(seed)

classifiers = {
    "random_forest": RandomForestClassifier,
    "sgd_classifier": SGDClassifier,
    "logistic_regression": LogisticRegression,
    "svc": SVC,
}

hyper_space = {
    "random_forest": {
        "n_estimators": hp.choice("n_estimators", np.arange(50, 100)),
        "max_depth": hp.choice("max_depth", np.arange(10,50)),
        "min_samples_leaf": hp.loguniform("min_samples_leaf", 0.01, 0.1),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
        "max_leaf_nodes": hp.choice("max_leaf_nodes", list(np.arange(10,30)) + [None]),
        "n_jobs": hp.choice("n_jobs", [2]),
        "class_weight": hp.choice("class_weight", ["balanced", "balanced_subsample"]),
        "random_state": hp.choice("random_state", [seed])
    }
}

# define objective function
def hyperparameter_tuning(clf, X, y):
    search_params = hyper_space[clf]
    acc = cross_val_score(clf(**search_params), X, y, scoring="f1").mean()
    return {"loss": -acc, "status": STATUS_OK}


@click.option('--classifier', type=str, help="Classifier name")
@click.option('--train-table-name', type=str, help="Training table name")
@click.option('--test-table-name', type=str, help="Test table name")
@click.option('--target-column', type=str, help="Target column name")
@click.option('--output-table-name', type=str, help="Predictions test table name")
@click.option('--num-iter', typ=int, help="Number of iterations")
def main(classifier,
         train_table_name,
         test_table_name,
         target_column,
         output_table_name,
         ):
    data_type = Path(train_table_name).suffix
    if data_type == ".csv":
        df_train = pd.read_csv(train_table_name)
        df_test = pd.read_csv(test_table_name)
    elif data_type == ".pkl":
        df_train = joblib.load(train_table_name)
        df_test = joblib.load(test_table_name)
    else:
        print("Unrecognized file format")
        exit()

    X_train = df_train

    cls = classifiers[classifier]


if __name__ == "__main__":
    main()