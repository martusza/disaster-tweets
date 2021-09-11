import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import click
from hyperopt import hp, tpe, Trials, space_eval, fmin, STATUS_OK, space_eval

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

seed = 42
np.random.seed(seed)

classifiers = {
    "random_forest": RandomForestClassifier,
    # "sgd_classifier": SGDClassifier,
    # "logistic_regression": LogisticRegression,
    # "svc": SVC,
}

hyper_space = {
    "random_forest": {
        "n_estimators": hp.choice("n_estimators", np.arange(50, 100)),
        "max_depth": hp.choice("max_depth", np.arange(10,50)),
        "min_samples_leaf": hp.uniform("min_samples_leaf", 0.01, 0.1),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2"]),
        "max_leaf_nodes": hp.choice("max_leaf_nodes", list(np.arange(10,30)) + [None]),
        # "n_jobs": hp.choice("n_jobs", [2]),
        # "class_weight": hp.choice("class_weight", ["balanced", "balanced_subsample"]),
        "random_state": hp.choice("random_state", [seed])
    }
}


@click.command()
@click.option('--classifier', type=str, help="Classifier name")
@click.option('--train-table-name', type=str, help="Training table name")
@click.option('--test-table-name', type=str, help="Test table name")
@click.option('--target-column', type=str, help="Target column name")
@click.option('--output-table-name', type=str, help="Predictions test table name")
@click.option('--num-iter', type=int, help="Number of iterations", default=1)
def main(classifier,
         train_table_name,
         test_table_name,
         target_column,
         output_table_name,
         num_iter,
         ):

    # Recognizing data type input
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

    df_train.fillna(0, inplace=True)
    y_train = df_train[target_column]
    X_train = df_train.drop([target_column], axis=1)

    df_test.fillna(0, inplace=True)
    y_test = df_test[target_column]
    X_test = df_test.drop([target_column], axis=1)

    clf = classifiers[classifier]
    space = hyper_space[classifier]

    trials = Trials()

    # define objective function
    def hyperparameter_tuning(space):
        clf = classifiers[classifier](**space)
        clf.fit(X_train, y_train)

        # Calculating scores
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        train_score = f1_score(y_train, y_train_pred, average='macro')
        test_score = f1_score(y_test, y_test_pred, average="macro")
        print(
            "Calculating f1 metric.\nTrain score: {}\nTest score: {}".format(train_score, test_score)
        )
        return -test_score


    best = fmin(
            fn=hyperparameter_tuning,
            space=space,
            algo=tpe.suggest,
            max_evals=num_iter,
            trials=trials,
    )

    print("Best: {}".format(best))

    best_model = classifiers[classifier](**best)
    best_model.fit(X_train, y_train)
    X_test['predicted'] = best_model.predict(X_test)
    X_test.to_csv(output_table_name)


if __name__ == "__main__":
    main()