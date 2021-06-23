import pandas as pd
from sklearn.pipeline import Pipeline
from src.data.preproces_dataset import TextCleanTransformer, DenseTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import mlflow
import mlflow.sklearn

df = pd.read_csv("data/raw/train.csv")

X, y = df['text'], df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def eval_metrics(actual, pred):
    c = confusion_matrix(actual, pred)
    a = accuracy_score(actual, pred)
    f = f1_score(actual, pred)
    return c, a, f


if __name__ == "__main__":

    with mlflow.start_run():
        tfidf_params = {'ngram_range': (1, 1),
                        'max_features': 3000}

        pipe = Pipeline([('clean', TextCleanTransformer()),
                         ('tfidf', TfidfVectorizer(**tfidf_params)),
                         ('dense', DenseTransformer()),
                         ('model', GaussianNB())])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        cm, a, f = eval_metrics(y_test, y_pred)
        for key in tfidf_params:
            mlflow.log_param(key, tfidf_params[key])

        # mlflow.log_metric("confusion", c)
        mlflow.log_metric("accuracy", a)
        mlflow.log_metric("f1", f)

        t_n, f_p, f_n, t_p = cm.ravel()
        mlflow.log_metric("tn", t_n)
        mlflow.log_metric("fp", f_p)
        mlflow.log_metric("fn", f_n)
        mlflow.log_metric("tp", t_p)

        mlflow.sklearn.log_model(pipe, "model")
