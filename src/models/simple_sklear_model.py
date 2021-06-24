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
    con_m = confusion_matrix(actual, pred)
    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred)
    return con_m, accuracy, f1


if __name__ == "__main__":

    """
    MLFlow tracking - more details:
    https://www.mlflow.org/docs/latest/tracking.html
    
    Logs to directory on local machine. There is possibility to tract to server.
    
    Log parameters:
    1. mlflow.log_params(dict) - log all parameters stored in dictionary
    2. mlflow.log_param(key, value) - log each parameter separately, input key and value as string 
    
    Log metrics:
    1. mlflow.log_metric(key, value) - log each metric separately, input key and value as string 
    2. mlflow.log_metrics(dict) - log all metrics stored in dictionary
    
    Log model:
    mlflow.sklearn.log_model(model, "model_name") - log model stored as .pkl file
    
    Autolog:
    mlflow.sklearn.autolog()
    """

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

        mlflow.log_params(tfidf_params)

        # Log metrics
        mlflow.log_metric("accuracy", a)
        mlflow.log_metric("f1", f)

        t_n, f_p, f_n, t_p = cm.ravel()
        mlflow.log_metric("tn", t_n)
        mlflow.log_metric("fp", f_p)
        mlflow.log_metric("fn", f_n)
        mlflow.log_metric("tp", t_p)

