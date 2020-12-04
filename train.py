import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from joblib import dump

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--training_data_file', type=str, default='train.json')
    args = parser.parse_args()

    training_data_file_name = args.training_data_file
    training_data_file = open(training_data_file_name, "r")
    training_data = json.load(training_data_file)
    X, y = [], []
    for line in training_data:
        X.append(line['data'])
        y.append(line['label'])

    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=False, max_df=0.375, ngram_range=(1, 2), sublinear_tf=True)),
        ('clf', OneVsRestClassifier(LinearSVC(C=1.175, class_weight='balanced')))])
    text_clf = text_clf.fit(X, y)
    dump(text_clf, 'model.pkl')
