import argparse
import json
from joblib import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--test_data_file', type=str, default='testdataexample')
    parser.add_argument('-m', '--model_file', type=str, default='model.pkl')
    args = parser.parse_args()

    test_data_file_name = args.test_data_file
    test_data_file = json.load(test_data_file_name)
    X_test = []
    for line in test_data_file:
        X_test.append(line)

    model_file_name = args.model_file
    model = load(model_file_name)

    predicted = model.predict(X_test)
    for line in predicted:
        print(line)
