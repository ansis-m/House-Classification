import os
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def download_file():
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")


def target_encode(X_train, X_test, y_train):
    columns_to_encode = ['Zip_area', 'Zip_loc', 'Room']

    encoder = TargetEncoder(cols=columns_to_encode)

    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()

    X_train_transformed = encoder.fit_transform(X_train_transformed, y_train)
    X_test_transformed = encoder.transform(X_test_transformed)

    return X_train_transformed, X_test_transformed


def print_summary(data: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(data[["Area", "Room", "Lon", "Lat", "Zip_area", "Zip_loc"]],
                                                        data["Price"], test_size=0.3, random_state=1,
                                                        stratify=data["Zip_loc"])

    X_train_transformed, X_test_transformed = target_encode(X_train.copy(), X_test.copy(), y_train)
    clf = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6, min_samples_split=4,
                                 random_state=3)

    clf.fit(X_train_transformed, y_train)
    y_pred = clf.predict(X_test_transformed)

    accuracy = accuracy_score(y_pred, y_test)
    print(accuracy)


def main():
    download_file()
    data = pd.read_csv("../Data/house_class.csv")
    print_summary(data)


if __name__ == '__main__':
    main()
