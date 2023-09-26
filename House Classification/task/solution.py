import os
import requests
import sys
import pandas as pd


def download_file():
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")


def print_summary(data: pd.DataFrame):
    print(data.shape[0])
    print(data.shape[1])
    print(data.isna().any().any())
    print(max(data["Room"]))
    print(data["Area"].mean())
    print(len(data["Zip_loc"].unique()))


def main():
    download_file()
    data = pd.read_csv("../Data/house_class.csv")
    print_summary(data)


if __name__ == '__main__':
    main()
