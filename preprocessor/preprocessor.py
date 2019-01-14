__author__ = "Emir Ombasic"

import pandas as pd

data_source_url = '../data/data_raw.csv'
export_url = '../data/data.csv'

dataset = pd.read_csv(data_source_url)
print(dataset.info())

dataset.drop_duplicates(keep='first', inplace=True)
print(dataset.info())

dataset.to_csv(export_url, index=False)