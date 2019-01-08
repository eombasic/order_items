import sys
import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from keras.models import model_from_json
from helpers import dataframe_helper
import json

args = sys.argv
response = {}

data_source_url = args[1]
dir_path = os.path.dirname(os.path.realpath(__file__))

dataFrameHelper = dataframe_helper.DataframeHelper()

columns_to_read = dataFrameHelper.getusedcols()
dataset = pd.read_csv(data_source_url, usecols=columns_to_read, converters={'delivery_address_zip': str})

dataset.dropna(subset=['requested_delivery_date'], inplace=True)
dataset.drop('requested_delivery_date', axis=1, inplace=True)

dataFrameHelper.fillNaVals(dataset, False)
dataFrameHelper.add_country_distances(dataset)
dataFrameHelper.add_eu_flag_customer(dataset)
dataFrameHelper.add_eu_flag_supplier(dataset)
dataFrameHelper.add_us_flag_customer(dataset)
dataFrameHelper.add_us_flag_supplier(dataset)
dataFrameHelper.add_asia_flag_customer(dataset)
dataFrameHelper.add_asia_flag_supplier(dataset)
dataset['deviation_type'] = 0
dataFrameHelper.set_deviation_types(dataset)
dataset.drop('delivery_deviation_in_days', axis=1, inplace=True)
dataset.drop('late', axis=1, inplace=True)
dataset.drop('early', axis=1, inplace=True)
dataset.drop('on_time', axis=1, inplace=True)

columns_to_encode = dataFrameHelper.getLabelCols()

for col_name_to_label in columns_to_encode:
    label_file = dir_path + '/classificator/' + col_name_to_label + '.npy'
    encoder_1 = joblib.load(label_file)
    dataset[col_name_to_label] = encoder_1.transform(dataset[col_name_to_label])


nr_of_cols = len(dataset.columns) - 1

X = dataset.iloc[:, 0:nr_of_cols].values
Y = dataset.iloc[:, nr_of_cols].values

encoder = joblib.load(dir_path + '/classificator/y_encoder.npy')
y1 = encoder.transform(Y)
encoder_classes = encoder.classes_

le_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
# print(le_name_mapping)


# load json and create model
json_file = open(dir_path + '/classificator/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(dir_path + "/classificator/model.h5")
# print("Loaded model from disk")

predictions = loaded_model.predict(X)

results = []

m,n = predictions.shape
for i in range(1,m):
    prediction_label = predictions[i]
    # print(predictions[i])
    max_val_key = np.argmax(prediction_label, axis=0)
    prediction_label = encoder_classes[max_val_key]

    res = {
        'prediction_label' : prediction_label,
        'probabilities' : predictions[i].tolist()
    }
    results.append(res)

response['results'] = results
response['encoder_classes'] = encoder_classes.tolist()
json_response = json.dumps(response)
print(json_response)
# print('lala')
