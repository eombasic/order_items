import sys
import os
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from keras.models import model_from_json
from helpers import dataframe_helper
from helpers import response_helper

args = sys.argv
response = {
    'error': 'no'
}

arg_names = ['command', 'data_source_url']
args = dict(zip(arg_names, sys.argv))

responseHelper = response_helper.ResponseHelper()
responseHelper.setResponse(response)

is_data_source_url_set = 'data_source_url' in args

if not is_data_source_url_set:
    responseHelper.flushError('no_data_source_url')

data_source_url = args['data_source_url']

if not len(data_source_url) > 0:
    responseHelper.flushError('data_source_url_empty')

if not os.path.isfile(data_source_url):
    responseHelper.flushError('data_source_file_does_not_exist')

dir_path = os.path.dirname(os.path.realpath(__file__))

dataFrameHelper = dataframe_helper.DataframeHelper()

columns_to_read = dataFrameHelper.getusedcols()

try:
    dataset = pd.read_csv(data_source_url, usecols=columns_to_read, converters={'delivery_address_zip': str})
except:
    e = sys.exc_info()[0]
    responseHelper.flushError('pandas_error : {}'.format(e))

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
    try:
        encoder_1 = joblib.load(label_file)
    except:
        responseHelper.flushError('cant_load_col_label_npy: ' + label_file)
    dataset[col_name_to_label] = encoder_1.transform(dataset[col_name_to_label])

nr_of_cols = len(dataset.columns) - 1

X = dataset.iloc[:, 0:nr_of_cols].values
encoder = joblib.load(dir_path + '/classificator/y_encoder.npy')
encoder_classes = encoder.classes_

le_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

# load json and create model
json_file = open(dir_path + '/classificator/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(dir_path + "/classificator/model.h5")

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
responseHelper.flush()
