import pandas as pd
import numpy as np
from sklearn.externals import joblib
from keras.models import model_from_json
from helpers import dataframe_helper

data_source_url = '../data/data_test.csv'

dataFrameHelper = dataframe_helper.DataframeHelper()

columns_to_read = dataFrameHelper.getusedcols()
dataset = pd.read_csv(data_source_url, usecols=columns_to_read, converters={'delivery_address_zip': str})

dataset.drop_duplicates(keep='first', inplace=True)

dataset.dropna(subset=['requested_delivery_date'], inplace=True)
dataset.drop('requested_delivery_date', axis=1, inplace=True)

dataFrameHelper.fillNaVals(dataset)
dataFrameHelper.add_country_distances(dataset)
dataFrameHelper.add_eu_flag_customer(dataset)
dataFrameHelper.add_eu_flag_supplier(dataset)
dataFrameHelper.add_us_flag_customer(dataset)
dataFrameHelper.add_us_flag_supplier(dataset)
dataFrameHelper.add_asia_flag_customer(dataset)
dataFrameHelper.add_asia_flag_supplier(dataset)

cat_columns = dataFrameHelper.get_cat_columns()
dataset = pd.get_dummies(dataset, prefix_sep="__", columns=cat_columns)


cat_dummies = joblib.load('cat_dummies.npy')
print('loaded cat_dummies:')
print(cat_dummies)

# Remove additional columns
for col in dataset.columns:
    if ("__" in col) and (col.split("__")[0] in cat_columns) and col not in cat_dummies:
        print("Removing additional feature {}".format(col))
        dataset.drop(col, axis=1, inplace=True)

# Add missing columns
for col in cat_dummies:
    if col not in dataset.columns:
        print("Adding missing feature {}".format(col))
        dataset[col] = 0

processed_columns = joblib.load('processed_columns.npy')
print('loaded processed_columns')
print(processed_columns)

dataset['deviation_type'] = 0
dataFrameHelper.set_deviation_types(dataset)
dataset.drop('delivery_deviation_in_days', axis=1, inplace=True)
dataset.drop('late', axis=1, inplace=True)
dataset.drop('early', axis=1, inplace=True)
dataset.drop('on_time', axis=1, inplace=True)

columns_to_encode = dataFrameHelper.getLabelCols()

for col_name_to_label in columns_to_encode:
    encoder_1 = joblib.load(col_name_to_label + '.npy')
    dataset[col_name_to_label] = encoder_1.transform(dataset[col_name_to_label])


dataset = dataset[processed_columns]

nr_of_cols = len(dataset.columns) - 1

X = dataset.iloc[:, 0:nr_of_cols].values
Y = dataset.iloc[:, nr_of_cols].values

encoder = joblib.load('y_encoder.npy')
y1 = encoder.transform(Y)
encoder_classes = encoder.classes_

le_name_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print(le_name_mapping)


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


predictions = loaded_model.predict(X)

# print(predictions)

total_predictions = 0
corect_predictions = 0

m,n = predictions.shape
for i in range(1,m):
    prediction_label = predictions[i]
    # print(predictions[i])
    max_val_key = np.argmax(prediction_label, axis=0)
    prediction_label = encoder_classes[max_val_key]
    actual_value = Y[i]

    total_predictions = total_predictions + 1
    not_valid_flag = '*'
    if actual_value == prediction_label:
        not_valid_flag = ''
        corect_predictions = corect_predictions + 1
    print('{}. actual value : {}, predicted value : {} {} {} '.format(i, actual_value, prediction_label, predictions[i], not_valid_flag))

percent = (corect_predictions / total_predictions) * 100
print('Predicted {} out of {} so : {} %'.format(corect_predictions, total_predictions, percent))
