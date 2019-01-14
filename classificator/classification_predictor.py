__author__ = "Emir Ombasic"

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from sklearn.externals import joblib
import numpy as np

from helpers import dataframe_helper

data_source_url = '../data/data.csv'

dataFrameHelper = dataframe_helper.DataframeHelper()

columns_to_read = dataFrameHelper.getusedcols()
dataset = pd.read_csv(data_source_url, usecols=columns_to_read, converters={'delivery_address_zip': str})
# dataFrameHelper.printDataFrameInfo(dataset)

# dataFrameHelper.preprocess(dataset)

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
# dataset = pd.concat([dataset, dummies], axis=1)

dataset['deviation_type'] = 0
dataFrameHelper.set_deviation_types(dataset)

dataset.drop('delivery_deviation_in_days', axis=1, inplace=True)
dataset.drop('late', axis=1, inplace=True)
dataset.drop('early', axis=1, inplace=True)
dataset.drop('on_time', axis=1, inplace=True)

columns_to_encode = dataFrameHelper.getLabelCols()

for col_name_to_label in columns_to_encode:
    # print(col_name_to_label)
    encoder_commodity = LabelEncoder()
    dataset[col_name_to_label] = dataset[col_name_to_label].fillna('0')
    encoder_commodity.fit(dataset[col_name_to_label])
    dataset[col_name_to_label] = encoder_commodity.transform(dataset[col_name_to_label])
    joblib.dump(encoder_commodity, col_name_to_label + '.npy')




#drop cat cols
'''
for cat_col in cat_columns:
    dataset.drop(cat_col, axis=1, inplace=True)
'''

nr_of_cols = len(dataset.columns) - 1


cat_dummies = [col for col in dataset
               if "__" in col
               and col.split("__")[0] in cat_columns]

print('cat_dummies {}'.format(cat_dummies))
joblib.dump(cat_dummies, 'cat_dummies.npy')

processed_columns = list(dataset.columns[:])
print('processed columns {}'.format(processed_columns))
joblib.dump(processed_columns, 'processed_columns.npy')

X = dataset.iloc[:, 0:nr_of_cols].values
y = dataset.iloc[:, nr_of_cols].values


encoder = LabelEncoder()
encoder.fit(y)
y1 = encoder.transform(y)
Y = pd.get_dummies(y1).values
joblib.dump(encoder, 'y_encoder.npy')

deviation_type_value_counts = dataset['deviation_type'].value_counts()

on_time_count = deviation_type_value_counts['on_time']
print('on time count {} '.format(on_time_count))
late_count = deviation_type_value_counts['late']
print('late count {} '.format(late_count))
total_count = len(dataset.index)
print('total count {} '.format(total_count))

late_ratio = late_count / total_count
on_time_ratio = on_time_count / total_count

late_weight = on_time_ratio
print('late weight : {}'.format(late_weight))
on_time_weight = late_ratio
print('on time weight : {}'.format(on_time_weight))

res = late_weight*late_count
print(res)
res = on_time_weight*on_time_count
print(res)
# exit()

# on_time_weight = 100 - (deviation_type_value_counts['on_time']*100)
# late_weight = 100 - (deviation_type_value_counts['late']*100)

on_time_index = np.where(encoder.classes_ == 'on_time')
late_index = np.where(encoder.classes_ == 'late')

class_weight = {on_time_index[0][0]: on_time_weight * 100,
                late_index[0][0]: late_weight * 100}


print("class weight {} ".format(class_weight))

model = Sequential()
model.add(Dense(nr_of_cols,input_shape=(nr_of_cols,),activation='relu', name='features'))
model.add(Dense(213,activation='relu'))
model.add(Dense(113,activation='relu'))
model.add(Dense(53,activation='relu'))
model.add(Dense(13,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(2,activation='sigmoid'))
model.compile(Adam(lr=0.0001),'binary_crossentropy',metrics=['accuracy', 'binary_crossentropy'])
model.fit(X, Y, epochs=20, batch_size=500, class_weight=class_weight)
model.summary()


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
