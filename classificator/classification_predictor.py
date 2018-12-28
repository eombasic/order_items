import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from sklearn.externals import joblib

from helpers import dataframe_helper

data_source_url = 'data.csv'

dataFrameHelper = dataframe_helper.DataframeHelper()

columns_to_read = dataFrameHelper.getusedcols()
dataset = pd.read_csv(data_source_url, usecols=columns_to_read, converters={'delivery_address_zip': str})
# nr_of_cols = 21
# dataFrameHelper.printDataFrameInfo(dataset)



dataFrameHelper.preprocess(dataset)

'''
pairs = dataset.groupby(['supplier_country', 'delivery_address_country']).size().sort_values(ascending=False)
print(pairs)
print(dataset['delivery_address_country'].value_counts())
print(dataset['supplier_country'].value_counts())
exit()
'''


# correlations = dataset.corr()
# dataFrameHelper.printDataFrameInfo(dataset)

dataFrameHelper.move_df_sunday(dataset, 'deliver_on_day')
dataFrameHelper.move_df_sunday(dataset, 'order_created_on_day')

dataFrameHelper.add_country_distances(dataset)

dataFrameHelper.add_eu_flag_customer(dataset)
dataFrameHelper.add_eu_flag_supplier(dataset)

dataFrameHelper.add_us_flag_customer(dataset)
dataFrameHelper.add_us_flag_supplier(dataset)

dataFrameHelper.add_asia_flag_customer(dataset)
dataFrameHelper.add_asia_flag_supplier(dataset)

# dataFrameHelper.add_intercontitental(dataset)
# print(dataset['is_intercontinental'].value_counts())
# exit()

'''
print(dataset['country_distance'].value_counts())
print(dataset['country_distance'].value_counts())
exit()
'''

dataset['deviation_type'] = 0
dataFrameHelper.set_deviation_types(dataset)

dataset.drop('delivery_deviation_in_days', axis=1, inplace=True)
dataset.drop('late', axis=1, inplace=True)
dataset.drop('early', axis=1, inplace=True)

columns_to_encode = dataFrameHelper.getLabelCols()

for col_name_to_label in columns_to_encode:
    encoder_commodity = LabelEncoder()
    encoder_commodity.fit(dataset[col_name_to_label])
    dataset[col_name_to_label] = encoder_commodity.transform(dataset[col_name_to_label])
    joblib.dump(encoder_commodity, col_name_to_label + '.npy')


nr_of_cols = len(dataset.columns) - 1

X = dataset.iloc[:, 0:nr_of_cols].values
y = dataset.iloc[:, nr_of_cols].values

encoder = LabelEncoder()
encoder.fit(y)
y1 = encoder.transform(y)
Y = pd.get_dummies(y1).values
joblib.dump(encoder, 'y_encoder.npy')


deviation_type_value_counts = dataset['deviation_type'].value_counts(normalize=True)

on_time_weight = 100 - (deviation_type_value_counts['on_time'])*100
early_weight = 100 - (deviation_type_value_counts['early'])*100
late_weight = 100 - (deviation_type_value_counts['late'])*100

print(on_time_weight)
print(early_weight)
print(late_weight)

print(encoder.classes_)
print(type(encoder.classes_))
exit()

class_weight = {0: 1.,
                1: 4.,
                2: 1.}

print(class_weight)



class_weight = {}
class_weight = 1

model = Sequential()
model.add(Dense(nr_of_cols,input_shape=(nr_of_cols,),activation='relu', name='features'))
model.add(Dense(13,activation='relu'))
# model.add(Dropout(0.4))
model.add(Dense(7,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(Adam(lr=0.02),'categorical_crossentropy',metrics=['accuracy'])
model.fit(X, Y, epochs=20, batch_size=500, class_weight=class_weight)
model.summary()


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
