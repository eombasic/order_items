import importlib
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from keras.models import model_from_json
from keras import backend as keras_Backend

class DeliveryPredictor:

    dataframeHelper = False

    basePath = False

    def __del__(self):
        keras_Backend.clear_session()

    def getDataframeHelper(self):
        if not self.dataframeHelper:
            dataframeModule = importlib.import_module('helpers.dataframe_helper')
            self.dataframeHelper = dataframeModule.DataframeHelper()
        return self.dataframeHelper;

    def info(self):
        return 'customer delivery predictor'

    def getNpyPath(self):
        return self.basePath + '/classificator/'

    def getModelPath(self):
        return self.basePath + '/classificator/'

    def setBasePath(self, path):
        self.basePath = path

    def predict(self, data):

        dataset = pd.DataFrame(data, index=[0])

        dataset.drop_duplicates(keep='first', inplace=True)

        dataset.dropna(subset=['requested_delivery_date'], inplace=True)
        dataset.drop('requested_delivery_date', axis=1, inplace=True)

        dataFrameHelper = self.getDataframeHelper()

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

        npyPath = self.getNpyPath()
        cat_dummies = joblib.load(npyPath + 'cat_dummies.npy')

        # Remove additional columns
        for col in dataset.columns:
            if ("__" in col) and (col.split("__")[0] in cat_columns) and col not in cat_dummies:
                # print("Removing additional feature {}".format(col))
                dataset.drop(col, axis=1, inplace=True)

        # Add missing columns
        for col in cat_dummies:
            if col not in dataset.columns:
                # print("Adding missing feature {}".format(col))
                dataset[col] = 0

        processed_columns = joblib.load(npyPath + 'processed_columns.npy')

        dataset['deviation_type'] = 0
        dataFrameHelper.set_deviation_types(dataset)
        dataset.drop('delivery_deviation_in_days', axis=1, inplace=True)
        dataset.drop('late', axis=1, inplace=True)
        dataset.drop('early', axis=1, inplace=True)
        dataset.drop('on_time', axis=1, inplace=True)

        columns_to_encode = dataFrameHelper.getLabelCols()

        for col_name_to_label in columns_to_encode:
            encoder_1 = joblib.load(npyPath + col_name_to_label + '.npy')
            dataset[col_name_to_label] = encoder_1.transform(dataset[col_name_to_label])

        dataset = dataset[processed_columns]

        nr_of_cols = len(dataset.columns) - 1

        X = dataset.iloc[:, 0:nr_of_cols].values

        encoder = joblib.load(npyPath + 'y_encoder.npy')
        encoder_classes = encoder.classes_

        modelPath = self.getModelPath()
        # load json and create model
        json_file = open(modelPath + 'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(modelPath + "model.h5")

        predictions = loaded_model.predict(X)
        print('predictions')
        print(predictions)
        prediction_label = predictions[0]

        print('prediction label')
        print(prediction_label)
        max_val_key = np.argmax(prediction_label, axis=0)

        print('max val key')
        print(max_val_key)

        prediction_label = encoder_classes[max_val_key]

        print('prediction label')
        print(prediction_label)

        lst = [dataset]

        del dataset
        del dataFrameHelper
        del lst
        del loaded_model
        del loaded_model_json
        del X

        return prediction_label
