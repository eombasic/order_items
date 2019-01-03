import pandas as pd
from helpers import dataframe_helper

dataFrameHelper = dataframe_helper.DataframeHelper()

columns_to_read = dataFrameHelper.getusedcols()
data_source_url = dataFrameHelper.get_raw_data_file_location()

dataset = pd.read_csv(data_source_url, usecols=columns_to_read, converters={'delivery_address_zip': str})

print(dataset.info())

