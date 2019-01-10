import pandas as pd
from helpers import dataframe_helper

data_source_url = '../data/data.csv'

dataset = pd.read_csv(data_source_url)
print(dataset.info())

dataFrameHelper = dataframe_helper.DataframeHelper()

value_count_cols = ['item_commodity_id', 'idx', 'unit_id', 'group_structure_id', 'purchasing_organisation', 'company_code', 'order_incoterm', 'zterm_name', 'order_type', 'cluster_id', 'parent_cluster_id', 'supplier_country', 'delivery_address_country', 'delivery_address_zip']

print('=====================')
print('value counts :')
for value_count_col in value_count_cols:
    val_count = dataset[value_count_col].nunique()
    print(" {} : {}".format(value_count_col, val_count))

print('=====================')


