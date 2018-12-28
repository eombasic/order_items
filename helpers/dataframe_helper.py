class DataframeHelper:

    eu_countries = [
        "DE", "GB", "AU", "GB", "NL", "HU", "FR", "IT", "NL", "CH", "CZ", "MT", "LI", "ES", "SE", "BE", "PT", "RO", "SK"
    ]

    us_countries = {
        "US", "CN", "SG", "MX", "BR"
    }

    asia_countries = {
        "IN", "JP", "MY", "HK", "KR", "TH", "KP"
    }

    distances = {
        "DE": {
            "CN": 6746,
            "US": 7857,
            "GB": 1423,
            "IN": 6748,
            "FR": 1413,
            "MX": 9465,
        },
        "CN": {
            "DE": 6746
        },
        "US": {
            "DE": 7857,
        },
        "HU": {
            "DE": 992
        },
        "AT": {
            "DE": 650
        },
        "NL": {
            "DE": 1131
        },
        "CH": {
            "DE": 662
        },
        "JP": {
            "US": 10144
        },
        "SG": {
            "DE": 10119
        },
        "IT": {
            "DE": 1340
        },
        "CZ": {
            "DE": 506
        }
    }

    def getusedcols(self):
        return [
            'item_total_quantity',
            'item_costs',
            'delivery_deviation_in_days',
            #'item_commodity_id',
            #'idx',
            #'unit_id',
            'group_structure_id',
            'purchasing_organisation',
            'company_code',
            'order_created_on_day',
            'deliver_on_day',
            'cluster_id',
            'delivery_address_country',
            'delivery_address_zip',
            'supplier_country',
            # 'supplier_zip',
            'item_count',
            'order_item_quantity_count',
            'original_lifespan',
            'early',
            'late',
        ]

    def getLabelCols(self):
        return [
            #'item_commodity_id',
           # 'idx',
            #'unit_id',
            'group_structure_id',
            'purchasing_organisation',
            'company_code',
            'cluster_id',
             'delivery_address_country',
            'delivery_address_zip',
            'supplier_country',
            # 'supplier_zip',
        ]

    def printDataFrameInfo(self, df):
        print("Number of rows {} ".format(len(df.index)))
        print(df.describe())
        # print("dupplicated rows {}".format(df.duplicated()))

    def preprocess(self, df):
        df.drop_duplicates(keep='first', inplace=True)
        df.dropna(inplace=True)

    def move_sunday(self, row):
        if row['deliver_on_day'] == 1:
            ret = 7
        else:
            ret = row['deliver_on_day'] - 1
        return ret

    def move_df_sunday(self, df, col_name):
        df[col_name] = df.apply(self.move_sunday, axis=1)

    def set_state(self, row):
        ret = 'on_time'
        if row['delivery_deviation_in_days'] > 0:
            ret = 'late'
        if row['delivery_deviation_in_days'] < 0:
            ret = 'early'
        return ret

    def set_deviation_types(self, df):
        df['deviation_type'] = df.apply(self.set_state, axis=1)

    def get_country_distances(self, row):

        distance = 0
        supplier_country = row['supplier_country']
        customer_country = row['delivery_address_country']
        if supplier_country != customer_country:
            if self.distances.get(supplier_country) != None:
                check_distances = self.distances.get(supplier_country)
                if check_distances.get(customer_country) != None:
                    distance = check_distances.get(customer_country)

        return distance

    def add_country_distances(self, df):
        df['country_distance'] = df.apply(self.get_country_distances, axis=1)

    def is_eu_country(self, col_value):
        is_eu = 0
        if col_value in self.eu_countries:
            is_eu = 1
        return is_eu

    def add_eu_flag_customer(self, df):
        df['customer_from_eu'] = df['delivery_address_country'].apply(self.is_eu_country)

    def add_eu_flag_supplier(self, df):
        df['supplier_from_eu'] = df['supplier_country'].apply(self.is_eu_country)

    def is_us_country(self, col_value):
        is_us = 0
        if col_value in self.us_countries:
            is_us = 1
        return is_us

    def add_us_flag_customer(self, df):
        df['customer_from_us'] = df['delivery_address_country'].apply(self.is_us_country)

    def add_us_flag_supplier(self, df):
        df['supplier_from_us'] = df['supplier_country'].apply(self.is_us_country)

    def is_asia_country(self, col_value):
        is_asian = 0
        if col_value in self.asia_countries:
            is_asian = 1
        return is_asian

    def add_asia_flag_customer(self, df):
        df['customer_from_asia'] = df['delivery_address_country'].apply(self.is_asia_country)

    def add_asia_flag_supplier(self, df):
        df['supplier_from_asia'] = df['supplier_country'].apply(self.is_asia_country)

    def is_intercontitnental(self, row):
        inter_contitnental = 0
        if row['customer_from_eu'] == 1 and ( row['supplier_from_us'] == 1  or row['supplier_from_asia'] == 1)  :
            inter_contitnental = 1

        if row['customer_from_us'] == 1 and ( row['supplier_from_eu'] == 1  or row['supplier_from_asia'] == 1)  :
            inter_contitnental = 1

        if row['customer_from_asia'] == 1 and ( row['supplier_from_eu'] == 1  or row['supplier_from_us'] == 1)  :
            inter_contitnental = 1

        return inter_contitnental

    def add_intercontitental(self, df):
        df['is_intercontinental'] = df.apply(self.is_intercontitnental, axis=1)
