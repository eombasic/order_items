class DataframeHelper:

    data_path = '../data/'

    def get_raw_data_file_location(self):
        return self.data_path + 'data_raw.csv'

    col_definitions = {
        "item_total_quantity":          {"use": True, "label_encode": False, "fillna": False},
        "item_costs":                   {"use": True, "label_encode": False, "fillna": False},
        "late":                         {"use": True, "label_encode": False, "fillna": False},
        "early":                        {"use": True, "label_encode": False, "fillna": False},
        "on_time":                      {"use": True, "label_encode": False, "fillna": False},
        "delivery_deviation_in_days":   {"use": True, "label_encode": False, "fillna": False},
        "item_commodity_id":            {"use": True, "label_encode": True, "fillna": True, "fillna_val": 0},
        "idx":                          {"use": True, "label_encode": True, "fillna": False},
        "unit_id":                      {"use": True, "label_encode": True, "fillna": False},
        "group_structure_id":           {"use": True, "label_encode": False, "fillna": True, "fillna_val": 0},
        "purchasing_organisation":      {"use": True, "label_encode": False, "fillna": False},
        "company_code":                 {"use": True, "label_encode": False, "fillna": False},
        "order_incoterm":               {"use": True, "label_encode": False, "fillna": False},
        "zterm_name":                   {"use": True, "label_encode": False, "fillna": True, "fillna_val": '-'},
        "order_type":                   {"use": True, "label_encode": False, "fillna": False},
        "item_created_by_supplier":     {"use": True, "label_encode": False, "fillna": False},
        "item_quantity_created_by_supplier":          {"use": True, "label_encode": False, "fillna": True, "fillna_val": '0'},
        "order_created_on_day":         {"use": True, "label_encode": False, "fillna": False},
        "deliver_on_day":               {"use": True, "label_encode": False, "fillna": False},
        "order_created_in_month":       {"use": True, "label_encode": False, "fillna": False},
        "deliver_in_month":             {"use": True, "label_encode": False, "fillna": False},
        "cluster_id":                   {"use": True, "label_encode": True, "fillna": True, "fillna_val": 0},
        "parent_cluster_id":            {"use": True, "label_encode": True, "fillna": False},
        "root_cluster":                 {"use": True, "label_encode": False, "fillna": False},
        "supplier_country":             {"use": True, "label_encode": False, "fillna": True, "fillna_val": '-'},
        "delivery_address_country":     {"use": True, "label_encode": False, "fillna": True, "fillna_val": '-'},
        "supplier_zip":                 {"use": False, "label_encode": True, "fillna": False},
        "delivery_address_zip":         {"use": True, "label_encode": True, "fillna": False},
        "same_country":                 {"use": True, "label_encode": False, "fillna": False},
        "same_zip":                     {"use": True, "label_encode": False, "fillna": False},
        "item_count":                   {"use": True, "label_encode": False, "fillna": False},
        "order_item_quantity_count":    {"use": True, "label_encode": False, "fillna": False},
        "requested_delivery_date":      {"use": True, "label_encode": False, "fillna": False},
        "delivery_date":                {"use": False, "label_encode": False, "fillna": False},
        "original_lifespan":            {"use": True, "label_encode": False, "fillna": False},
    }

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

        use_cols = []

        for col_name, col_definition in self.col_definitions.items():
            if col_definition["use"]:
                use_cols.append(col_name)

        return use_cols

    def getLabelCols(self):

        use_cols = []
        for col_name, col_definition in self.col_definitions.items():
            if col_definition["use"] and col_definition["label_encode"]:
                use_cols.append(col_name)

        return use_cols

    def printDataFrameInfo(self, df):
        print("Number of rows {} ".format(len(df.index)))
        print(df.describe())
        # print("dupplicated rows {}".format(df.duplicated()))

    def fillNaVals(self, df, verbose = True):
        for col_name, col_definition in self.col_definitions.items():
            if col_definition["fillna"]:
                if verbose:
                    print("Fill {} with {}".format(col_name, col_definition['fillna_val']))
                df[col_name] = df[col_name].fillna(col_definition['fillna_val'])

    def printFirstNa(self, df):
        nulls = df[df.isnull().any(axis=1)]
        print(nulls.iloc[0])

    def preprocess(self, df):
        # df[['early', 'on_time', 'late']] = df[['early', 'on_time', 'late']].fillna(value=0)
        # df.drop_duplicates(keep='first', inplace=True)
        #df.dropna(inplace=True)
        return True

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
        if row['late'] > 0:
            ret = 'late'
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

    def get_cat_columns(self):
        return ['order_type', 'purchasing_organisation', 'group_structure_id', 'company_code', 'order_incoterm', 'zterm_name',  'supplier_country', 'delivery_address_country']
