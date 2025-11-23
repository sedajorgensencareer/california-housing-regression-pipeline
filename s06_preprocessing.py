from s01_load_data import load_housing_data, add_income_cat
from s03_create_test_set import create_stratified_test_set

strat_train_set, strat_test_set = create_stratified_test_set(load_housing_data())

# Remove median_house_value (target feature) from housing
housing = strat_train_set.drop("median_house_value", axis = 1)
# Extracts median_house_value as a target vector
housing_labels = strat_train_set["median_house_value"].copy()

# Drops only the rows where total_bedrooms is missing
# housing.dropna(subset=["total_bedrooms"], inplace=True)

# Drops the entire total_bedrooms column
# housing.drop("total_bedrooms", axis=1, inplace=True)

# Impute (fill in) missing values with median
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)

# Finds rows with missing values and shows first five
# null_rows_idx = housing.isnull().any(axis=1)
# print(housing.loc[null_rows_idx].head())

from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

imputer = SimpleImputer(strategy="median")

# Creates a copy of the data containing only the numerical attributes
housing_num = housing.select_dtypes(include=[np.number])
# null_rows_idx = housing.isnull().any(axis=1)
# print(housing_num.loc[null_rows_idx].head())
imputer.fit(housing_num)

# print(imputer.statistics_)
# print(housing_num.median().values)

X = imputer.transform(housing_num)
# print(imputer.feature_names_in_)

# Converted inputed NumPy array X back to a DataFrame object with the same column names and indices as the original housing_num DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

# Makes it so transformers output Pandas DataFrame and not NumPy array
# from sklearn import set_config
# set_config(transform_output="pandas")

# print(housing_tr.loc[null_rows_idx].head())

# Detects outliers in DataFrame
from sklearn.ensemble import IsolationForest
isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(X)

# Code to drop outliers 
# housing = housing.iloc[outerlier_pred == 1]
# housing_labels = housing_labels.iloc[outlier_pred == 1]

housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(8))


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

print(housing_cat_encoded[:8])
print(ordinal_encoder.categories_)

# Encodes ocean_proximity values with One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)


