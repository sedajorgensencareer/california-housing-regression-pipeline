"""Experiment with preprocessing techniques for the California housing data, including imputation, scaling, encoding, feature engineering, and prototype pipelines for numerical and categorical features."""


from matplotlib import pyplot as plt
from s01_load_data import load_housing_data, add_income_cat
from s03_create_test_set import create_stratified_test_set


# This file is just my rough preprocessing sandbox. 
# I used it to experiment with imputation, scaling, encoding, outliers, 
# and feature engineering while learning the data. 
# The final preprocessing now lives in the full pipeline file.




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


from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1,1))
housing_num_min_Max_scaled = min_max_scaler.fit_transform(housing_num)



from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# Shows transformation of feature to make it closer to a Gaussian distribution
# fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
# housing["population"].hist(ax=axs[0], bins=50)
# housing["population"].apply(np.log).hist(ax=axs[1], bins=50)
# axs[0].set_xlabel("Population")
# axs[1].set_xlabel("Log of population")
# axs[0].set_ylabel("Number of districts")



from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)



from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())

model.fit(housing[["median_income"]], housing_labels)
some_new_data = housing[["median_income"]].iloc[:5]
predictions = model.predict(some_new_data)


from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

rbf_transformer = FunctionTransformer(rbf_kernel,
                                      kw_args=dict(Y=[[35.]], gamma=0.1))

age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

# print(age_simil_35)

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        # No *args or **kwargs → keeps compatibility with get/set_params
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        # Create and fit the KMeans model
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self

    def transform(self, X):
        # Compute Gaussian RBF similarity between samples and cluster centers
        return rbf_kernel(
            X,
            self.kmeans_.cluster_centers_,
            gamma=self.gamma
        )

    def get_feature_names_out(self, names=None):
        # Nice readable feature names for pipelines / DataFrames
        return [
            f"Cluster {i} similarity"
            for i in range(self.n_clusters)
        ]

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]], sample_weight=housing_labels)

# Outputs the first three rows (district/sample) and their corresponding similarities to each K-Means cluster center
# print(similarities[:3].round(2))


# The plot shows California’s districts colored by their RBF similarity to the nearest K-Means cluster center, 
# with larger dots for more populated areas and black X’s marking the learned geographic cluster centers.
housing_renamed = housing.rename(columns={
    "latitude": "Latitude", "longitude": "Longitude",
    "population": "Population",
    "median_house_value": "Median house value (ᴜsᴅ)"})
housing_renamed["Max cluster similarity"] = similarities.max(axis=1)

housing_renamed.plot(kind="scatter", x="Longitude", y="Latitude", grid=True,
                     s=housing_renamed["Population"] / 100, label="Population",
                     c="Max cluster similarity",
                     cmap="jet", colorbar=True,
                     legend=True, sharex=False, figsize=(10, 7))
plt.plot(cluster_simil.kmeans_.cluster_centers_[:, 1],
         cluster_simil.kmeans_.cluster_centers_[:, 0],
         linestyle="", color="black", marker="X", markersize=20,
         label="Cluster centers")
plt.legend(loc="upper right")

plt.show()

from sklearn.pipeline import Pipeline, make_pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])

# Creates a pipeline without the need for naming.
# num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


from sklearn import set_config
set_config(display="diagram")

# print(num_pipeline)

housing_num_prepared = num_pipeline.fit_transform(housing_num)
# print("Housing num prepared:\n", housing_num_prepared[:2].round(2))

df_housing_num_prepared = pd.DataFrame(housing_num_prepared, 
                                       columns=num_pipeline.get_feature_names_out(),
                                       index=housing_num.index)
# print("Housing num prepared DataFrame:\n", df_housing_num_prepared.head(2))

from sklearn.compose import ColumnTransformer

num_attributes = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]

cat_attributes = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

# preprocessing = ColumnTransformer([
#     ("num", num_pipeline, num_attributes),
#     ("cat", cat_pipeline, cat_attributes)
# ])

from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object))
)

housing_prepared = preprocessing.fit_transform(housing)