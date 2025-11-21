import sys

# Assures Python 3.10 or above
assert sys.version_info >= (3, 10)

from packaging.version import Version
import sklearn
# Assures Scikit-Learn version 1.6.1 or above
assert Version(sklearn.__version__) >= Version("1.6.1")

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    # Looks for datasets/housing.tgz
    tarball_path = Path("datasets/housing.tgz")
    # If file is not found, creates directory and copies file from GitHub
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing_full = load_housing_data()

# Output first five rows of DataFrame, showing data structure.
print(housing_full.head())

# Output a quick description of data 
# Output shows 20433 values for total_bedroms, highlighting that 207 districts are missing this feature.
print(housing_full.info())

# Output from ocean_proximity revealed that it's data type is 'object' (text attribute)
# It was also revealed that all of the five districts had the same ocean_proximity values, suggesting categorical nature
# Outputs categories and their corresponding district counts
print(housing_full["ocean_proximity"].value_counts())

# Summary of numerical attributes
print(housing_full.describe())

import matplotlib.pyplot as plt

# Outputs numerical variable distribution, showing how many data points fall into each range of values (bins)
# Revealed that median income is NOT measured in USD, and was preprocessed with a cap at 15 and 0.5.
# Revealed that housing_median_age and median_house_value were capped.
# Revealed that many histograms are right skewed
housing_full.hist(bins=200, figsize=(12, 8))
# plt.show()

import numpy as np
# Function to randomly shuffle the dataset, split the dataset into train and test sets, using a random number generator 'rng' for reproducability.
# Example of homemade function
def shuffle_and_split_data(data, test_ratio, rng):
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

rng = np.random.default_rng(seed=42)

# train_set, test_set = shuffle_and_split_data(housing_full, 0.2, rng)


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing_full, test_size=0.2, random_state=42)

print(len(train_set)) # Outputs 16512
print(len(test_set)) # Outputs 4128

print(test_set["total_bedrooms"].isnull().sum())


# Example code for estimating the probability of a bad sample
from scipy.stats import binom

# sample_size = 1000
# ratio_female = 0.516
# # Probability of sample containing 489 or fewer females
# proba_too_small = binom(sample_size, ratio_female).cdf(490-1)
# # Probability of sample containing 541 or more females
# proba_too_large = 1 - binom(sample_size, ratio_female).cdf(540)
# print(proba_too_small + proba_too_large)

# *Scenario*: Discussion with experts who informed me that the median income is very a important
# attribute to predict median housing prices. The following code create income attribute with five income categories. 
housing_full["income_cat"] = pd.cut(housing_full["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
cat_counts = housing_full["income_cat"].value_counts().sort_index()
cat_counts.plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
# plt.show()


from sklearn.model_selection import StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing_full, housing_full["income_cat"]):
    strat_train_set_n = housing_full.iloc[train_index]
    strat_test_set_n = housing_full.iloc[test_index]
    strat_splits.append([strat_test_set_n, strat_test_set_n])
    

# strat_train_set, strat_test_set = strat_splits[0]

# Get a single split using the train_test_split function with the stratify argument
start_train_set, strat_test_set = train_test_split(housing_full, test_size=0.2, stratify=housing_full["income_cat"], random_state=42)

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))



## Code to compares income category proportions in full dataset, stratified sample, random sample. Showing stratified error percentage and random error percentage
# def income_cat_proportions(data):
#     return data["income_cat"].value_counts() / len(data)


# # Train–test split
# train_set, test_set = train_test_split(
#     housing_full, test_size=0.2, random_state=42
# )

# # Compare category proportions
# compare_props = pd.DataFrame({
#     "Overall %": income_cat_proportions(housing_full),
#     "Stratified %": income_cat_proportions(strat_test_set),
#     "Random %": income_cat_proportions(test_set)
# }).sort_index()

# compare_props.index.name = "Income Category"

# # Compute proportional errors
# compare_props["Strat. Error %"] = (
#     compare_props["Stratified %"] / compare_props["Overall %"] - 1
# )
# compare_props["Rand. Error %"] = (
#     compare_props["Random %"] / compare_props["Overall %"] - 1
# )

# # Display results as percentages
# print((compare_props * 100).round(2))

