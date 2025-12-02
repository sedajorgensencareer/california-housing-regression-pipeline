from s01_load_data import load_housing_data
import matplotlib.pyplot as plt

"""Explore and summarize the raw California Housing dataset with basic info, statistics, and visual inspection."""

housing_full = load_housing_data()


print(housing_full.head())
print(housing_full.info())
print(housing_full["ocean_proximity"].value_counts())
print(housing_full.describe())

housing_full.hist(bins=50, figsize=(12, 8))
plt.show()