from s01_load_data import load_housing_data, add_income_cat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


housing_full = load_housing_data()
train_set, test_set = train_test_split(housing_full, test_size=0.2, random_state=42)

# print(len(train_set), len(test_set))
# print(test_set["total_bedrooms"].isnull().sum())


def create_stratified_test_set(housing_full: pd.DataFrame):
    # Create income attribute with five income categories
    housing_full = add_income_cat(housing_full)
    #housing_full["income_cat"].value_counts().sort_index().plot.bar()
    


    strat_train_set, strat_test_set = train_test_split(housing_full, test_size=0.2, stratify=housing_full["income_cat"], random_state=42)
    return strat_train_set, strat_test_set
    #print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))