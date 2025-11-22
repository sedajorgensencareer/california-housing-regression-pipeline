import pandas as pd
from sklearn.model_selection import train_test_split
from s01_load_data import load_housing_data, add_income_cat

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

housing_full = add_income_cat(load_housing_data())

_, test_set = train_test_split(housing_full, test_size=0.2, random_state=42)
_, strat_test_set = train_test_split(housing_full, test_size=0.2, stratify=housing_full["income_cat"], random_state=42)

compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing_full),
    "Stratified %": income_cat_proportions(strat_test_set),
    "Random %": income_cat_proportions(test_set),
}).sort_index()

compare_props["Strat. Error %"] = compare_props["Stratified %"] / compare_props["Overall %"] - 1
compare_props["Rand. Error %"] = compare_props["Random %"] / compare_props["Overall %"] - 1

print((compare_props * 100).round(2))
