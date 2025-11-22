import matplotlib.pyplot as plt
from s01_load_data import load_housing_data, add_income_cat
from s03_create_test_set import create_stratified_test_set

strat_train_set, strat_test_set = create_stratified_test_set(load_housing_data())

housing = strat_train_set.copy()

# Plot of latitude and longitude
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)

# Plot of latitude and longitude with alpha of 0.2
# Represents latitude and longitude, while showing densit of data point.
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)


# Plot of housing prices across latitude and longitude.
# Circle size represents population (option S)
# Color represents the price (option C)
# The plot shows that house prices are very much related to location (e.g., population and ocean proximity)
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, 
             s=housing["population"]/ 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10,7))


# Generates correlation coefficient between every pair of numerical attributes
corr_matrix = housing.corr(numeric_only=True)
# Outputs how much each attribute correlates to median house value
print(corr_matrix["median_house_value"].sort_values(ascending=False))


from pandas.plotting import scatter_matrix

