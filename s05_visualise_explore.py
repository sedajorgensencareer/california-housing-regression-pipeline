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


# Scatter plot - plots every numerical attribute against every other numerical attribute
# Attributes chosen include top 4 attributes from correlation matrix
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))


# Median income versus median house value scatter plot
# Clearly shows the price cap at 500,000
# Less obvious straight lines at 450,000 and around 350,000, possibly around 280,000, and maybe a few more below that
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha = 0.1, grid=True)



# Creates new attribute rooms_per_house reflecting the average rooms per household
housing["rooms_per_house"]= housing["total_rooms"] / housing["households"]
# Creates new attribute bedrooms_ratio reflecting what fraction of all rooms are bedrooms
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
# Creates new attribute people_per_house reflecting how many people on average live in each household
housing["people_per_house"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

plt.show()